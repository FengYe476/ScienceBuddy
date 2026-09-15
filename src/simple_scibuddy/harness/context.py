"""Host-owned observation and tool-history budgets."""

import copy

# Fixed observation processing shared with the host/RL adapter.
MARKER = "\n[Tool output truncated; showing head and tail. Re-run a filtered query for omitted details.]\n"


def token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def clip_text(text, limit, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= limit:
        return text
    marker = MARKER if limit >= 128 else "[Tool output omitted]"
    room = max(0, limit - token_count(marker, tokenizer) - 8)
    # Decode and re-tokenize: BPE boundaries may change after joining fragments.
    while room >= 0:
        head = room * 3 // 4
        tail = room - head
        value = (tokenizer.decode(tokens[:head], skip_special_tokens=False) if head else "") + marker
        value += tokenizer.decode(tokens[-tail:], skip_special_tokens=False) if tail else ""
        size = token_count(value, tokenizer)
        if size <= limit:
            return value
        room -= max(1, size - limit)
    return ""


def clip_observation(observation, limit, tokenizer):
    """Keep the dict contract and error status; cap its serialized visible size."""
    assert limit >= 256
    visible = copy.deepcopy(observation)
    before = token_count(str(visible), tokenizer)
    if before <= limit:
        return visible, None
    # Standard runtime observations contain stdout, error and optional status flags.
    for key in ("error", "stdout"):
        if isinstance(visible.get(key), str):
            cap = min(512, limit // 3) if key == "error" else limit
            visible[key] = clip_text(visible[key], cap, tokenizer)
    visible["output_truncated"] = True
    while token_count(str(visible), tokenizer) > limit:
        strings = [k for k, v in visible.items() if isinstance(v, str)]
        key = max(strings, key=lambda k: token_count(visible[k], tokenizer))
        current = token_count(visible[key], tokenizer)
        excess = token_count(str(visible), tokenizer) - limit
        visible[key] = clip_text(visible[key], max(0, current - excess - 8), tokenizer)
    return visible, {
        "original_tokens": before,
        "visible_tokens": token_count(str(visible), tokenizer),
        "limit": limit,
    }


def bound_tool_history(messages, tokenizer, *, per_message=2048, total=8192, input_limit=27648):
    """Preserve instructions/questions/drafts; shrink oldest tool messages first."""
    output = copy.deepcopy(messages)
    indexes = [
        i
        for i, m in enumerate(output)
        if m.get("source") == "tool"
        or (
            m.get("source") is None
            and m["role"] in {"user", "tool"}
            and isinstance(m.get("content"), str)
            and (m["role"] == "tool" or m["content"].startswith("Tool observation"))
        )
    ]
    if not indexes:
        return output, []
    original = {i: token_count(output[i]["content"], tokenizer) for i in indexes}
    without = copy.deepcopy(output)
    for i in indexes:
        without[i]["content"] = ""
    visible = [{"role": m["role"], "content": m["content"]} for m in without]
    base = len(
        tokenizer.apply_chat_template(
            visible, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
    )
    remaining = max(0, min(total, input_limit - base - 64))
    # Leave a small notice for each old observation while favouring recent results.
    for position, i in enumerate(reversed(indexes)):
        older = len(indexes) - position - 1
        budget = max(0, min(per_message, remaining - 32 * older))
        output[i]["content"] = clip_text(output[i]["content"], budget, tokenizer)
        remaining -= token_count(output[i]["content"], tokenizer)
    changes = [
        {
            "message_index": i,
            "original_tokens": original[i],
            "visible_tokens": token_count(output[i]["content"], tokenizer),
        }
        for i in indexes
        if output[i]["content"] != messages[i]["content"]
    ]
    return output, changes
