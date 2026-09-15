"""Run an immutable program with host-owned budgets and grading."""

import asyncio
import json
import time
from pathlib import Path

from simple_scibuddy.artifacts import digest, write_json
from simple_scibuddy.environments.runtime import Container, ControllerFailure, ExecutionMemoryBudget
from simple_scibuddy.harness.context import bound_tool_history, clip_observation
from simple_scibuddy.harness.messages import classify_messages, model_messages
from simple_scibuddy.inference.client import ContextBudget


async def run_episode(
    task,
    harness,
    policy,
    folder,
    system,
    *,
    actions=9,
    seconds=900,
    tool_seconds=120,
    container_factory=Container,
    runtime_image=None,
    baked_lake=False,
    observation_chars=8000,
    tool_response_tokens=None,
    tool_history_tokens=8192,
    feedback=None,
    user_turns=0,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    controller = container_factory(runtime_image or task.environment["image"], folder / "controller", "controller")
    runtime = container_factory(
        runtime_image or task.environment["image"],
        folder / "execution",
        "execution",
        lake=None if baked_lake else str(task.lake_path),
    )
    if observation_chars < 256:
        raise ValueError("Observation budget must be at least 256 characters")
    tools, submissions, replies = [], [], []
    candidate_error = None
    execution_failure = None
    timings = {"setup": 0.0, "model": 0.0, "tools": 0.0, "collect": 0.0}
    reward, stop, started = 0.0, "program_ended", time.monotonic()
    files = [p.name for p in sorted(task.assets.iterdir())]
    public = {
        "prompt": task.public["prompt"],
        "family": task.public["family"],
        "subtask": task.public.get("subtask"),
        "files": files,
        "system": system + "\nPublic files: " + ", ".join("/workspace/assets/" + f for f in files),
        "budgets": {
            "model_calls": actions,
            "tool_calls": actions,
            "completion_tokens": policy.max_tokens,
            "context_tokens": policy.context,
            "seconds": seconds,
            "tool_seconds": tool_seconds,
            "user_turns": user_turns,
        },
    }
    if tool_response_tokens is not None:
        public["budgets"].update(tool_response_tokens=tool_response_tokens,
                                 tool_history_tokens=tool_history_tokens)
    try:
        await runtime.start(assets=task.assets)
        await controller.start(harness=harness)
        timings["setup"] = time.monotonic() - started
        try:
            async with asyncio.timeout(seconds):
                await controller.send(public)
                while True:
                    event = await controller.receive()
                    with (folder / "events.jsonl").open("a") as stream:
                        stream.write(json.dumps(event) + "\n")
                    if not isinstance(event, dict):
                        raise ControllerFailure("Harness events must be JSON objects")
                    op = event.get("op")
                    if op == "generate":
                        if len(policy.calls) >= actions:
                            stop = "model_call_budget"
                            break
                        try:
                            tick = time.monotonic()
                            messages, provenance = classify_messages(
                                event.get('messages'), public['prompt'], [r['reply'] for r in replies], len(tools))
                            changes = []
                            if tool_response_tokens is not None:
                                messages, changes = bound_tool_history(
                                    messages, policy.tokenizer, per_message=tool_response_tokens,
                                    total=tool_history_tokens,
                                    input_limit=policy.context - policy.max_tokens,
                                )
                            call = await policy.generate(model_messages(messages))
                            call['input_provenance'] = provenance
                            call["tool_history_clipping"] = changes
                            timings["model"] += time.monotonic() - tick
                        except ContextBudget:
                            stop = "context_budget"
                            break
                        write_json(folder / f"call-{len(policy.calls):03d}.json", call)
                        if call["finish_reason"] == "length":
                            stop = "output_truncated"
                            break
                        await controller.send(call["text"])
                    elif op == "execute":
                        if not isinstance(event.get("code"), str):
                            raise ControllerFailure("execute requires Python source text")
                        if len(tools) >= actions:
                            stop = "tool_call_budget"
                            break
                        try:
                            tick = time.monotonic()
                            observation = await runtime.execute(event["code"], timeout=tool_seconds)
                            tool_elapsed = time.monotonic() - tick
                            timings["tools"] += tool_elapsed
                        except TimeoutError:
                            timings["tools"] += time.monotonic() - tick
                            stop = "tool_timeout"
                            break
                        except ExecutionMemoryBudget as exc:
                            elapsed = time.monotonic() - tick
                            timings['tools'] += elapsed
                            observation = {'stdout': '', 'error': str(exc), 'execution_memory_budget': True}
                            raw_path = folder / f'tool-{len(tools) + 1:03d}.json'
                            write_json(raw_path, observation)
                            tools.append({'call_index': len(policy.calls) - 1, 'code': event['code'],
                                          'elapsed_seconds': elapsed, 'observation': observation,
                                          'raw_observation': raw_path.name, 'clipping': None})
                            execution_failure = exc.state
                            stop = 'execution_memory_budget'
                            break
                        raw_path = folder / f"tool-{len(tools) + 1:03d}.json"
                        write_json(raw_path, observation)
                        clipping = None
                        if tool_response_tokens is not None:
                            observation, clipping = clip_observation(
                                observation, tool_response_tokens, policy.tokenizer
                            )
                        else:
                            observation = bounded_observation(observation, observation_chars)
                        tools.append(
                            {
                                "call_index": len(policy.calls) - 1,
                                "code": event["code"],
                                "elapsed_seconds": tool_elapsed,
                                "observation": observation,
                                "raw_observation": raw_path.name,
                                "clipping": clipping,
                            }
                        )
                        await controller.send(observation)
                    elif op == "submit":
                        if not isinstance(event.get("text"), str):
                            raise ControllerFailure("submit requires answer text")
                        outcome = task.verify(event["text"])
                        if not submissions:
                            reward = outcome["score"]
                        submissions.append(
                            {"call_index": len(policy.calls) - 1, "text": event["text"], "outcome": outcome}
                        )
                        if feedback is not None and len(replies) < user_turns:
                            # The reviewer gets public content only, never reference answers or scores.
                            reply = await feedback(task.public, policy.calls, tools, event["text"])
                            replies.append(reply)
                            if reply.get("reply") and not reply.get("accept", False):
                                await controller.send({"done": False, "reply": reply["reply"]})
                                continue
                        stop = "first_answer_evaluation"
                        break
                    elif op == "end":
                        if len(tools) >= actions:
                            stop = "tool_call_budget"
                        elif len(policy.calls) >= actions:
                            stop = "model_call_budget"
                        elif not submissions:
                            stop = 'no_submission'
                        break
                    else:
                        raise ControllerFailure("Harness protocol failure: " + str(event))
        except ControllerFailure as exc:
            candidate_error, stop, reward = str(exc), "candidate_error", 0.0
        except TimeoutError:
            stop = "time_budget"
        if not policy.calls:
            candidate_error, stop, reward = "Harness produced no trainable model calls", "candidate_error", 0.0
        tick = time.monotonic()
        await runtime.collect()
        timings["collect"] = time.monotonic() - tick
    finally:
        cleanup = await asyncio.gather(controller.close(), runtime.close(), return_exceptions=True)
        failures = [str(x) for x in cleanup if isinstance(x, BaseException)]
        write_json(folder / "cleanup.json", {"errors": failures})
        if failures:
            raise RuntimeError("Owned container cleanup failed: " + "; ".join(failures))
    episode = {
        "task_id": task.public["id"],
        "family": task.public["family"],
        "source_group": task.public["source_group"],
        "split": task.public["split"],
        "harness_sha256": digest(Path(harness).read_bytes()),
        "reward": reward,
        "calls": policy.calls,
        "tool_calls": tools,
        "submissions": submissions,
        "user_feedback": replies,
        "final_reward": submissions[-1]["outcome"]["score"] if submissions else 0.0,
        "stop_reason": stop,
        "elapsed_seconds": time.monotonic() - started,
        "timings": timings,
    }
    if candidate_error is not None:
        episode["error"] = candidate_error
    if execution_failure is not None:
        episode['execution_failure'] = execution_failure
    write_json(folder / "episode.json", episode)
    write_json(
        folder / "public.json",
        {
            k: episode[k]
            for k in ("task_id", "family", "harness_sha256", "calls", "tool_calls", "stop_reason")
        },
    )
    return episode


def bounded_observation(observation, limit=8000):
    """Bound model-visible text; raw results are kept separately in the episode."""
    result = dict(observation)
    notice = "\n[TRUNCATED: print a smaller slice or summary.]"
    for key, budget in (("stdout", limit * 3 // 4), ("error", limit // 4)):
        value = result.get(key)
        if isinstance(value, str) and len(value) > budget:
            result[key] = value[:budget - len(notice)] + notice
    return result
