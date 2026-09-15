"""Evidence-driven online harness evolution, implemented locally.

Public evidence and execution preflight helpers. Test results never enter proposals.
"""

import copy

from simple_scibuddy.coevolve.context import execution_fields, validate_assessment

RUNTIME_FAILURES = {
    "output_truncated",
    "context_budget",
    "model_call_budget",
    "tool_call_budget",
    "tool_timeout",
    "time_budget",
    "candidate_error",
    "execution_memory_budget",
    "no_submission",
}


def public_evidence(episode):
    feedback = [
        {"reply": f["reply"]}
        for f in episode.get("user_feedback", [])
        if f.get("valid") and f.get("feedback_type") == "correction"
    ]
    acceptance = [
        {"reply": f["reply"]}
        for f in episode.get("user_feedback", [])
        if f.get("valid") and f.get("feedback_type") == "acceptance"
    ]
    annotations = []
    # Check field addresses against the original submission prefix too. Later tool calls
    # must not legitimize a field unavailable to the earlier assessment. Quote text is unchecked.
    for i, item in enumerate(episode.get("user_feedback", [])):
        if not item.get("valid") or not item.get("assessment"):
            continue
        submissions = episode.get("submissions", [])
        if i >= len(submissions):
            continue
        submission = submissions[i]
        call_index = submission.get("call_index", len(episode["calls"]) - 1)
        tools = [t for t in episode.get("tool_calls", []) if t.get("call_index", 0) <= call_index]
        fields = execution_fields(
            episode.get("public_task", {}).get("prompt", ""), episode["calls"][: call_index + 1], tools
        )
        fields.update(answer=submission["text"], reply=item["reply"])
        valid, _ = validate_assessment(item["assessment"], fields, item["reply"], item["feedback_type"])
        if valid:
            assessment = copy.deepcopy(item["assessment"])
            aliases = {"answer": f"submissions/{i}/text", "reply": f"user_feedback/{i}/reply"}
            for observation in assessment["observations"]:
                observation["field"] = aliases.get(observation["field"], observation["field"])
            assessment["related_fields"] = [
                aliases.get(field, field) for field in assessment["related_fields"]
            ]
            annotations.append(
                {
                    "feedback_index": i,
                    "submission_call_index": call_index,
                    "reply": item["reply"],
                    "submission_text": submission["text"],
                    "assessment": assessment,
                    "provenance": "Field addresses checked; quote text is not matched. PRM interpretations are hypotheses, not verified causes.",
                }
            )
    diagnostics = []
    if episode["stop_reason"] in RUNTIME_FAILURES:
        diagnostics.append(episode["stop_reason"])
    for tool in episode.get("tool_calls", []):
        error = tool["observation"].get("error")
        if error and not tool["observation"].get("infrastructure_error"):
            diagnostics.append(str(error)[:600])
    return {
        "task_id": episode["task_id"],
        "question": episode.get("public_task", {}).get("prompt", ""),
        "subtask": episode.get("public_task", {}).get("subtask"),
        "harness_sha256": episode.get("harness_sha256"),
        "prm_annotations": annotations,
        "execution": {
            "stop_reason": episode["stop_reason"],
            "model_calls": len(episode["calls"]),
            "tool_calls": len(episode.get("tool_calls", [])),
            "response_tokens": sum(len(c.get("completion_ids", [])) for c in episode["calls"]),
            "submitted": bool(episode.get("submissions")),
        },
        "acceptance": acceptance,
        "tools": [
            {"code": t.get("code", ""), "observation": t["observation"], "call_index": t.get("call_index")}
            for t in episode.get("tool_calls", [])
        ],
        "feedback": feedback,
        "diagnostics": diagnostics,
        "trace": [
            {
                "response": c["text"],
                "input_messages": [
                    {"role": m["role"], "content": m["content"]} for m in c.get("messages", [])
                ],
                "input_provenance": c.get("input_provenance", []),
                "tool_history_clipping": c.get("tool_history_clipping", []),
            }
            for c in episode["calls"]
        ],
    }


def evidence_preview(record):
    """Show causal action/observation pairs, not detached error messages."""
    pairs, seen = [], set()
    tools = record.get("tools", [])
    for index, tool in enumerate(tools):
        observation = tool["observation"]
        error = observation.get("error")
        if not error or observation.get("infrastructure_error"):
            continue
        signature = (str(error), tool.get("code", ""))
        if signature in seen:
            continue
        seen.add(signature)
        pairs.append(
            {
                "tool_index": index,
                "code": tool.get("code", "")[:1000],
                "observation": str(observation)[:500],
                "truncated": len(tool.get("code", "")) > 1000 or len(str(observation)) > 500,
            }
        )
        if len(pairs) == 1:
            break
    # A generic correction does not identify the faulty reasoning. Show the
    # initial attempt as well as the eventual response, including successful
    # tool observations that can support (or contradict) the answer.
    trace = record.get("trace", [])
    initial = str(trace[0].get("response", "")) if trace else ""
    final = str(trace[-1].get("response", "")) if trace else ""
    observed_action = None
    if tools and not pairs:
        tool = tools[-1]
        observed_action = {
            "tool_index": len(tools) - 1,
            "code": tool.get("code", "")[:1000],
            "observation": str(tool["observation"])[:500],
            "truncated": len(tool.get("code", "")) > 1000 or len(str(tool["observation"])) > 500,
        }
    return {
        "task_id": record["task_id"],
        "question_preview": record.get("question", "")[:750],
        "subtask": record.get("subtask"),
        "failed_actions": pairs,
        "execution": record.get("execution", {}),
        "feedback": record.get("feedback", []),
        "observed_action": observed_action,
        "acceptance": record.get("acceptance", []),
        "first_response": initial[:600],
        "last_response": final[-600:],
        "response_preview_truncated": len(initial) > 600 or len(final) > 600,
    }


def preflight_passed(episode):
    return (
        not episode.get("error")
        and episode["stop_reason"] == "first_answer_evaluation"
        and bool(episode["submissions"])
        and episode["submissions"][-1]["outcome"].get("answer_format_valid", False)
    )
