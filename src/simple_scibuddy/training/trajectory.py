"""Pure conversion: one model call per row, one reward per episode."""


def to_generator_output(episodes, trajectory_ids):
    if len(episodes) != len(trajectory_ids):
        raise ValueError("Episode/identity mismatch")
    keys = (
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_logprobs",
        "trajectory_ids",
        "is_last_step",
        "env_metrics",
    )
    result = {k: [] for k in keys}
    for episode, identity in zip(episodes, trajectory_ids):
        if not episode["calls"]:
            if episode.get("stop_reason") == "candidate_error" and episode["reward"] == 0:
                # No policy action means no gradient row; retain failure in episode metrics.
                continue
            raise ValueError("Empty model trajectory")
        for i, call in enumerate(episode["calls"]):
            ids, logprobs = call["completion_ids"], call["old_logprobs"]
            if not ids or len(ids) != len(logprobs):
                raise ValueError("Missing or misaligned completion tokens/logprobs")
            last = i == len(episode["calls"]) - 1
            result["prompt_token_ids"].append(call["prompt_ids"])
            result["response_ids"].append(ids)
            result["loss_masks"].append([1] * len(ids))
            result["rollout_logprobs"].append(logprobs)
            result["rewards"].append(([0.0] * (len(ids) - 1)) + [episode["reward"] if last else 0.0])
            result["trajectory_ids"].append(identity)
            result["is_last_step"].append(last)
            result["stop_reasons"].append(call["finish_reason"])
            result["env_metrics"].append(
                {
                    "family": episode["family"],
                    "stop_reason": episode["stop_reason"],
                    "tool_calls": len(episode["tool_calls"]),
                    "elapsed_seconds": episode["elapsed_seconds"],
                }
            )
    if not result["response_ids"]:
        raise ValueError("Entire rollout batch has no trainable model actions")
    count = len(episodes)
    metrics = {
        "sciencebuddy/accuracy": sum(e["reward"] for e in episodes) / count,
        "sciencebuddy/tool_calls": sum(len(e["tool_calls"]) for e in episodes) / count,
        "sciencebuddy/model_calls": sum(len(e["calls"]) for e in episodes) / count,
        "sciencebuddy/episode_seconds": sum(e["elapsed_seconds"] for e in episodes) / count,
        "sciencebuddy/generated_tokens": sum(len(c["completion_ids"]) for e in episodes for c in e["calls"])
        / count,
        "sciencebuddy/tool_errors": sum(
            bool(t["observation"].get("error")) for e in episodes for t in e["tool_calls"]
        )
        / count,
    }
    for family in sorted({e["family"] for e in episodes}):
        members = [e for e in episodes if e["family"] == family]
        metrics[f"sciencebuddy/{family}/accuracy"] = sum(e["reward"] for e in members) / len(members)
    for reason in sorted(set(LIMIT_REASONS.values()) | {"first_answer_evaluation", "program_ended"}):
        metrics[f"sciencebuddy/stop/{reason}"] = sum(e["stop_reason"] == reason for e in episodes) / count
    groups = {}
    for episode, identity in zip(episodes, trajectory_ids):
        groups.setdefault(getattr(identity, "instance_id", str(identity)), []).append(episode["reward"])
    metrics["sciencebuddy/mixed_reward_groups"] = sum(
        len(set(values)) > 1 for values in groups.values()
    ) / len(groups)
    # SkyRL infers aggregation from key names when concatenating rollout batches.
    result["rollout_metrics"] = {key + "_mean": value for key, value in metrics.items()}
    result["rollout_metrics"].update(outcome_metrics(episodes, trajectory_ids))
    return result


LIMIT_REASONS = {
    "tool": "tool_call_budget",
    "model_calls": "model_call_budget",
    "context": "context_budget",
    "response": "output_truncated",
    "tool_timeout": "tool_timeout",
    "wall_time": "time_budget",
    "execution_memory": "execution_memory_budget",
}


def outcome_metrics(episodes, identities):
    """Percentages use attempts or task groups explicitly, never model-call rows."""
    if len(episodes) != len(identities):
        raise ValueError("Episode/identity mismatch")
    if not episodes:
        return {}
    groups = {}
    for episode, identity in zip(episodes, identities):
        groups.setdefault(getattr(identity, "instance_id", str(identity)), []).append(episode)
    count, tasks = len(episodes), len(groups)
    metrics = {"sciencebuddy/episodes/count": count, "sciencebuddy/tasks/count": tasks}
    for name, reason in LIMIT_REASONS.items():
        attempts_hit = sum(e["stop_reason"] == reason for e in episodes)
        tasks_hit = sum(any(e["stop_reason"] == reason for e in group) for group in groups.values())
        for scope, hits, total in (("episodes", attempts_hit, count), ("tasks", tasks_hit, tasks)):
            prefix = f"sciencebuddy/{scope}/limits/{name}"
            metrics[prefix + "_count"] = hits
            metrics[prefix + "_pct_mean"] = 100 * hits / total
    solved = [sum(e["reward"] == 1 for e in group) for group in groups.values()]
    all_count = sum(n == len(group) for n, group in zip(solved, groups.values()))
    none_count = sum(n == 0 for n in solved)
    for name, hits in (("solve_all", all_count), ("solve_none", none_count),
                       ("solve_mixed", tasks - all_count - none_count)):
        metrics[f"sciencebuddy/tasks/{name}_count"] = hits
        metrics[f"sciencebuddy/tasks/{name}_pct_mean"] = 100 * hits / tasks
    metrics["sciencebuddy/episodes/solved_pct_mean"] = 100 * sum(solved) / count
    return metrics
