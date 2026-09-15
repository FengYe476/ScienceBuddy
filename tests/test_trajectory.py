import pytest

from simple_scibuddy.training.trajectory import to_generator_output


def call(prompt, ids):
    return {
        "prompt_ids": prompt,
        "completion_ids": ids,
        "old_logprobs": [-0.5] * len(ids),
        "finish_reason": "stop",
    }


def episode(calls, reward):
    return {
        "calls": calls,
        "reward": reward,
        "family": "SeqQA",
        "tool_calls": [],
        "elapsed_seconds": 1.0,
        "stop_reason": "first_answer_evaluation",
    }


def test_rewritten_context_keeps_call_ids_and_single_terminal_reward():
    a = episode([call([1, 2], [3, 4]), call([90, 91], [5])], 1)
    b = episode([call([1, 2], [6])], 0)
    out = to_generator_output([a, b], ["a", "b"])
    assert out["prompt_token_ids"] == [[1, 2], [90, 91], [1, 2]]
    assert out["response_ids"] == [[3, 4], [5], [6]]
    assert out["rewards"] == [[0, 0], [1], [0]]
    assert out["trajectory_ids"] == ["a", "a", "b"]
    assert out["is_last_step"] == [False, True, True]
    assert out["loss_masks"] == [[1, 1], [1], [1]]


def test_missing_alignment_aborts_instead_of_dropping_sample():
    c = call([1], [2, 3])
    c["old_logprobs"] = [0]
    with pytest.raises(ValueError, match="misaligned"):
        to_generator_output([episode([c], 1)], ["a"])
    with pytest.raises(ValueError, match="Empty"):
        to_generator_output([episode([], 0)], ["a"])


def test_task_success_groups_and_limits_use_episode_denominators():
    from types import SimpleNamespace

    episodes, identities = [], []
    reasons = ["output_truncated"] * 5 + ["context_budget"] * 2 + ["tool_call_budget"]
    for group in range(3):
        for i in range(8):
            reward = int(group == 0 or (group == 2 and i < 4))
            e = episode([call([1], [2])] * (1 + i % 3), reward)
            if group == 1:
                e["stop_reason"] = reasons[i]
            if group == 2 and i >= 4:
                e["stop_reason"] = ["model_call_budget", "time_budget", "tool_timeout", "output_truncated"][i - 4]
            episodes.append(e)
            identities.append(SimpleNamespace(instance_id=str(group)))
    metrics = to_generator_output(episodes, identities)["rollout_metrics"]
    assert metrics["sciencebuddy/episodes/count"] == 24
    assert metrics["sciencebuddy/tasks/count"] == 3
    for name in ("solve_all", "solve_none", "solve_mixed"):
        assert metrics[f"sciencebuddy/tasks/{name}_count"] == 1
    assert metrics["sciencebuddy/episodes/solved_pct_mean"] == 50
    assert metrics["sciencebuddy/episodes/limits/response_pct_mean"] == 25
    assert metrics["sciencebuddy/tasks/limits/response_pct_mean"] == pytest.approx(200 / 3)
    assert metrics["sciencebuddy/episodes/limits/context_count"] == 2
    assert metrics["sciencebuddy/episodes/limits/tool_count"] == 1
    assert metrics["sciencebuddy/episodes/limits/tool_timeout_count"] == 1
    # A later clean batch emits zeros, not missing/stale limit values.
    clean = to_generator_output([episode([call([1], [2])], 1)], ["clean"])["rollout_metrics"]
    assert clean["sciencebuddy/episodes/limits/response_pct_mean"] == 0
    assert clean["sciencebuddy/stop/context_budget_mean"] == 0
    assert clean["sciencebuddy/tasks/solve_none_count"] == 0


def test_no_action_candidate_failure_has_no_fabricated_gradient():
    failed = dict(episode([], 0), stop_reason='candidate_error')
    successful = episode([call([1], [2])], 1)
    output = to_generator_output([failed, successful], ['failed', 'successful'])
    assert output['trajectory_ids'] == ['successful']
    assert output['rollout_metrics']['sciencebuddy/accuracy_mean'] == 0.5
    with pytest.raises(ValueError, match='no trainable model actions'):
        to_generator_output([failed], ['failed'])


def test_memory_budget_failure_keeps_its_policy_tokens_and_sample_identity():
    failed = dict(episode([call([1], [2, 3])], 0), stop_reason='execution_memory_budget')
    healthy = episode([call([1], [4])], 1)
    output = to_generator_output([failed, healthy], ['oom', 'healthy'])
    assert output['trajectory_ids'] == ['oom', 'healthy']
    assert output['loss_masks'] == [[1, 1], [1]]
    assert output['rewards'] == [[0, 0], [1]]
    assert output['rollout_metrics']['sciencebuddy/episodes/limits/execution_memory_count'] == 1
    assert output['rollout_metrics']['sciencebuddy/stop/execution_memory_budget_mean'] == 0.5
