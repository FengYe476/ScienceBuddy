"""Small contract checks against the pinned SkyRL interfaces; no GPU/Ray launch."""
import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("skyrl")
from simple_scibuddy.training.callbacks import RunCallbacks


def test_performance_smoke_skips_final_save_and_eval(tmp_path):
    callback = RunCallbacks(tmp_path, no_eval=True, no_checkpoint=True)
    control = NS(should_save=True, should_evaluate=True)
    trainer = NS(cfg=NS(trainer=NS(max_training_steps=1)))
    callback.on_step_end(trainer, NS(global_step=1, total_steps=100), control)
    assert not control.should_save
    assert not control.should_evaluate


def test_save_cadence_ignores_epoch_boundaries_and_saves_final(tmp_path):
    callback = RunCallbacks(tmp_path, 50)
    trainer = NS(cfg=NS(trainer=NS(max_training_steps=102)))
    saved = []
    for step in range(1, 103):
        control = NS(should_save=False)
        callback.on_step_end(trainer, NS(global_step=step, total_steps=200, steps_per_epoch=1), control)
        assert control.should_evaluate == control.should_save
        if control.should_save:
            saved.append(step)
    assert saved == [50, 100, 102]
    trainer.cfg.trainer.max_training_steps = 2
    control = NS(should_save=False)
    callback.on_step_end(trainer, NS(global_step=2, total_steps=100), control)
    assert control.should_save


def test_only_latest_export_retained_and_no_best_checkpoint_copy(tmp_path):
    exports = tmp_path / "exports"
    for name in ("global_step_50", "global_step_100", "dumped_evals"):
        (exports / name).mkdir(parents=True)
    calls = []
    trainer = NS(cfg=NS(trainer=NS(export_path=str(exports))),
                 dispatch=NS(finalize_pending_saves=lambda role: calls.append(role)),
                 save_models=lambda: calls.append("export"))
    callback = RunCallbacks(tmp_path)
    callback.on_save(trainer, NS(global_step=100, ckpt_path="checkpoint100"), None)
    assert calls == [], "on_save must not export a second time"
    assert json.loads((tmp_path / "latest-checkpoint.json").read_text())["optimizer_state_saved"] is False
    assert sorted(p.name for p in exports.iterdir()) == ["dumped_evals", "global_step_100"]
    callback.on_eval_end(trainer, NS(global_step=100, metrics={"eval/all/avg_score": .5}), None)
    assert not (tmp_path / "best-checkpoint").exists()
    assert json.loads((tmp_path / "best-validation.json").read_text())["score"] == .5


def test_startup_http_session_is_closed_before_loop_exit(monkeypatch):
    import asyncio

    import aiohttp
    from skyrl.backends.skyrl_train.inference_servers import setup

    from simple_scibuddy.inference.lifecycle import InferenceLifecycle

    sessions = []

    class Client:
        async def sleep(self, level=2):
            sessions.append(aiohttp.ClientSession())

        async def aclose(self):
            assert not asyncio.get_running_loop().is_closed()
            await sessions[-1].close()

    client = Client()
    servers = NS(router=None, server_groups=[], prefill_server_groups=[], decode_server_groups=[])
    monkeypatch.setattr(setup, "build_new_inference_client", lambda *a, **kw: (client, servers))
    exp = InferenceLifecycle()
    exp.cfg = NS(trainer=NS(placement=NS(colocate_all=True)))
    exp.tokenizer = exp.colocate_pg = None
    assert exp._get_new_inference_client() is client
    assert sessions[0].closed


def test_eval_metrics_cover_all_batches_and_reset_between_evaluations(monkeypatch):
    import asyncio

    from skyrl.train.trainer import RayPPOTrainer

    from simple_scibuddy.training.entrypoint import ManagedTrainer
    from simple_scibuddy.training.generator import ScienceBuddyGenerator

    generator = object.__new__(ScienceBuddyGenerator)
    generator.evaluation_outcomes = []
    trainer = object.__new__(ManagedTrainer)
    trainer.generator = generator

    async def evaluate(self, **kwargs):
        # Unequal evaluation batches: calculate percentages over 3 tasks, not
        # an unweighted average of per-batch percentages (which would be 50%).
        for i in range(3):
            generator.evaluation_outcomes.append((
                {"reward": int(i == 2), "stop_reason": "first_answer_evaluation" if i == 2 else "context_budget"},
                NS(instance_id=str(i)),
            ))
        return {"eval/all/avg_score": 1 / 3}

    monkeypatch.setattr(RayPPOTrainer, "eval", evaluate)
    for _ in range(2):
        metrics = asyncio.run(trainer.eval())
        assert metrics["eval/sciencebuddy/tasks/count"] == 3
        assert metrics["eval/sciencebuddy/episodes/limits/context_pct_mean"] == pytest.approx(200 / 3)
        assert metrics["eval/sciencebuddy/tasks/solve_none_count"] == 2


def test_five_step_evaluation_does_not_increase_checkpoint_saves(tmp_path):
    callback = RunCallbacks(tmp_path, 50, eval_interval=5)
    trainer = NS(cfg=NS(trainer=NS(max_training_steps=20)))
    evaluated, saved = [], []
    for step in range(1, 21):
        control = NS(should_save=False, should_evaluate=False)
        callback.on_step_end(trainer, NS(global_step=step, total_steps=100), control)
        if control.should_evaluate:
            evaluated.append(step)
        if control.should_save:
            saved.append(step)
    assert evaluated == [5, 10, 15, 20]
    assert saved == [20]


def test_final_only_save_across_longer_stage(tmp_path):
    callback = RunCallbacks(tmp_path, 0, eval_interval=5)
    trainer = NS(cfg=NS(trainer=NS(max_training_steps=62)))
    saved, evaluated = [], []
    for step in range(1, 63):
        control = NS()
        callback.on_step_end(trainer, NS(global_step=step, total_steps=100), control)
        if control.should_save:
            saved.append(step)
        if control.should_evaluate:
            evaluated.append(step)
    assert saved == [62]
    assert evaluated == [*range(5, 61, 5), 62]


def test_training_save_uses_only_hf_export(tmp_path):
    from skyrl.train.trainer import RayPPOTrainer

    from simple_scibuddy.training.entrypoint import ManagedTrainer

    saved = []
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer checkpoint path must never be called")
    trainer = NS(cfg=NS(trainer=NS(export_path=str(tmp_path / 'exports'))),
                 global_step=20, tokenizer=object(), has_critic=False,
                 dispatch=NS(save_hf_model=lambda *args: saved.append(args), save_checkpoint=forbidden))
    trainer.save_models = lambda: RayPPOTrainer.save_models(trainer)
    path = ManagedTrainer.save_checkpoints(trainer)
    assert path == str(tmp_path / 'exports/global_step_20/policy')
    assert saved == [('policy', path, trainer.tokenizer)]
