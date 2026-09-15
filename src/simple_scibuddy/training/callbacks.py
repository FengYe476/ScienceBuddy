"""Strict save cadence, bounded retention and evaluation history."""

import json
import shutil
from pathlib import Path

from skyrl.train.utils.callbacks import TrainingCallback

from simple_scibuddy.artifacts import write_json


def prune_exports(folder, keep_step):
    """Delete only this run's older numbered exports, never evaluation dumps."""
    for path in Path(folder).glob("global_step_*"):
        suffix = path.name.removeprefix("global_step_")
        if path.is_dir() and not path.is_symlink() and suffix.isdigit() and int(suffix) != keep_step:
            shutil.rmtree(path)


class RunCallbacks(TrainingCallback):
    def __init__(self, folder, interval=50, *, eval_interval=None, no_eval=False, no_checkpoint=False):
        if interval < 0:
            raise ValueError("Checkpoint interval must be nonnegative (0 means final only)")
        self.folder = Path(folder)
        self.interval = interval
        self.eval_interval = (interval or 50) if eval_interval is None else eval_interval
        if self.eval_interval < 1:
            raise ValueError("Evaluation interval must be positive")
        self.best = float("-inf")
        self.last_update = 0
        self.no_eval, self.no_checkpoint = no_eval, no_checkpoint

    def on_step_end(self, trainer, callback_input, control):
        # Built-in intervals are zero: upstream otherwise saves at every epoch end.
        maximum = trainer.cfg.trainer.max_training_steps
        final_step = min(callback_input.total_steps, maximum) if maximum > 0 else callback_input.total_steps
        step = callback_input.global_step
        self.last_update = step
        control.should_save = not self.no_checkpoint and ((self.interval > 0 and step % self.interval == 0) or step == final_step)
        control.should_evaluate = not self.no_eval and (
            step % self.eval_interval == 0 or step == final_step or control.should_save)

    def on_save(self, trainer, callback_input, control):
        # ManagedTrainer has already saved the model before firing on_save.
        prune_exports(trainer.cfg.trainer.export_path, callback_input.global_step)
        write_json(self.folder / "latest-checkpoint.json", {
            "step": callback_input.global_step,
            "checkpoint": callback_input.ckpt_path,
            "format": "hf_model", "optimizer_state_saved": False, "resumable": False,
            "export": str(Path(trainer.cfg.trainer.export_path) / f"global_step_{callback_input.global_step}" / "policy"),
        })

    def on_eval_end(self, trainer, callback_input, control):
        metrics = callback_input.metrics or {}
        score = metrics.get("eval/all/avg_score")
        if score is None:
            return
        record = {"step": self.last_update, "phase": "baseline" if self.last_update == 0 else "trained", "metrics": metrics}
        with (self.folder / "evaluation-history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if score > self.best:
            self.best = score
            # Record the best score without retaining an extra model export.
            write_json(self.folder / "best-validation.json", {
                "step": self.last_update, "score": score,
                "note": "Metrics only; only the latest checkpoint is retained",
            })
