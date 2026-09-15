"""Import-only SkyRL extension; no upstream changes."""

import os
import sys
from pathlib import Path

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from simple_scibuddy.inference.lifecycle import InferenceLifecycle
from simple_scibuddy.training.generator import ScienceBuddyGenerator, load_settings
from simple_scibuddy.training.tracking import ExperimentTracking


class ManagedTrainer(RayPPOTrainer):
    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        if self.cfg.trainer.strategy == 'fsdp':
            from simple_scibuddy.training.weights import PolicyWorker as RecurrentPolicyWorker

            PolicyWorker = RecurrentPolicyWorker
        return super().build_models(PolicyWorker, CriticWorker, RefWorker)

    def save_checkpoints(self):
        """Use SkyRL's HF export path; never serialize optimizer/trainer state."""
        self.save_models()
        return str(Path(self.cfg.trainer.export_path) / f"global_step_{self.global_step}" / "policy")

    async def eval(self, **kwargs):
        # Co-evolution measures the final exported model in the handoff worker.
        # Keep only the initial evaluation here, avoiding duplicate end measurements.
        if getattr(self, "stage_boundary_eval", False) and self.global_step > 0:
            return {}
        self.generator.evaluation_outcomes.clear()
        metrics = await super().eval(**kwargs)
        metrics.update(self.generator.evaluation_metrics())
        return metrics

    async def train(self):
        try:
            await super().train()
        finally:
            await self.inference_engine_client.aclose()


class ScienceBuddyExp(ExperimentTracking, InferenceLifecycle, BasePPOExp):
    def get_trainer(self, **kwargs):
        from simple_scibuddy.training.callbacks import RunCallbacks

        trainer = ManagedTrainer(**kwargs)
        settings = load_settings(os.environ["SKYRL_SIMPLE_SCIBUDDY_SETTINGS"])
        trainer.stage_boundary_eval = bool(settings.get("coevolve"))
        trainer.add_callback(RunCallbacks(settings["run_dir"], settings["checkpoint_interval"],
                            eval_interval=settings.get("eval_interval"),
                            no_eval=settings.get("no_eval", False),
                            no_checkpoint=settings.get("no_checkpoint", False)))
        return trainer

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return ScienceBuddyGenerator(
            cfg.generator,
            inference_engine_client,
            tokenizer,
            load_settings(os.environ["SKYRL_SIMPLE_SCIBUDDY_SETTINGS"]),
        )


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    ScienceBuddyExp(cfg).run()


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
