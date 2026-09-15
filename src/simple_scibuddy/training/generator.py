"""SkyRL adapter around arbitrary synchronous broker-driven harness programs."""

import asyncio
import json
import time
import uuid
from pathlib import Path

from skyrl.backends.skyrl_train.inference_servers.engine_utils import get_sampling_params_for_backend
from skyrl.train.generators.base import GeneratorInterface

from simple_scibuddy.artifacts import digest, write_json
from simple_scibuddy.harness.broker import run_episode
from simple_scibuddy.inference.client import RecordingPolicy
from simple_scibuddy.training.data import assigned_split
from simple_scibuddy.training.trajectory import outcome_metrics, to_generator_output


class ScienceBuddyGenerator(GeneratorInterface):
    def __init__(self, cfg, client, tokenizer, settings):
        self.cfg, self.client, self.tokenizer, self.settings = cfg, client, tokenizer, settings
        self.limit = asyncio.Semaphore(settings["concurrency"])
        manifest = load_settings(settings["split_manifest"])
        self.task_ids = {split: set(ids) for split, ids in manifest["task_ids"].items()}
        self.evaluation_outcomes = []
        from simple_scibuddy.data.dataset import TaskDataset
        from simple_scibuddy.harness.scientific import SYSTEM

        self.dataset, self.system = TaskDataset(settings["dataset"]), SYSTEM

    async def generate(self, input_batch):
        started = time.monotonic()
        started_at = time.time()
        completed = 0
        phase = getattr(input_batch.get("batch_metadata"), "training_phase", "train")
        step = getattr(input_batch.get("batch_metadata"), "global_step", 0) or 0
        identities = input_batch["trajectory_ids"]
        extras = input_batch["env_extras"]
        if identities is None or len(identities) != len(extras):
            raise RuntimeError("Complete trajectory identities required")
        sampling = input_batch.get("sampling_params")
        if sampling is None:
            sampling = get_sampling_params_for_backend(
                self.cfg.inference_engine.backend, self.cfg.sampling_params
            )

        async def one(extra, identity):
            nonlocal completed
            async with self.limit:
                session = uuid.uuid4().hex
                folder = Path(self.settings["run_dir"]) / "episodes" / session
                sample_seed = int(
                    digest(
                        [
                            self.settings["seed"],
                            extra["extra_info"]["task_id"],
                            identity.repetition_id,
                            (getattr(input_batch.get("batch_metadata"), "global_step", 0)
                             if getattr(input_batch.get("batch_metadata"), "training_phase", "train") == "train" else 0),
                        ]
                    )[:8],
                    16,
                ) % (2**31)
                policy = RecordingPolicy(
                    self.client,
                    self.tokenizer,
                    dict(sampling, seed=sample_seed),
                    self.client.model_name,
                    session,
                    self.settings["context_tokens"],
                    self.settings["max_tokens"],
                )
                try:
                    task_id = extra["extra_info"]["task_id"]
                    task = self.dataset.load(task_id)
                    split = assigned_split(task_id, self.task_ids, phase)
                    episode = await run_episode(
                        task,
                        self.settings["harness"],
                        policy,
                        folder,
                        self.system,
                        actions=self.settings["actions"],
                        seconds=self.settings["seconds"],
                        tool_seconds=self.settings.get("tool_seconds", 120),
                        runtime_image=self.settings.get("runtime_image"),
                        baked_lake=self.settings.get("baked_lake", False),
                        observation_chars=self.settings["observation_chars"],
                        tool_response_tokens=self.settings.get("tool_response_tokens"),
                        tool_history_tokens=self.settings.get("tool_history_tokens", 8192),
                    )
                    episode.update(
                        split=split, source_split=task.public["split"],
                        model_version=f"update-{max(0, step - int(phase == 'train'))}",
                        training_step=step,
                        model_path=self.settings["model"],
                        seed=sample_seed,
                        trajectory_id=identity.to_string(),
                    )
                    write_json(folder / "episode.json", episode)
                    completed += 1
                    if completed % 8 == 0:
                        print(f"Rollout progress: {completed}/{len(extras)} episodes, "
                              f"{time.monotonic() - started:.1f}s", flush=True)
                    return episode
                finally:
                    await self.client.finish_session(session)

        tasks = []
        async with asyncio.TaskGroup() as group:
            for extra, identity in zip(extras, identities):
                tasks.append(group.create_task(one(extra, identity)))
        episodes = [task.result() for task in tasks]
        if phase == "eval":
            self.evaluation_outcomes.extend(
                ({"reward": e["reward"], "stop_reason": e["stop_reason"]}, identity)
                for e, identity in zip(episodes, identities)
            )
        output = to_generator_output(episodes, identities)
        elapsed = time.monotonic() - started
        tokens = sum(len(c["completion_ids"]) for e in episodes for c in e["calls"])
        profile = {"phase": phase, "step": step, "started_at": started_at, "seconds": elapsed, "episodes": len(episodes),
                   "generated_tokens": tokens, "tokens_per_second": tokens / elapsed,
                   "concurrency": self.settings["concurrency"],
                   "mean_model_seconds": sum(e["timings"]["model"] for e in episodes) / len(episodes),
                   "mean_tool_seconds": sum(e["timings"]["tools"] for e in episodes) / len(episodes),
                   "mean_setup_seconds": sum(e["timings"]["setup"] for e in episodes) / len(episodes)}
        with (Path(self.settings["run_dir"]) / "rollout-profile.jsonl").open("a") as stream:
            stream.write(json.dumps(profile) + "\n")
        output["rollout_metrics"].update({
            "sciencebuddy/profile/" + k + "_mean": v for k, v in profile.items()
            if isinstance(v, (int, float)) and k != "started_at"
        })
        print("Rollout profile: " + json.dumps(profile), flush=True)
        return output

    def evaluation_metrics(self):
        return {"eval/" + key: value for key, value in outcome_metrics(
            [e for e, _ in self.evaluation_outcomes], [i for _, i in self.evaluation_outcomes]
        ).items()}


def load_settings(path):
    return json.loads(Path(path).read_text())
