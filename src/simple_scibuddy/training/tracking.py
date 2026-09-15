"""One W&B writer per experiment; child trainers emit local metric events."""

import json
import os
import tomllib
import traceback
from pathlib import Path


def publish_config(run, folder, text):
    if not text:
        return
    path = Path(folder) / "config.toml"
    path.write_bytes(text.encode("utf-8"))
    run.config.update({"launch": tomllib.loads(text)})
    run.save(str(path), base_path=str(path.parent), policy="now")


class EventTracker:
    """SkyRL's tracker interface without initializing or finishing a W&B run."""

    backend = "wandb"

    def __init__(self, settings):
        self.path = Path(settings["run_dir"]) / "tracking.jsonl"
        round_id = settings["coevolve"]["round_id"]
        stage = "evaluation" if settings["mode"] == "evaluate" else "rl"
        self.prefix = f"{stage}/{round_id}"
        self.tables = {}

    def emit(self, event):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"prefix": self.prefix, **event}, default=lambda v: v.item()) + "\n")

    def log(self, data, step, commit=False):
        evaluation = {k: v for k, v in data.items() if k.startswith("eval/")}
        if self.prefix.startswith("rl/") and evaluation:
            self.emit({"kind": "metrics", "step": step, "data": evaluation,
                       "prefix": self.prefix.replace("rl/", "evaluation/", 1) + "/checkpoint"})
            data = {k: v for k, v in data.items() if k not in evaluation}
        if data:
            self.emit({"kind": "metrics", "step": step, "data": data})

    def log_samples_to_table(self, key, columns, samples, step):
        rows = self.tables.setdefault(key, [])
        rows.extend(samples)
        self.emit({"kind": "table", "step": step, "key": key, "columns": columns, "rows": rows})

    def log_exception(self, error, step=0):
        self.log_samples_to_table("errors", ["step", "type", "traceback"],
                                  [(step, type(error).__name__, traceback.format_exc()[-10000:])], step)

    def finish(self):
        pass


class MetricRelay:
    """Drain complete events into the parent run, with an axis per round/stage."""

    def __init__(self, run):
        self.run = run
        self.offsets = {}
        self.prefixes = set()

    def drain(self, folder):
        path = Path(folder) / "tracking.jsonl"
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self.offsets.get(path, 0))
            while True:
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                prefix = event["prefix"]
                axis = prefix + "/step"
                if prefix not in self.prefixes:
                    self.run.define_metric(prefix + "/*", step_metric=axis)
                    self.prefixes.add(prefix)
                if event["kind"] == "table":
                    import wandb

                    data = {prefix + "/" + event["key"]:
                            wandb.Table(columns=event["columns"], data=event["rows"])}
                else:
                    data = {prefix + "/" + key: value for key, value in event["data"].items()}
                self.run.log(dict(data, **{axis: event["step"]}))
                self.offsets[path] = stream.tell()


class ExperimentTracking:
    def get_tracker(self):
        if not hasattr(self, "_simple_scibuddy_tracker"):
            settings = json.loads(Path(os.environ["SKYRL_SIMPLE_SCIBUDDY_SETTINGS"]).read_text())
            if settings.get("single_wandb_run"):
                self._simple_scibuddy_tracker = EventTracker(settings)
            else:
                self._simple_scibuddy_tracker = super().get_tracker()
                if self._simple_scibuddy_tracker.backend == "wandb":
                    publish_config(self._simple_scibuddy_tracker.logger.run, settings["run_dir"],
                                   settings.get("launch_config_toml"))
        return self._simple_scibuddy_tracker
