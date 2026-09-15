"""Prepare frozen data and validate the local container runtime before training."""

import json
import subprocess
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, digest, file_digest, write_json
from simple_scibuddy.data.dataset import load_environment
from simple_scibuddy.environments.build_runtime import build
from simple_scibuddy.paths import local_path


def preflight(cfg):
    from transformers import AutoTokenizer

    from simple_scibuddy.training.data import prepare

    frozen_manifest = json.loads((Path(cfg["dataset"]) / "manifest.json").read_text())
    for name, sha in frozen_manifest.get("payload_hashes", {}).items():
        if file_digest(local_path(name, base=cfg["dataset"], field="payload hash path")) != sha:
            raise RuntimeError(f"Frozen task payload changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    destination = ROOT / "data/sciencebuddy"
    destination.mkdir(parents=True, exist_ok=True)
    prepare(cfg, destination, tokenizer)
    lock = load_environment(cfg["dataset"])
    lake = Path(lock["data_lake"]["path"])
    for name, metadata in lock["data_lake"]["files"].items():
        path = local_path(name, base=lake, field="data lake file")
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise RuntimeError(f"Missing/incomplete scientific asset: {name}")
    if (
        not (Path(cfg["dataset"]) / "runtime/env/Dockerfile").exists()
        and subprocess.run(
            ["docker", "image", "inspect", lock["image"]], capture_output=True, timeout=30
        ).returncode
    ):
        raise RuntimeError("Missing local runtime image; restore the frozen runtime before setup")
    build(cfg)
    local = json.loads((ROOT / "configs/local.json").read_text())
    cfg.update(
        runtime_image=local["runtime_image"],
        baked_lake=local["baked_lake"],
        runtime_base_image=local.get("runtime_base_image", lock["image"]),
    )
    write_json(
        ROOT / "runs/logs/sciencebuddy-preflight.json",
        {
            "status": "passed",
            "settings": cfg,
            "image": cfg["runtime_image"],
            "harness_hash": digest(Path(cfg["harness"]).read_bytes()),
        },
    )
    print("ScienceBuddy preflight passed", flush=True)
