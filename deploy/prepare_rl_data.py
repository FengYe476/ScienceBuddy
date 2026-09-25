#!/usr/bin/env python3
"""Produce data/sciencebuddy/ for RL training on a machine without Docker.

    bash deploy/prepare_rl_data.sh

launch.py:113 refuses to start training unless data/sciencebuddy/manifest.json exists,
and that directory is normally written by `cli.py preflight`. Upstream preflight does five
things, and two of them need Docker:

    1. verify the frozen payload hashes                     no Docker
    2. training/data.py:prepare -> parquet + manifest.json  no Docker
    3. verify every data-lake file is present and complete  no Docker
    4. `docker image inspect` on the pinned runtime image   Docker
    5. build_runtime.py:build, which bakes the lake into    Docker
       an image

Steps 4 and 5 exist for the case build_runtime.py's own docstring names: "nested Docker
rejects read-only binds". The Apptainer backend bind-mounts the lake directly, which is
why configs/local.json carries baked_lake=false, so there is nothing for them to do here.

This script runs steps 1-3 by calling the same upstream functions, so the parquet files and
manifest are identical to what preflight would have written, and skips 4-5. It does not
modify any source file.
"""

import json
import sys
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, file_digest
from simple_scibuddy.configuration import settings
from simple_scibuddy.data.dataset import load_environment
from simple_scibuddy.paths import local_path

BAR = "=" * 74


def main():
    cfg = settings()
    dataset = Path(cfg["dataset"])
    print(BAR)
    print(f"dataset  {dataset}")
    print(f"model    {cfg['model']}")
    print(BAR)

    # 1. frozen payloads unchanged (preflight.py:19-22)
    frozen = json.loads((dataset / "manifest.json").read_text())
    hashes = frozen.get("payload_hashes", {})
    for name, sha in hashes.items():
        if file_digest(local_path(name, base=dataset, field="payload hash path")) != sha:
            sys.exit(f"frozen task payload changed: {name}")
    print(f"  1/3  payload hashes verified ({len(hashes)} entries)")

    # 2. the parquet files and manifest RL reads (preflight.py:23-25)
    from transformers import AutoTokenizer

    from simple_scibuddy.training.data import prepare

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    destination = ROOT / "data/sciencebuddy"
    destination.mkdir(parents=True, exist_ok=True)
    prepare(cfg, destination, tokenizer)
    manifest = json.loads((destination / "manifest.json").read_text())
    print(f"  2/3  wrote {destination.relative_to(ROOT)}")
    print(f"       counts {manifest['counts']}   seed {manifest['seed']}")
    print(f"       held-out validation IDs excluded: {len(manifest['validation_task_ids'])}")

    # 3. every data-lake file present at its recorded size (preflight.py:26-31)
    lock = load_environment(dataset)
    lake = Path(lock["data_lake"]["path"])
    for name, metadata in lock["data_lake"]["files"].items():
        path = local_path(name, base=lake, field="data lake file")
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            sys.exit(f"missing or incomplete scientific asset: {name}")
    print(f"  3/3  data lake verified ({len(lock['data_lake']['files'])} files)")

    print(f"\n  skipped the Docker image build; the runtime is {cfg['runtime_image']}")
    print(f"  baked_lake={cfg.get('baked_lake')}, so the lake is bind-mounted at run time\n")


if __name__ == "__main__":
    main()
