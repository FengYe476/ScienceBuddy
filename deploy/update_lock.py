"""Rewrite environment.lock.json from the actual data lake."""
import argparse, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--release", required=True)
ap.add_argument("--image", required=True)
a = ap.parse_args()

REL = Path(a.release)
LAKE = REL / "resources"
files, total = {}, 0
for p in sorted(LAKE.rglob("*")):
    if not p.is_file():
        continue
    with p.open("rb") as s:
        files[str(p.relative_to(LAKE))] = {"sha256": hashlib.file_digest(s, "sha256").hexdigest()}
    total += p.stat().st_size

lock = json.loads((REL / "environment.lock.json").read_text())
lock["image"] = a.image
lock["network"] = "none"
lock["data_lake"] = {
    "path": "resources",
    "digest": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    "files": files,
}
lock["runtime"] = {"kind": "apptainer", "definition": "deploy/runtime.def",
                   "mounts": {"/workspace": "per-episode private workspace",
                              "/opt/scitrace": "TOOLS.md and DATA.md",
                              "/opt/data/biomni_data/data_lake": "the release resources/ directory, read-only"}}
(REL / "environment.lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
print("  data lake: %d files, %.1f GB, digest %s" % (len(files), total / 1e9, lock["data_lake"]["digest"][:16]))
