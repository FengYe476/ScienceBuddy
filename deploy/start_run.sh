#!/usr/bin/env bash
# Prepare and launch one harness-evolution run.
#
#   bash deploy/start_run.sh configs/h-only-v2.toml id200-val30-test30-v2
#   bash deploy/start_run.sh configs/h-only-v2.toml id200-val30-test30-v2 id200-val30-test30-v1
#
# The third argument is an existing release whose data lake can be reused; leave it out and
# the lake is downloaded by deploy/bootstrap.sh instead.
#
# Idempotent: finished steps are skipped, so the script is safe to rerun. Seven steps:
#   1. preflight checks (venv / improver.env / experiment directory still free)
#   2. deterministic release rebuild (same SPLIT_SEED)
#   3. populate resources/ with hard links -- symlinks cannot be used here:
#      local_tree in paths.py:29-33 rejects symlinks that resolve outside the release
#      directory, and `ln -s ../<other-release>/resources` points at a sibling directory,
#      so it is rejected. A hard link reports is_symlink() as false and shares the inode
#      with the original, costing no extra space.
#   4. rewrite data_lake.files in environment.lock.json
#   5. point configs/local.json at the new release
#   6. validate by loading the release through upstream's own loader
#   7. launch under nohup and print the pid plus monitoring commands
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG=${1:?usage: bash deploy/start_run.sh <config.toml> <release-name> [source-release]}
NEW_NAME=${2:?usage: bash deploy/start_run.sh <config.toml> <release-name> [source-release]}
OLD_NAME=${3:-}
PROFILE=${PROFILE:-small}

NEW="$ROOT/data/releases/$NEW_NAME"
OLD=${OLD_NAME:+$ROOT/data/releases/$OLD_NAME}
PY="$ROOT/.venv/bin/python"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. checks
say "1/7 preflight checks"
[ -x "$PY" ] || die "$PY not found; run bash deploy/bootstrap.sh first"
[ -f "$ROOT/$CONFIG" ] || die "$CONFIG not found; git pull first"
[ -f "$HOME/.sciencebuddy/improver.env" ] \
    || die "~/.sciencebuddy/improver.env not found; the harness phase fails at its first proposal"
[ -z "$OLD" ] || [ -d "$OLD/resources" ] || die "$OLD/resources not found"

EXP=$("$PY" - "$ROOT/$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'^experiment\s*=\s*"([^"]+)"', text, re.M)
print((m.group(1) if m else "").replace("../", "", 1))
PY
)
[ -n "$EXP" ] || die "no experiment path in $CONFIG"
LOG="$ROOT/logs/$(basename "$EXP").log"
[ -d "$ROOT/$EXP" ] && die "experiment directory already exists: $ROOT/$EXP
loop.py:56 refuses to reuse one. Change experiment in $CONFIG, or delete that directory."

printf '  config      %s\n' "$CONFIG"
printf '  release     %s\n' "$NEW_NAME"
printf '  experiment  %s\n' "$EXP"
printf '  disk free   %s\n' "$(df -BG --output=avail "$ROOT" | tail -1 | tr -d ' ')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  GPU         /'

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 1000 ]; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/    /'
    die "${USED} MiB of GPU memory is already in use, so vLLM cannot start.
inference/server.py launches vLLM with start_new_session=True, so it outlives a plain
kill of the experiment process. Release it with:

    pkill -f 'vllm.entrypoints.openai.api_server'

then rerun this script."
fi

# ---------------------------------------------------------------- 2. dataset
say "2/7 rebuild release (profile=$PROFILE)"
if [ -f "$NEW/manifest.json" ]; then
    echo "  already present, skipping"
else
    "$PY" deploy/build_release.py --out "$NEW" --profile "$PROFILE" \
        --cache "$ROOT/.cache/sources" 2>&1 | tail -16
fi
[ -f "$NEW/manifest.json" ] || die "release build failed"

# ---------------------------------------------------------------- 3. data lake
say "3/7 hard-link the data lake"
if [ -n "$OLD" ]; then
    # One file at a time rather than `cp -al`: cp returns non-zero when the destination
    # already exists, which cannot distinguish "already linked on an earlier run" (success)
    # from "cannot be linked" (failure). os.link plus an explicit inode comparison can.
    "$PY" - "$OLD/resources" "$NEW/resources" <<'PY'
import os, pathlib, shutil, sys

old, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
new.mkdir(parents=True, exist_ok=True)
linked = shared = copied = independent = 0
for src in sorted(old.iterdir()):
    if not src.is_file():
        continue
    dst = new / src.name
    if dst.exists():
        a, b = src.stat(), dst.stat()
        if (a.st_ino, a.st_dev) == (b.st_ino, b.st_dev):
            shared += 1          # already linked on an earlier run: the idempotent path
        else:
            independent += 1     # an independent copy, also usable; leave it alone
        continue
    try:
        os.link(src, dst)
        linked += 1
    except OSError:              # cross-device and similar: fall back to a real copy
        shutil.copy2(src, dst)
        copied += 1

total = linked + shared + copied + independent
print(f"  newly linked {linked} - already sharing inodes {shared} - copied {copied} - "
      f"independent copies {independent}")
print(f"  {total} files in total")
missing = {p.name for p in old.iterdir() if p.is_file()} - {p.name for p in new.iterdir() if p.is_file()}
if missing:
    sys.exit(f"  {len(missing)} files missing: {sorted(missing)[:5]}")
PY
else
    echo "  no source release given; expecting bootstrap.sh to have filled $NEW/resources"
    [ -d "$NEW/resources" ] || die "$NEW/resources not found"
fi
find "$NEW/resources" -type l -print -quit | grep -q . \
    && die "symlinks found under resources/; local_tree will reject them"
printf '  disk used %s (hard links do not add to this)\n' "$(du -sh "$NEW/resources" | cut -f1)"

# ---------------------------------------------------------------- 4. lock
say "4/7 rewrite environment.lock.json"
"$PY" deploy/update_lock.py --release "$NEW" --image "$ROOT/runtime.sif"

# ---------------------------------------------------------------- 5. config
say "5/7 point configs/local.json at the new release"
"$PY" - "$NEW_NAME" <<'PY'
import json, pathlib, sys
path = pathlib.Path("configs/local.json")
cfg = json.loads(path.read_text())
cfg["dataset"] = f"data/releases/{sys.argv[1]}"
path.write_text(json.dumps(cfg, indent=2) + "\n")
print("  dataset =", cfg["dataset"])
PY

# ---------------------------------------------------------------- 6. validation
say "6/7 validate through upstream's loader"
PYTHONPATH="$ROOT/src" SCIBUDDY_RUNTIME=apptainer "$PY" - "$ROOT/$CONFIG" <<'PY'
import sys, tomllib, pathlib
from simple_scibuddy.configuration import settings
from simple_scibuddy.data.dataset import TaskDataset   # the class loop.py:17 uses

cfg = settings()
# The constructor runs local_tree and the split_counts check, so a symlink or a count
# mismatch fails here rather than after the experiment directory has been created.
data = TaskDataset(cfg["dataset"])
counts = {s: len(data.tasks(s)) for s in ("train", "val", "test")}
print("  split      ", counts)

harness = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["harness_evolve"]
want = harness["selection_tasks"]
if counts["val"] != want:
    sys.exit(f"  selection_tasks={want} does not match val={counts['val']}; "
             "phase.py:126 refuses to start")

steps, feedback, candidates = harness["steps_per_phase"], harness["feedback_every"], harness["candidates"]
n = 1 + counts["test"] + steps * (feedback + candidates + (1 + candidates) * counts["val"]) + counts["test"]
print(f"  episodes    {n} expected")
print("  data lake  ", pathlib.Path(data.environment['data_lake']['path']).name)
PY

# ---------------------------------------------------------------- 7. launch
say "7/7 launch"
mkdir -p "$ROOT/logs"
nohup bash deploy/run.sh "$CONFIG" > "$LOG" 2>&1 &
PID=$!
sleep 5
kill -0 "$PID" 2>/dev/null || { tail -30 "$LOG"; die "the process exited immediately after launch; log above"; }

cat <<EOF

  pid   $PID
  log   $LOG

Monitoring:
  tail -f $LOG
  grep -c 'Harness rollout' $LOG          # episodes completed so far
  python3 deploy/analyze_episodes.py $EXP
  python3 deploy/estimate_runtime.py $EXP

Stop:
  kill $PID && pkill -f 'vllm.entrypoints.openai.api_server'

EOF
