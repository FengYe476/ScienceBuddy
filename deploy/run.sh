#!/usr/bin/env bash
# Start one experiment on a machine without Docker, preparing the runtime environment.
#
#   bash deploy/run.sh configs/h-only.toml
#   bash deploy/run.sh configs/rl-only.toml
#
# Calls coevolve.experiment with the .venv interpreter directly, bypassing the
# `uv run --isolated --frozen --extra train` in scripts/train.py: harness_evolve mode never
# touches skyrl/ray/torch (zero references in loop.py, phase.py and worker.py), so there is no
# reason to rebuild the 39 GB training stack for it. RL mode should still go through
# scripts/train.py.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${1:?usage: bash deploy/run.sh configs/<name>.toml}
[ -f "$CONFIG" ] || { echo "configuration not found: $CONFIG"; exit 1; }

export SCIBUDDY_RUNTIME=apptainer
export SCIBUDDY_SCITRACE=$ROOT/scitrace
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$ROOT/logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$ROOT/src
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ -f "$HOME/.sciencebuddy/improver.env" ]; then
    set -a; . "$HOME/.sciencebuddy/improver.env"; set +a
else
    echo "warning: ~/.sciencebuddy/improver.env not found; the harness phase will fail at its first proposal"
fi

# loop.py:56 uses mkdir(exist_ok=False) and refuses to reuse an experiment directory; fail early with a readable message
EXP=$(python3 - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'^experiment\s*=\s*"([^"]+)"', text, re.M)
print(m.group(1) if m else "")
PY
)
if [ -n "$EXP" ] && [ -d "$ROOT/$(echo "$EXP" | sed 's|^\.\./||')" ]; then
    echo "experiment directory already exists: $EXP"
    echo "upstream does not allow reuse (loop.py:56). Change experiment in $CONFIG, or delete that directory."
    exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.tmp"
rm -f "$ROOT/.tmp/sciencebuddy.lock"
echo "config     $CONFIG"
echo "experiment $EXP"
echo "runtime    apptainer + $SCIBUDDY_SCITRACE"
echo "GPU        ${CUDA_VISIBLE_DEVICES}"
exec "$ROOT/.venv/bin/python" -m simple_scibuddy.coevolve.experiment "$CONFIG"
