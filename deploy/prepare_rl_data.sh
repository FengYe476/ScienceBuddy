#!/usr/bin/env bash
# Generate data/sciencebuddy/ so RL training can start. Same environment as deploy/run.sh,
# because training/data.py:31 embeds harness/scientific.py:SYSTEM into every prompt and the
# dataset path comes from configs/local.json.
#
#   bash deploy/prepare_rl_data.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export SCIBUDDY_RUNTIME=apptainer
export SCIBUDDY_SCITRACE=$ROOT/scitrace
export PYTHONPATH=$ROOT/src
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=${PY:-$ROOT/.venv/bin/python}
[ -x "$PY" ] || PY=$ROOT/.venv-skyrl/bin/python
[ -x "$PY" ] || { echo "no interpreter found; run bash deploy/bootstrap.sh first"; exit 1; }

exec "$PY" deploy/prepare_rl_data.py
