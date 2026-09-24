#!/usr/bin/env bash
# Measure intermediate-generation harnesses on the test split. The environment matches
# deploy/run.sh exactly: if the seed, the container backend or the data-lake path differs in
# any way, the results can no longer be placed beside the original run.
#
#   bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01
#   bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01 H1
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${1:?usage: bash deploy/rerun_on_test.sh <config.toml> <run dir> [H1,H2]}
RUN=${2:?usage: bash deploy/rerun_on_test.sh <config.toml> <run dir> [H1,H2]}
ONLY=${3:-}

[ -f "$CONFIG" ] || { echo "configuration not found: $CONFIG"; exit 1; }
[ -d "$RUN" ] || { echo "experiment directory not found: $RUN"; exit 1; }

export SCIBUDDY_RUNTIME=apptainer
export SCIBUDDY_SCITRACE=$ROOT/scitrace
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$ROOT/logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$ROOT/src
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# This measurement never calls the improver, so no API key is needed; improver.env is not loaded.

mkdir -p "$ROOT/logs"
echo "config     $CONFIG"
echo "experiment $RUN"
echo "runtime    apptainer + $SCIBUDDY_SCITRACE"
echo "GPU        ${CUDA_VISIBLE_DEVICES}"

ARGS=("$CONFIG" "$RUN")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
exec "$ROOT/.venv/bin/python" deploy/rerun_on_test.py "${ARGS[@]}"
