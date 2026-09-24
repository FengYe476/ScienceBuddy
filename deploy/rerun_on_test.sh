#!/usr/bin/env bash
# 把中间世代的 harness 补测到 test 上。环境变量与 deploy/run.sh 一致，
# 否则 seed、容器后端或数据湖路径任一不同，结果就不能和原实验并排比较。
#
#   bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01
#   bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01 H1
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${1:?用法: bash deploy/rerun_on_test.sh <config.toml> <run 目录> [H1,H2]}
RUN=${2:?用法: bash deploy/rerun_on_test.sh <config.toml> <run 目录> [H1,H2]}
ONLY=${3:-}

[ -f "$CONFIG" ] || { echo "找不到配置 $CONFIG"; exit 1; }
[ -d "$RUN" ] || { echo "找不到实验目录 $RUN"; exit 1; }

export SCIBUDDY_RUNTIME=apptainer
export SCIBUDDY_SCITRACE=$ROOT/scitrace
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$ROOT/logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$ROOT/src
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 补测不调用 improver，所以不需要 API key —— 不加载 improver.env

mkdir -p "$ROOT/logs"
echo "配置    $CONFIG"
echo "实验    $RUN"
echo "运行时  apptainer + $SCIBUDDY_SCITRACE"
echo "GPU     ${CUDA_VISIBLE_DEVICES}"

ARGS=("$CONFIG" "$RUN")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
exec "$ROOT/.venv/bin/python" deploy/rerun_on_test.py "${ARGS[@]}"
