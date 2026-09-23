#!/usr/bin/env bash
# 在无 Docker 的机器上启动一次实验，自动准备好运行时环境变量。
#
#   bash deploy/run.sh configs/h-only.toml
#   bash deploy/run.sh configs/rl-only.toml
#
# 直接用 .venv 里的解释器调用 coevolve.experiment，绕开 scripts/train.py 的
# `uv run --isolated --frozen --extra train`：harness_evolve 模式不碰 skyrl/ray/torch
# （loop.py、phase.py、worker.py 里零引用），没必要为它重建 39 GB 的训练栈。
# RL 模式仍应走 scripts/train.py。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${1:?用法: bash deploy/run.sh configs/<name>.toml}
[ -f "$CONFIG" ] || { echo "找不到配置 $CONFIG"; exit 1; }

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
    echo "警告：找不到 ~/.sciencebuddy/improver.env，harness 阶段会在第一次提案时失败"
fi

# loop.py:56 用 mkdir(exist_ok=False) 拒绝复用实验目录，提前给出可读的提示
EXP=$(python3 - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'^experiment\s*=\s*"([^"]+)"', text, re.M)
print(m.group(1) if m else "")
PY
)
if [ -n "$EXP" ] && [ -d "$ROOT/$(echo "$EXP" | sed 's|^\.\./||')" ]; then
    echo "实验目录已存在: $EXP"
    echo "上游不允许复用（loop.py:56）。请改 $CONFIG 里的 experiment，或删除该目录。"
    exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.tmp"
rm -f "$ROOT/.tmp/sciencebuddy.lock"
echo "配置    $CONFIG"
echo "实验    $EXP"
echo "运行时  apptainer + $SCIBUDDY_SCITRACE"
echo "GPU     ${CUDA_VISIBLE_DEVICES}"
exec "$ROOT/.venv/bin/python" -m simple_scibuddy.coevolve.experiment "$CONFIG"
