#!/usr/bin/env bash
# 在一台新的 Linux GPU 机器上准备 harness 进化实验（不含 RL）。
#
#   bash deploy/bootstrap.sh
#
# 做六件事，全部幂等，可重复执行：
#   1. 环境自检（GPU 显存 / 磁盘 / Apptainer / unshare）
#   2. 用 uv 建 Python 3.12 环境并装 vLLM（harness 阶段不需要 SkyRL）
#   3. 下载 Qwen3.5-4B（检查软链接，paths.py:local_tree 不允许）
#   4. 下载 Biomni 数据湖（76 个文件，约 15 GB）
#   5. 构建 Apptainer 镜像，生成 TOOLS.md / DATA.md
#   6. 写 configs/local.json 并校验 release
#
# 需要的磁盘：模型 9.3G + 数据湖 15G + venv 15G ≈ 40G
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DEPLOY="$ROOT/deploy"
RELEASE_NAME=${RELEASE_NAME:-id40-val10-test10-v1}
RELEASE="$ROOT/data/releases/$RELEASE_NAME"
LAKE="$RELEASE/resources"
SCITRACE="$ROOT/scitrace"
SIF="$ROOT/runtime.sif"
BUCKET=https://biomni-release.s3.amazonaws.com

export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/.cache/uv}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$ROOT/.cache/apptainer}
export APPTAINER_TMPDIR=$APPTAINER_CACHEDIR/tmp
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$APPTAINER_TMPDIR" "$ROOT/logs"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. 自检
say "1/6 环境自检"
command -v nvidia-smi >/dev/null || { echo "找不到 nvidia-smi"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU /'
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 20000 ]; then
    echo "  显存 ${VRAM}MiB 不足：harness 阶段需要约 13.7 GB（权重 8.4 + KV 4.3 + 开销 1），"
    echo "  按 gpu_memory_utilization=0.75 计算至少需要 20 GB 卡。"
    exit 1
fi
AVAIL=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "  可用磁盘 ${AVAIL}G（需要约 40G）"
[ "${AVAIL:-0}" -ge 40 ] || { echo "  磁盘不足"; exit 1; }
for t in apptainer unshare curl git; do
    printf '  %-10s %s\n' "$t" "$(command -v $t || echo '缺失')"
done
command -v apptainer >/dev/null || { echo "  需要 Apptainer（本 fork 用它替代 Docker）"; exit 1; }

# ---------------------------------------------------------------- 2. Python 环境
say "2/6 Python 3.12 + vLLM"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    if ! command -v uv >/dev/null && [ ! -x "$ROOT/.tools/bin/uv" ]; then
        echo "  安装 uv（很多 HPC 没有 Python 3.12 模块，uv 能自己拉取）"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools/bin" sh 2>&1 | tail -2
    fi
    UV=$(command -v uv || echo "$ROOT/.tools/bin/uv")
    "$UV" python install 3.12 2>&1 | tail -2
    "$UV" venv --python 3.12 "$ROOT/.venv" 2>&1 | tail -2
    # harness 阶段只需要推理栈，不需要 SkyRL 那 39G 的 CUDA 训练栈
    "$UV" pip install --python "$ROOT/.venv/bin/python" \
        "vllm==0.28.0" jsonschema wandb huggingface_hub pandas pyarrow 2>&1 | tail -3
fi
"$ROOT/.venv/bin/python" -c "
import sys, vllm, torch
print('  python', sys.version.split()[0], '| vllm', vllm.__version__, '| torch', torch.__version__)
print('  CUDA 可用:', torch.cuda.is_available())
"

# ---------------------------------------------------------------- 3. 模型
say "3/6 Qwen3.5-4B"
MODEL="$ROOT/models/Qwen3.5-4B"
if [ ! -f "$MODEL/config.json" ]; then
    mkdir -p "$ROOT/models"
    "$ROOT/.venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B", local_dir="$MODEL", max_workers=8)
PY
fi
LINKS=$(find "$MODEL" -type l | head -3)
[ -z "$LINKS" ] || { echo "  发现软链接，paths.py:local_tree 会拒绝："; echo "$LINKS"; exit 1; }
echo "  $(du -sh "$MODEL" | cut -f1)，无软链接"

# ---------------------------------------------------------------- 4. 数据湖
say "4/6 Biomni 数据湖"
if [ ! -f "$RELEASE/manifest.json" ]; then
    echo "  release 不存在，从公开数据源确定性重建（seed 20260911）"
    "$ROOT/.venv/bin/python" "$DEPLOY/build_release.py" --out "$RELEASE" \
        --cache "$ROOT/.cache/sources" 2>&1 | tail -12
fi
[ -f "$RELEASE/manifest.json" ] || { echo "  release 构建失败"; exit 1; }
mkdir -p "$LAKE"
NEED=$("$ROOT/.venv/bin/python" - <<PY
import json, pathlib
names = json.load(open("$DEPLOY/lake_files.json"))
have = {p.name for p in pathlib.Path("$LAKE").iterdir() if p.is_file() and p.stat().st_size > 0}
print("\n".join(n for n in names if n not in have))
PY
)
if [ -n "$NEED" ]; then
    echo "  待下载 $(echo "$NEED" | wc -l) 个文件"
    echo "$NEED" | xargs -P 8 -I{} sh -c "
        curl -sfL --retry 3 -o '$LAKE/{}.part' '$BUCKET/data_lake/{}' \
          && mv '$LAKE/{}.part' '$LAKE/{}' || { rm -f '$LAKE/{}.part'; echo '失败 {}'; }"
fi
find "$LAKE" -name '._*' -delete 2>/dev/null || true
find "$LAKE" -size 0 -delete 2>/dev/null || true
echo "  $(ls -1 "$LAKE" | wc -l) 个文件，$(du -sh "$LAKE" | cut -f1)"

# ---------------------------------------------------------------- 5. 镜像
say "5/6 Apptainer 镜像与文档"
mkdir -p "$SCITRACE"
cp -f "$DEPLOY/TOOLS.md" "$SCITRACE/TOOLS.md"
"$ROOT/.venv/bin/python" "$DEPLOY/gen_datamd.py" --lake "$LAKE" --desc "$DEPLOY/lake_desc.json" --out "$SCITRACE/DATA.md"
if [ ! -f "$SIF" ]; then
    sed -e "s|@SCITRACE@|$SCITRACE|g" "$DEPLOY/runtime.def" > "$ROOT/.cache/runtime.def"
    apptainer build --fakeroot "$SIF" "$ROOT/.cache/runtime.def" 2>&1 | tail -4
fi
ls -lh "$SIF" | awk '{print "  镜像 " $5}'
apptainer exec --containall --bind "$SCITRACE:/opt/scitrace:ro" --bind "$LAKE:/opt/data/biomni_data/data_lake:ro" \
    "$SIF" python -c "
from pathlib import Path
for p in ('/opt/scitrace/TOOLS.md','/opt/scitrace/DATA.md','/opt/data/biomni_data/data_lake'):
    print('  ', p, '存在' if Path(p).exists() else '缺失!')
import numpy, pandas, Bio; print('   库 OK')
"

# ---------------------------------------------------------------- 6. 配置与校验
say "6/6 配置与校验"
cat > "$ROOT/configs/local.json" <<EOF
{
  "dataset": "data/releases/$RELEASE_NAME",
  "model": "models/Qwen3.5-4B",
  "runtime_image": "$SIF",
  "baked_lake": false,
  "concurrency": 8,
  "workers": 8
}
EOF
"$ROOT/.venv/bin/python" "$DEPLOY/update_lock.py" --release "$RELEASE" --image "$SIF"
PYTHONPATH="$ROOT/src" SCIBUDDY_RUNTIME=apptainer "$ROOT/.venv/bin/python" -c "
from simple_scibuddy.configuration import settings
from simple_scibuddy.harness.broker import run_episode
import inspect
c = settings()
print('  容器后端:', inspect.signature(run_episode).parameters['container_factory'].default.__name__)
print('  数据集  :', c['dataset'].split('/')[-1])
print('  镜像    :', c['runtime_image'].split('/')[-1])
"

cat <<EOF

准备完成。下一步跑 harness 进化：

  export SCIBUDDY_RUNTIME=apptainer
  export SCIBUDDY_SCITRACE=$SCITRACE
  export WANDB_MODE=offline WANDB_DIR=$ROOT/logs
  set -a; . ~/.sciencebuddy/improver.env; set +a
  export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
  cd $ROOT
  nohup .venv/bin/python scripts/train.py configs/h-only.toml > logs/h-only.log 2>&1 &
EOF
