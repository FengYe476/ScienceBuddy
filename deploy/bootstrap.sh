#!/usr/bin/env bash
# Prepare the harness-evolution experiment on a fresh Linux GPU machine (no RL).
#
#   bash deploy/bootstrap.sh
#
# Six steps, all idempotent and safe to rerun:
#   1. environment check (GPU memory / disk / Apptainer / unshare)
#   2. build a Python 3.12 environment with uv and install vLLM (the harness phase
#      does not need SkyRL)
#   3. download Qwen3.5-4B (checking for symlinks, which paths.py:local_tree rejects)
#   4. download the Biomni data lake (76 files, about 15 GB)
#   5. build the Apptainer image and generate TOOLS.md / DATA.md
#   6. write configs/local.json and validate the release
#
# Disk required: model 9.3G + data lake 15G + venv 15G, roughly 40G
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

# ---------------------------------------------------------------- 1. environment check
say "1/6 environment check"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU /'
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 20000 ]; then
    echo "  ${VRAM}MiB of GPU memory is not enough: the harness phase needs about 13.7 GB"
    echo "  (weights 8.4 + KV 4.3 + overhead 1); at gpu_memory_utilization=0.75 that means a 20 GB card."
    exit 1
fi
AVAIL=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "  disk available ${AVAIL}G (about 40G needed)"
[ "${AVAIL:-0}" -ge 40 ] || { echo "  not enough disk"; exit 1; }
for t in apptainer unshare curl git; do
    printf '  %-10s %s\n' "$t" "$(command -v $t || echo 'missing')"
done
command -v apptainer >/dev/null || { echo "  Apptainer is required (this fork uses it instead of Docker)"; exit 1; }

# ---------------------------------------------------------------- 2. Python environment
say "2/6 Python 3.12 + vLLM"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    if ! command -v uv >/dev/null && [ ! -x "$ROOT/.tools/bin/uv" ]; then
        echo "  installing uv (many HPC systems have no Python 3.12 module; uv can fetch one)"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools/bin" sh 2>&1 | tail -2
    fi
    UV=$(command -v uv || echo "$ROOT/.tools/bin/uv")
    "$UV" python install 3.12 2>&1 | tail -2
    "$UV" venv --python 3.12 "$ROOT/.venv" 2>&1 | tail -2
    # The harness phase only needs the inference stack, not SkyRL's 39G CUDA training stack
    "$UV" pip install --python "$ROOT/.venv/bin/python" \
        "vllm==0.28.0" jsonschema wandb huggingface_hub pandas pyarrow 2>&1 | tail -3
fi
"$ROOT/.venv/bin/python" -c "
import sys, vllm, torch
print('  python', sys.version.split()[0], '| vllm', vllm.__version__, '| torch', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
"

# ---------------------------------------------------------------- 3. model
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
[ -z "$LINKS" ] || { echo "  symlinks found; paths.py:local_tree will reject them:"; echo "$LINKS"; exit 1; }
echo "  $(du -sh "$MODEL" | cut -f1), no symlinks"

# ---------------------------------------------------------------- 4. data lake
say "4/6 Biomni data lake"
if [ ! -f "$RELEASE/manifest.json" ]; then
    echo "  release not present; rebuilding deterministically from public sources (seed 20260911)"
    "$ROOT/.venv/bin/python" "$DEPLOY/build_release.py" --out "$RELEASE" \
        --cache "$ROOT/.cache/sources" 2>&1 | tail -12
fi
[ -f "$RELEASE/manifest.json" ] || { echo "  release build failed"; exit 1; }
mkdir -p "$LAKE"
NEED=$("$ROOT/.venv/bin/python" - <<PY
import json, pathlib
names = json.load(open("$DEPLOY/lake_files.json"))
have = {p.name for p in pathlib.Path("$LAKE").iterdir() if p.is_file() and p.stat().st_size > 0}
print("\n".join(n for n in names if n not in have))
PY
)
if [ -n "$NEED" ]; then
    echo "  $(echo "$NEED" | wc -l) files to download"
    echo "$NEED" | xargs -P 8 -I{} sh -c "
        curl -sfL --retry 3 -o '$LAKE/{}.part' '$BUCKET/data_lake/{}' \
          && mv '$LAKE/{}.part' '$LAKE/{}' || { rm -f '$LAKE/{}.part'; echo 'failed {}'; }"
fi
find "$LAKE" -name '._*' -delete 2>/dev/null || true
find "$LAKE" -size 0 -delete 2>/dev/null || true
echo "  $(ls -1 "$LAKE" | wc -l) files, $(du -sh "$LAKE" | cut -f1)"

# ---------------------------------------------------------------- 5. image
say "5/6 Apptainer image and documentation"
mkdir -p "$SCITRACE"
cp -f "$DEPLOY/TOOLS.md" "$SCITRACE/TOOLS.md"
"$ROOT/.venv/bin/python" "$DEPLOY/gen_datamd.py" --lake "$LAKE" --desc "$DEPLOY/lake_desc.json" --out "$SCITRACE/DATA.md"
if [ ! -f "$SIF" ]; then
    sed -e "s|@SCITRACE@|$SCITRACE|g" "$DEPLOY/runtime.def" > "$ROOT/.cache/runtime.def"
    apptainer build --fakeroot "$SIF" "$ROOT/.cache/runtime.def" 2>&1 | tail -4
fi
ls -lh "$SIF" | awk '{print "  image " $5}'
apptainer exec --containall --bind "$SCITRACE:/opt/scitrace:ro" --bind "$LAKE:/opt/data/biomni_data/data_lake:ro" \
    "$SIF" python -c "
from pathlib import Path
for p in ('/opt/scitrace/TOOLS.md','/opt/scitrace/DATA.md','/opt/data/biomni_data/data_lake'):
    print('  ', p, 'present' if Path(p).exists() else 'MISSING!')
import numpy, pandas, Bio; print('   libraries OK')
"

# ---------------------------------------------------------------- 6. configuration and validation
say "6/6 configuration and validation"
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
print('  container backend:', inspect.signature(run_episode).parameters['container_factory'].default.__name__)
print('  dataset          :', c['dataset'].split('/')[-1])
print('  image            :', c['runtime_image'].split('/')[-1])
"

cat <<EOF

Setup complete. Next, run harness evolution:

  export SCIBUDDY_RUNTIME=apptainer
  export SCIBUDDY_SCITRACE=$SCITRACE
  export WANDB_MODE=offline WANDB_DIR=$ROOT/logs
  set -a; . ~/.sciencebuddy/improver.env; set +a
  export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
  cd $ROOT
  nohup .venv/bin/python scripts/train.py configs/h-only.toml > logs/h-only.log 2>&1 &
EOF
