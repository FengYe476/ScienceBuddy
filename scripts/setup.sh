#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if ! command -v uv >/dev/null && [[ ! -x .skyrl-tools/bin/uv ]]; then
    echo 'Install uv into a local tools environment, then rerun this script.' >&2
    exit 1
fi
UV_BIN=$(command -v uv || echo "$ROOT/.skyrl-tools/bin/uv")
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-skyrl"
export UV_LINK_MODE=hardlink
export UV_HTTP_TIMEOUT=600
mkdir -p runs/logs .tmp
exec 9>.tmp/sciencebuddy.lock
flock -n 9 || { echo "A Simple-SciBuddy command is already running." >&2; exit 1; }
git submodule update --init --recursive
echo "Installing locked dependencies; details: runs/logs/install.log"
"$UV_BIN" sync --frozen --extra train > runs/logs/install.log 2>&1

# Reuse the launcher's project-local environment without acquiring a second lock.
"$ROOT/.venv-skyrl/bin/python" - "$UV_BIN" >> runs/logs/install.log 2>&1 <<'PYTHON'
import subprocess
import sys
sys.path.insert(0, "scripts")
from train import environment
subprocess.run([sys.argv[1], "run", "--isolated", "--frozen", "--extra", "train",
                "-m", "simple_scibuddy.cli", "preflight"], env=environment(), check=True)
PYTHON
echo 'Setup complete. Run: python scripts/train.py configs/train.toml'
