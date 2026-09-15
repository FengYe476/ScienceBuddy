#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CHECK_PYTHON=${SIMPLE_SCIBUDDY_CHECK_PYTHON:-"$ROOT/.venv-check/bin/python"}
mkdir -p runs/logs
trap 'cat runs/logs/lint.log runs/logs/tests.log 2>/dev/null || true' ERR
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
"$CHECK_PYTHON" -m ruff check src tests > runs/logs/lint.log 2>&1
"$CHECK_PYTHON" -m pytest -q --ignore=tests/integration > runs/logs/tests.log 2>&1
cat runs/logs/lint.log runs/logs/tests.log
