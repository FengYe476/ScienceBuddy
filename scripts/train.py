#!/usr/bin/env python3
"""Run one TOML experiment in the locked project environment, with a foreground log."""

import fcntl
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simple_scibuddy.paths import local_path


def environment():
    env = dict(os.environ)
    # Do not import code from another checkout through the calling shell.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    local = {
        "UV_CACHE_DIR": ".cache/uv", "UV_PROJECT_ENVIRONMENT": ".venv-skyrl",
        "TMPDIR": ".tmp", "XDG_CACHE_HOME": ".cache", "HF_HOME": ".cache/huggingface",
        "TORCH_HOME": ".cache/torch", "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torch_inductor", "VLLM_CACHE_ROOT": ".cache/vllm",
        "TRITON_CACHE_DIR": ".cache/triton", "CUDA_CACHE_PATH": ".cache/cuda", "RAY_TMPDIR": ".tmp",
    }
    env.update({key: str(local_path(value, base=ROOT, field=key)) for key, value in local.items()})
    env.update(SKYRL_WORKSPACE=str(ROOT), UV_PYTHON_DOWNLOADS="never", UV_LINK_MODE="hardlink",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", RAY_USAGE_STATS_ENABLED="0", MAX_JOBS="4",
               SKYRL_LD_LIBRARY_PATH_EXPORT="1",
               RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook")
    env["PATH"] = f"{ROOT}/.skyrl-tools/bin:{ROOT}/.venv-skyrl/bin:" + env.get("PATH", "")
    # Use the installed system toolchain unless a repository-local override is provided.
    if env.get("SIMPLE_SCIBUDDY_CUDA_HOME"):
        env["CUDA_HOME"] = str(local_path(env["SIMPLE_SCIBUDDY_CUDA_HOME"], base=ROOT, field="CUDA override"))
    if env.get("SIMPLE_SCIBUDDY_CUDA_COMPAT"):
        compatibility = str(local_path(env["SIMPLE_SCIBUDDY_CUDA_COMPAT"], base=ROOT, field="CUDA compatibility"))
        env["LD_LIBRARY_PATH"] = compatibility + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3,4,5,6,7"
    env.setdefault("WANDB_MODE", "online")
    for name in ("wandb.env", "improver.env"):
        secrets = local_path(Path(".secrets") / name, base=ROOT, field="credentials file")
        if secrets.exists():
            # Read literal KEY=value / export KEY=value entries without executing shell code.
            for line in secrets.read_text().splitlines():
                words = shlex.split(line, comments=True)
                if words[:1] == ["export"]:
                    words = words[1:]
                if not words:
                    continue
                if len(words) != 1 or "=" not in words[0]:
                    raise ValueError(f"{name} must contain literal KEY=value assignments")
                key, value = words[0].split("=", 1)
                if not key.isidentifier():
                    raise ValueError(f"Invalid environment variable name in {name}")
                env.setdefault(key, value)
    return env


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/train.py config.toml [--dry-run]")
    config = str(local_path(sys.argv[1], base=ROOT, relative=False, field="configuration file")) if sys.argv[1] not in {"-h", "--help"} else sys.argv[1]
    for directory in (ROOT / "runs/logs", ROOT / ".tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    env = environment()
    uv = shutil.which("uv", path=env["PATH"])
    if not uv:
        raise SystemExit("Run scripts/setup.sh first: project uv is unavailable")
    with (ROOT / ".tmp/sciencebuddy.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A ScienceBuddy launcher is already running") from None
        log = ROOT / "runs/logs" / datetime.now(timezone.utc).strftime("sciencebuddy-%Y%m%d-%H%M%S.log")
        command = [uv, "run", "--isolated", "--frozen", "--extra", "train",
                   "-m", "simple_scibuddy.coevolve.experiment", config, *sys.argv[2:]]
        with log.open("w") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            def interrupt(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupt)
            signal.signal(signal.SIGHUP, interrupt)
            try:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                status = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGINT)
                process.wait()
                status = 130
            raise SystemExit(status if status >= 0 else 128 - status)


if __name__ == "__main__":
    main()
