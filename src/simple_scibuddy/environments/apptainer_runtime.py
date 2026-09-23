"""Apptainer 实现的容器后端，用于没有 Docker 的 HPC 环境。

与 environments/runtime.py:Container 同接口（start / send / receive / execute /
collect / close / closed_stream），协议完全一致，所以 broker.py、phase.py、
training/generator.py 都不需要改动。

对齐上游 Docker 实现的三点：
  * bind mount 做出 harness/scientific.py:SYSTEM 承诺的绝对路径
    /workspace/assets、/opt/scitrace、/opt/data/biomni_data/data_lake
  * `unshare -rn` 提供网络隔离，errno 101 (ENETUNREACH)，与 `docker --network none` 一致
  * 8 MiB 超长协议记录的丢弃语义（runtime.py:99-125）

已知偏差（见 adapter/DEVIATIONS.md）：
  * 无 `--memory 16g` cgroup 限制，因此不产生 execution_memory_budget 停止原因；
    作业级内存由 Slurm 约束。

通过环境变量选择：
    SCIBUDDY_RUNTIME=apptainer       启用本后端（runtime.py 据此切换 Container）
    SCIBUDDY_SCITRACE=<dir>          bind 到 /opt/scitrace 的目录
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from simple_scibuddy.environments.runtime import ControllerFailure

GUEST_WORKSPACE = "/workspace"
GUEST_SCITRACE = "/opt/scitrace"
GUEST_LAKE = "/opt/data/biomni_data/data_lake"

CONTROLLER_SOURCE = '''
import json, runpy, sys, traceback


class API:
    def call(self, op, **payload):
        print(json.dumps({"op": op, **payload}), flush=True)
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("Broker disconnected")
        return json.loads(line)

    def generate(self, messages):
        return self.call("generate", messages=messages)

    def execute(self, code):
        return self.call("execute", code=code)

    def submit(self, text):
        return self.call("submit", text=text)


if __name__ == "__main__":
    try:
        task = json.loads(sys.stdin.readline())
        runpy.run_path(sys.argv[1])["run"](task, API())
        print(json.dumps({"op": "end"}), flush=True)
    except BaseException:
        print(json.dumps({"op": "error", "error": traceback.format_exc()}), flush=True)
'''

# 执行容器侧的常驻 REPL。上游把它烤在 release 镜像里，仓库中没有；
# 协议按 runtime.py:74-77 的期望：读 {"code": ...}，回 {"stdout", "error"}。
REPL_SOURCE = '''
import contextlib, io, json, sys, traceback

state = {"__name__": "__scitrace__"}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        code = json.loads(line).get("code", "")
    except ValueError:
        continue
    buffer = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(code, "<tool>", "exec"), state)
    except BaseException:
        error = traceback.format_exc(limit=8)
    print(json.dumps({"stdout": buffer.getvalue(), "error": error}), flush=True)
'''


def runtime_binary():
    return shutil.which("apptainer") or shutil.which("singularity")


def _network_isolation():
    """`unshare -rn` 在用户命名空间里断网，非特权可用，errno 与 Docker 一致。"""
    if not shutil.which("unshare"):
        return []
    try:
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20)
        return ["unshare", "-rn"] if probe.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


class ApptainerContainer:
    _prefix = None

    def __init__(self, image, folder, kind, *, lake=None):
        self.image = image
        self.folder = Path(folder)
        self.kind = kind
        self.lake = lake
        self.process = None
        self.stderr = None
        self.workspace = None
        self.runtime = runtime_binary()
        if not self.runtime:
            raise ControllerFailure("找不到 apptainer/singularity")
        candidate = Path(str(image).replace("apptainer:", "", 1))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_file():
            raise ControllerFailure(f"镜像不存在: {candidate}")
        self.sif = candidate

    def _argv(self, script, harness_target):
        if ApptainerContainer._prefix is None:
            ApptainerContainer._prefix = _network_isolation()
        binds = [f"{self.workspace}:{GUEST_WORKSPACE}"]
        scitrace = os.environ.get("SCIBUDDY_SCITRACE", "")
        if scitrace and Path(scitrace).is_dir():
            binds.append(f"{scitrace}:{GUEST_SCITRACE}:ro")
        if self.lake and Path(self.lake).is_dir():
            binds.append(f"{self.lake}:{GUEST_LAKE}:ro")

        argv = list(ApptainerContainer._prefix)
        argv += [self.runtime, "exec", "--containall", "--cleanenv"]
        for b in binds:
            argv += ["--bind", b]
        argv += ["--pwd", GUEST_WORKSPACE, str(self.sif), "python", "-u",
                 f"{GUEST_WORKSPACE}/.runner/{script.name}"]
        if harness_target:
            argv.append(f"{GUEST_WORKSPACE}/.runner/{harness_target.name}")
        return argv

    async def start(self, assets=None, harness=None):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix=f"sb-{self.kind}-"))
        scripts = self.workspace / ".runner"
        scripts.mkdir()
        (self.workspace / "assets").mkdir(exist_ok=True)
        (self.workspace / "tmp").mkdir(exist_ok=True)
        if assets and Path(assets).is_dir():
            shutil.copytree(assets, self.workspace / "assets", dirs_exist_ok=True)

        harness_target = None
        if self.kind == "controller":
            if harness is None:
                raise ControllerFailure("controller 需要 harness 程序")
            harness_target = scripts / "harness.py"
            shutil.copyfile(harness, harness_target)
            script = scripts / "controller.py"
            script.write_text(CONTROLLER_SOURCE)
        else:
            script = scripts / "repl.py"
            script.write_text(REPL_SOURCE)

        argv = self._argv(script, harness_target)
        self.stderr = (self.folder / "stderr.log").open("wb")
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            limit=8 * 1024 * 1024,
        )
        (self.folder / "container.json").write_text(json.dumps({
            "runtime": "apptainer",
            "kind": self.kind,
            "image": str(self.sif),
            "workspace": str(self.workspace),
            "lake_bound": bool(self.lake),
            "network_isolated": bool(ApptainerContainer._prefix),
            "argv": argv,
        }, indent=2) + "\n")

    async def send(self, value):
        self.process.stdin.write((json.dumps(value) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self):
        while True:
            oversized = False
            while True:
                try:
                    line = await self.process.stdout.readuntil(b"\n")
                    break
                except asyncio.LimitOverrunError as exc:
                    oversized = True
                    await self.process.stdout.readexactly(exc.consumed)
                except asyncio.IncompleteReadError as exc:
                    await self.closed_stream(exc)
            if oversized:
                if self.kind == "execution":
                    return {"stdout": "", "error": "Tool output exceeded 8 MiB. Print a smaller summary."}
                raise ControllerFailure("Harness protocol record exceeded 8 MiB")
            if not line:
                await self.closed_stream()
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self.kind == "controller" or ("stdout" in value and "error" in value):
                return value

    async def closed_stream(self, cause=None):
        self.folder.mkdir(parents=True, exist_ok=True)
        state = {"runtime": "apptainer",
                 "returncode": self.process.returncode if self.process else None}
        (self.folder / "exit-state.json").write_text(json.dumps(state, indent=2) + "\n")
        error = ControllerFailure if self.kind == "controller" else RuntimeError
        raise error(f"{self.kind} 容器流已关闭; state={state}") from cause

    async def execute(self, code, timeout=120):
        try:
            await self.send({"code": code})
        except (BrokenPipeError, ConnectionResetError) as exc:
            await self.closed_stream(exc)
        result = await asyncio.wait_for(self.receive(), timeout)
        if result.get("error") is None and result["stdout"].startswith("Error:"):
            result["error"] = result["stdout"]
        return result

    async def collect(self):
        output = self.folder / "outputs"
        output.mkdir(exist_ok=True)
        if self.workspace and self.workspace.is_dir():
            for item in self.workspace.iterdir():
                if item.name == ".runner":
                    continue
                target = output / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copyfile(item, target)

    async def close(self):
        try:
            if self.process and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        finally:
            if self.stderr:
                self.stderr.close()
            if self.workspace and self.workspace.is_dir():
                shutil.rmtree(self.workspace, ignore_errors=True)
