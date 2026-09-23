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

from simple_scibuddy.environments.runtime import ControllerFailure, ExecutionMemoryBudget

GUEST_WORKSPACE = "/workspace"
GUEST_SCITRACE = "/opt/scitrace"
GUEST_LAKE = "/opt/data/biomni_data/data_lake"

# 两侧都先把协议通道 dup 到私有 fd，再把 sys.stdin/stdout 换成安全替身。
# 被执行的代码（harness 程序、模型生成的工具代码）因此无法关闭或消费协议流。
# 不做这一步时，一段 input()/sys.stdout.close() 就会让容器以 returncode 1 退出，
# 进而触发 closed_stream 的 RuntimeError，炸掉整个 TaskGroup。
CHANNEL_PRELUDE = '''
import io, json, os, sys

_pin = os.fdopen(os.dup(0), "r")
_pout = os.fdopen(os.dup(1), "w")
sys.stdin = io.StringIO("")          # 用户代码读不到协议输入
sys.stdout = io.StringIO()           # 误用的 print 落进废纸篓而非协议流


def _emit(value):
    _pout.write(json.dumps(value) + "\\n")
    _pout.flush()
'''

CONTROLLER_SOURCE = CHANNEL_PRELUDE + '''
import runpy, traceback


class API:
    def call(self, op, **payload):
        _emit({"op": op, **payload})
        line = _pin.readline()
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
        task = json.loads(_pin.readline())
        runpy.run_path(sys.argv[1])["run"](task, API())
        _emit({"op": "end"})
    except BaseException:
        _emit({"op": "error", "error": traceback.format_exc()})
'''

# 执行容器侧的常驻 REPL。上游把它烤在 release 镜像里，仓库中没有；
# 协议按 runtime.py:74-77 的期望：读 {"code": ...}，回 {"stdout", "error"}。
REPL_SOURCE = CHANNEL_PRELUDE + '''
import contextlib, resource, traceback

# 对齐 Docker 的 --memory 16g（见 upstream runtime.py:57）。RLIMIT_AS 限制的是
# 虚拟地址空间而非 RSS，所以放宽到 32 GiB：目的是让失控分配抛出可捕获的
# MemoryError，而不是被内核 OOM-kill 掉整个容器（那会丢掉整轮实验）。
try:
    _cap = int(os.environ.get("SCIBUDDY_MEM_BYTES", str(32 * 1024 ** 3)))
    resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))
except (ValueError, OSError):
    pass

state = {"__name__": "__scitrace__"}
for line in _pin:
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
        try:
            error = traceback.format_exc(limit=8)
        except BaseException:
            error = "Tool raised an exception that could not be formatted."
    try:
        # 代码可能把 redirect 用的缓冲区关掉（sys.stdout.close()），
        # 此时 getvalue() 会抛 ValueError —— 不能让它中断循环。
        out = buffer.getvalue()
    except BaseException:
        out = ""
        error = error or "Tool closed its output stream; captured output was lost."
    try:
        _emit({"stdout": out, "error": error})
    except BaseException:
        break                         # 只有协议流真的断了才退出循环
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
        """对齐 runtime.py:126-138 的分流，但用可观测的证据代替 docker inspect。

        上游只把确认的内存耗尽降级成单题失败，其他容器死亡一律中止实验
        （"unknown transport/daemon failures still abort"）。Apptainer 非特权
        模式设不了 cgroup，拿不到 OOMKilled 标志，所以改用退出码判断：
        被信号杀死（128+N，OOM-kill 是 137）视为资源耗尽，可恢复；其余
        仍然当作基础设施故障上抛。
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        code = self.process.returncode if self.process else None
        stderr_tail = ""
        try:
            log = self.folder / "stderr.log"
            if log.is_file():
                stderr_tail = log.read_text(errors="replace")[-2000:]
        except OSError:
            pass
        state = {"runtime": "apptainer", "returncode": code,
                 "killed_by_signal": code is not None and code < 0 or (code or 0) > 128,
                 "stderr_tail": stderr_tail}
        (self.folder / "exit-state.json").write_text(json.dumps(state, indent=2) + "\n")
        if self.kind == "execution" and state["killed_by_signal"]:
            raise ExecutionMemoryBudget(state) from cause
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
