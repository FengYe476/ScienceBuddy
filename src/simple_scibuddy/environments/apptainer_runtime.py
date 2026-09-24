"""Apptainer container backend for HPC environments without Docker.

Same interface as environments/runtime.py:Container (start / send / receive /
execute / collect / close / closed_stream) and the same wire protocol, so
broker.py, phase.py and training/generator.py need no changes.

Three things are kept aligned with the upstream Docker implementation:
  * bind mounts recreate the absolute paths promised by
    harness/scientific.py:SYSTEM -- /workspace/assets, /opt/scitrace and
    /opt/data/biomni_data/data_lake
  * `unshare -rn` provides network isolation with errno 101 (ENETUNREACH),
    matching `docker --network none`
  * the discard semantics for protocol records over 8 MiB (runtime.py:99-125)

Known deviation (see adapter/DEVIATIONS.md):
  * no `--memory 16g` cgroup limit, so the cgroup path to the
    execution_memory_budget stop reason does not exist; job-level memory is
    bounded by Slurm instead.

Selected through environment variables:
    SCIBUDDY_RUNTIME=apptainer   enable this backend (runtime.py switches Container on it)
    SCIBUDDY_SCITRACE=<dir>      directory bind-mounted at /opt/scitrace
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from simple_scibuddy.environments.runtime import ControllerFailure, ExecutionMemoryBudget


class ApptainerMemoryBudget(ExecutionMemoryBudget):
    """Same type as upstream (broker.py:126 catches ExecutionMemoryBudget), truthful message.

    Upstream runtime.py:18 hardcodes "16 GiB ... (Docker OOMKilled)". That text reaches the
    model as a tool observation and then becomes improver diagnostic evidence through
    evidence.py:public_evidence. In this fork it is wrong in three ways: there is no Docker,
    the limit comes from RLIMIT_AS rather than a cgroup, and the number is 32 GiB rather than
    16 GiB. Passing it through unchanged would have the improver revise the harness on a
    false premise.
    """

    def __init__(self, state, cap_bytes):
        RuntimeError.__init__(
            self,
            f"Execution container was killed (returncode {state.get('returncode')}); "
            f"its address space limit is {cap_bytes // 1024 ** 3} GiB (Apptainer, RLIMIT_AS). "
            "Load a slice or an aggregate instead of the whole file.",
        )
        self.state = state

GUEST_WORKSPACE = "/workspace"
GUEST_SCITRACE = "/opt/scitrace"
GUEST_LAKE = "/opt/data/biomni_data/data_lake"

# Both sides dup the protocol channel onto private file descriptors first, then replace
# sys.stdin/sys.stdout with harmless stand-ins. Executed code -- the harness program and any
# tool code the model generates -- therefore cannot close or consume the protocol stream.
# Without this, a single input() or sys.stdout.close() makes the container exit with
# returncode 1, which raises RuntimeError from closed_stream and tears down the TaskGroup.
CHANNEL_PRELUDE = '''
import io, json, os, sys

_pin = os.fdopen(os.dup(0), "r")
_pout = os.fdopen(os.dup(1), "w")
sys.stdin = io.StringIO("")          # user code cannot read the protocol input
sys.stdout = io.StringIO()           # stray print() goes to a sink, not the protocol stream


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

# The persistent REPL on the execution-container side. Upstream bakes this into the release
# image and it is not in the repository; the protocol follows what runtime.py:74-77 expects:
# read {"code": ...}, reply {"stdout", "error"}.
REPL_SOURCE = CHANNEL_PRELUDE + '''
import contextlib, resource, traceback

# Stands in for Docker's --memory 16g (see upstream runtime.py:57). RLIMIT_AS bounds virtual
# address space rather than RSS, so it is loosened to 32 GiB: the goal is for a runaway
# allocation to raise a catchable MemoryError instead of having the kernel OOM-kill the whole
# container, which would lose the entire round.
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
        # The code may have closed the redirect buffer (sys.stdout.close()), in which case
        # getvalue() raises ValueError -- that must not break the loop.
        out = buffer.getvalue()
    except BaseException:
        out = ""
        error = error or "Tool closed its output stream; captured output was lost."
    try:
        _emit({"stdout": out, "error": error})
    except BaseException:
        break                         # only leave the loop if the protocol stream is truly gone
'''


def runtime_binary():
    return shutil.which("apptainer") or shutil.which("singularity")


def _network_isolation():
    """`unshare -rn` drops networking in a user namespace: unprivileged, same errno as Docker."""
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
            raise ControllerFailure("apptainer/singularity not found")
        candidate = Path(str(image).replace("apptainer:", "", 1))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_file():
            raise ControllerFailure(f"image does not exist: {candidate}")
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
                raise ControllerFailure("the controller requires a harness program")
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
        """Mirrors the triage in runtime.py:126-138, using observable evidence instead of
        docker inspect.

        Upstream downgrades only a confirmed memory exhaustion to a single-task failure and
        aborts the experiment on any other container death ("unknown transport/daemon failures
        still abort"). Unprivileged Apptainer cannot set a cgroup, so there is no OOMKilled
        flag to read; the exit code is used instead. Death by signal (128+N, and OOM-kill is
        137) counts as resource exhaustion and is recoverable; everything else is still raised
        as an infrastructure failure.
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
            cap = int(os.environ.get("SCIBUDDY_MEM_BYTES", str(32 * 1024**3)))
            raise ApptainerMemoryBudget(state, cap) from cause
        error = ControllerFailure if self.kind == "controller" else RuntimeError
        raise error(f"{self.kind} container stream closed; state={state}") from cause

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
