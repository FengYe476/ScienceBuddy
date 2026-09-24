"""Bounded Docker lifecycle; never execute agent code on the host."""

import asyncio
import json
import os
import uuid
from pathlib import Path


class ControllerFailure(RuntimeError):
    """The generated program exited or violated its RPC protocol."""


class ExecutionMemoryBudget(RuntimeError):
    """Docker confirmed this tool process exceeded its isolated memory budget."""

    def __init__(self, state):
        super().__init__('Execution container exceeded its 16 GiB memory budget (Docker OOMKilled)')
        self.state = state


async def command(*args, timeout=60):
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError(f"{args[0:2]} exited {proc.returncode}: {err.decode()[-2000:]}")
    return out.decode()


class Container:
    def __init__(self, image, folder, kind, *, lake=None):
        self.image, self.folder, self.kind, self.lake = image, Path(folder), kind, lake
        self.name = f"simple_scibuddy-{kind}-{uuid.uuid4().hex[:16]}"
        self.process = None
        self.created = False

    async def start(self, assets=None, harness=None):
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "container.json").write_text(json.dumps({"name": self.name, "image": self.image}))
        args = [
            "docker",
            "create",
            "--name",
            self.name,
            "--network",
            "none",
            "--cpus",
            "1" if self.kind == "controller" else "4",
            "--memory",
            "512m" if self.kind == "controller" else "16g",
            "-i",
        ]
        if self.lake:
            args += ["--mount", f"type=bind,src={self.lake},dst=/opt/data/biomni_data/data_lake,readonly"]
        if self.kind == "controller":
            args += ["--entrypoint", "python", self.image, "-u", "/tmp/controller.py"]
        else:
            args += [self.image]
        # Record ownership before create: a timeout can occur after Docker created the container.
        self.created = True
        await command(*args)
        if assets:
            await command("docker", "cp", str(assets), f"{self.name}:/workspace/assets")
        if harness:
            await command("docker", "cp", str(harness), f"{self.name}:/tmp/harness.py")
            controller = Path(__file__).parents[1] / "harness/controller.py"
            await command("docker", "cp", str(controller), f"{self.name}:/tmp/controller.py")
        self.stderr = (self.folder / "stderr.log").open("wb")
        await command("docker", "start", self.name)
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "attach",
            self.name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=self.stderr,
        )
        self.reader = await asyncio.create_subprocess_exec(
            "docker",
            "logs",
            "--follow",
            self.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            limit=8 * 1024 * 1024,
        )

    async def send(self, value):
        self.process.stdin.write((json.dumps(value) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self):
        while True:
            # Drain an oversized JSON record without keeping it in host memory.
            # The next execute call must start at the next complete record.
            oversized = False
            while True:
                try:
                    line = await self.reader.stdout.readuntil(b"\n")
                    break
                except asyncio.LimitOverrunError as exc:
                    oversized = True
                    await self.reader.stdout.readexactly(exc.consumed)
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
        # Inspect before cleanup removes the evidence. Only confirmed tool-memory
        # exhaustion becomes a task failure; unknown transport/daemon failures still abort.
        self.folder.mkdir(parents=True, exist_ok=True)
        try:
            state = json.loads(await command('docker', 'inspect', '--format', '{{json .State}}', self.name))
        except (RuntimeError, ValueError, TimeoutError) as exc:
            state = {'inspection_error': str(exc)}
        (self.folder / 'exit-state.json').write_text(json.dumps(state, indent=2) + '\n')
        if self.kind == 'execution' and state.get('OOMKilled') is True and state.get('Running') is False:
            raise ExecutionMemoryBudget(state) from cause
        error = ControllerFailure if self.kind == 'controller' else RuntimeError
        raise error(f'{self.kind} container stream closed; state={state}') from cause

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
        await command("docker", "cp", f"{self.name}:/workspace/.", str(output), timeout=60)

    async def close(self):
        try:
            if self.created:
                # Only this UUID-owned container; absent containers are already clean.
                exists = await command("docker", "ps", "-aq", "--filter", f"name=^/{self.name}$")
                if exists.strip():
                    await command("docker", "rm", "--force", self.name)
                self.created = False
        finally:
            if self.process and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            if hasattr(self, "reader") and self.reader.returncode is None:
                self.reader.terminate()
                await self.reader.wait()
            if hasattr(self, "stderr"):
                self.stderr.close()


# --- backend switch (added by this fork) -------------------------------------
# HPC clusters such as Anvil do not provide Docker. When SCIBUDDY_RUNTIME=apptainer is set,
# Container points at the Apptainer implementation. This has to happen at module level:
# broker.py:25 binds Container as a default argument of run_episode, which is evaluated at
# function-definition time, so rebinding the attribute later has no effect.
# See B1/B3 in adapter/DEVIATIONS.md.
if os.environ.get("SCIBUDDY_RUNTIME") == "apptainer":  # noqa: E402
    from simple_scibuddy.environments.apptainer_runtime import (  # noqa: E402,F401
        ApptainerContainer as Container,
    )
