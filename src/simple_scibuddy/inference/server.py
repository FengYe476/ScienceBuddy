"""Phase-owned vLLM services. Stop only the process group started here."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager

from simple_scibuddy.artifacts import write_json


@contextmanager
def serve(model, gpu, folder, context=32768, memory=0.75):
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
               "--served-model-name", "sciencebuddy", "--host", "127.0.0.1", "--port", str(port),
               "--language-model-only", "--reasoning-parser", "qwen3", "--max-model-len", str(context), "--max-num-seqs", "1",
               "--gpu-memory-utilization", str(memory), "--no-enable-prefix-caching",
               "--no-enable-chunked-prefill", "--no-enable-log-requests"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD="spawn")
    with (folder / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / "service.json", {"pid": process.pid, "model": str(model), "gpu": gpu,
                                             "base_url": url, "command": command})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {process.returncode}; see {log.name}")
                try:
                    with opener.open(url + "/models", timeout=2) as response:
                        if json.load(response).get("data"):
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"vLLM startup timed out; see {log.name}")
                time.sleep(2)
            print(f"Local vLLM ready: GPU {gpu}, {model}", flush=True)
            yield url
        finally:
            # A dead API parent may still have owned workers in its process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@contextmanager
def serve_pool(model, devices, folder, context, share_last=False):
    """Start one replica per GPU concurrently; clean up every successful startup."""
    with ExitStack() as stack, ThreadPoolExecutor(max_workers=len(devices)) as executor:
        contexts = [serve(model, gpu, folder / f"gpu-{i}", context,
                          memory=0.4 if share_last and i == len(devices) - 1 else 0.75)
                    for i, gpu in enumerate(devices)]
        futures = [executor.submit(cm.__enter__) for cm in contexts]
        urls, errors = [], []
        for cm, future in zip(contexts, futures):
            try:
                urls.append(future.result())
                stack.callback(cm.__exit__, None, None, None)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        yield urls


class CompletionClient:
    """The RecordingPolicy interface over the local vLLM completions endpoint."""

    model_name = "sciencebuddy"

    def __init__(self, url):
        import httpx

        self.client = httpx.AsyncClient(base_url=url + "/", timeout=300, trust_env=False)

    async def generate(self, request, model):
        import httpx

        payload = dict(request["sampling_params"], model=model, prompt=request["prompt_token_ids"][0],
                       return_token_ids=True)
        # Retry only the read-only generation request, never a tool action or submission.
        for attempt in range(3):
            try:
                response = await self.client.post("completions", json=payload)
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                print(f"Local inference transport error; retry {attempt + 1}/2", flush=True)
                await asyncio.sleep(attempt + 1)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return {"response_ids": [choice["token_ids"]], "responses": [choice["text"]],
                "response_logprobs": [choice["logprobs"]["token_logprobs"]],
                "stop_reasons": [choice["finish_reason"]]}

    async def close(self):
        await self.client.aclose()
