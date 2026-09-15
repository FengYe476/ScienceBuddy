"""Explicit GPU diagnostic: compare checkpoint loading with BF16 weight updates.

Run only on a free, explicitly selected CUDA device. Inputs must be recorded
training prompts; this compares token distributions, not heldout task scores.
Launch with scripts/train.py's environment() so CUDA compatibility libraries
and project-local caches match the experiment runtime.
This local-only inspection also requires VLLM_ALLOW_INSECURE_SERIALIZATION=1
for apply_model callbacks; never expose this diagnostic engine to remote clients.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

from simple_scibuddy.inference.weight_audit import fingerprint as parameter_fingerprint


def fingerprint(model):
    import torch

    result = parameter_fingerprint(model)
    result["context_buffers"] = {}
    for name, tensor in model.named_buffers():
        if name.endswith("rotary_emb.cos_sin_cache"):
            value = tensor[:24576].detach().cpu().contiguous()
            result["context_buffers"][name] = {
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
    return result


def round_alog(model):
    import torch

    model._reload_probe_alog = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                model._reload_probe_alog[name] = param.detach().cpu().clone()
                param.copy_(param.to(torch.bfloat16).to(param.dtype))
    return len(model._reload_probe_alog)


def restore_alog(model):
    import torch

    with torch.no_grad():
        for name, original in model._reload_probe_alog.items():
            model.get_parameter(name).copy_(original)
    del model._reload_probe_alog


def reload_checkpoint(model, checkpoint, cast_bf16, preserve_recurrent=False, layerwise=False):
    import torch
    from safetensors import safe_open
    from vllm.config import set_current_vllm_config

    # One tensor per loader call mirrors the unbucketed SkyRL extractor. This
    # isolates casting/loading; it intentionally does not claim to test CUDA IPC.
    loaded = set()
    config = getattr(model, "language_model", model).vllm_config
    with set_current_vllm_config(config), torch.device(next(model.parameters()).device), torch.no_grad():
        if layerwise:
            from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import patch_numel_loaded
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            patch_numel_loaded()
            initialize_layerwise_reload(model)
        for shard in sorted(Path(checkpoint).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    value = handle.get_tensor(name)
                    if cast_bf16 and not (preserve_recurrent and name.endswith('.A_log')):
                        value = value.to(torch.bfloat16)
                    target_name = "language_model." + name if hasattr(model, "language_model") else name
                    loaded.update(model.load_weights([(target_name, value)]))
        if layerwise:
            finalize_layerwise_reload(model, config.model_config)
    return {"loaded_parameter_count": len(loaded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-model", help="Construct this wrapper, then load --model weights")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--fingerprint-only", action="store_true",
                        help="Record direct-loaded tensors without generating or reloading")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous diagnostic measurements"
    records = json.loads(args.prompts.read_text())["prompts"]
    assert records and all(item["split"] == "train" for item in records)
    assert len({item["task_id"] for item in records}) == len(records)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.initial_model or args.model, dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=24576, max_num_seqs=1, seed=42,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enable_sleep_mode=True, language_model_only=True, gpu_memory_utilization=0.65,
              generation_config="vllm")
    if args.initial_model:
        llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=True,
                                preserve_recurrent=True))
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=args.max_tokens,
                              seed=42, logprobs=5)
    prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in records]
    report = {"model": args.model, "initial_model": args.initial_model, "prompt_file": str(args.prompts),
              "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              "max_tokens": args.max_tokens, "conditions": [],
              "scope": "Single engine, training inputs; no optimizer or IPC transfer."}
    if args.fingerprint_only:
        report['conditions'].append({'name': 'direct_load', 'weights': llm.apply_model(fingerprint),
                                     'outputs': []})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        return

    def measure(name):
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        sequences = []
        for record, output in zip(records, outputs, strict=True):
            completion = output.outputs[0]
            first = completion.logprobs[0] if completion.logprobs else {}
            sequences.append({"task_id": record["task_id"],
                              "tokens": list(completion.token_ids),
                              "first_token_logprobs": {str(k): v.logprob for k, v in first.items()}})
        report["conditions"].append({"name": name, "weights": llm.apply_model(fingerprint),
                                     "outputs": sequences})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, "recorded", flush=True)

    measure("direct_load")
    measure("repeat_unchanged")
    if args.generation_only:
        return
    llm.sleep(level=1)
    llm.wake_up()
    measure("sleep_wake_unchanged")
    counts = llm.apply_model(round_alog)
    assert all(count > 0 for count in counts), "Expected Qwen recurrent parameters"
    measure("A_log_bf16_rounded")
    llm.apply_model(restore_alog)
    measure("A_log_restored")
    llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=False))
    measure("checkpoint_reloaded_original_dtype")
    llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=True))
    measure("checkpoint_reloaded_bf16")
    llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=True,
                            preserve_recurrent=True))
    measure("checkpoint_reloaded_bf16_preserve_recurrent")
    # Training initially discards weights with level-2 sleep. Restoring weights
    # and then KV storage must also preserve non-checkpoint model buffers.
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=True,
                            preserve_recurrent=True))
    llm.wake_up(tags=["kv_cache"])
    measure("sleep2_reload_preserve_recurrent")
    llm.apply_model(partial(reload_checkpoint, checkpoint=args.model, cast_bf16=True,
                            preserve_recurrent=True, layerwise=True))
    measure("layerwise_reload_preserve_recurrent")


if __name__ == "__main__":
    main()
