"""Preserve shared parameters during FSDP setup and recurrent precision during sync."""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import ray
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.weight_sync import CudaIpcTransferStrategy, get_transfer_strategy_cls
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPWeightExtractor
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase


@contextmanager
def preserve_conversion_aliases(model):
    groups = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    aliases = [names for names in groups.values() if len(names) > 1]
    original = model.to
    original_load = model.load_state_dict

    def restore_aliases():
        for names in aliases:
            shared = model.get_parameter(names[0])
            for name in names[1:]:
                parent, _, leaf = name.rpartition('.')
                setattr(model.get_submodule(parent), leaf, shared)

    def convert(*args, **kwargs):
        result = original(*args, **kwargs)
        restore_aliases()
        return result

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        # Loading meta parameters uses assign=True, which replaces each alias
        # separately. FSDP's post-load hook has refreshed the canonical parameter;
        # reconnect its other names before the optimizer is constructed.
        restore_aliases()
        return result

    with patch.object(model, 'to', convert), patch.object(model, 'load_state_dict', load):
        yield aliases


class SharedParameterStrategy(FSDPStrategy):
    def _fsdp_init_model(self, model, is_train=True, is_wrapped=False):
        module = model.model if is_wrapped else model
        # Module.to(meta) creates separate Parameters for tied embeddings. Restore
        # aliases before fully_shard builds its parameter groups and optimizer.
        with preserve_conversion_aliases(module) as aliases:
            result = super()._fsdp_init_model(model, is_train=is_train, is_wrapped=is_wrapped)
            for names in aliases:
                if any(module.get_parameter(name) is not module.get_parameter(names[0]) for name in names[1:]):
                    raise RuntimeError(f'FSDP initialization broke shared parameters: {names}')
            return result


class RecurrentWeightExtractor(FSDPWeightExtractor):
    def extract_weights(self, dtype):
        # vLLM explicitly stores A_log in FP32, even for BF16 inference. Casting
        # it to BF16 changes generation and makes live sync differ from HF load.
        preserved = {
            self.weight_prefix + name: self._gather_tensor(param).float().detach().contiguous()
            for name, param in self.model.state_dict().items() if name.endswith('.A_log')
        }
        for chunk in super().extract_weights(dtype):
            tensors = [preserved.get(name, tensor)
                       for name, tensor in zip(chunk.names, chunk.tensors, strict=True)]
            yield replace(chunk, tensors=tensors, dtypes=[str(t.dtype) for t in tensors])

    def get_weight_metadata(self, dtype):
        metadata = super().get_weight_metadata(dtype)
        metadata['dtype_names'] = [
            'float32' if name.endswith('.A_log') else item
            for name, item in zip(metadata['names'], metadata['dtype_names'], strict=True)
        ]
        return metadata


class RecurrentPolicyWorker(FSDPPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps=None):
        # Upstream has no strategy-factory hook. Scope this substitution to this
        # worker's synchronous initialization; each Ray worker has its own process.
        with patch('skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker.FSDPStrategy', SharedParameterStrategy):
            return super().init_model(model_path, num_training_steps)

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg):
        # Keep upstream grouping and transfer selection, but install the extractor
        # before sender initialization (some backends capture it at construction).
        bucketing = get_transfer_strategy_cls(
            weight_sync_backend=inference_engine_cfg.weight_sync_backend,
            colocate_all=self.cfg.placement.colocate_all,
        ) is CudaIpcTransferStrategy
        self.weight_extractor = RecurrentWeightExtractor(
            self.model.model, enable_bucketing=bucketing,
            batch_size_threshold_gb=(inference_engine_cfg.weight_transfer_threshold_cuda_ipc_GB
                                     if bucketing else 0.0),
            weight_prefix='language_model.' if self._is_multimodal_lm_only else '',
        )
        await PolicyWorkerBase.init_weight_sync_state(self, inference_engine_client, inference_engine_cfg)


PolicyWorker = ray.remote(num_gpus=1)(RecurrentPolicyWorker)
