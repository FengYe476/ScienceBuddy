"""Opt-in receiver audit for explicit distributed weight-sync diagnostics."""

import hashlib
import json
import os
from pathlib import Path

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import NewInferenceWorkerWrap


def fingerprint(model):
    import torch

    result = {}
    for group, tensors in (("parameters", model.named_parameters()), ("buffers", model.named_buffers())):
        result[group] = {}
        for name, tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            result[group][name] = {
                'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                'dtype': str(value.dtype), 'shape': list(value.shape), 'stride': list(tensor.stride()),
            }
    return result


class AuditedInferenceWorker(NewInferenceWorkerWrap):
    def skyrl_finish_weight_update(self):
        super().skyrl_finish_weight_update()
        settings = json.loads(Path(os.environ['SKYRL_SIMPLE_SCIBUDDY_SETTINGS']).read_text())
        folder = Path(settings['run_dir']) / 'weight-audit'
        folder.mkdir(exist_ok=True)
        sequence = getattr(self, '_simple_scibuddy_audit_sequence', 0)
        record = {'pid': os.getpid(), 'update': sequence,
                  'weights': fingerprint(self.model_runner.model)}
        (folder / f'{os.getpid()}-{sequence:04d}.json').write_text(json.dumps(record))
        self._simple_scibuddy_audit_sequence = sequence + 1
