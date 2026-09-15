"""Exercise upstream step-wise GRPO on unequal-length episodes without starting Ray."""
from types import SimpleNamespace

import torch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer


class Batch(dict):
    def to(self, device):
        return self

cfg = SkyRLTrainConfig.from_cli_overrides([
    'generator.step_wise_trajectories=true', 'trainer.algorithm.advantage_estimator=grpo'])
trainer = SimpleNamespace(cfg=cfg, all_metrics={})
for n in (8, 16):
    for varied in (True, False):
        rewards, endings = [], []
        # One task, n sampled episodes, different call counts per episode.
        for i in range(n):
            for j in range(1 + i % 3):
                final = j == i % 3
                endings.append(final)
                rewards.append([0., float(i % 2) if varied and final else 0.])
        batch = Batch(rewards=torch.tensor(rewards), response_mask=torch.ones(len(rewards), 2), values=None)
        batch.metadata = {'uids': ['task']*len(rewards), 'is_last_step': endings, 'avg_response_length': 2}
        result = RayPPOTrainer.compute_advantages_and_returns(trainer, batch)
        assert torch.isfinite(result['advantages']).all()
        if not varied:
            assert torch.count_nonzero(result['advantages']) == 0
        cursor = 0
        for i in range(n):
            count = 1 + i % 3
            assert torch.allclose(result['advantages'][cursor:cursor+count],
                                  result['advantages'][cursor].expand(count, 2))
            if varied:
                assert (result['advantages'][cursor,0] > 0) == bool(i % 2)
            cursor += count
print('STEP-WISE GRPO PASS: 8/16 episodes, unequal calls, terminal credit, zero variance')
