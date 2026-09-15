"""Real-container check using the actual seed harness and a scripted policy."""
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from simple_scibuddy.artifacts import ROOT
from simple_scibuddy.configuration import settings
from simple_scibuddy.harness.broker import run_episode


async def main():
    cfg = settings()
    from simple_scibuddy.data.dataset import TaskDataset
    from simple_scibuddy.harness.scientific import SYSTEM
    dataset = TaskDataset(cfg['dataset'])
    task = dataset.load(dataset.tasks('train')[0]['id'])
    responses = ['<execute>x = 6 * 7\nprint(x)</execute>', '<execute>print("x" * (9 * 1024 * 1024))</execute>', '<execute>print(x + 1)</execute>',
                 '<answer>INVALID_FIXTURE_ANSWER</answer>']
    policy = SimpleNamespace(calls=[], max_tokens=4096, context=32768)
    async def generate(messages):
        index = len(policy.calls)
        if index == 2:
            assert "exceeded" in messages[-1]["content"], "Output limit was not returned to harness"
        if index == 3:
            assert '43' in messages[-1]['content'], 'Persistent runtime state lost'
        value = {'text': responses[index], 'finish_reason': 'stop', 'messages': messages}
        policy.calls.append(value)
        return value
    policy.generate = generate
    folder = Path(tempfile.mkdtemp(prefix='container-contract-', dir=ROOT/'runs'))
    result = await run_episode(task, cfg['harness'], policy, folder, SYSTEM, runtime_image=cfg.get('runtime_image'),
                               baked_lake=cfg.get('baked_lake', False))
    assert result['reward'] == 0 and len(result['tool_calls']) == 3
    assert result['stop_reason'] == 'first_answer_evaluation'
    print(json.dumps({'passed': True, 'run_dir': str(folder)}))


asyncio.run(main())
