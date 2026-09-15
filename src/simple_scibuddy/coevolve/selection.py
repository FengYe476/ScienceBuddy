"""Compare independent proposals with their parent on held-out validation tasks."""

import asyncio

from simple_scibuddy.artifacts import file_digest, write_json


def select_candidates(parent, folder, count, propose, preflight, evaluate, task_ids):
    if count < 1 or not task_ids:
        raise ValueError("Candidate search requires proposals and a validation set")
    candidates = []
    seen = {file_digest(parent)}
    # Generate every hypothesis before revealing any validation results.
    for index in range(1, count + 1):
        directory = folder / f"candidate-{index:02d}"
        directory.mkdir()
        path = directory / "harness.py"
        result = propose(path, index)
        valid = result['status'] == 'validated'
        executable = valid and preflight(path, directory / 'preflight')
        if executable and file_digest(path) in seen:
            result.update(status='duplicate')
            executable = False
        if executable:
            seen.add(file_digest(path))
        candidates.append({**result, 'id': index, 'path': str(path), 'executable': executable})

    async def score():
        async with asyncio.TaskGroup() as group:
            jobs = {'parent': group.create_task(evaluate(parent, folder / 'validation-parent'))}
            for candidate in candidates:
                if candidate['executable']:
                    jobs[candidate['id']] = group.create_task(evaluate(
                        candidate['path'], folder / f"validation-{candidate['id']:02d}"))
        return {key: job.result() for key, job in jobs.items()}

    results = asyncio.run(score())
    scores = {}
    for identity, episodes in results.items():
        if len(episodes) != len(task_ids) or {e['task_id'] for e in episodes} != set(task_ids):
            raise RuntimeError('Validation must cover the same complete task set')
        scores[identity] = {'correct': sum(e['reward'] == 1 for e in episodes),
                            'errors': sum(bool(e.get('error')) for e in episodes),
                            'tasks': len(episodes)}
    winner, best = None, scores['parent']['correct']
    for candidate in candidates:
        score = scores.get(candidate['id'])
        candidate['validation'] = score
        if score is not None and score['errors'] == 0 and score['correct'] > best:
            winner, best = candidate, score['correct']
    result = {'status': 'applied' if winner else 'rejected',
              'selected': winner['path'] if winner else str(parent),
              'selected_candidate': winner['id'] if winner else 0,
              'reason': winner.get('reason') if winner else 'No valid candidate strictly improved validation accuracy',
              'operations': winner.get('operations', []) if winner else [],
              'hypothesis': winner.get('hypothesis') if winner else None,
              'candidates': candidates,
              'validation': {'parent_correct': scores['parent']['correct'], 'selected_correct': best,
                             'tasks': len(task_ids), 'accuracy_delta': (best - scores['parent']['correct']) / len(task_ids)},
              'task_ids': list(task_ids)}
    write_json(folder / 'selection.json', result)
    return result
