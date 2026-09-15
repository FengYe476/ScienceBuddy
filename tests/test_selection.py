import pytest

from simple_scibuddy.coevolve.selection import select_candidates


@pytest.mark.parametrize('scores,winner', [([1, 3, 2], 2), ([1, 1, 0], 0), ([0, 0, 0], 0), ([2, 2, 1], 1)])
def test_candidates_share_parent_and_best_validation_wins(tmp_path, scores, winner):
    parent = tmp_path / 'parent.py'
    parent.write_text('parent')
    proposals, evaluations = [], []
    def propose(path, index):
        assert not evaluations, 'No validation information may inform this candidate pool'
        proposals.append(index)
        path.write_text(str(index))
        return {'status': 'validated'}
    async def evaluate(path, destination):
        assert proposals == [1, 2, 3]
        from pathlib import Path
        contents = Path(path).read_text()
        correct = 1 if contents == 'parent' else scores[int(contents) - 1]
        evaluations.append(contents)
        return [{'task_id': str(i), 'reward': int(i < correct)} for i in range(3)]
    result = select_candidates(parent, tmp_path, 3, propose, lambda *a: True, evaluate, ['0', '1', '2'])
    assert result['selected_candidate'] == winner
    assert len(evaluations) == 4


def test_validation_errors_cannot_win(tmp_path):
    parent = tmp_path/'parent.py'
    parent.write_text('parent')
    def propose(path, index):
        path.write_text('candidate')
        return {'status': 'validated'}
    async def evaluate(path, destination):
        return [{'task_id': 'x', 'reward': int('parent' not in destination.name),
                 'error': None if 'parent' in destination.name else 'container failed'}]
    result = select_candidates(parent, tmp_path, 1, propose, lambda *a: True, evaluate, ['x'])
    assert result['selected_candidate'] == 0
