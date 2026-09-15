"""Public resource inventory adapted from ScienceBuddy-RSI algorithms/capabilities.py."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

from simple_scibuddy.data.dataset import load_environment
from simple_scibuddy.paths import local_path


def dataset_capabilities(dataset):
    return public_capabilities(load_environment(dataset.root))


def public_capabilities(environment):
    lake = Path(environment['data_lake']['path']).resolve()
    files = []
    for name, metadata in sorted(environment['data_lake']['files'].items()):
        path = local_path(name, base=lake, field='resource file')
        if not path.is_relative_to(lake):
            raise ValueError('Resource path escapes the public data lake')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != metadata['sha256']:
            raise ValueError('Public resource hash mismatch: ' + name)
        item = {'path': '/opt/data/biomni_data/data_lake/'+name,
                'bytes': path.stat().st_size}
        if name.endswith('.parquet'):
            import pyarrow.parquet as pq
            item.update(format='parquet', reader='pandas.read_parquet', columns=pq.read_schema(path).names)
        elif name.endswith('.pkl'):
            import pandas as pd
            frame = pd.read_pickle(path)
            item.update(format='trusted frozen pandas pickle', reader='pandas.read_pickle', columns=list(frame.columns))
        elif name.endswith('.gmt'):
            item.update(format='tab-separated gene sets',
                        fields=['gene set name', 'source description/URL', 'gene symbols from column 3 onward'])
        elif name.endswith(('.tsv', '.tsv.gz')) or name == 'hgnc_complete_set.txt':
            opener = gzip.open if name.endswith('.gz') else open
            with opener(path, 'rt') as stream:
                columns = next(csv.reader(stream, delimiter='\t'))
            item.update(format='TSV', reader="pandas.read_csv(path, sep='\\t')", columns=columns)
        elif name.endswith('.json'):
            value = json.loads(path.read_text())
            item.update(format='JSON', top_level_keys=sorted(value) if isinstance(value, dict) else ['array'])
        else:
            item['format'] = 'FASTA sequence' if name.endswith('.fasta') else 'text'
        files.append(item)
    return {'scope': 'Public runtime capabilities, not solutions or evaluation results',
            'verified_files': len(files), 'image': environment['image'],
            'network': environment.get('network', 'none'),
            'execution': 'Persistent Python, not a shell. Submit Python code directly; no python -c or heredoc commands.',
            'libraries': ['Python standard library', 'numpy', 'pandas', 'pyarrow', 'BioPython', 'requests', 'BeautifulSoup'],
            'task_files': '/workspace/assets (includes task_prompt.txt and provided articles)',
            'resource_root': '/opt/data/biomni_data/data_lake',
            'public_resource_guide': (lake/'README.md').read_text() if (lake/'README.md').exists() else '',
            'files': files,
            'interface': {'api.execute(code)': 'Run Python in the scientific interpreter; returns stdout and error',
                          'api.generate(messages)': 'Request the solver under the unchanged broker budget'},
            'boundary': 'Scientific files are available even when no domain-specific helper is registered. '
                        'A harness may organize lookup, expose a used helper, or correct tool usage. '
                        'Do not infer missing data merely from zero tool calls. '
                        'File availability does not prove every benchmark label; use Val to test hypotheses.'}
