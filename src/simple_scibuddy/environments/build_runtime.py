"""Bake the frozen lake into a local image when nested Docker rejects read-only binds."""
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, write_json
from simple_scibuddy.configuration import settings
from simple_scibuddy.data.dataset import load_environment
from simple_scibuddy.paths import local_path


def build(cfg):
    lock = load_environment(cfg['dataset'])
    lake = Path(lock['data_lake']['path'])
    base_image = cfg.get('runtime_base_image', lock['image'])
    record = ROOT / 'runs/logs/runtime-image.json'
    if cfg.get('baked_lake') and cfg.get('runtime_image') and record.exists():
        previous = json.loads(record.read_text())
        if (previous.get('image') == cfg['runtime_image'] and previous.get('source_image') == lock['image']
                and previous.get('base_image', lock['image']) == base_image
                and previous.get('lake_digest') == lock['data_lake']['digest']):
            found = subprocess.run(['docker', 'image', 'inspect', cfg['runtime_image']], capture_output=True)
            if found.returncode == 0:
                print('Using existing frozen runtime', cfg['runtime_image'])
                return
    (ROOT / '.tmp').mkdir(exist_ok=True)
    if subprocess.run(['docker', 'image', 'inspect', base_image], capture_output=True).returncode:
        runtime = Path(cfg['dataset']) / 'runtime'
        if not (runtime / 'env/Dockerfile').is_file():
            raise RuntimeError('Missing pinned runtime image and frozen runtime build files; rerun setup with the source release available')
        subprocess.run(['docker', 'build', '-t', 'simple_scibuddy-runtime:release-base',
                        '-f', str(runtime / 'env/Dockerfile'), str(runtime)], check=True)
        base_image = subprocess.check_output(['docker', 'image', 'inspect', 'simple_scibuddy-runtime:release-base',
                                             '--format', '{{.Id}}'], text=True).strip()

    context = ROOT / '.tmp/sciencebuddy-runtime-context.tar'
    with tarfile.open(context, 'w') as tar:
        dockerfile = f"FROM {base_image}\nCOPY lake/ /opt/data/biomni_data/data_lake/\nUSER agent\n".encode()
        info = tarfile.TarInfo('Dockerfile')
        info.size = len(dockerfile)
        tar.addfile(info, io.BytesIO(dockerfile))
        for name, metadata in lock['data_lake']['files'].items():
            path = local_path(name, base=lake, field='data lake file')
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata['sha256']:
                    raise RuntimeError(f'Asset hash mismatch: {name}')
            info = tar.gettarinfo(str(path), arcname='lake/' + name)
            info.uid = info.gid = 0
            info.uname = info.gname = 'root'
            info.mode = 0o444
            with path.open('rb') as stream:
                tar.addfile(info, stream)
    with context.open('rb') as stream:
        subprocess.run(['docker', 'build', '-t', 'simple_scibuddy-runtime:frozen-lake', '-'],
                       stdin=stream, check=True, timeout=900, env=dict(os.environ, DOCKER_BUILDKIT='0'))
    identity = subprocess.check_output(['docker','image','inspect','simple_scibuddy-runtime:frozen-lake',
                                        '--format','{{.Id}}'], text=True).strip()
    local = ROOT / 'configs/local.json'
    value = json.loads(local.read_text()) if local.exists() else {}
    value.update(runtime_image=identity, runtime_base_image=base_image, baked_lake=True)
    write_json(local, value)
    write_json(ROOT / 'runs/logs/runtime-image.json', {'image': identity, 'source_image': lock['image'],
                                                'base_image': base_image,
                                                'lake_digest': lock['data_lake']['digest']})
    print('Built pinned local runtime', identity)

    context.unlink()


if __name__ == "__main__":
    build(settings())
