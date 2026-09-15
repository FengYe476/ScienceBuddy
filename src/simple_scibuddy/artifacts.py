"""Run identities and JSON artifacts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_identity(folder, pattern="*"):
    folder = Path(folder)
    return digest({str(p.relative_to(folder)): file_digest(p) for p in sorted(folder.rglob(pattern))
                   if p.is_file() and "__pycache__" not in p.parts})
