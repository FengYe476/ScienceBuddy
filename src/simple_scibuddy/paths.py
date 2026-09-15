"""Resolve portable project references without borrowing files from other checkouts."""

import os
from pathlib import Path, PureWindowsPath


def local_path(value, *, base, root=None, relative=True, field="path"):
    """Resolve a reference and reject absolute settings or symlinks escaping its root."""
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{field} must be a relative filesystem path")
    value = str(value)
    if not value.strip() or value.startswith("~") or "://" in value:
        raise ValueError(f"{field} must be a relative filesystem path")
    path = Path(value)
    if relative and (path.is_absolute() or PureWindowsPath(value).drive):
        raise ValueError(f"{field} must be relative, not an absolute path")
    boundary = Path(root if root is not None else base).resolve()
    resolved = (Path(base) / path).resolve()
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"{field} must stay inside the repository or resource directory")
    return resolved


def local_tree(directory, *, root=None):
    """Check nested symlinks without reading model weights or task contents."""
    directory = Path(directory)
    boundary = Path(root if root is not None else directory).resolve()
    local_path(directory, base=boundary, root=boundary, relative=False)
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            item = Path(parent) / name
            if item.is_symlink():
                local_path(item, base=boundary, root=boundary, relative=False, field="resource symlink")
