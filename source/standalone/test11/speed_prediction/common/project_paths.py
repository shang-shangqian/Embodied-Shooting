"""Resolve references saved before the project moved to its current directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
LEGACY_PROJECT_ROOT = Path('/home/jby/IsaacLab-2.3.2')


def resolve_project_path(value: str | Path) -> Path:
    """Use this checkout for old project paths, even if the old copy still exists.

    External paths retain their locations. Missing migrated files never fall
    back to the original copy, which could change an experiment's inputs.
    """
    path = Path(value).expanduser()
    try:
        relative = path.relative_to(LEGACY_PROJECT_ROOT)
    except ValueError:
        return path
    return PROJECT_ROOT / relative


def remap_project_paths(value: Any) -> Any:
    """Normalize loaded values without changing the original record on disk."""
    if isinstance(value, str) and value.startswith(str(LEGACY_PROJECT_ROOT)):
        return str(resolve_project_path(value))
    if isinstance(value, list):
        return [remap_project_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: remap_project_paths(item) for key, item in value.items()}
    return value


def load_project_json(path: str | Path) -> Any:
    return remap_project_paths(json.loads(resolve_project_path(path).read_text()))
