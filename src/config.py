"""YAML configuration loading with small, explicit defaults inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config and resolve ``defaults: [name]`` from the same directory."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}

    defaults = current.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        name = item if isinstance(item, str) else next(iter(item))
        default_path = path.parent / f"{name}.yaml"
        if default_path.resolve() == path:
            raise ValueError(f"Configuration {path} inherits itself")
        merged = deep_merge(merged, load_config(default_path))
    return deep_merge(merged, current)

