from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got: {type(payload)!r}")
    return payload


def load_experiment_config(experiment_name: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "experiments" / f"{experiment_name}.yaml"
    payload = _load_yaml(path)
    payload.setdefault("name", experiment_name)
    payload.setdefault("tactile_sensor_type", "gsmini")
    payload.setdefault("collection", {})
    payload.setdefault("evaluation", {})
    payload.setdefault("env", {})
    return payload


def load_named_config(group: str, name: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / group / f"{name}.yaml"
    return _load_yaml(path)
