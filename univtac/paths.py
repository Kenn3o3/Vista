from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_root() -> Path:
    return REPO_ROOT


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def require_experiment_name(experiment_name: str | None = None) -> str:
    experiment = experiment_name or os.environ.get("EXPERIMENT_NAME")
    if not experiment:
        raise ValueError("experiment_name is required. Pass --experiment-name or set EXPERIMENT_NAME.")
    return experiment


def raw_data_root(repo_root: Path, experiment_name: str) -> Path:
    return repo_root / "data" / experiment_name


def raw_task_dir(repo_root: Path, experiment_name: str, task_name: str) -> Path:
    return raw_data_root(repo_root, experiment_name) / task_name


def raw_hdf5_dir(repo_root: Path, experiment_name: str, task_name: str) -> Path:
    return raw_task_dir(repo_root, experiment_name, task_name) / "hdf5"


def raw_metadata_path(repo_root: Path, experiment_name: str, task_name: str) -> Path:
    return raw_task_dir(repo_root, experiment_name, task_name) / "metadata.json"


def raw_video_dir(repo_root: Path, experiment_name: str, task_name: str) -> Path:
    return raw_data_root(repo_root, experiment_name) / "videos" / task_name


def raw_runtime_dir(repo_root: Path, experiment_name: str, task_name: str) -> Path:
    return repo_root / ".cache" / "collect" / experiment_name / task_name


def prepared_data_dir(
    repo_root: Path,
    experiment_name: str,
    task_name: str,
    policy_name: str,
    config_name: str | None = None,
) -> Path:
    root = repo_root / "prepared_data" / experiment_name / task_name / policy_name
    if config_name is not None:
        root = root / config_name
    return root


def isp_cache_dir(repo_root: Path, experiment_name: str, task_name: str, config_name: str) -> Path:
    return prepared_data_dir(repo_root, experiment_name, task_name, "ISP", config_name) / "cache"


def policy_cache_dir(repo_root: Path, experiment_name: str, task_name: str, policy_name: str, config_name: str) -> Path:
    return prepared_data_dir(repo_root, experiment_name, task_name, policy_name, config_name) / "cache"


def checkpoint_run_dir(
    repo_root: Path,
    experiment_name: str,
    task_name: str,
    policy_name: str,
    config_name: str,
    run_id: str,
) -> Path:
    return repo_root / "checkpoints" / experiment_name / task_name / policy_name / config_name / run_id


def eval_run_dir(
    repo_root: Path,
    experiment_name: str,
    task_name: str,
    policy_name: str,
    config_name: str,
    train_run_id: str,
    inference_config_name: str,
    eval_id: str,
) -> Path:
    return (
        repo_root
        / "eval_result"
        / experiment_name
        / task_name
        / policy_name
        / config_name
        / train_run_id
        / inference_config_name
        / eval_id
    )


def test_data_run_dir(
    repo_root: Path,
    policy_name: str,
    task_name: str,
    variant: str,
    n_demo: int | str | None = None,
    case_name: str = "normal_1",
    run_id: str | None = None,
) -> Path:
    task_token = task_name
    if n_demo is not None and str(n_demo) != "":
        task_token = f"{task_name}_demo{n_demo}"
    return (
        repo_root
        / "test_data"
        / policy_name
        / task_token
        / variant
        / case_name
        / (run_id or timestamp_id())
    )


def config_name_with_demo(config_name: str, n_demo: int) -> str:
    return f"{config_name}_demo{n_demo}"


def act_variant(config_name: str, n_demo: int) -> str:
    return config_name_with_demo(config_name, n_demo)


def isp_variant(config_name: str, n_demo: int) -> str:
    return config_name_with_demo(config_name, n_demo)


def tactileact_variant(backbone: str, n_demo: int) -> str:
    return f"{backbone}_demo{n_demo}"


@dataclass(frozen=True)
class CheckpointMetadata:
    experiment_name: str
    task_name: str
    policy_name: str
    config_name: str
    run_id: str
    ckpt_dir: Path
    ckpt_path: Path
    train_metadata: dict[str, Any] | None = None

    @property
    def inferred_n_demo(self) -> int | None:
        marker = "_demo"
        if marker not in self.config_name:
            return None
        suffix = self.config_name.rsplit(marker, 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return None

    @property
    def inferred_train_config(self) -> str | None:
        if self.policy_name == "ACT":
            if self.train_metadata:
                value = self.train_metadata.get("train_config")
                if isinstance(value, str) and value:
                    return value
            marker = "_demo"
            if marker not in self.config_name:
                return None
            base = self.config_name.rsplit(marker, 1)[0]
            if base == "act":
                return "train_config_vision_ee"
            if base == "univtac":
                return "train_config_ee"
            return None
        return None


def infer_checkpoint_metadata(repo_root: Path, ckpt_path: str | Path) -> CheckpointMetadata:
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    ckpt_root = (repo_root / "checkpoints").resolve()
    relative = ckpt_path.relative_to(ckpt_root)
    parts = relative.parts

    if len(parts) < 6:
        raise ValueError(
            "Checkpoint path does not match checkpoints/<experiment>/<task>/<policy>/<config>/<run_id>/... layout: "
            f"{ckpt_path}"
        )

    experiment_name, task_name, policy_name, config_name, run_id = parts[:5]
    train_metadata = None
    metadata_path = ckpt_root / experiment_name / task_name / policy_name / config_name / run_id / "run_metadata.json"
    if metadata_path.is_file():
        train_metadata = load_json(metadata_path)

    return CheckpointMetadata(
        experiment_name=experiment_name,
        task_name=task_name,
        policy_name=policy_name,
        config_name=config_name,
        run_id=run_id,
        ckpt_dir=ckpt_path.parent,
        ckpt_path=ckpt_path,
        train_metadata=train_metadata,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
