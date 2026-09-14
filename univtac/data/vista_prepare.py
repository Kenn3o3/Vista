from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VISTA_ROOT = REPO_ROOT / "policy" / "VISTA"
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from univtac.experiments import load_named_config
from univtac.paths import (
    config_name_with_demo,
    policy_cache_dir,
    prepared_data_dir,
    raw_data_root,
    require_experiment_name,
    write_json,
)
from univtac.registry import require_known_task
from univtac.subprocess_utils import ensure_absent
from univtac.vista_config import normalize_vista_config_name


def _register_vista_resolvers() -> None:
    OmegaConf.register_new_resolver("get_ws_x_center", lambda task_name: 0.0, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda task_name: 0.0, replace=True)
    OmegaConf.register_new_resolver("get_ws_z_center", lambda task_name: 0.8, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def _normalization_mode(config: dict) -> str:
    normalization = str(config.get("normalization", "default")).lower()
    return {
        "on": "default",
        "default": "default",
        "legacy": "legacy",
        "off": "off",
        "identity": "off",
    }[normalization]


def _compose_dataset_cfg(
    *,
    hydra_config: str,
    experiment_name: str,
    task_name: str,
    n_demo: int,
    dataset_root: Path,
    cache_dir: Path,
    config: dict,
):
    action_horizon = config.get("action_horizon")
    n_obs_steps = int(config.get("n_obs_steps", 2))
    n_action_steps = int(config.get("n_action_steps", action_horizon or 8))
    horizon = int(config.get("horizon", n_obs_steps - 1 + n_action_steps))
    hydra_overrides = config.get("hydra_overrides", []) or []
    if not isinstance(hydra_overrides, list):
        raise TypeError(f"Expected hydra_overrides list in configs/vista/{hydra_config}.yaml")

    config_dir = VISTA_ROOT / "vista" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=hydra_config,
            overrides=[
                f"task_name={task_name}",
                f"n_demo={n_demo}",
                f"dataset_root={dataset_root}",
                "data_split=clean",
                f"exp_name={experiment_name}",
                f"+task.dataset.cache_dir={cache_dir}",
                f"task.dataset.normalization_mode={_normalization_mode(config)}",
                f"n_obs_steps={n_obs_steps}",
                f"dataset_obs_steps={n_obs_steps}",
                f"horizon={horizon}",
                f"n_action_steps={n_action_steps}",
                *[str(override) for override in hydra_overrides],
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg, {
        "action_horizon": action_horizon,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "n_obs_steps": n_obs_steps,
        "hydra_overrides": hydra_overrides,
    }


def prepare_vista_data(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    n_demo: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    require_known_task(task_name)
    _register_vista_resolvers()

    requested_config_name = config_name
    config_name = normalize_vista_config_name(config_name)
    config = load_named_config("vista", config_name)
    hydra_config = str(config.get("hydra_config", "train_vista"))
    full_config_name = config_name_with_demo(config_name, n_demo)
    dataset_root = raw_data_root(REPO_ROOT, experiment_name)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw VISTA dataset root not found: {dataset_root}")

    output_dir = prepared_data_dir(REPO_ROOT, experiment_name, task_name, "VISTA", full_config_name)
    cache_dir = policy_cache_dir(REPO_ROOT, experiment_name, task_name, "VISTA", full_config_name)
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not dry_run:
        ensure_absent(output_dir, label="prepared VISTA data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg, resolved = _compose_dataset_cfg(
        hydra_config=hydra_config,
        experiment_name=experiment_name,
        task_name=task_name,
        n_demo=n_demo,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        config=config,
    )

    if not dry_run:
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        del dataset

    metadata = {
        "experiment_name": experiment_name,
        "policy_name": "VISTA",
        "task_name": task_name,
        "config_name": full_config_name,
        "config_base_name": config_name,
        "requested_config_base_name": requested_config_name,
        "hydra_config": hydra_config,
        "n_demo": n_demo,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
        "dry_run": dry_run,
        **resolved,
    }
    write_json(output_dir / "prepared_data.json", metadata)
    print(f"[VISTA prepare] task={task_name} config={full_config_name}")
    print(f"[VISTA prepare] output_dir={output_dir}")
    print(f"[VISTA prepare] cache_dir={cache_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VISTA cached data in the clean UniVTAC layout")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--n-demo", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepare_vista_data(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        config_name=args.config_name,
        n_demo=args.n_demo,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
