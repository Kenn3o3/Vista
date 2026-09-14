from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.experiments import load_named_config
from univtac.paths import (
    checkpoint_run_dir,
    config_name_with_demo,
    policy_cache_dir,
    raw_data_root,
    require_experiment_name,
    timestamp_id,
)
from univtac.registry import require_known_task
from univtac.subprocess_utils import (
    cleanup_prepared_artifact,
    ensure_absent,
    python_executable_for_env,
    run_checked,
    write_run_metadata,
)
from univtac.vista_config import normalize_vista_config_name


VISTA_ROOT = REPO_ROOT / "policy" / "VISTA"

_CROSS_ENV_COMPILER_VARS = (
    "CC",
    "CXX",
    "CPPFLAGS",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "LD",
)

_CROSS_ENV_CONDA_VARS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
    "CONDA_PROMPT_MODIFIER",
    "_CE_M",
    "_CE_CONDA",
)

_VAL_LOSS_CKPT_RE = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)\.ckpt$")


def _preflight_vista_import(conda_env: str, env: dict[str, str | None], hydra_config: str) -> None:
    policy_modules = {
        "train_vista": "vista.policy.vista_so3_policy",
        "train_vista_so2": "vista.policy.vista_so2_policy",
        "train_vista_so3_global_l0": "vista.policy.vista_so3_global_l0_policy",
        "train_vista_so2_global_l0": "vista.policy.vista_so2_global_l0_policy",
        "train_vista_so3_early_fusion": "vista.policy.vista_so3_early_fusion_policy",
        "train_vista_so2_early_fusion": "vista.policy.vista_so2_early_fusion_policy",
        "train_vista_so3_concate": "vista.policy.vista_so3_concate_policy",
        "train_vista_so2_concate": "vista.policy.vista_so2_concate_policy",
    }
    policy_module = policy_modules.get(hydra_config, "vista.policy.vista_so3_policy")
    command = [
        python_executable_for_env(conda_env),
        "-c",
        (
            "import importlib; "
            f"m = importlib.import_module('{policy_module}'); "
            "print('VISTA policy module:', m.__file__)"
        ),
    ]
    run_checked(command, env=env, cwd=VISTA_ROOT)


def _best_vista_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    direct_best = ckpt_dir / "best.ckpt"
    if direct_best.is_file():
        return direct_best
    candidates: list[tuple[float, Path]] = []
    for path in ckpt_dir.glob("*.ckpt"):
        if path.name in {"latest.ckpt", "best.ckpt"}:
            continue
        match = _VAL_LOSS_CKPT_RE.search(path.name)
        if match is not None:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1]


def train_vista(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    n_demo: int,
    run_id: str | None = None,
    seed: int = 0,
    gpu: str | None = None,
    conda_env: str | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_epochs: int | None = None,
    wandb_mode: str | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    num_workers: int | None = None,
    allow_existing_run_dir: bool = False,
) -> Path:
    require_known_task(task_name)
    requested_config_name = config_name
    config_name = normalize_vista_config_name(config_name)
    config = load_named_config("vista", config_name)
    hydra_config = str(config.get("hydra_config", "train_vista"))
    normalization = str(config.get("normalization", "default")).lower()
    normalization_mode = {"on": "default", "default": "default", "legacy": "legacy", "off": "off", "identity": "off"}[
        normalization
    ]

    action_horizon = config.get("action_horizon")
    n_obs_steps = int(config.get("n_obs_steps", 2))
    n_action_steps = int(config.get("n_action_steps", action_horizon or 8))
    horizon = int(config.get("horizon", n_obs_steps - 1 + n_action_steps))
    hydra_overrides = config.get("hydra_overrides", []) or []
    if not isinstance(hydra_overrides, list):
        raise TypeError(f"Expected hydra_overrides list in configs/vista/{config_name}.yaml")

    dataset_root = raw_data_root(REPO_ROOT, experiment_name)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw VISTA dataset root not found: {dataset_root}")

    resolved_run_id = run_id or timestamp_id()
    full_config_name = config_name_with_demo(config_name, n_demo)
    wandb_project = task_name
    wandb_run_name = full_config_name
    run_dir = checkpoint_run_dir(REPO_ROOT, experiment_name, task_name, "VISTA", full_config_name, resolved_run_id)
    if not allow_existing_run_dir:
        ensure_absent(run_dir, label="VISTA checkpoint run directory")
    cache_dir = policy_cache_dir(REPO_ROOT, experiment_name, task_name, "VISTA", full_config_name)

    env = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and current_env and current_env != conda_env:
        for key in _CROSS_ENV_COMPILER_VARS:
            env[key] = None
        for key in _CROSS_ENV_CONDA_VARS:
            env[key] = None
        for key in tuple(os.environ):
            if key.startswith("CONDA_PREFIX"):
                env[key] = None

    command = [
        python_executable_for_env(conda_env),
        "train.py",
        f"--config-name={hydra_config}",
        f"task_name={task_name}",
        f"n_demo={n_demo}",
        f"dataset_root={dataset_root}",
        "data_split=clean",
        f"exp_name={experiment_name}",
        f"+task.dataset.cache_dir={cache_dir}",
        "training.device=cuda:0",
        f"training.seed={seed}",
        f"hydra.run.dir={run_dir}",
        f"hydra.sweep.dir={run_dir}",
        f"multi_run.run_dir={run_dir}",
        f"task.dataset.normalization_mode={normalization_mode}",
        f"n_obs_steps={n_obs_steps}",
        f"dataset_obs_steps={n_obs_steps}",
        f"horizon={horizon}",
        f"n_action_steps={n_action_steps}",
    ]
    if batch_size is not None:
        command.extend([f"dataloader.batch_size={batch_size}", f"val_dataloader.batch_size={val_batch_size or batch_size}"])
    elif val_batch_size is not None:
        command.append(f"val_dataloader.batch_size={val_batch_size}")
    if num_epochs is not None:
        command.append(f"training.num_epochs={num_epochs}")
    command.extend(
        [
            "training.checkpoint_every=1",
            "checkpoint.topk.k=1",
            "checkpoint.topk.format_str=best.ckpt",
            "checkpoint.save_last_ckpt=False",
            "checkpoint.save_last_snapshot=False",
        ]
    )
    if max_train_steps is not None:
        command.append(f"training.max_train_steps={max_train_steps}")
    if max_val_steps is not None:
        command.append(f"training.max_val_steps={max_val_steps}")
    if num_workers is not None:
        for loader in ("dataloader", "val_dataloader"):
            command.extend([f"{loader}.num_workers={num_workers}",
                            f"{loader}.persistent_workers={str(num_workers > 0).lower()}"])
    effective_wandb_mode = wandb_mode or "offline"
    command.append(f"logging.mode={effective_wandb_mode}")
    command.append(f"logging.project={wandb_project}")
    command.append(f"logging.name={wandb_run_name}")
    command.extend(str(override) for override in hydra_overrides)

    _preflight_vista_import(conda_env, env, hydra_config)
    run_checked(command, env=env, cwd=VISTA_ROOT)
    best_ckpt_path = None
    best_ckpt = _best_vista_checkpoint(run_dir)
    if best_ckpt is not None:
        best_ckpt_path = run_dir / "checkpoints" / "best.ckpt"
        if best_ckpt.resolve() != best_ckpt_path.resolve():
            shutil.copy2(best_ckpt, best_ckpt_path)

    metadata = {
            "experiment_name": experiment_name,
            "policy_name": "VISTA",
            "task_name": task_name,
            "config_name": full_config_name,
            "config_base_name": config_name,
            "requested_config_base_name": requested_config_name,
            "run_id": resolved_run_id,
            "hydra_config": hydra_config,
            "n_demo": n_demo,
            "normalization": normalization,
            "normalization_mode": normalization_mode,
            "action_horizon": action_horizon,
            "horizon": horizon,
            "n_action_steps": n_action_steps,
            "n_obs_steps": n_obs_steps,
            "hydra_overrides": hydra_overrides,
            "seed": seed,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size or batch_size,
            "num_epochs": num_epochs,
            "requested_wandb_mode": wandb_mode,
            "wandb_mode": effective_wandb_mode,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "checkpoint_dir": str(run_dir),
            "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
    }
    write_run_metadata(run_dir, metadata)
    if best_ckpt_path is not None:
        cleanup_prepared_artifact(cache_dir.parent, label="VISTA prepared cache")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VISTA in the clean UniVTAC layout")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--n-demo", type=int, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--allow-existing-run-dir", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    train_vista(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        config_name=args.config_name,
        n_demo=args.n_demo,
        run_id=args.run_id,
        seed=args.seed,
        gpu=args.gpu,
        conda_env=args.conda_env,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        wandb_mode=args.wandb_mode,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        num_workers=args.num_workers,
        allow_existing_run_dir=args.allow_existing_run_dir,
    )


if __name__ == "__main__":
    main()
