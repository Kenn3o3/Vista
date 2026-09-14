from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.experiments import load_named_config
from univtac.paths import checkpoint_run_dir, config_name_with_demo, prepared_data_dir, require_experiment_name, timestamp_id
from univtac.registry import require_known_task
from univtac.subprocess_utils import cleanup_prepared_artifact, ensure_absent, run_checked, write_run_metadata
from univtac.data.vital_prepare import prepare_vital_data
from policy.DP.train import train_dp


VITAL_ROOT = REPO_ROOT / "policy" / "ViTAL"


def _load_norm_stats(dataset_dir: Path) -> dict:
    stats_path = dataset_dir / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"ViTAL norm_stats.json not found: {stats_path}")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _vital_dataset_is_complete(dataset_dir: Path, expected_episodes: int) -> bool:
    if not dataset_dir.is_dir():
        return False
    stats_path = dataset_dir / "norm_stats.json"
    metadata_path = dataset_dir / "prepared_data.json"
    if not stats_path.is_file() or not metadata_path.is_file():
        return False
    return len(list(dataset_dir.glob("episode_*.hdf5"))) >= expected_episodes


def _archive_incomplete_dataset(dataset_dir: Path) -> None:
    archive_name = f"{dataset_dir.name}.incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_path = dataset_dir.with_name(archive_name)
    suffix = 1
    while archive_path.exists():
        archive_path = dataset_dir.with_name(f"{archive_name}_{suffix}")
        suffix += 1
    shutil.move(str(dataset_dir), str(archive_path))
    print(f"[ViTAL] archived incomplete prepared dataset: {dataset_dir} -> {archive_path}")


def _write_training_config(path: Path, *, config: dict, dataset_dir: Path, run_name: str, gpu: str | None) -> None:
    payload = {
        "eval": False,
        "checkpoint": "policy_best.ckpt",
        "onscreen_render": False,
        "save_dir": str(dataset_dir),
        "name": run_name,
        "policy_class": "ACT",
        "batch_size": int(config.get("batch_size", 64)),
        "seed": int(config.get("seed", 0)),
        "num_epochs": int(config.get("num_epochs", 4000)),
        "lr": float(config.get("lr", 1e-5)),
        "kl_weight": float(config.get("kl_weight", 10.0)),
        "start_kl_epoch": int(config.get("start_kl_epoch", 0)),
        "kl_scale_epochs": int(config.get("kl_scale_epochs", 0)),
        "chunk_size": int(config.get("chunk_size", 20)),
        "hidden_dim": int(config.get("hidden_dim", 512)),
        "dim_feedforward": int(config.get("dim_feedforward", 3200)),
        "temporal_agg": bool(config.get("temporal_agg", True)),
        "z_dimension": int(config.get("z_dimension", 32)),
        "gpu": -1,
        "lr_backbone": float(config.get("lr_backbone", 1e-5)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
        "backbone": str(config.get("backbone", "clip_backbone")),
        "dilation": bool(config.get("dilation", False)),
        "position_embedding": str(config.get("position_embedding", "sine")),
        "enc_layers": int(config.get("enc_layers", 4)),
        "dec_layers": int(config.get("dec_layers", 7)),
        "dropout": float(config.get("dropout", 0.025)),
        "nheads": int(config.get("nheads", 8)),
        "pre_norm": bool(config.get("pre_norm", False)),
        "masks": bool(config.get("masks", False)),
        "gelsight_backbone_path": str(_resolve_repo_path(config.get("gelsight_backbone_path", "none"))),
        "vision_backbone_path": str(_resolve_repo_path(config.get("vision_backbone_path", "none"))),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_repo_path(path_value: str) -> str:
    if not path_value or path_value == "none":
        return "none"
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"ViTAL pretrained encoder checkpoint not found: {path}")
    return str(path)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def _link_or_copy_if_exists(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def train_vital(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    n_demo: int,
    run_id: str | None = None,
    seed: int = 0,
    gpu: str | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_epochs: int | None = None,
    conda_env: str | None = None,
    wandb_mode: str | None = None,
    allow_existing_run_dir: bool = False,
) -> Path:
    require_known_task(task_name)
    config = load_named_config("vital", config_name)
    if str(config.get("kind", "act")).lower() == "dp":
        return train_dp(
            experiment_name=experiment_name,
            task_name=task_name,
            config_name=config_name,
            n_demo=n_demo,
            policy_name="ViTAL",
            config_group="vital",
            run_id=run_id,
            seed=seed,
            gpu=gpu,
            conda_env=conda_env,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_epochs=num_epochs,
            wandb_mode=wandb_mode,
            allow_existing_run_dir=allow_existing_run_dir,
        )

    if batch_size is not None:
        config["batch_size"] = batch_size
    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    config["seed"] = seed

    full_config_name = config_name_with_demo(config_name, n_demo)
    dataset_dir = prepared_data_dir(REPO_ROOT, experiment_name, task_name, "ViTAL", full_config_name)
    if dataset_dir.exists() and not _vital_dataset_is_complete(dataset_dir, n_demo):
        _archive_incomplete_dataset(dataset_dir)
    if not dataset_dir.is_dir():
        dataset_dir = prepare_vital_data(
            experiment_name=experiment_name,
            task_name=task_name,
            config_name=config_name,
            n_demo=n_demo,
            overwrite=False,
        )
    stats = _load_norm_stats(dataset_dir)
    num_episodes = int(stats.get("num_episodes", n_demo))

    resolved_run_id = run_id or timestamp_id()
    run_dir = checkpoint_run_dir(REPO_ROOT, experiment_name, task_name, "ViTAL", full_config_name, resolved_run_id)
    if not allow_existing_run_dir:
        ensure_absent(run_dir, label="ViTAL checkpoint run directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    model_name = f"univtac_{experiment_name}_{task_name}_{full_config_name}_{resolved_run_id}"
    legacy_ckpt_dir = VITAL_ROOT / "act-ckpt" / model_name
    if legacy_ckpt_dir.exists():
        shutil.rmtree(legacy_ckpt_dir)
    with tempfile.TemporaryDirectory(prefix="train_vital_cfg_") as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        _write_training_config(config_path, config=config, dataset_dir=dataset_dir, run_name=model_name, gpu=gpu)
        run_checked([sys.executable, "imitate_episodes.py", "--config", str(config_path)], env=env, cwd=VITAL_ROOT)

    if not legacy_ckpt_dir.is_dir():
        raise FileNotFoundError(f"ViTAL training did not produce checkpoint directory: {legacy_ckpt_dir}")

    keep_files = {
        "args.json",
        "best.info.json",
        "dataset_stats.pkl",
        "loss_history.jsonl",
        "policy_best.ckpt",
        "train_summary.json",
    }
    for path in legacy_ckpt_dir.iterdir():
        if path.is_file() and path.name in keep_files:
            shutil.copy2(path, run_dir / path.name)
    shutil.rmtree(legacy_ckpt_dir)

    _link_or_copy_if_exists(run_dir / "policy_best.ckpt", run_dir / "best.ckpt")
    _copy_if_exists(dataset_dir / "norm_stats.json", run_dir / "norm_stats.json")

    metadata = {
            "experiment_name": experiment_name,
            "policy_name": "ViTAL",
            "task_name": task_name,
            "config_name": full_config_name,
            "config_base_name": config_name,
            "run_id": resolved_run_id,
            "n_demo": n_demo,
            "num_episodes": num_episodes,
            "seed": seed,
            "batch_size": int(config.get("batch_size", 64)),
            "val_batch_size": val_batch_size,
            "num_epochs": int(config.get("num_epochs", 4000)),
            "chunk_size": int(config.get("chunk_size", 20)),
            "dataset_dir": str(dataset_dir),
            "checkpoint_dir": str(run_dir),
            "legacy_checkpoint_dir": str(legacy_ckpt_dir),
            "best_ckpt_path": str(run_dir / "best.ckpt"),
    }
    write_run_metadata(run_dir, metadata)
    cleanup_prepared_artifact(dataset_dir, label="ViTAL prepared dataset")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViTAL in the clean UniVTAC layout")
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
    args = parser.parse_args()

    train_vital(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        config_name=args.config_name,
        n_demo=args.n_demo,
        run_id=args.run_id,
        seed=args.seed,
        gpu=args.gpu,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        conda_env=args.conda_env,
        wandb_mode=args.wandb_mode,
        allow_existing_run_dir=args.allow_existing_run_dir,
    )


if __name__ == "__main__":
    main()
