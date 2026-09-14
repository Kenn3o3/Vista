from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.ViTAL.clip_pretraining import ClipDataset, clip_pretraining
from univtac.data.vital_prepare import prepare_vital_data
from univtac.subprocess_utils import cleanup_prepared_artifact


DEFAULT_TASKS = [
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
]


class ViTALConcatDataset(ConcatDataset):
    """ConcatDataset with the small metadata expected by ViTAL pretraining."""

    def __init__(self, datasets):
        super().__init__(datasets)
        if not datasets:
            raise ValueError("ViTALConcatDataset requires at least one dataset.")
        self.n_cameras = datasets[0].n_cameras


def _split_indices(num_episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_episodes)
    split = max(1, int(0.8 * num_episodes))
    if split >= num_episodes:
        split = max(1, num_episodes - 1)
    return indices[:split], indices[split:]


def _load_task_datasets(
    *,
    experiment_name: str,
    task_name: str,
    n_demo: int,
    n_clip_images: int,
    min_distance: int,
    seed: int,
    overwrite_prepared: bool,
) -> tuple[ClipDataset, ClipDataset, Path]:
    dataset_dir = prepare_vital_data(
        experiment_name=experiment_name,
        task_name=task_name,
        config_name="encoder",
        n_demo=n_demo,
        overwrite=overwrite_prepared,
    )
    norm_stats = json.loads((dataset_dir / "norm_stats.json").read_text(encoding="utf-8"))
    camera_names = ["cam_wrist"]
    tactile_names = ["cam_left_tactile", "cam_right_tactile"]
    num_episodes = int(norm_stats["num_episodes"])
    train_indices, val_indices = _split_indices(num_episodes, seed)
    if len(val_indices) == 0:
        val_indices = train_indices[-1:]

    train_dataset = ClipDataset(
        train_indices.tolist(),
        str(dataset_dir),
        camera_names,
        tactile_names,
        norm_stats,
        n_images=n_clip_images,
        min_distance=min_distance,
    )
    val_dataset = ClipDataset(
        val_indices.tolist(),
        str(dataset_dir),
        camera_names,
        tactile_names,
        norm_stats,
        n_images=n_clip_images,
        min_distance=min_distance,
    )
    return train_dataset, val_dataset, dataset_dir


def train_vital_encoder(
    *,
    experiment_name: str,
    tasks: list[str],
    n_demo: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    n_clip_images: int,
    min_distance: int,
    seed: int,
    gpu: str | None,
    overwrite: bool,
    overwrite_prepared: bool,
) -> Path:
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    train_datasets = []
    val_datasets = []
    prepared_dirs = []
    for task_name in tasks:
        train_dataset, val_dataset, dataset_dir = _load_task_datasets(
            experiment_name=experiment_name,
            task_name=task_name,
            n_demo=n_demo,
            n_clip_images=n_clip_images,
            min_distance=min_distance,
            seed=seed,
            overwrite_prepared=overwrite_prepared,
        )
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        prepared_dirs.append(dataset_dir)

    train_dataset = ViTALConcatDataset(train_datasets)
    val_dataset = ViTALConcatDataset(val_datasets)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if gpu is not None:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8 if device.type == "cuda" else 0
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        dataloader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    metadata = {
        "experiment_name": experiment_name,
        "tasks": tasks,
        "n_demo": n_demo,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_clip_images": n_clip_images,
        "min_distance": min_distance,
        "seed": seed,
        "output_dir": str(output_dir),
        "device": str(device),
    }
    (output_dir / "pretrain_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    env_device = str(device)
    print(f"[ViTAL encoder] device={env_device} train_sets={len(train_datasets)} output={output_dir}")
    clip_pretraining(
        train_loader,
        val_loader,
        device,
        save_dir=str(output_dir),
        save_freq=max(1, epochs),
        plot_freq=max(1, min(50, epochs)),
        n_epochs=epochs,
    )

    produced = {
        "vision_encoder": output_dir / "best_vision_encoder.pth",
        "vision_projection": output_dir / "best_vision_projection.pth",
        "gelsight_encoder": output_dir / "best_gelsight_encoder.pth",
        "gelsight_projection": output_dir / "best_gelsight_projection.pth",
    }
    for name, src in produced.items():
        if not src.is_file():
            raise FileNotFoundError(f"Expected ViTAL pretrain artifact not found: {src}")
        shutil.copy2(src, output_dir / f"{name}.pth")
    for dataset_dir in prepared_dirs:
        cleanup_prepared_artifact(dataset_dir, label="ViTAL encoder prepared dataset")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain ViTAL vision and tactile encoders on UniVTAC data")
    parser.add_argument("--experiment-name", type=str, default="MIDFOV_EXPERIMENT")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS), help="Comma-separated task list or 'all'.")
    parser.add_argument("--n-demo", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "VITAL_encoder")
    parser.add_argument("--epochs", type=int, default=1501)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-clip-images", type=int, default=3)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-prepared", action="store_true")
    args = parser.parse_args()

    tasks = DEFAULT_TASKS if args.tasks.strip().lower() == "all" else [item.strip() for item in args.tasks.split(",") if item.strip()]
    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_vital_encoder(
        experiment_name=args.experiment_name,
        tasks=tasks,
        n_demo=args.n_demo,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_clip_images=args.n_clip_images,
        min_distance=args.min_distance,
        seed=args.seed,
        gpu=args.gpu,
        overwrite=args.overwrite,
        overwrite_prepared=args.overwrite_prepared or args.overwrite,
    )


if __name__ == "__main__":
    main()
