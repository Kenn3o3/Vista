from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.common.gripper import gripper_scalar_from_qpos
from univtac.paths import prepared_data_dir, raw_hdf5_dir, raw_task_dir, require_experiment_name, write_json
from univtac.registry import require_known_task
from univtac.subprocess_utils import count_hdf5_files, ensure_absent


def _sorted_hdf5_files(hdf5_dir: Path) -> list[Path]:
    return sorted(hdf5_dir.glob("*.hdf5"), key=lambda path: int(path.stem))


def _resolve_tactile_paths(example_hdf5: Path) -> tuple[str, str]:
    candidates = [
        ("tactile/left_tactile/rgb_marker", "tactile/right_tactile/rgb_marker"),
        ("tactile/left_gsmini/rgb_marker", "tactile/right_gsmini/rgb_marker"),
    ]
    with h5py.File(str(example_hdf5), "r") as f:
        for left_path, right_path in candidates:
            if left_path in f and right_path in f:
                return left_path, right_path
    raise KeyError(f"No supported tactile rgb_marker pair found in {example_hdf5}")


def _stream_to_img(data, *, resize: bool, path: str) -> np.ndarray:
    flat = data.ravel()
    imgs = None
    for idx, buf in enumerate(flat):
        if isinstance(buf, (bytes, bytearray, np.bytes_)):
            arr = np.frombuffer(bytes(buf), dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported image buffer type: {type(buf)}")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image buffer for {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (256, 256))
        if imgs is None:
            imgs = np.empty((len(flat), *img.shape), dtype=np.uint8)
        imgs[idx] = img
    if imgs is None:
        raise ValueError(f"No image data found for {path}")
    return imgs


def _load_raw_dataset(f: h5py.File, data_path: str, *, resize: bool) -> np.ndarray:
    endpoint = data_path.rsplit("/", 1)[-1]
    data = f[data_path][()]
    if "rgb" in endpoint:
        return _stream_to_img(data, resize=resize, path=data_path)
    return np.asarray(data, dtype=np.float32)


def _batch_gather_hdf5(
    hdf5_paths: list[Path],
    *,
    data_paths: list[str],
    resize: bool,
    downsample_factor: int,
) -> dict[str, np.ndarray]:
    episode_start = 0
    episode_ends: list[int] = []
    for path in hdf5_paths:
        with h5py.File(str(path), "r") as f:
            episode_start += len(np.arange(0, len(f["embodiment/joint"]) - 1, downsample_factor))
            episode_ends.append(episode_start)

    total_length = episode_start
    print(f"Total data pairs: {total_length}")
    result: dict[str, np.ndarray] = {}

    for eid, path in enumerate(tqdm(hdf5_paths, desc="[ViTAL prepare] load", unit="episode")):
        start = 0 if eid == 0 else episode_ends[eid - 1]
        end = episode_ends[eid]
        with h5py.File(str(path), "r") as f:
            for data_path in data_paths:
                data = _load_raw_dataset(f, data_path, resize=resize)
                if data_path == "embodiment/joint":
                    if "embodiment/joint_state" not in result:
                        result["embodiment/joint_action"] = np.empty((total_length, *data.shape[1:]), dtype=data.dtype)
                        result["embodiment/joint_state"] = np.empty((total_length, *data.shape[1:]), dtype=data.dtype)
                elif data_path == "embodiment/ee":
                    if "embodiment/ee_state" not in result:
                        result["embodiment/ee_action"] = np.empty((total_length, *data.shape[1:]), dtype=data.dtype)
                        result["embodiment/ee_state"] = np.empty((total_length, *data.shape[1:]), dtype=data.dtype)
                elif data_path not in result:
                    result[data_path] = np.empty((total_length, *data.shape[1:]), dtype=data.dtype)

                downsample_idx = np.arange(0, len(data) - 1, downsample_factor)
                if data_path == "embodiment/joint":
                    result["embodiment/joint_action"][start:end] = data[1:][downsample_idx]
                    result["embodiment/joint_state"][start:end] = data[:-1][downsample_idx]
                elif data_path == "embodiment/ee":
                    result["embodiment/ee_action"][start:end] = data[1:][downsample_idx]
                    result["embodiment/ee_state"][start:end] = data[:-1][downsample_idx]
                else:
                    result[data_path][start:end] = data[:-1][downsample_idx]

    result["episode_ends"] = np.asarray(episode_ends, dtype=np.int64)
    return result


def _compute_norm_stats(
    qpos_data: np.ndarray,
    action_data: np.ndarray,
    left_tac_data: np.ndarray,
    right_tac_data: np.ndarray,
) -> dict:
    def tactile_stats(data: np.ndarray, chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        total = 0
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        for start in range(0, data.shape[0], chunk_size):
            chunk = data[start : start + chunk_size].astype(np.float32) / 255.0
            reduce_axes = tuple(range(chunk.ndim - 1))
            total += int(np.prod(chunk.shape[:-1]))
            channel_sum += chunk.sum(axis=reduce_axes, dtype=np.float64)
            channel_sq_sum += np.square(chunk, dtype=np.float32).sum(
                axis=reduce_axes,
                dtype=np.float64,
            )
        mean = channel_sum / max(total, 1)
        var = np.maximum(channel_sq_sum / max(total, 1) - np.square(mean), 0.0)
        std = np.clip(np.sqrt(var), 1e-2, np.inf)
        return mean.astype(np.float32), std.astype(np.float32)

    qpos_mean = qpos_data.mean(axis=0).astype(np.float32)
    qpos_std = np.clip(qpos_data.std(axis=0), 1e-2, np.inf).astype(np.float32)
    action_mean = action_data.mean(axis=0).astype(np.float32)
    action_std = np.clip(action_data.std(axis=0), 1e-2, np.inf).astype(np.float32)

    left_tac_mean, left_tac_std = tactile_stats(left_tac_data)
    right_tac_mean, right_tac_std = tactile_stats(right_tac_data)

    return {
        "qpos_mean": qpos_mean.tolist(),
        "qpos_std": qpos_std.tolist(),
        "qpos_min": qpos_data.min(axis=0).astype(np.float32).tolist(),
        "qpos_max": qpos_data.max(axis=0).astype(np.float32).tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_min": action_data.min(axis=0).astype(np.float32).tolist(),
        "action_max": action_data.max(axis=0).astype(np.float32).tolist(),
        "left_tac_mean": left_tac_mean.tolist(),
        "left_tac_std": left_tac_std.tolist(),
        "right_tac_mean": right_tac_mean.tolist(),
        "right_tac_std": right_tac_std.tolist(),
        "gelsight_mean": ((left_tac_mean + right_tac_mean) / 2.0).astype(np.float32).tolist(),
        "gelsight_std": ((left_tac_std + right_tac_std) / 2.0).astype(np.float32).tolist(),
    }


def _save_episode_hdf5(
    output_path: Path,
    *,
    qpos: np.ndarray,
    action: np.ndarray,
    ee: np.ndarray,
    head: np.ndarray,
    left_tac: np.ndarray,
    right_tac: np.ndarray,
    wrist: np.ndarray | None = None,
) -> None:
    with h5py.File(str(output_path), "w") as f:
        f.create_dataset("action", data=action.astype(np.float32), dtype="float32", compression="gzip", compression_opts=4)
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos.astype(np.float32), dtype="float32", compression="gzip", compression_opts=4)
        obs.create_dataset("ee", data=ee.astype(np.float32), dtype="float32", compression="gzip", compression_opts=4)

        images = obs.create_group("images")
        images.create_dataset("cam_high", data=head.astype(np.uint8), dtype="uint8", compression="gzip", compression_opts=4)
        if wrist is not None:
            images.create_dataset("cam_wrist", data=wrist.astype(np.uint8), dtype="uint8", compression="gzip", compression_opts=4)
        images.create_dataset("gelsight", data=left_tac.astype(np.uint8), dtype="uint8", compression="gzip", compression_opts=4)
        images.create_dataset("cam_left_tactile", data=left_tac.astype(np.uint8), dtype="uint8", compression="gzip", compression_opts=4)
        images.create_dataset("cam_right_tactile", data=right_tac.astype(np.uint8), dtype="uint8", compression="gzip", compression_opts=4)

        f.attrs["sim"] = True
        f.attrs["image_height"] = int(head.shape[1])
        f.attrs["image_width"] = int(head.shape[2])
        f.attrs["gelsight_height"] = int(left_tac.shape[1])
        f.attrs["gelsight_width"] = int(left_tac.shape[2])
        f.attrs["num_timesteps"] = int(qpos.shape[0])


def _prepare_episodes(
    *,
    hdf5_paths: list[Path],
    output_dir: Path,
    camera_type: str,
    downsample_factor: int,
) -> tuple[int, dict]:
    if not hdf5_paths:
        raise FileNotFoundError("No input HDF5 files provided for ViTAL prepare.")

    left_tactile_path, right_tactile_path = _resolve_tactile_paths(hdf5_paths[0])
    data_paths = ["embodiment/joint", "embodiment/ee"]
    data_paths.append("observation/wrist/rgb")
    data_paths.extend([left_tactile_path, right_tactile_path])

    batch_data = _batch_gather_hdf5(
        hdf5_paths,
        data_paths=data_paths,
        resize=True,
        downsample_factor=downsample_factor,
    )

    episode_ends = batch_data.get("episode_ends")
    if episode_ends is None:
        episode_ends = np.array([len(batch_data["embodiment/joint_state"])], dtype=np.int64)

    ee_state = batch_data["embodiment/ee_state"][:, :7]
    ee_action = batch_data["embodiment/ee_action"][:, :7]
    joint_state = batch_data["embodiment/joint_state"].astype(np.float32)
    joint_action = batch_data["embodiment/joint_action"].astype(np.float32)
    qpos = np.concatenate(
        [ee_state.astype(np.float32), gripper_scalar_from_qpos(joint_state[:, -2:]).astype(np.float32)],
        axis=-1,
    )
    action = np.concatenate(
        [ee_action.astype(np.float32), gripper_scalar_from_qpos(joint_action[:, -2:]).astype(np.float32)],
        axis=-1,
    )
    if not np.isfinite(qpos).all():
        raise ValueError("ViTAL prepared qpos contains non-finite values.")
    if not np.isfinite(action).all():
        raise ValueError("ViTAL prepared action contains non-finite values.")
    head_cam = batch_data["observation/wrist/rgb"]
    wrist_cam = batch_data["observation/wrist/rgb"]
    left_tac = batch_data[left_tactile_path]
    right_tac = batch_data[right_tactile_path]

    start_idx = 0
    successful = 0
    for ep_idx, end_idx in tqdm(enumerate(episode_ends), total=len(episode_ends), desc="[ViTAL prepare] write"):
        end_idx = int(end_idx)
        try:
            _save_episode_hdf5(
                output_dir / f"episode_{ep_idx}.hdf5",
                qpos=qpos[start_idx:end_idx],
                action=action[start_idx:end_idx],
                ee=ee_state[start_idx:end_idx],
                head=head_cam[start_idx:end_idx],
                wrist=None if wrist_cam is None else wrist_cam[start_idx:end_idx],
                left_tac=left_tac[start_idx:end_idx],
                right_tac=right_tac[start_idx:end_idx],
            )
            successful += 1
        finally:
            start_idx = end_idx

    stats = _compute_norm_stats(qpos, action, left_tac, right_tac)
    return successful, stats


def prepare_vital_data(
    *,
    experiment_name: str,
    task_name: str,
    n_demo: int,
    config_name: str = "vital",
    overwrite: bool = False,
) -> Path:
    require_known_task(task_name)

    input_dir = raw_task_dir(REPO_ROOT, experiment_name, task_name)
    hdf5_dir = raw_hdf5_dir(REPO_ROOT, experiment_name, task_name)
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Raw HDF5 directory not found: {hdf5_dir}")

    available = count_hdf5_files(hdf5_dir)
    resolved_n_demo = n_demo if n_demo > 0 else available
    if resolved_n_demo > available:
        raise ValueError(f"Requested n_demo={resolved_n_demo}, but only found {available} episodes in {hdf5_dir}")

    output_dir = prepared_data_dir(REPO_ROOT, experiment_name, task_name, "ViTAL", f"{config_name}_demo{resolved_n_demo}")
    if overwrite and output_dir.exists():
        stats_path = output_dir / "norm_stats.json"
        metadata_path = output_dir / "prepared_data.json"
        episode_count = len(list(output_dir.glob("episode_*.hdf5")))
        if stats_path.is_file() and metadata_path.is_file() and episode_count >= resolved_n_demo:
            return output_dir
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        ensure_absent(output_dir, label="prepared ViTAL data directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    settings_path = REPO_ROOT / "policy" / "task_settings.json"
    task_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if task_name not in task_settings:
        raise KeyError(f"Task '{task_name}' not found in {settings_path}")
    camera_type = str(task_settings[task_name].get("camera_type", "head"))
    downsample_factor = int(task_settings[task_name].get("downsample", 1))

    hdf5_paths = _sorted_hdf5_files(hdf5_dir)[:resolved_n_demo]
    successful, stats = _prepare_episodes(
        hdf5_paths=hdf5_paths,
        output_dir=output_dir,
        camera_type=camera_type,
        downsample_factor=downsample_factor,
    )

    stats.update(
        {
            "task_name": task_name,
            "task_config": "clean",
            "num_episodes": successful,
            "camera": ["cam_wrist"],
            "tactile": ["cam_left_tactile", "cam_right_tactile"],
        }
    )
    write_json(output_dir / "norm_stats.json", stats)
    write_json(
        output_dir / "prepared_data.json",
        {
            "experiment_name": experiment_name,
            "policy_name": "ViTAL",
            "task_name": task_name,
            "config_name": f"{config_name}_demo{resolved_n_demo}",
            "config_base_name": config_name,
            "n_demo": resolved_n_demo,
            "num_episodes": successful,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "camera_type": camera_type,
            "downsample_factor": downsample_factor,
        },
    )
    print(f"[ViTAL prepare] task={task_name} episodes={successful} output_dir={output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ViTAL data in the clean UniVTAC layout")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="vital")
    parser.add_argument("--n-demo", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_vital_data(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        config_name=args.config_name,
        n_demo=args.n_demo,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
