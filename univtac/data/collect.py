from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from shutil import ExecError
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

from univtac.experiments import load_experiment_config
from univtac.paths import raw_metadata_path, raw_runtime_dir, raw_task_dir, raw_video_dir, require_experiment_name, write_json
from univtac.registry import require_known_task
from univtac.subprocess_utils import count_hdf5_files

if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg


DEFAULT_OBS_DATA_TYPE = {
    "camera": ["rgb"],
    "tactile": ["rgb", "rgb_marker", "marker", "depth", "pose"],
    "embodiment": ["joint", "ee"],
    "actor": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect data into the clean UniVTAC layout")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--episode-num", type=int, default=-1)
    parser.add_argument("--start-seed", type=int, default=-1)
    parser.add_argument("--max-seed", type=int, default=-1)
    parser.add_argument("--gpu", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


log_path = Path("./log")


def log(msg: str) -> None:
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def _parse_indexed_stem(stem: str, *, prefix: str | None = None) -> tuple[int, str] | None:
    if prefix is None:
        head, sep, tail = stem.partition("_")
        if not head.isdigit():
            return None
        return int(head), f"_{tail}" if sep else ""

    token = f"{prefix}_"
    if not stem.startswith(token):
        return None
    remainder = stem[len(token):]
    head, sep, tail = remainder.partition("_")
    if not head.isdigit():
        return None
    return int(head), f"_{tail}" if sep else ""


def _indexed_files(directory: Path, suffix: str, *, prefix: str | None = None) -> list[tuple[int, str, Path]]:
    files: list[tuple[int, str, Path]] = []
    if not directory.is_dir():
        return files

    for path in directory.glob(f"*{suffix}"):
        parsed = _parse_indexed_stem(path.stem, prefix=prefix)
        if parsed is None:
            continue
        idx, stem_suffix = parsed
        files.append((idx, stem_suffix, path))
    return sorted(files, key=lambda item: (item[0], item[2].name))


def _format_indexed_name(idx: int, stem_suffix: str, suffix: str, *, prefix: str | None = None) -> str:
    stem = f"{idx}{stem_suffix}" if prefix is None else f"{prefix}_{idx}{stem_suffix}"
    return f"{stem}{suffix}"


def _renumber_indexed_files(directory: Path, suffix: str, *, prefix: str | None = None) -> None:
    if not directory.is_dir():
        return

    files = _indexed_files(directory, suffix, prefix=prefix)
    if not files:
        return

    expected_names = [
        _format_indexed_name(idx, stem_suffix, suffix, prefix=prefix)
        for idx, (_, stem_suffix, _) in enumerate(files)
    ]
    if all(path.name == expected for (_, _, path), expected in zip(files, expected_names)):
        return

    temp_paths: list[tuple[Path, str]] = []
    prefix_token = prefix or "numeric"
    for idx, (_, stem_suffix, path) in enumerate(files):
        temp_path = path.with_name(f".__renaming__{prefix_token}_{idx}{suffix}")
        path.rename(temp_path)
        temp_paths.append((temp_path, stem_suffix))
    for idx, (temp_path, stem_suffix) in enumerate(temp_paths):
        target_path = directory / _format_indexed_name(idx, stem_suffix, suffix, prefix=prefix)
        temp_path.rename(target_path)


def _count_indexed_files(directory: Path, suffix: str, *, prefix: str | None = None) -> int:
    return len(_indexed_files(directory, suffix, prefix=prefix))


def _should_save_failed_videos(collection_cfg: dict, task_name: str) -> bool:
    scoped_tasks = collection_cfg.get("save_failed_videos_for_tasks")
    if isinstance(scoped_tasks, (list, tuple, set)):
        if task_name in {str(item) for item in scoped_tasks}:
            return True

    save_failed_videos = collection_cfg.get("save_failed_videos", False)
    if isinstance(save_failed_videos, (list, tuple, set)):
        return task_name in {str(item) for item in save_failed_videos}
    return bool(save_failed_videos)


def _normalize_collection_layout(task_root: Path, video_root: Path, metadata_path: Path) -> None:
    hdf5_dir = task_root / "hdf5"
    _renumber_indexed_files(hdf5_dir, ".hdf5")
    _renumber_indexed_files(video_root, ".mp4")
    _renumber_indexed_files(video_root, ".mp4", prefix="fail")

    if not metadata_path.is_file():
        return

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if isinstance(payload, dict) and "episodes" in payload:
        episodes = payload.get("episodes", {})
        normalized = {str(idx): episodes[key] for idx, key in enumerate(sorted(episodes, key=int))}
        payload["episodes"] = normalized
        write_json(metadata_path, payload)
        return

    legacy_items = []
    for key in sorted(payload, key=int):
        value = payload[key]
        if isinstance(value, dict):
            item = dict(value)
            item.setdefault("source_seed", int(key))
            legacy_items.append(item)
    if legacy_items:
        write_json(
            metadata_path,
            {
                "summary": {},
                "episodes": {str(idx): item for idx, item in enumerate(legacy_items)},
            },
        )


def _load_collection_metadata(metadata_path: Path) -> tuple[dict, dict[str, dict]]:
    if not metadata_path.is_file():
        return {}, {}

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    if isinstance(payload, dict) and isinstance(payload.get("episodes"), dict):
        summary = payload.get("summary", {})
        episodes = {
            str(key): dict(value)
            for key, value in payload["episodes"].items()
            if isinstance(value, dict)
        }
        return dict(summary) if isinstance(summary, dict) else {}, episodes

    legacy_episodes = {}
    for idx, key in enumerate(sorted(payload, key=int)):
        value = payload[key]
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item.setdefault("source_seed", int(key))
        legacy_episodes[str(idx)] = item
    return {}, legacy_episodes


def _write_collection_metadata(
    metadata_path: Path,
    *,
    episodes: dict[str, dict],
    num_attempts: int,
    next_seed: int,
) -> None:
    steps = [
        float(item["cost_step"])
        for item in episodes.values()
        if isinstance(item.get("cost_step"), (int, float))
    ]
    times = [
        float(item["cost_time"])
        for item in episodes.values()
        if isinstance(item.get("cost_time"), (int, float))
    ]
    source_seeds = [
        int(item["source_seed"])
        for item in episodes.values()
        if isinstance(item.get("source_seed"), int)
    ]
    num_successes = len(episodes)
    summary = {
        "num_attempts": int(num_attempts),
        "num_successes": int(num_successes),
        "num_failures": int(max(0, num_attempts - num_successes)),
        "success_rate": float(num_successes / num_attempts) if num_attempts > 0 else 0.0,
        "next_seed": int(next_seed),
        "max_source_seed": int(max(source_seeds)) if source_seeds else -1,
        "min_steps": int(min(steps)) if steps else None,
        "max_steps": int(max(steps)) if steps else None,
        "avg_steps": float(sum(steps) / len(steps)) if steps else None,
        "min_time": float(min(times)) if times else None,
        "max_time": float(max(times)) if times else None,
        "avg_time": float(sum(times) / len(times)) if times else None,
    }
    write_json(
        metadata_path,
        {
            "summary": summary,
            "episodes": {str(idx): episodes[str(idx)] for idx in range(num_successes)},
        },
    )


def run(
    task: "BaseTask",
    *,
    experiment_name: str,
    task_name: str,
    episode_num: int,
    use_seed: bool,
    start_seed: int,
    max_seed: int,
    save_failed_videos: bool,
) -> None:
    metadata_path = raw_metadata_path(REPO_ROOT, experiment_name, task_name)
    video_root = task.video_root
    hdf5_dir = task.save_root / "hdf5"
    _normalize_collection_layout(task.save_root, video_root, metadata_path)
    summary, episodes = _load_collection_metadata(metadata_path)

    suc_num = count_hdf5_files(hdf5_dir) if hdf5_dir.is_dir() else 0
    if len(episodes) != suc_num:
        log(
            f"Metadata episode count ({len(episodes)}) and hdf5 count ({suc_num}) differ. "
            f"Using hdf5 count as source of truth."
        )
        normalized_episodes: dict[str, dict] = {}
        for idx in range(suc_num):
            if str(idx) in episodes:
                normalized_episodes[str(idx)] = episodes[str(idx)]
            else:
                normalized_episodes[str(idx)] = {
                    "episode_id": idx,
                    "source_seed": None,
                    "result": "success",
                }
        episodes = normalized_episodes

    failed_video_num = _count_indexed_files(video_root, ".mp4", prefix="fail")

    if start_seed != -1:
        seed = start_seed
        log(f"Starting from explicit seed {seed}.")
    elif use_seed:
        seed = int(summary.get("next_seed", -1))
        if seed < 0:
            source_seeds = [
                int(item["source_seed"])
                for item in episodes.values()
                if isinstance(item.get("source_seed"), int)
            ]
            seed = (max(source_seeds) + 1) if source_seeds else suc_num
        log(f"Resuming from seed {seed} with {suc_num} successful episodes already collected.")
    else:
        seed = 0

    mean_steps = float(summary.get("avg_steps") or 0.0)
    num_attempts = int(summary.get("num_attempts", suc_num))
    while suc_num < episode_num and (max_seed == -1 or seed <= max_seed):
        task.cfg.record_id = suc_num
        try:
            start_t = time.perf_counter()
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception:
            num_attempts += 1
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            if save_failed_videos:
                failed_video_path = video_root / f"fail_{failed_video_num}.mp4"
                task.clean_cache(mean_steps=mean_steps, result=None, video_output_path=failed_video_path)
                log(f"[{suc_num:<3d}] Saved failed video to {failed_video_path.name}.")
                failed_video_num += 1
            else:
                task.clean_cache(mean_steps=mean_steps, result=None, discard_video=True)
        else:
            num_attempts += 1
            if task.plan_success and task.check_success() and not task.check_early_stop():
                task.save_to_hdf5()
                log(
                    f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n"
                )
                previous_successes = suc_num
                if mean_steps > 0:
                    mean_steps = ((previous_successes * mean_steps) + task.step_count) / (previous_successes + 1)
                else:
                    mean_steps = task.step_count
                task.clean_cache(mean_steps=mean_steps, result="success")
                episode_meta = dict(task.metadata)
                episode_meta["episode_id"] = previous_successes
                episode_meta["source_seed"] = seed
                episodes[str(previous_successes)] = episode_meta
                suc_num += 1
            else:
                log(
                    f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}"
                )
                if save_failed_videos:
                    failed_video_path = video_root / f"fail_{failed_video_num}.mp4"
                    task.clean_cache(mean_steps=mean_steps, result=None, video_output_path=failed_video_path)
                    log(f"[{suc_num:<3d}] Saved failed video to {failed_video_path.name}.")
                    failed_video_num += 1
                else:
                    task.clean_cache(mean_steps=mean_steps, result=None, discard_video=True)

        _write_collection_metadata(
            metadata_path,
            episodes=episodes,
            num_attempts=num_attempts,
            next_seed=seed + 1,
        )

        seed += 1

    log(
        f"Complete collection, success rate: "
        f"{suc_num}/{num_attempts} ({((suc_num / num_attempts) * 100.0) if num_attempts > 0 else 0.0:.2f}%)"
    )


def main() -> None:
    global log_path
    parser = build_parser()
    args_cli = parser.parse_args()
    if args_cli.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu

    experiment_name = require_experiment_name(args_cli.experiment_name)
    experiment_cfg = load_experiment_config(experiment_name)
    collection_cfg = experiment_cfg.get("collection", {})

    args_cli.enable_cameras = True
    args_cli.num_envs = 1
    if int(collection_cfg.get("render_frequency", 0)) == 0:
        args_cli.livestream = 0

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    task_name = require_known_task(args_cli.task_name)
    os.environ["UNIVTAC_EXPERIMENT_NAME"] = experiment_name

    task_module = importlib.import_module(f"envs.{task_name}")
    env_cfg: "BaseTaskCfg" = task_module.TaskCfg()
    env_cfg.tactile_sensor_type = experiment_cfg.get("tactile_sensor_type", "gsmini")
    env_cfg.save_dir = raw_task_dir(REPO_ROOT, experiment_name, task_name)
    env_cfg.video_save_dir = raw_video_dir(REPO_ROOT, experiment_name, task_name)
    env_cfg.runtime_dir = raw_runtime_dir(REPO_ROOT, experiment_name, task_name)
    env_cfg.auto_write_metadata = False
    env_cfg.decimation = collection_cfg.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = collection_cfg.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = collection_cfg.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = collection_cfg.get("render_frequency", env_cfg.render_frequency)
    env_cfg.video_size = tuple(collection_cfg.get("video_size", env_cfg.video_size))
    env_cfg.obs_data_type = collection_cfg.get("observations", DEFAULT_OBS_DATA_TYPE)
    env_cfg.random_texture = collection_cfg.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    save_failed_videos = _should_save_failed_videos(collection_cfg, task_name)

    episode_num = (
        args_cli.episode_num
        if args_cli.episode_num != -1
        else int(collection_cfg.get("episode_num", 50))
    )
    start_seed = (
        args_cli.start_seed
        if args_cli.start_seed != -1
        else int(collection_cfg.get("start_seed", -1))
    )
    max_seed = (
        args_cli.max_seed
        if args_cli.max_seed != -1
        else int(collection_cfg.get("max_seed", -1))
    )

    init_start = time.perf_counter()
    task: "BaseTask" = task_module.Task(env_cfg, mode="collect")
    init_cost = time.perf_counter() - init_start

    log_path = task.runtime_root / "logs" / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_name}")
    log(f"Experiment Name: {experiment_name}")
    log(
        "Collection Config: \n"
        + json.dumps(
            {
                "episode_num": episode_num,
                "start_seed": start_seed,
                "max_seed": max_seed,
                **collection_cfg,
            },
            ensure_ascii=False,
            indent=4,
        )
        + "\n--------------------\n"
    )
    log(f"Env Config: \n{env_cfg}\n--------------------\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        run(
            task,
            experiment_name=experiment_name,
            task_name=task_name,
            episode_num=episode_num,
            use_seed=bool(collection_cfg.get("use_seed", True)),
            start_seed=start_seed,
            max_seed=max_seed,
            save_failed_videos=save_failed_videos,
        )
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
