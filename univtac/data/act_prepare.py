from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.paths import prepared_data_dir, raw_hdf5_dir, raw_task_dir, require_experiment_name, write_json
from univtac.registry import require_known_task
from univtac.subprocess_utils import count_hdf5_files, ensure_absent, run_checked


PREPARE_SCRIPTS = {
    "joint": "policy/ACT/process_data.py",
    "ee": "policy/ACT/process_data_ee.py",
}


def prepare_act_data(
    *,
    experiment_name: str,
    task_name: str,
    representation: str,
    num_episodes: int = -1,
    overwrite: bool = False,
) -> Path:
    require_known_task(task_name)
    if representation not in PREPARE_SCRIPTS:
        raise ValueError(f"Unsupported ACT representation: {representation}")

    input_dir = raw_task_dir(REPO_ROOT, experiment_name, task_name)
    hdf5_dir = raw_hdf5_dir(REPO_ROOT, experiment_name, task_name)
    output_dir = prepared_data_dir(REPO_ROOT, experiment_name, task_name, "ACT", representation)
    resolved_episodes = num_episodes if num_episodes > 0 else count_hdf5_files(hdf5_dir)

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        ensure_absent(output_dir, label="prepared ACT data directory")

    run_checked(
        [
            sys.executable,
            PREPARE_SCRIPTS[representation],
            task_name,
            "clean",
            str(resolved_episodes),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--skip-sim-task-configs",
        ]
    )
    write_json(
        output_dir / "prepared_data.json",
        {
            "experiment_name": experiment_name,
            "policy_name": "ACT",
            "task_name": task_name,
            "representation": representation,
            "num_episodes": resolved_episodes,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ACT data in the clean UniVTAC layout")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--representation", choices=sorted(PREPARE_SCRIPTS), required=True)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_act_data(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        representation=args.representation,
        num_episodes=args.num_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
