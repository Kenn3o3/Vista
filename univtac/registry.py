from __future__ import annotations

from pathlib import Path

from .paths import REPO_ROOT


TASKS = [
    "insert_hole",
    "grasp_classify",
    "insert_HDMI",
    "insert_HDMI_D1",
    "insert_HDMI_D2",
    "insert_tube",
    "insert_tube_D1",
    "insert_tube_D2",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
    "put_bottle_in_shelf_D1",
    "put_bottle_in_shelf_D2",
]

TASK_SETTINGS_PATH = REPO_ROOT / "policy" / "task_settings.json"


def resolve_tasks(raw_tasks: list[str] | None) -> list[str]:
    if not raw_tasks:
        return list(TASKS)
    if len(raw_tasks) == 1 and raw_tasks[0] == "all":
        return list(TASKS)
    return raw_tasks


def require_known_task(task_name: str) -> str:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task_name: {task_name}. Expected one of: {TASKS}")
    return task_name
