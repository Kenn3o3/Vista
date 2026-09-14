#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME TASK_NAME POLICY_NAME CONFIG_NAME INFERENCE_CONFIG TOTAL_NUM [debug]" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT insert_HDMI VISTA vista vista_exec8 2 debug" >&2
    echo "Optional env: UNIVTAC_EVAL_AUTO_DRY_RUN=1 to print the resolved command without running eval." >&2
    exit 1
fi

EXPERIMENT_NAME=$1
TASK_NAME=$2
POLICY_NAME=$3
CONFIG_NAME=$4
INFERENCE_CONFIG=$5
TOTAL_NUM=$6
shift 6

CKPT_PATH=$(
python - "$REPO_ROOT" "$EXPERIMENT_NAME" "$TASK_NAME" "$POLICY_NAME" "$CONFIG_NAME" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


def _config_matches(config_dir_name: str, wanted: str) -> bool:
    if config_dir_name == wanted:
        return True
    return bool(re.fullmatch(re.escape(wanted) + r"_demo\d+", config_dir_name))


def _candidate_priority(path: Path) -> tuple[int, float, str]:
    name = path.name
    if name == "best.ckpt":
        rank = 0
    elif name == "policy_best.ckpt":
        rank = 1
    else:
        rank = 2
    return (-path.stat().st_mtime, rank, str(path))


def _latest_best_in_run(run_dir: Path) -> Path | None:
    direct_best = run_dir / "best.ckpt"
    if direct_best.is_file():
        return direct_best

    direct_policy_best = run_dir / "policy_best.ckpt"
    if direct_policy_best.is_file():
        return direct_policy_best

    nested_best = run_dir / "checkpoints" / "best.ckpt"
    if nested_best.is_file():
        return nested_best

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        pattern = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)\.ckpt$")
        scored: list[tuple[float, float, Path]] = []
        for path in ckpt_dir.glob("*.ckpt"):
            match = pattern.search(path.name)
            if match is None:
                continue
            scored.append((float(match.group(1)), -path.stat().st_mtime, path))
        if scored:
            scored.sort(key=lambda item: (item[0], item[1], str(item[2])))
            return scored[0][2]
    return None


repo_root = Path(sys.argv[1]).resolve()
experiment_name = sys.argv[2]
task_name = sys.argv[3]
policy_name = sys.argv[4]
config_name = sys.argv[5]

checkpoints_root = repo_root / "checkpoints"
if not checkpoints_root.is_dir():
    raise SystemExit(f"Checkpoint root not found: {checkpoints_root}")

experiment_root = checkpoints_root / experiment_name
if not experiment_root.is_dir():
    raise SystemExit(
        f"Experiment {experiment_name} does not exist under {checkpoints_root}"
    )

candidates: list[Path] = []
policy_root = experiment_root / task_name / policy_name
if policy_root.is_dir():
    for config_dir in policy_root.iterdir():
        if not config_dir.is_dir():
            continue
        if not _config_matches(config_dir.name, config_name):
            continue
        for run_dir in config_dir.iterdir():
            if not run_dir.is_dir():
                continue
            best = _latest_best_in_run(run_dir)
            if best is not None:
                candidates.append(best)

if not candidates:
    available_configs: list[str] = []
    if policy_root.is_dir():
        available_configs = sorted(
            config_dir.name
            for config_dir in policy_root.iterdir()
            if config_dir.is_dir()
        )
    config_hint = ""
    if available_configs:
        config_hint = f" Available configs under {policy_root}: {', '.join(available_configs)}."
    raise SystemExit(
        f"No checkpoint found for experiment={experiment_name}, task={task_name}, "
        f"policy={policy_name}, config={config_name}.{config_hint}"
    )

candidates.sort(key=_candidate_priority)
print(candidates[0])
PY
)

echo "Resolved checkpoint: $CKPT_PATH"

COMMAND=(
    bash
    "$SCRIPT_DIR/eval_ckpt.sh"
    "$CKPT_PATH"
    "$INFERENCE_CONFIG"
    "$TOTAL_NUM"
    "$@"
)

if [[ "${UNIVTAC_EVAL_AUTO_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry run:'
    for arg in "${COMMAND[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
    exit 0
fi

"${COMMAND[@]}"
