#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASKS=(
  grasp_classify
  insert_HDMI
  insert_hole
  insert_tube
  lift_bottle
  lift_can
  pull_out_key
  put_bottle_in_shelf
)

resolve_tasks() {
    if [[ $# -eq 0 || "$1" == "all" ]]; then
        SELECTED_cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASKS=("${TASKS[@]}")
    else
        SELECTED_cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASKS=("$@")
    fi
}

print_selected_tasks() {
    printf 'Tasks:'
    for task in "${SELECTED_TASKS[@]}"; do
        printf ' %s' "$task"
    done
    printf '\n'
}
