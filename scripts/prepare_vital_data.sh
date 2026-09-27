#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME CONFIG_NAME N_DEMO TASK|all [TASK ...] [-- extra prepare args...]" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT act 50 lift_can -- --overwrite" >&2
    exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_NAME=$2
N_DEMO=$3
shift 3

TASK_ARGS=()
EXTRA_ARGS=()
SEEN_DELIM=0
for arg in "$@"; do
    if [[ $SEEN_DELIM -eq 0 && "$arg" == "--" ]]; then
        SEEN_DELIM=1
        continue
    fi
    if [[ $SEEN_DELIM -eq 0 ]]; then
        TASK_ARGS+=("$arg")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

resolve_tasks "${TASK_ARGS[@]}"
print_selected_tasks

for task in "${SELECTED_TASKS[@]}"; do
    python -m univtac.data.vital_prepare \
        "$task" \
        --experiment-name "$EXPERIMENT_NAME" \
        --config-name "$CONFIG_NAME" \
        --n-demo "$N_DEMO" \
        "${EXTRA_ARGS[@]}"
done
