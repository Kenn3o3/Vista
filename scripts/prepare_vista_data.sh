#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME CONFIG_NAME N_DEMO TASK|all [TASK ...] [-- extra prepare args...]" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT vista 50 lift_can -- --overwrite" >&2
    echo "Optional env: UNIVTAC_VISTA_for task in "${SELECTED_TASKS[@]}"; do
    python -m univtac.data.vista_prepare \
        "$task" \
        --experiment-name "$EXPERIMENT_NAME" \
        --config-name "$CONFIG_NAME" \
        --n-demo "$N_DEMO" \
        "${EXTRA_ARGS[@]}"
done
