#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME TASK_NAME|all CONFIG_NAME INFERENCE_CONFIG TOTAL_NUM [debug]" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT insert_HDMI vista vista_exec8 20 debug" >&2
    echo "Optional env: UNIVTAC_EVAL_AUTO_DRY_RUN=1 to print the resolved command without running eval." >&2
    exit 1
fi

EXPERIMENT_NAME=$1
TASK_NAME=$2
CONFIG_NAME=$3
INFERENCE_CONFIG=$4
TOTAL_NUM=$5
shift 5

resolve_tasks "$TASK_NAME"
print_selected_tasks

for task in "${SELECTED_TASKS[@]}"; do
    bash "$SCRIPT_DIR/eval_ckpt_auto.sh" \
        "$EXPERIMENT_NAME" \
        "$task" \
        VISTA \
        "$CONFIG_NAME" \
        "$INFERENCE_CONFIG" \
        "$TOTAL_NUM" \
        "$@"
done
