#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME TASK|all [TASK ...]" >&2
    exit 1
fi

EXPERIMENT_NAME=$1
shift
resolve_tasks "$@"
print_selected_tasks

for task in "${SELECTED_TASKS[@]}"; do
    python -m univtac.data.act_prepare \
        "$task" \
        --experiment-name "$EXPERIMENT_NAME" \
        --representation joint \
        --overwrite
    python -m univtac.data.act_prepare \
        "$task" \
        --experiment-name "$EXPERIMENT_NAME" \
        --representation ee \
        --overwrite
done
