#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME NUM_EPISODES TASK|all [TASK ...]" >&2
    exit 1
fi

EXPERIMENT_NAME=$1
NUM_EPISODES=$2
shift 2
resolve_tasks "$@"
print_selected_tasks

for task in "${SELECTED_TASKS[@]}"; do
    python -m univtac.data.collect "$task" \
        --experiment-name "$EXPERIMENT_NAME" \
        --episode-num "$NUM_EPISODES"
done
