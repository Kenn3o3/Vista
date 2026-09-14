#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME SEED TASK [EXTRA_ARGS ...]" >&2
    exit 1
fi

EXPERIMENT_NAME=$1
SEED=$2
TASK_NAME=$3
shift 3

python -m univtac.diagnose "$TASK_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    --seed "$SEED" \
    "$@"
