#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CHECKPOINT_PATH INFERENCE_CONFIG TOTAL_NUM [debug]" >&2
    exit 1
fi

CKPT_PATH=$1
INFERENCE_CONFIG=$2
TOTAL_NUM=$3
DEBUG_FLAG=${4:-}

command=(
    python
    -m
    univtac.eval.runner
    --ckpt
    "$CKPT_PATH"
    --inference-config
    "$INFERENCE_CONFIG"
    --total-num
    "$TOTAL_NUM"
)

if [[ "$DEBUG_FLAG" == "debug" ]]; then
    command+=(--debug)
fi

"${command[@]}"
