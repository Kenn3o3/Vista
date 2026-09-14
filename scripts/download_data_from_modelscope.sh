#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PATH_IN_REPO_OR_EXPERIMENT [LOCAL_PATH]" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT" >&2
    echo "Example: $0 MIDFOV_EXPERIMENT data/MIDFOV_EXPERIMENT" >&2
    exit 1
fi

PATH_IN_REPO=$1
PATH_IN_REPO=${PATH_IN_REPO#/}
LOCAL_PATH=${2:-"$REPO_ROOT/data/$PATH_IN_REPO"}

MODELSCOPE_REPO_ID=${MODELSCOPE_REPO_ID:-kenn3o3/MIDFOV_EXPERIMENT}
MODELSCOPE_REPO_TYPE=${MODELSCOPE_REPO_TYPE:-dataset}
MODELSCOPE_TOKEN=${MODELSCOPE_TOKEN:-}
MODELSCOPE_REVISION=${MODELSCOPE_REVISION:-master}
MODELSCOPE_MAX_WORKERS=${MODELSCOPE_MAX_WORKERS:-8}
MODELSCOPE_DRY_RUN=${MODELSCOPE_DRY_RUN:-0}

if [[ -z "$MODELSCOPE_REPO_ID" ]]; then
    echo "MODELSCOPE_REPO_ID is required." >&2
    exit 1
fi

echo "ModelScope download"
echo "  repo id    : $MODELSCOPE_REPO_ID"
echo "  repo type  : $MODELSCOPE_REPO_TYPE"
echo "  repo path  : ${PATH_IN_REPO:-.}"
echo "  local path : $LOCAL_PATH"
echo "  revision   : $MODELSCOPE_REVISION"
echo "  dry run    : $MODELSCOPE_DRY_RUN"

export PATH_IN_REPO
export LOCAL_PATH
export MODELSCOPE_REPO_ID
export MODELSCOPE_REPO_TYPE
export MODELSCOPE_TOKEN
export MODELSCOPE_REVISION
export MODELSCOPE_MAX_WORKERS
export MODELSCOPE_DRY_RUN

python - <<'PY'
from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path

from modelscope import dataset_snapshot_download, snapshot_download
from modelscope.hub.api import HubApi


def _call_snapshot_download(
    download_fn,
    *,
    repo_id: str,
    revision: str,
    local_dir: str,
    allow_patterns,
    max_workers: int,
    token: str | None,
):
    signature = inspect.signature(download_fn)
    supported = set(signature.parameters)

    kwargs = {}
    if "revision" in supported:
        kwargs["revision"] = revision
    if "local_dir" in supported:
        kwargs["local_dir"] = local_dir
    if "allow_patterns" in supported:
        kwargs["allow_patterns"] = allow_patterns
    elif "allow_file_pattern" in supported:
        kwargs["allow_file_pattern"] = allow_patterns
    if "max_workers" in supported:
        kwargs["max_workers"] = max_workers

    if token:
        if "token" in supported:
            kwargs["token"] = token
        else:
            HubApi().login(token)

    return download_fn(repo_id, **kwargs)


repo_id = os.environ["MODELSCOPE_REPO_ID"]
repo_type = os.environ["MODELSCOPE_REPO_TYPE"]
token = os.environ.get("MODELSCOPE_TOKEN") or None
revision = os.environ["MODELSCOPE_REVISION"]
max_workers = int(os.environ["MODELSCOPE_MAX_WORKERS"])
dry_run = os.environ["MODELSCOPE_DRY_RUN"] == "1"
path_in_repo = os.environ["PATH_IN_REPO"].strip("/")
local_path = Path(os.environ["LOCAL_PATH"]).expanduser().resolve()

summary = {
    "repo_id": repo_id,
    "repo_type": repo_type,
    "path_in_repo": path_in_repo,
    "local_path": str(local_path),
    "revision": revision,
}

if dry_run:
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

allow_patterns = None
if path_in_repo:
    allow_patterns = [
        path_in_repo,
        f"{path_in_repo}/**",
    ]

download_fn = dataset_snapshot_download if repo_type == "dataset" else snapshot_download

with tempfile.TemporaryDirectory(prefix="modelscope_download_") as tmp_dir:
    snapshot_root = _call_snapshot_download(
        download_fn,
        repo_id=repo_id,
        revision=revision,
        local_dir=tmp_dir,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
        token=token,
    )

    source_path = Path(snapshot_root)
    if path_in_repo:
        source_path = source_path / path_in_repo
    if not source_path.exists():
        raise SystemExit(
            f"Downloaded snapshot does not contain requested path: {path_in_repo or '.'}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, local_path, dirs_exist_ok=True)
    else:
        if local_path.is_dir():
            destination = local_path / source_path.name
        else:
            destination = local_path
            destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

print("Download complete.")
print(local_path)
PY
