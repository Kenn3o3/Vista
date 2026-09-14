#!/usr/bin/env bash
# Run with the active simulator Python, bundled TacEx sources and UIPC libraries.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
SIM_PYTHON=${VISTA_SIM_PYTHON:-python}
TACEX_SOURCE="$REPO_ROOT/third_party/TacEx/source"
export PYTHONPATH="$REPO_ROOT:$TACEX_SOURCE/tacex:$TACEX_SOURCE/tacex_assets:$TACEX_SOURCE/tacex_tasks:$TACEX_SOURCE/tacex_uipc${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${UIPC_LIB_DIR:-}" ]]; then
    UIPC_LIB_DIR=$("$SIM_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('uipc')
if spec and spec.origin:
    root = Path(spec.origin).parent / 'modules'
    for configuration in ('Release/bin', 'RelWithDebInfo/bin', 'release', 'releasedbg'):
        candidate = root / configuration
        if (candidate / 'libuipc_core.so').is_file():
            print(candidate)
            break
PY
)
fi
if [[ -n "$UIPC_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$UIPC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$SIM_PYTHON" "$@"
