from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT, write_json


def conda_executable() -> str:
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.extend(
            [
                prefix.parent.parent / "bin" / "conda",
                prefix.parent / "condabin" / "conda",
            ]
        )
    executable = Path(sys.executable).resolve()
    conda_root_from_executable = executable.parents[3] if len(executable.parents) > 3 else executable.parent
    candidates.extend(
        [
            conda_root_from_executable / "bin" / "conda",
            conda_root_from_executable / "condabin" / "conda",
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "miniconda3" / "condabin" / "conda",
            Path.home() / "miniforge3" / "bin" / "conda",
            Path.home() / "miniforge3" / "condabin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "condabin" / "conda",
            Path.home() / ".conda" / "bin" / "conda",
            Path.home() / ".conda" / "condabin" / "conda",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe
    resolved = shutil.which("conda")
    if resolved:
        return resolved
    return "conda"


def conda_env_args(conda_env: str) -> list[str]:
    """Return conda-run env selector args, preferring the active env prefix.

    Some machines have more than one conda installation with an env named
    ``isp``. Selecting by prefix avoids accidentally entering the wrong one.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        if prefix.name == conda_env and prefix.is_dir():
            return ["-p", str(prefix)]

    home_prefix = Path.home() / ".conda" / "envs" / conda_env
    if home_prefix.is_dir():
        return ["-p", str(home_prefix)]

    return ["-n", conda_env]


def python_executable_for_env(conda_env: str | None = None) -> str:
    """Use the active Python unless an environment is explicitly requested."""
    if not conda_env:
        return sys.executable
    candidate = Path(conda_env).expanduser() / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    import json
    result = subprocess.run([conda_executable(), "env", "list", "--json"],
                            capture_output=True, text=True, check=True)
    matches = [Path(prefix) / "bin" / "python"
               for prefix in json.loads(result.stdout)["envs"]
               if Path(prefix).name == conda_env]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"Cannot uniquely resolve Conda environment {conda_env!r}; pass its full prefix.")
    return str(matches[0])


def run_checked(command: list[str], env: dict[str, str | None] | None = None, cwd: Path | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        for key, value in env.items():
            if value is None:
                merged_env.pop(key, None)
            else:
                merged_env[key] = str(value)
    subprocess.run(command, cwd=str(cwd or REPO_ROOT), env=merged_env, check=True)


def count_hdf5_files(hdf5_dir: Path) -> int:
    files = sorted(hdf5_dir.glob("*.hdf5"), key=lambda path: int(path.stem))
    return len(files)


def latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return sorted(candidates)[-1]


def ensure_absent(path: Path, *, label: str = "path") -> None:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def write_run_metadata(path: Path, payload: dict) -> None:
    write_json(path / "run_metadata.json", payload)


def cleanup_prepared_artifact(path: str | os.PathLike, *, label: str = "prepared data") -> bool:
    """Remove a prepared-data artifact after a successful training run.

    Caches are kept by default. Set UNIVTAC_KEEP_PREPARED_DATA=0 to remove them.
    """
    keep_value = os.environ.get("UNIVTAC_KEEP_PREPARED_DATA", "1").strip().lower()
    if keep_value in {"1", "true", "yes", "on"}:
        print(f"[cleanup] keep {label}: {path} (UNIVTAC_KEEP_PREPARED_DATA={keep_value})")
        return False

    target = Path(path).expanduser()
    if not target.exists():
        return False

    resolved = target.resolve()
    if "prepared_data" not in resolved.parts:
        raise ValueError(f"Refusing to delete non-prepared-data path: {resolved}")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print(f"[cleanup] removed {label}: {resolved}")
    return True
