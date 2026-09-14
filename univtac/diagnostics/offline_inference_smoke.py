from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _add_policy_paths() -> None:
    for rel in ("policy/DP", "policy/ISP", "policy/VISTA"):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_policy_state(payload: dict) -> tuple[dict, str]:
    state_dicts = payload.get("state_dicts", {})
    for key in ("ema_model", "model"):
        if key in state_dicts:
            return state_dicts[key], key
    raise KeyError(f"Checkpoint has no model weights. state_dicts={sorted(state_dicts)}")


def _build_dataset(
    cfg,
    *,
    n_demo: int | None,
    use_cache: bool,
    dataset_root: Path | None,
    cache_dir: Path | None,
):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    if n_demo is not None:
        dataset_cfg.n_demo = n_demo
    dataset_cfg.use_cache = use_cache
    if dataset_root is not None:
        dataset_cfg.dataset_root = str(dataset_root)
    if cache_dir is not None:
        dataset_cfg.cache_dir = str(cache_dir)
    return hydra.utils.instantiate(dataset_cfg)


def smoke_checkpoint(
    *,
    ckpt: Path,
    n_demo: int | None,
    sample_index: int,
    device_name: str,
    use_cache: bool,
    dataset_root: Path | None,
    cache_dir: Path | None,
) -> dict:
    _add_policy_paths()
    from univtac.data.vista_prepare import _register_vista_resolvers
    _register_vista_resolvers()
    payload = torch.load(ckpt.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    model = hydra.utils.instantiate(cfg.policy)
    state, state_key = _resolve_policy_state(payload)
    model.load_state_dict(state)

    if dataset_root is None:
        task_name = str(cfg.task.dataset.task_name)
        configured_root = Path(str(cfg.task.dataset.dataset_root)).expanduser()
        configured_task_dir = configured_root / task_name / "hdf5"
        local_root = REPO_ROOT / "data" / str(getattr(cfg, "exp_name", "MIDFOV_EXPERIMENT"))
        if not configured_task_dir.is_dir() and (local_root / task_name / "hdf5").is_dir():
            dataset_root = local_root
    dataset = _build_dataset(
        cfg,
        n_demo=n_demo,
        use_cache=use_cache,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
    )
    # The checkpoint includes the training normalizer and workspace center.
    # Recomputing them from a small evaluation subset changes the policy.

    device = torch.device(device_name)
    model.to(device)
    model.eval()

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")
    sample_index = min(max(0, sample_index), len(dataset) - 1)
    sample = dataset[sample_index]
    batch = {
        "obs": {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()},
        "action": sample["action"].unsqueeze(0).to(device),
    }

    with torch.no_grad():
        result = model.predict_action(batch["obs"])

    if "action" not in result:
        raise KeyError(f"predict_action did not return 'action'. keys={sorted(result.keys())}")
    action = result["action"]
    if not torch.isfinite(action).all():
        raise RuntimeError("Predicted action contains NaN or Inf")

    gt = batch["action"]
    compare_len = min(action.shape[1], gt.shape[1])
    mse = torch.nn.functional.mse_loss(action[:, :compare_len], gt[:, :compare_len]).item()

    action_cpu = action.detach().cpu()
    summary = {
        "ckpt": str(ckpt),
        "state_key": state_key,
        "dataset_len": len(dataset),
        "sample_index": sample_index,
        "device": str(device),
        "obs_keys": sorted(batch["obs"].keys()),
        "pred_shape": list(action_cpu.shape),
        "gt_shape": list(gt.shape),
        "pred_min": float(action_cpu.min().item()),
        "pred_max": float(action_cpu.max().item()),
        "pred_mean": float(action_cpu.mean().item()),
        "pred_std": float(action_cpu.std().item()),
        "gt_compare_mse": float(mse),
        "first_pred_action": action_cpu[0, 0].tolist(),
        "first_gt_action": gt.detach().cpu()[0, 0].tolist(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline checkpoint inference smoke test.")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--n-demo", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = smoke_checkpoint(
        ckpt=args.ckpt,
        n_demo=args.n_demo,
        sample_index=args.sample_index,
        device_name=args.device,
        use_cache=not args.no_cache,
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
