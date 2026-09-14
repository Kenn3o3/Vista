from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.experiments import load_named_config
from univtac.paths import eval_run_dir, infer_checkpoint_metadata, require_experiment_name, timestamp_id
from univtac.subprocess_utils import ensure_absent, run_checked, write_run_metadata


def _setup_isp_hydra_eval_env(
    *,
    env: dict[str, str],
    metadata,
    inference_cfg: dict,
    gpu: str | None,
    device: str | None,
) -> str:
    env["ISP_CKPT_PATH"] = str(metadata.ckpt_path)
    env["ISP_EXEC_HORIZON"] = str(inference_cfg.get("exec_horizon", 1))
    env["ISP_MAX_INFERENCES"] = str(inference_cfg.get("max_inferences", 20))
    env["ISP_ACTION_TYPE"] = "ee_servo"
    env["UNIVTAC_EE_SERVO_MODE"] = "single_step"
    env["ISP_POSTPROCESSING"] = str(inference_cfg.get("postprocessing", "normal"))
    env["UNIVTAC_DEPLOY_POLICY_FAMILY"] = str(metadata.policy_name)
    if "temporal_agg_k" in inference_cfg:
        env["ISP_TEMPORAL_AGG_K"] = str(inference_cfg["temporal_agg_k"])
    if "scheduler" in inference_cfg:
        env["ISP_INFERENCE_SCHEDULER"] = str(inference_cfg["scheduler"])
    if "num_inference_steps" in inference_cfg:
        env["ISP_NUM_INFERENCE_STEPS"] = str(inference_cfg["num_inference_steps"])
    if metadata.inferred_n_demo is not None:
        env["EP_NUM"] = str(metadata.inferred_n_demo)
    if device is not None:
        env["ISP_DEVICE"] = device
    elif gpu is not None:
        env["ISP_DEVICE"] = "cuda:0"
    return "policy/ISP/deploy"


def _setup_dp_hydra_eval_env(
    *,
    env: dict[str, str],
    metadata,
    inference_cfg: dict,
    gpu: str | None,
    device: str | None,
) -> str:
    env["DP_CKPT_PATH"] = str(metadata.ckpt_path)
    env["DP_EXEC_HORIZON"] = str(inference_cfg.get("exec_horizon", 1))
    env["DP_MAX_INFERENCES"] = str(inference_cfg.get("max_inferences", 20))
    env["DP_ACTION_TYPE"] = "ee_servo"
    env["UNIVTAC_EE_SERVO_MODE"] = "single_step"
    env["DP_POSTPROCESSING"] = str(inference_cfg.get("postprocessing", "normal"))
    env["UNIVTAC_DEPLOY_POLICY_FAMILY"] = str(metadata.policy_name)
    if "temporal_agg_k" in inference_cfg:
        env["DP_TEMPORAL_AGG_K"] = str(inference_cfg["temporal_agg_k"])
    if metadata.inferred_n_demo is not None:
        env["EP_NUM"] = str(metadata.inferred_n_demo)
    if device is not None:
        env["DP_DEVICE"] = device
    elif gpu is not None:
        env["DP_DEVICE"] = "cuda:0"
    return "policy/DP/deploy"


def _setup_act_hydra_eval_env(
    *,
    env: dict[str, str],
    metadata,
    inference_cfg: dict,
    gpu: str | None,
    device: str | None,
) -> str:
    env["ACT_CKPT_PATH"] = str(metadata.ckpt_path)
    train_config = metadata.inferred_train_config or "train_config_ee"
    if metadata.train_metadata:
        train_config = str(metadata.train_metadata.get("train_config", train_config))
    env["ACT_TRAIN_CONFIG"] = train_config
    env["ACT_EXEC_HORIZON"] = str(inference_cfg.get("exec_horizon", 1))
    env["ACT_MAX_INFERENCES"] = str(inference_cfg.get("max_inferences", 20))
    env["ACT_ACTION_TYPE"] = "ee_servo"
    env["UNIVTAC_EE_SERVO_MODE"] = "single_step"
    env["ACT_POSTPROCESSING"] = str(inference_cfg.get("postprocessing", "normal"))
    env["UNIVTAC_DEPLOY_POLICY_FAMILY"] = str(metadata.policy_name)
    if "temporal_agg_k" in inference_cfg:
        env["ACT_TEMPORAL_AGG_K"] = str(inference_cfg["temporal_agg_k"])
    if metadata.inferred_n_demo is not None:
        env["EP_NUM"] = str(metadata.inferred_n_demo)
    if device is not None:
        env["ACT_DEVICE"] = device
    elif gpu is not None:
        env["ACT_DEVICE"] = "cuda:0"
    if metadata.train_metadata:
        return str(metadata.train_metadata.get("deploy_config", "policy/ACT/deploy_ee"))
    return "policy/ACT/deploy_ee"


def evaluate_checkpoint(
    *,
    ckpt_path: str,
    inference_config_name: str,
    total_num: int,
    experiment_name: str | None = None,
    start_seed: int = 1000,
    gpu: str | None = None,
    device: str | None = None,
    debug: bool = False,
) -> Path:
    metadata = infer_checkpoint_metadata(REPO_ROOT, ckpt_path)
    train_meta = metadata.train_metadata or {}
    resolved_experiment_name = experiment_name or metadata.experiment_name
    if resolved_experiment_name != metadata.experiment_name:
        raise ValueError(
            f"Checkpoint belongs to experiment {metadata.experiment_name}, got explicit {resolved_experiment_name}"
        )

    inference_cfg = load_named_config("inference", inference_config_name)
    eval_id = timestamp_id()
    save_dir = eval_run_dir(
        REPO_ROOT,
        resolved_experiment_name,
        metadata.task_name,
        metadata.policy_name,
        metadata.config_name,
        metadata.run_id,
        inference_config_name,
        eval_id,
    )
    ensure_absent(save_dir, label="evaluation output directory")

    env = {
        "UNIVTAC_EVAL_SAVE_DIR": str(save_dir),
        "UNIVTAC_EXPERIMENT_NAME": resolved_experiment_name,
    }
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    debug_dump_enabled = debug or bool(inference_cfg.get("debug_dump", False))
    env["UNIVTAC_DEBUG_DUMP"] = "1" if debug_dump_enabled else "0"
    env["UNIVTAC_DEBUG_DIR"] = str(save_dir / "debug")

    if metadata.policy_name == "DP":
        deploy_config = _setup_dp_hydra_eval_env(
            env=env,
            metadata=metadata,
            inference_cfg=inference_cfg,
            gpu=gpu,
            device=device,
        )
    elif metadata.policy_name == "ACT":
        deploy_config = _setup_act_hydra_eval_env(
            env=env,
            metadata=metadata,
            inference_cfg=inference_cfg,
            gpu=gpu,
            device=device,
        )
    elif metadata.policy_name == "ISP":
        deploy_config = _setup_isp_hydra_eval_env(
            env=env,
            metadata=metadata,
            inference_cfg=inference_cfg,
            gpu=gpu,
            device=device,
        )
    elif metadata.policy_name == "VISTA":
        env["VISTA_CKPT_PATH"] = str(metadata.ckpt_path)
        env["VISTA_EXEC_HORIZON"] = str(inference_cfg.get("exec_horizon", 1))
        env["VISTA_MAX_INFERENCES"] = str(inference_cfg.get("max_inferences", 20))
        env["VISTA_ACTION_TYPE"] = "ee_servo"
        env["UNIVTAC_EE_SERVO_MODE"] = "single_step"
        env["VISTA_POSTPROCESSING"] = str(inference_cfg.get("postprocessing", "normal"))
        if "scheduler" in inference_cfg:
            env["VISTA_INFERENCE_SCHEDULER"] = str(inference_cfg["scheduler"])
        if "num_inference_steps" in inference_cfg:
            env["VISTA_NUM_INFERENCE_STEPS"] = str(inference_cfg["num_inference_steps"])
        if "temporal_agg_k" in inference_cfg:
            env["VISTA_TEMPORAL_AGG_K"] = str(inference_cfg["temporal_agg_k"])
        if metadata.inferred_n_demo is not None:
            env["EP_NUM"] = str(metadata.inferred_n_demo)
        if device is not None:
            env["VISTA_DEVICE"] = device
        elif gpu is not None:
            env["VISTA_DEVICE"] = "cuda:0"
        deploy_config = "policy/VISTA/deploy"
    elif metadata.policy_name == "ViTAL" and "hydra_config" in train_meta:
        deploy_config = _setup_dp_hydra_eval_env(
            env=env,
            metadata=metadata,
            inference_cfg=inference_cfg,
            gpu=gpu,
            device=device,
        )
    elif metadata.policy_name == "ViTAL":
        env["VITAL_CKPT_PATH"] = str(metadata.ckpt_path)
        env["VITAL_EXEC_HORIZON"] = str(inference_cfg.get("exec_horizon", 1))
        env["VITAL_MAX_INFERENCES"] = str(inference_cfg.get("max_inferences", 20))
        env["VITAL_ACTION_TYPE"] = "ee_servo"
        env["UNIVTAC_EE_SERVO_MODE"] = "single_step"
        if metadata.inferred_n_demo is not None:
            env["EP_NUM"] = str(metadata.inferred_n_demo)
        if device is not None:
            env["VITAL_DEVICE"] = device
        elif gpu is not None:
            env["VITAL_DEVICE"] = "cuda:0"
        deploy_config = "policy/ViTAL/deploy_clean"
    else:
        raise ValueError(f"Unsupported policy_name: {metadata.policy_name}")

    command = [
        sys.executable,
        "scripts/eval_policy.py",
        metadata.task_name,
        "clean",
        deploy_config,
        "--total_num",
        str(total_num),
        "--start_seed",
        str(start_seed),
        "--headless",
    ]
    if device is not None:
        command.extend(["--device", device])

    run_checked(command, env=env)
    write_run_metadata(
        save_dir,
        {
            "experiment_name": resolved_experiment_name,
            "policy_name": metadata.policy_name,
            "task_name": metadata.task_name,
            "config_name": metadata.config_name,
            "train_run_id": metadata.run_id,
            "eval_id": eval_id,
            "ckpt_path": str(metadata.ckpt_path),
            "eval_dir": str(save_dir),
            "inference_config_name": inference_config_name,
            "inference_config": inference_cfg,
        },
    )
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint in the clean UniVTAC layout")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--inference-config", required=True, type=str)
    parser.add_argument("--total-num", type=int, default=20)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    evaluate_checkpoint(
        ckpt_path=args.ckpt,
        inference_config_name=args.inference_config,
        total_num=args.total_num,
        experiment_name=require_experiment_name(args.experiment_name)
        if args.experiment_name is not None
        else None,
        start_seed=args.start_seed,
        gpu=args.gpu,
        device=args.device,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
