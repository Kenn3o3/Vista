# Simulation

Installation, asset download, data collection, and rollout commands are in the [README](../README.md).

Collection and evaluation run headless. `HEADLESS=1` is required for collection. The evaluator passes `--headless` itself. Scene queries stay enabled in `envs/_base_task.py` so the gel pads remain attached to the fingers.

Copy `configs/experiments/MIDFOV_EXPERIMENT.yaml` when you collect a new experiment. Do not overwrite the paper dataset. Rollouts are written under `eval_result/`.
