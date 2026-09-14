# Simulation setup

Policy training and offline checkpoint inference use `environment.yml` and do not require simulation. Rollouts additionally need Isaac Sim 4.5.0, Isaac Lab 2.1.1, TacEx/UIPC and cuRobo in a Python 3.10 simulator environment on the same computer. The simulator environment must also contain the policy's PyTorch3D, escnn, e3nn, Hydra and diffusion dependencies. Do not install the Python 3.9 training lock blindly into Isaac Sim.

The bundled TacEx integration documents its build in [Local-Installation.md](../third_party/TacEx/docs/source/installation/Local-Installation.md). Core installation, after Isaac Sim/Isaac Lab and UIPC build dependencies are installed:

```bash
# Activate your Isaac Sim environment first.
python -m pip install -e third_party/TacEx/source/tacex
python -m pip install -e third_party/TacEx/source/tacex_assets
python -m pip install -e third_party/TacEx/source/tacex_tasks
python -m pip install -e third_party/TacEx/source/tacex_uipc
```

`scripts/sim_python.sh` uses the active simulator Python and discovers UIPC shared libraries beside the installed bindings. Set `UIPC_LIB_DIR` only for a nonstandard build location. It also makes the bundled TacEx source packages importable.

The last installation command builds libuipc and requires the documented CUDA/CMake/Vcpkg toolchain. `envs/robot/robot.py` additionally imports cuRobo for motion planning. Install cuRobo following its upstream instructions and use `assets/embodiments/franka/curobo.yml`.

Large TacEx sensor/robot assets, calibration arrays and the environment texture are hosted in the public MIDFOV ModelScope repository under `sim_assets/`. Download the exact version with SHA256 verification:

```bash
python -m pip install modelscope
python scripts/download_sim_assets.py
```

This restores the customized paper assets; a stock upstream TacEx download is not assumed equivalent. Existing asset files are protected unless `--overwrite` is supplied. Compiled libraries are built locally and are not part of the asset archive.

```bash
# Dataset collection into a new experiment; do not overwrite the paper dataset.
bash scripts/collect_data.sh MY_EXPERIMENT 10 lift_can

# Evaluation of a trained VISTA checkpoint.
bash scripts/sim_python.sh -m univtac.eval.runner \
  --ckpt checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt \
  --experiment-name MIDFOV_EXPERIMENT \
  --inference-config vista_exec8_ddim8 --total-num 40 --start-seed 1000
```

Define `configs/experiments/MY_EXPERIMENT.yaml` by copying `MIDFOV_EXPERIMENT.yaml` before collection. Rollout outputs remain under the ignored `eval_result/` directory. The default evaluator is headless. Camera rendering remains enabled for observations; livestreaming is opt-in.
