<p align="center">
  <img src="https://vista-paper.github.io/assets/icons/vista-sphere.svg" width="84" alt="VISTA" />
</p>

<h1 align="center">VISTA</h1>

<p align="center">
  <strong>Equivariant Visual-Tactile Diffusion Policy<br>for Contact-Rich Manipulation</strong>
</p>

<p align="center">
  <a href="https://vista-paper.github.io/"><img src="https://img.shields.io/badge/CoRL-2026-1463ff" alt="CoRL 2026" /></a>
  <a href="https://vista-paper.github.io/"><img src="https://img.shields.io/badge/Project-website-25bfd3" alt="Project website" /></a>
  <a href="https://github.com/Kenn3o3/vt_franka"><img src="https://img.shields.io/badge/Real_robot-vt__franka-444444" alt="Real robot code" /></a>
  <a href="https://modelscope.cn/datasets/kenn3o3/MIDFOV_EXPERIMENT"><img src="https://img.shields.io/badge/Dataset-ModelScope-d96c2c" alt="Simulation dataset" /></a>
</p>

<p align="center">
  <a href="https://kenn3o3.github.io/">Lik Hang Kenny Wong</a><sup>1</sup> ·
  <a href="https://yiyao-ma.github.io/">Yiyao Ma</a><sup>1</sup> ·
  Xiu-Shen Wei<sup>2</sup> ·
  Zelong Tan<sup>1</sup> ·
  Zhuheng Song<sup>1</sup> ·
  Dongsheng Xie<sup>1</sup> ·
  <a href="https://ck-kai.github.io/">Kai Chen</a><sup>1</sup> ·
  <a href="https://www.cse.cuhk.edu.hk/~qdou/index.html">Qi Dou</a><sup>1</sup>
  <br>
  <sup>1</sup> The Chinese University of Hong Kong · <sup>2</sup> Southeast University
</p>

VISTA fuses wrist vision and tactile spherical features, aligns that representation with the workspace using the end-effector orientation, and predicts action chunks with a finite-group equivariant diffusion policy.

<p align="center">
  <img src="https://vista-paper.github.io/assets/figures/method.png" width="100%" alt="VISTA lifts wrist RGB and tactile images to spheres, fuses them, and predicts actions with an equivariant diffusion policy." />
</p>

<p align="center">
  <img src="https://vista-paper.github.io/assets/figures/sim-environments.png" width="100%" alt="Eight simulated contact-rich tasks: grasp classify, insert HDMI, insert tube, insert hole, lift bottle, lift can, pull out key, and put bottle in shelf." />
</p>

<p align="center">
  Rollout videos for all eight tasks, plus the real-robot demos, are on the <a href="https://vista-paper.github.io/#experiments">project website</a>.
</p>

Run every command below from this repository root. The shell wrappers use the active Python. The internal package name remains `univtac`.

| | |
| --- | --- |
| [Environment](#environment) | One Python 3.10 conda env named `vista` |
| [Simulation assets](#simulation-assets) | Robot, gel, and scene files |
| [Dataset](#dataset) | Download the paper data, or collect your own |
| [Train VISTA](#train-vista) | `so3` (I60) and `so2` (C8) |
| [Offline inference](#offline-inference) | Check a checkpoint without Isaac |
| [Simulation evaluation](#simulation-evaluation) | Headless rollouts |
| [Baselines](#baselines) | ACT, ISP, ManiFeel, EquiDiff+Tac, ViTAL |

## Environment

`environment.yml` is an older Python 3.9 lock. Do not install it. Collection, training, offline inference, and simulation rollouts share one Python 3.10 environment named `vista`.

Linux with an NVIDIA GPU is required for simulation. Training and offline inference need the same PyTorch stack and do not start Isaac Sim. A bootstrapped [vcpkg](https://github.com/microsoft/vcpkg) checkout is required to build libuipc. If your shell uses `set -u`, turn that off before `conda activate`.

```bash
conda create -n vista python=3.10 -y
conda env update -n vista -f third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml
conda activate vista

python -m pip install -U pip 'setuptools<82' wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
python -m pip install flatdict==4.0.1 --no-build-isolation
```

Isaac Lab v2.1.1 lives outside this repository. Its installer upgrades PyTorch, so reinstall the CUDA 12.4 wheel before building cuRobo.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.1
TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install
cd -

python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install warp-lang==1.0.0
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout 0a50de1ba72db304195d59d9d0b1ed269696047f
CUDA_HOME="$CONDA_PREFIX" python -m pip install -e . --no-build-isolation
cd -
```

Install the policy stack, then TacEx. `CUDA_HOME` must stay on the conda CUDA 12.4 toolkit while libuipc builds. Point `CMAKE_TOOLCHAIN_FILE` at your bootstrapped vcpkg tree.

```bash
python -m pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
python -m pip install -r requirements.txt
python -m pip install -r requirements-baselines.txt

python -m pip install -e third_party/TacEx/source/tacex
python -m pip install -e third_party/TacEx/source/tacex_assets
python -m pip install -e third_party/TacEx/source/tacex_tasks
python -m pip uninstall -y torch_scatter
python -m pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install transforms3d trimesh tetgen
export CUDA_HOME="$CONDA_PREFIX"
export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
python -m pip install -e third_party/TacEx/source/tacex_uipc -v --no-build-isolation

python -c "import torch, pytorch3d, escnn, e3nn; print(torch.__version__, torch.cuda.is_available())"
```

`import tacex` before Isaac's `SimulationApp` starts fails, because Omniverse has not been launched yet. Collection and evaluation start the simulator first.

## Simulation assets

Large sensor, robot, and scene files are hosted with the dataset. Existing files are kept unless you pass `--overwrite`.

```bash
python -m pip install modelscope
python scripts/download_sim_assets.py
```

## Dataset

Download the paper demonstrations:

```bash
bash scripts/download_data_from_modelscope.sh MIDFOV_EXPERIMENT
```

Expected layout:

```text
data/MIDFOV_EXPERIMENT/<task>/hdf5/0.hdf5
data/MIDFOV_EXPERIMENT/<task>/metadata.json
```

Tasks: `grasp_classify`, `insert_HDMI`, `insert_hole`, `insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, `put_bottle_in_shelf`.

To collect a new experiment, copy `configs/experiments/MIDFOV_EXPERIMENT.yaml` to `configs/experiments/MY_EXPERIMENT.yaml` and set `name: MY_EXPERIMENT`. Collection is headless. Scene queries stay enabled so the gel pads remain attached to the fingers.

```bash
HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES bash scripts/collect_data.sh MY_EXPERIMENT 50 lift_can
```

Use `all` instead of a task name to run the eight tasks. Collect into a new experiment name so the paper dataset stays intact.

## Train VISTA

The main configs are `so3` (I60) and `so2` (C8). Training keeps the last 20% of episodes for validation, so use at least two demonstrations. Checkpoints contain pickled Python objects; load only checkpoints you trust.

```bash
bash scripts/prepare_vista_data.sh MIDFOV_EXPERIMENT so3 50 lift_can
bash scripts/train_vista.sh MIDFOV_EXPERIMENT so3 50 lift_can -- \
  --batch-size 32 --num-epochs 300 --seed 0 --run-id vista_so3_seed0
```

Use `all` in place of the task to train the eight tasks one after another. `--gpu 0` selects a GPU. The default W&B mode is offline. `--wandb-mode disabled` turns logging off. Caches stay on disk for the next run.

A short check that optimization and checkpoint writing run:

```bash
bash scripts/train_vista.sh MIDFOV_EXPERIMENT so3 2 lift_can -- \
  --batch-size 1 --num-epochs 1 --num-workers 0 \
  --max-train-steps 1 --max-val-steps 1 \
  --wandb-mode disabled --run-id smoke
```

The 50-demo, 300-epoch command is the paper setting. The short check does not reproduce those results. Checkpoints are written to:

```text
checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt
```

## Offline inference

```bash
python -m univtac.diagnostics.offline_inference_smoke \
  --ckpt checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt \
  --dataset-root data/MIDFOV_EXPERIMENT --n-demo 2
```

This loads the checkpoint EMA weights, normalizer, and workspace center, reads one observation, and checks that the predicted action is finite. It does not move a robot.

## Simulation evaluation

The evaluator is headless. `vista_exec8_ddim8` is the main profile: execute 8 actions, then replan, using 8 DDIM steps. Other profiles are `vista_exec8`, `vista_exec4`, `vista_exec8_ddim16`, and `vista_temporal_agg_exec1`. Results go under `eval_result/`.

```bash
OMNI_KIT_ACCEPT_EULA=YES bash scripts/sim_python.sh -m univtac.eval.runner \
  --ckpt checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt \
  --experiment-name MIDFOV_EXPERIMENT \
  --inference-config vista_exec8_ddim8 \
  --total-num 40 --start-seed 1000
```

`--total-num 1` runs a single seed and confirms that the simulator, policy, and logger start. Task success still depends on a fully trained checkpoint.

Summarize finished rollouts with `python scripts/summarize_evals.py --help`.

## Baselines

Install `requirements-baselines.txt` from the [environment](#environment) section first. ACT and ViTAL also need encoder weights:

```bash
bash scripts/download_data_from_modelscope.sh encoders checkpoints
```

That places `UniVTAC_encoder/best.pth`, `VITAL_encoder/vision_encoder.pth`, and `VITAL_encoder/gelsight_encoder.pth` under `checkpoints/`.

| Method | Prepare | Train |
| --- | --- | --- |
| ACT | `bash scripts/prepare_data.sh MIDFOV_EXPERIMENT lift_can` | `bash scripts/train_act.sh MIDFOV_EXPERIMENT univtac 50 lift_can` |
| ISP | built during training | `bash scripts/train_isp.sh MIDFOV_EXPERIMENT so3 50 lift_can` |
| ManiFeel | built during training | `bash scripts/train_dp.sh MIDFOV_EXPERIMENT manifeel 50 lift_can` |
| EquiDiff+Tac | built during training | `bash scripts/train_dp.sh MIDFOV_EXPERIMENT equidiff_tact 50 lift_can` |
| ViTAL ACT | `bash scripts/prepare_vital_data.sh MIDFOV_EXPERIMENT act 50 lift_can` | `bash scripts/train_vital.sh MIDFOV_EXPERIMENT act 50 lift_can` |
| ViTAL DP | `bash scripts/prepare_vital_data.sh MIDFOV_EXPERIMENT dp 50 lift_can` | `bash scripts/train_vital.sh MIDFOV_EXPERIMENT dp 50 lift_can` |

A one-epoch check uses the same commands with `2` demonstrations, `--batch-size 1 --num-epochs 1 --wandb-mode disabled --run-id smoke`. ACT maps `--num-epochs` to optimizer steps, so `1` is a single update. VISTA ablations (`so3_concate`, `so3_early_fusion`, `so3_global_l0`, and the `so2_*` variants) use `scripts/train_vista.sh`. See [docs/BASELINES.md](docs/BASELINES.md).

Evaluate a baseline checkpoint with the same `univtac.eval.runner` command and an inference config such as `isp_exec8_ddim8` or `temporal_agg`.

## Layout

| Directory | Purpose |
| --- | --- |
| `policy/VISTA/` | VISTA model, diffusion decoder, datasets, and Hydra configs |
| `configs/vista/` | `so3`, `so2`, and fusion ablations |
| `configs/experiments/` | Simulator experiment settings. The paper config is `MIDFOV_EXPERIMENT.yaml` |
| `univtac/`, `scripts/` | Data preparation, training, and evaluation |
| `policy/ACT`, `policy/ISP`, `policy/DP`, `policy/ViTAL` | Baselines |
| `envs/`, `assets/`, `third_party/TacEx/` | Simulation tasks and tactile simulation |

Real-robot collection and deployment: [vt_franka](https://github.com/Kenn3o3/vt_franka).

## Citation

```bibtex
@inproceedings{wong2026vista,
  title={Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation},
  author={Wong, Lik Hang Kenny and Ma, Yiyao and Wei, Xiu-Shen and Tan, Zelong and Song, Zhuheng and Xie, Dongsheng and Chen, Kai and Dou, Qi},
  booktitle={Conference on Robot Learning},
  year={2026}
}
```

## Acknowledgments and license

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original and third-party license files are preserved. The root [Apache 2.0 license](LICENSE) does not replace the licenses of bundled dependencies.
