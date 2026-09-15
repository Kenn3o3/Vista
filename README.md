# Vista

Official code for **Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation**, CoRL 2026.

VISTA fuses wrist vision and tactile spherical features, aligns the representation with the workspace using the end-effector orientation, and predicts action chunks with a finite-group equivariant diffusion policy.

[Project website](https://kenn3o3.github.io/vista-paper.github.io/) · [Real robot code](https://github.com/Kenn3o3/vt_franka) · [Simulation dataset](https://modelscope.cn/datasets/kenn3o3/MIDFOV_EXPERIMENT)

## Installation

Linux with an NVIDIA GPU is recommended. The reference policy environment uses Python 3.9, PyTorch 2.1.0, CUDA 11.8 and PyTorch3D 0.7.5.

```bash
git clone https://github.com/Kenn3o3/Vista.git
cd Vista
conda env create -f environment.yml
conda activate vista
python -c "import torch, pytorch3d, escnn, e3nn; print(torch.cuda.is_available())"
```

Run the following commands from this repository root. The shell wrappers use the active Python environment. `univtac` remains the internal Python package name for compatibility with existing experiments and checkpoints. Training and offline inference do not require Isaac Sim or a robot.

## Dataset

The ModelScope repository contains a top-level `MIDFOV_EXPERIMENT` directory. Download it with:

```bash
python -m pip install modelscope
bash scripts/download_data_from_modelscope.sh MIDFOV_EXPERIMENT
```

Expected layout:

```text
data/MIDFOV_EXPERIMENT/<task>/hdf5/0.hdf5
data/MIDFOV_EXPERIMENT/<task>/metadata.json
```

Tasks: `grasp_classify`, `insert_HDMI`, `insert_hole`, `insert_tube`, `lift_bottle`, `lift_can`, `pull_out_key`, `put_bottle_in_shelf`. Existing data may be symlinked to `data/`; it is ignored by Git.

## Train VISTA

```bash
# Cache the data (training can also build the cache automatically).
bash scripts/prepare_vista_data.sh MIDFOV_EXPERIMENT so3 50 lift_can

# I60 model; use so2 for the C8 model.
bash scripts/train_vista.sh MIDFOV_EXPERIMENT so3 50 lift_can -- \
  --batch-size 32 --num-epochs 300 --seed 0 --run-id vista_so3_seed0
```

Use `all` in place of the task to run the eight tasks sequentially on one computer. `--gpu 0` selects a GPU. The default is offline W&B logging; `--wandb-mode disabled` disables it. Data caches are retained for repeated runs.

For a short executable pipeline check on real demonstrations:

```bash
bash scripts/train_vista.sh MIDFOV_EXPERIMENT so3 2 lift_can -- \
  --batch-size 1 --num-epochs 1 --num-workers 0 \
  --max-train-steps 1 --max-val-steps 1 \
  --wandb-mode disabled --run-id smoke
```

The smoke run checks optimization and checkpoint creation; it does not reproduce paper performance. Use a new run ID for each run. Checkpoints are written to:

```text
checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt
```

## Offline inference

```bash
python -m univtac.diagnostics.offline_inference_smoke \
  --ckpt checkpoints/MIDFOV_EXPERIMENT/lift_can/VISTA/so3_demo50/vista_so3_seed0/checkpoints/best.ckpt \
  --dataset-root data/MIDFOV_EXPERIMENT --n-demo 2 --no-cache
```

This loads the checkpoint's EMA weights, normalization and workspace center, reads an observation from the dataset, and verifies finite predicted actions. It does not actuate hardware. Checkpoints contain serialized Python objects: load only checkpoints you trust.

## Simulation, baselines and real robots

- [Simulation setup and rollouts](docs/SIMULATION.md)
- [Baselines, encoders and ablations](docs/BASELINES.md)
- [Validation record](docs/VALIDATION.md)
- [Real robot collection, training and deployment](https://github.com/Kenn3o3/vt_franka)

VISTA's equivariant image encoders are trained with the policy and require no separate pretrained encoder download. Data and model weights are kept outside GitHub. Baseline encoders are available in the dataset repository’s `encoders/` directory; see [BASELINES.md](docs/BASELINES.md) for download instructions.

## Layout

| Directory | Purpose |
| --- | --- |
| `policy/VISTA/` | VISTA model, diffusion decoder, datasets and Hydra configs |
| `configs/vista/` | Main configurations and fusion ablations |
| `univtac/`, `scripts/` | Local data preparation, training and evaluation |
| `policy/ACT`, `policy/ISP`, `policy/DP`, `policy/ViTAL` | Baselines |
| `envs/`, `assets/`, `third_party/TacEx/` | Simulation tasks and tactile simulation integration |

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
