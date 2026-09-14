# Baselines and ablations

Use the reference Vista environment. The named configs record each model's own observation and action conventions; changing those conventions invalidates checkpoint compatibility.

| Method | Preparation | Training |
| --- | --- | --- |
| ACT / UniVTAC | `bash scripts/prepare_data.sh MIDFOV_EXPERIMENT lift_can` | `bash scripts/train_act.sh MIDFOV_EXPERIMENT univtac 50 lift_can` |
| ISP | Cache built during training | `bash scripts/train_isp.sh MIDFOV_EXPERIMENT so3 50 lift_can` |
| ManiFeel / EquiDiff+Tac | Cache built during training | `bash scripts/train_dp.sh MIDFOV_EXPERIMENT manifeel 50 lift_can` or config `equidiff_tact` |
| ViTAL | `bash scripts/prepare_vital_data.sh MIDFOV_EXPERIMENT act 50 lift_can` | `bash scripts/train_vital.sh MIDFOV_EXPERIMENT act 50 lift_can` or config `dp` |

ACT/UniVTAC and ViTAL need pretrained encoders. Put `UniVTAC_encoder/best.pth`, `VITAL_encoder/vision_encoder.pth` and `VITAL_encoder/gelsight_encoder.pth` under `checkpoints/`. `best_vision_encoder.pth` and `best_gelsight_encoder.pth` are the same ViTAL weights with their original filenames. These binaries must not be committed to GitHub. Download them from the public ModelScope dataset repository (the script also installs the SHA256 manifest):

```bash
python -m pip install modelscope
bash scripts/download_data_from_modelscope.sh encoders checkpoints
```

ViTAL encoder pretraining is exposed by `python -m univtac.train.vital_encoder --help`. The `encoder/` package contains the UniVTAC tactile reconstruction model. Main VISTA and ISP training do not load these baseline checkpoints.

Additional baseline dependencies are listed in `requirements-baselines.txt`; ACT uses torchvision image backbones and may download ImageNet weights on first use. Check the selected config before running a baseline.

VISTA ablations use the same `train_vista.sh` wrapper with configs `so3_concate` (ISP+Concat-Tac), `so3_early_fusion`, and `so3_global_l0`; corresponding `so2_*` configs are included. The main configs are `so3` (I60) and `so2` (C8). The main model uses 2 observation frames, predicts 16 steps and normally executes 8 steps before replanning. Faster DDIM inference profiles are in `configs/inference/`.

Finite-group equivariance is conditional on invariant sensor-local observations under the considered joint workspace transformation. It is not a guarantee for arbitrary changes to background, lighting or contact. Offline smoke tests verify executability, not the success rates or timing reported in the paper.
