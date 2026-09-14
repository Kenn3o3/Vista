# Release validation

Checked locally on 2026-09-15 with an NVIDIA RTX 3090 (24 GB), Python 3.9, PyTorch 2.1.0 and the reference ISP/e3nn dependency stack. A separate virtual environment inherited the existing policy packages; a completely fresh Conda installation has not been timed or certified.

- Parsed the release Python sources and checked the staged file list for private paths, common credential patterns, symlinks to local storage and model/data binaries.
- Trained the full I60 VISTA model on two real `MIDFOV_EXPERIMENT/lift_can` demonstrations with batch size 1: one optimizer step, one validation step, and checkpoint save completed.
- Reloaded the saved EMA checkpoint and performed offline inference on dataset observations. Output shape was `[1, 8, 10]`; all values were finite. Normalization and workspace center came from the checkpoint.
- Large raw data, prepared caches, checkpoints and generated logs remain outside Git.

This checks the executable pipeline, not convergence, reported success rates, every baseline, or all task/configuration combinations. The local simulator (Python 3.10, Isaac Sim 4.5 / Isaac Lab 2.1.1) also loaded the policy, rendered observations, executed an action, and saved video/metadata for seed 1000. The one-step-trained policy triggered task early stopping (0/1 successes); this is an end-to-end execution check, not a successful manipulation demonstration. The relocated UIPC libraries are discovered by `scripts/sim_python.sh`.
