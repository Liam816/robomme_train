# E2E learned memory on worker36021 / MUSA

Code: /home/mccuseradmin/heping/projects/robomme_train_wd, branch codex/musa-percmem-e2e. Base: codex/musa-percmem-v1 (4cdc1b6); cherry-pick b4432b1 as 86840b7.

The source configs/train/robomme_percmem_v1_e2e_repro.yaml is byte-identical to b4432b1. configs/train/robomme_musa_worker36021.yaml only resolves host paths and MUSA microbatch2 x accumulation8 x four devices = global64. Model, original seed initialization, 80k budget, LR/warmup, optimizer, EMA and the FM-only objective are preserved. Scorer is trainable; only vision is frozen. Hard argmin forward and the original softmax straight-through gradient traverse every write. No trained P0/P2 weight or offline bank is loaded.

CPU and MUSA each pass 184 numerical checks; the original eight E2E tests and 33 P0 checks pass. Production dimensions exercise 41 real historical frames, including evictions; bank IDs/ticks/masks/tokens match JAX. Full forward and nonzero-memory integration pass. Raw reports and assets: /data/heping/robomme_musa/e2e. This host uses the existing verified 20fps/77562-keyframe native cache.

Launch with the isolated /data/heping/robomme_musa/venv Python: scripts/launch_musa_e2e.py. It requires validation reports, runs a three-update four-card/checkpoint preflight, then begins a fresh 80000-update experiment. Start this supervisor with stdin DEVNULL and start_new_session=True for detached execution. It writes /data/heping/robomme_musa/e2e/training.status.json and per-run logs under logs/. A failed stage stops the pipeline.

Direct entry: bash scripts/run_musa_percmem_e2e.sh train --steps 80000 --run-name UNIQUE_NAME. Selects only physical GPUs 4,5,6,7 and CPU64-95, with four MCCL/FSDP ranks. Checkpoints use this Torch/MUSA backend and same four-rank layout; they are not Orbax checkpoints.

Production warmup note: the first three actual updates have finite loss/gradients and reader updates, but scorer/A_feat gradients remain zero. A production-dimension controlled diagnostic reproduces this in original JAX: with reader output scale2e-8, FP16 scorer gradient is zero in both backends; an FP32 diagnostic has norm9.727e-9. With scale0.03, FP16 gradients are 0.01457849 (JAX) and 0.01457844 (MUSA). The original zero initialization, FP16 boundary and actual warmup LR remain unchanged. Therefore a nonzero scorer gradient within the first three production steps is not a valid gate; controlled connectivity/update checks and ongoing gradient logging are retained.
