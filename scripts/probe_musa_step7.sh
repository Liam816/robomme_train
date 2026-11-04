#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROBOMME_MUSA_ROOT:-/data/heping/robomme_musa/e2e}"
PROFILE="${ROBOMME_MUSA_CONFIG:-$REPO/configs/train/robomme_percmem_v1_e2e_repro.yaml}"
source /data/heping/miniconda3/etc/profile.d/conda.sh
conda activate quicforge
export ROBOMME_MUSA_ROOT="$ROOT"
export ROBOMME_MUSA_REPO="$REPO"
export ROBOMME_MUSA_CONFIG="$PROFILE"
export ROBOMME_MUSA_RUNTIME="${ROBOMME_MUSA_RUNTIME:-$REPO/configs/train/robomme_musa_worker36021.yaml}"
export MUSA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME=/data/heping/robomme_musa/assets
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export ROBOMME_CPU_POOL=64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cpu_cache"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
PY=/data/heping/robomme_musa/venv/bin/python
cd "$REPO"
export MUSA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES=4
exec taskset -c 64-71 "$PY" -u -m mme_vla_suite.musa.probe_inference_gradients --checkpoint /data/heping/robomme_musa/e2e/runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42/profile_resume_e2e_20261104/step_000007 --output /data/heping/robomme_musa/e2e/profiles/inference20_gradrepair_20261104 --mode both
