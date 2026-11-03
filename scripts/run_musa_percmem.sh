#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROBOMME_MUSA_ROOT:-/data/heping/robomme_musa}"
REPO="$ROOT/repo"
source /data/heping/miniconda3/etc/profile.d/conda.sh
conda activate quicforge
export ROBOMME_MUSA_ROOT="$ROOT"
export MUSA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$ROOT/assets"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export ROBOMME_CPU_POOL="${ROBOMME_CPU_POOL:-64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cpu_cache"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/venv/bin/python"
cd "$REPO"
mode="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  index|norm|export)
    exec "$PY" -u -m mme_vla_suite.musa.prepare_assets "$mode" "$@"
    ;;
  validate-cpu)
    exec "$PY" -u -m mme_vla_suite.musa.validate_cpu "$@"
    ;;
  validate-production)
    export MUSA_VISIBLE_DEVICES=4
    export CUDA_VISIBLE_DEVICES=4
    exec "$PY" -u -m mme_vla_suite.musa.validate_production "$@"
    ;;
  extract)
    exec "$PY" -u -m torch.distributed.run --standalone --nproc-per-node=4 -m mme_vla_suite.musa.extract_cache "$@"
    ;;
  train)
    exec "$PY" -u -m torch.distributed.run --standalone --nproc-per-node=4 -m mme_vla_suite.musa.train --config "$REPO/configs/train/robomme_percmem_v1_teacher_musa.yaml" "$@"
    ;;
  *)
    echo "Usage: $0 {index|norm|export|validate-cpu|validate-production|extract|train} [args]" >&2
    exit 2
    ;;
esac
