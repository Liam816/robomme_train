#!/usr/bin/env bash
# Run from any directory after installing the independent project environment.
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET_ROOT=${DATASET_ROOT:-/mnt/cpfs/heping/data/public/robomme}
CACHE_ROOT=${CACHE_ROOT:-/mnt/cpfs/heping/data/public/robomme_framesamp_cache}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/mnt/cpfs/heping/checkpoints}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PROJECT/src:$PROJECT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$PROJECT/.venv/bin/python"
LOG_ROOT="$PROJECT/runs/framesamp_preparation"
mkdir -p "$LOG_ROOT"
cd "$PROJECT"
"$PYTHON" scripts/build_lerobot_framesamp.py --source "$DATASET_ROOT" --cache "$CACHE_ROOT" --openpi-data-home "$OPENPI_DATA_HOME" --index-only
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3}"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true; wait || true; exit 130' INT TERM
for shard in "${!GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$PYTHON" -u scripts/build_lerobot_framesamp.py \
    --source "$DATASET_ROOT" --cache "$CACHE_ROOT" --openpi-data-home "$OPENPI_DATA_HOME" \
    --num-shards "${#GPU_IDS[@]}" --shard-id "$shard" --batch-size 32 \
    > "$LOG_ROOT/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$LOG_ROOT/shard_pids.txt"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  printf 'Feature generation failed. See %s/shard_*.log\n' "$LOG_ROOT" >&2
  exit "$status"
fi
"$PYTHON" scripts/build_lerobot_framesamp.py --source "$DATASET_ROOT" --cache "$CACHE_ROOT" --openpi-data-home "$OPENPI_DATA_HOME" --verify-only
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$PYTHON" scripts/check_lerobot_framesamp.py --cache "$CACHE_ROOT" --output "$LOG_ROOT/data_check.json"
date --iso-8601=seconds > "$LOG_ROOT/COMPLETE"
