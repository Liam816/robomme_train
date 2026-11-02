#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$PROJECT/.venv/bin/python" "$PROJECT/scripts/run_training_profile.py" \
  --config "$PROJECT/configs/train/robomme_framesamp_modul_repro.yaml" "$@"
