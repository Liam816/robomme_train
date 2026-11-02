#!/usr/bin/env bash
# Use the reviewed profile and a single formal training pass.
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PROJECT/scripts/train_framesamp_modul_repro.sh" --execute "$@"
