#!/usr/bin/env bash
# Re-run Phase 1 on the hand dataset, then open the labelled 3D viewer.
#
# Usage:
#   bash scripts/run_hand.sh          # pipeline + interactive viewer
#   bash scripts/run_hand.sh save     # pipeline + off-screen PNG only
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ulimit -v 8000000 || true
python3 -m dataset_foundation.pipeline --config config/hand.yaml
if [ "${1:-}" = "save" ]; then
  exec python3 scripts/visualize.py --config config/hand.yaml --save
else
  exec python3 scripts/visualize.py --config config/hand.yaml --interactive
fi
