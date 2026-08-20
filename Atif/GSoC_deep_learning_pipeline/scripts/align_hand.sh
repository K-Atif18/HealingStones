#!/usr/bin/env bash
# Manual 3-point alignment for the hand dataset (config/hand.yaml).
#
# Usage:
#   bash scripts/align_hand.sh 2 6 7 8 9   # align only these part numbers
#   bash scripts/align_hand.sh             # align all parts
#
# Opens Open3D point-picking windows (needs a display). For each part:
#   1) fragment window: shift+click >=3 landmarks, press Q
#   2) model window:    click the SAME points in the SAME order, press Q
#   3) preview window:  close to accept
# Captured transforms are written into config/hand.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
if [ "$#" -gt 0 ]; then
  PARTS=$(IFS=,; echo "$*")
  exec python3 scripts/manual_align.py --config config/hand.yaml --parts "$PARTS"
else
  exec python3 scripts/manual_align.py --config config/hand.yaml
fi
