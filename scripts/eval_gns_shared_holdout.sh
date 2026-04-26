#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
CHECKPOINT="${CHECKPOINT:-runs/gns_shared_holdout_seed0/best.pt}"
DATASET="${DATASET:-data/transitions/gns_shared/pong_breakout_seed0}"
OUTPUT="${OUTPUT:-runs/gns_shared_holdout_seed0/eval.json}"
DEVICE="${DEVICE:-auto}"

"$PYTHON" eval_gns_shared_simulator.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --holdout-combos pong:teleport breakout:gravity \
  --output "$OUTPUT" \
  --device "$DEVICE"

echo "Saved evaluation to $OUTPUT"
