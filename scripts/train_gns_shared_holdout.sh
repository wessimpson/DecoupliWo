#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
DATASET="${DATASET:-data/transitions/gns_shared/pong_breakout_seed0}"
OUTPUT="${OUTPUT:-runs/gns_shared_holdout_seed0}"
SEED="${SEED:-0}"
MODEL_SIZE="${MODEL_SIZE:-large}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"

"$PYTHON" train_gns_shared_simulator.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --model-size "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --train-combos pong:normal pong:gravity breakout:normal breakout:teleport \
  --holdout-combos pong:teleport breakout:gravity

echo "Trained GNS shared simulator at $OUTPUT"
