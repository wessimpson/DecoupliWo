#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-pong_normal_smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-/tmp/decoupliwo_${RUN_ID}_data}"
RUN_DIR="${RUN_DIR:-/tmp/decoupliwo_${RUN_ID}_model}"
EPISODES="${EPISODES:-20}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-80}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cpu}"

"$PYTHON" data/collect_editable_world_transitions.py \
  --output "$DATASET" \
  --games pong \
  --modes normal \
  --policy heuristic \
  --episodes "$EPISODES" \
  --steps-per-episode "$STEPS_PER_EPISODE" \
  --no-counterfactual \
  --val-fraction 0.2 \
  --seed "$SEED"

"$PYTHON" data/inspect_pong_transitions.py "$DATASET" --split train --examples 3

"$PYTHON" train_pong_world_model.py \
  --dataset "$DATASET" \
  --output "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --train-combos pong:normal \
  --device "$DEVICE"

"$PYTHON" eval_pong_world_model.py \
  --checkpoint "$RUN_DIR/latest.pt" \
  --dataset "$DATASET" \
  --output "$RUN_DIR/eval.json" \
  --eval-modes normal \
  --rollout-episodes 2 \
  --rollout-horizon 20 \
  --device "$DEVICE"

"$PYTHON" play_pong_world_model.py \
  --checkpoint "$RUN_DIR/latest.pt" \
  --mode normal \
  --device "$DEVICE" \
  --headless-steps 5

echo "Pong-normal smoke completed."
echo "Dataset: $DATASET"
echo "Run dir: $RUN_DIR"
