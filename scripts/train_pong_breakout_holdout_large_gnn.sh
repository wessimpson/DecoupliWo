#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-/T7/users/soyuj/runs}"

# Large mixed Pong+Breakout dataset. Set SKIP_DATASET=1 if it already exists.
DATASET="${DATASET:-${RUN_ROOT}/data/transitions/editable_world/pong_breakout_large_seed${SEED}}"
SKIP_DATASET="${SKIP_DATASET:-0}"
EPISODES="${EPISODES:-5000}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-900}"
RARE_SAMPLES_PER_SOURCE="${RARE_SAMPLES_PER_SOURCE:-20000}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"

# Compositional holdout:
# Train where each game and each rule are seen somewhere, but two game-rule pairs are held out.
TRAIN_COMBOS="${TRAIN_COMBOS:-pong:normal pong:gravity breakout:normal breakout:teleport}"
HOLDOUT_COMBOS="${HOLDOUT_COMBOS:-pong:teleport breakout:gravity}"

OUTPUT="${OUTPUT:-${RUN_ROOT}/pong_breakout_holdout_large_gnn_seed${SEED}}"
MODEL_SIZE="${MODEL_SIZE:-large}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.01}"
MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-0.1}"
EVENT_WEIGHT="${EVENT_WEIGHT:-4.0}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ "$SKIP_DATASET" != "1" ]]; then
  OUTPUT="$DATASET" \
  SEED="$SEED" \
  GAMES="pong breakout" \
  MODES="normal gravity teleport" \
  POLICIES="random heuristic mixed" \
  EPISODES="$EPISODES" \
  STEPS_PER_EPISODE="$STEPS_PER_EPISODE" \
  RARE_SAMPLES_PER_SOURCE="$RARE_SAMPLES_PER_SOURCE" \
  CHUNK_SIZE="$CHUNK_SIZE" \
  PYTHON="$PYTHON" \
    bash scripts/generate_large_editable_world_dataset.sh
else
  echo "Skipping dataset generation."
  echo "Using existing dataset: $DATASET"
fi

DATASET="$DATASET" \
OUTPUT="$OUTPUT" \
MODEL_SIZE="$MODEL_SIZE" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
LR="$LR" \
WEIGHT_DECAY="$WEIGHT_DECAY" \
CONTRASTIVE_WEIGHT="$CONTRASTIVE_WEIGHT" \
MASK_LOSS_WEIGHT="$MASK_LOSS_WEIGHT" \
EVENT_WEIGHT="$EVENT_WEIGHT" \
DEVICE="$DEVICE" \
NUM_WORKERS="$NUM_WORKERS" \
TRAIN_COMBOS="$TRAIN_COMBOS" \
HOLDOUT_COMBOS="$HOLDOUT_COMBOS" \
SEED="$SEED" \
PYTHON="$PYTHON" \
  bash scripts/train_scaled_world_model.sh

echo "Pong+Breakout holdout GNN run complete."
echo "Dataset: $DATASET"
echo "Output:  $OUTPUT"
echo "Train combos:   $TRAIN_COMBOS"
echo "Holdout combos: $HOLDOUT_COMBOS"
