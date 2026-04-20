#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
DATASET="${DATASET:-data/transitions/debug/pong_normal_large_seed0}"
OUTPUT="${OUTPUT:-runs/pong_normal_large_slot_gnn_scaled_seed0}"
MODEL_SIZE="${MODEL_SIZE:-large}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.02}"
MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-0.1}"
EVENT_WEIGHT="${EVENT_WEIGHT:-3.0}"
TRAIN_COMBOS="${TRAIN_COMBOS:-pong:normal}"
HOLDOUT_COMBOS="${HOLDOUT_COMBOS:-}"
SEED="${SEED:-0}"

cmd=(
  "$PYTHON" train_pong_world_model.py
  --dataset "$DATASET"
  --output "$OUTPUT"
  --model-size "$MODEL_SIZE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --contrastive-weight "$CONTRASTIVE_WEIGHT"
  --mask-loss-weight "$MASK_LOSS_WEIGHT"
  --event-weight "$EVENT_WEIGHT"
  --device "$DEVICE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ -n "$TRAIN_COMBOS" ]]; then
  read -r -a combo_args <<< "$TRAIN_COMBOS"
  cmd+=(--train-combos "${combo_args[@]}")
fi

if [[ -n "$HOLDOUT_COMBOS" ]]; then
  read -r -a holdout_args <<< "$HOLDOUT_COMBOS"
  cmd+=(--holdout-combos "${holdout_args[@]}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "Scaled world model training complete."
echo "Output: $OUTPUT"
