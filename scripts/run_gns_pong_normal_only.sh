#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
SEED="${SEED:-0}"
DATASET="${DATASET:-data/transitions/gns_shared/pong_normal_seed${SEED}}"
OUTPUT="${OUTPUT:-runs/gns_pong_normal_seed${SEED}}"
EPISODES="${EPISODES:-5000}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-900}"
RARE_SAMPLES_PER_SOURCE="${RARE_SAMPLES_PER_SOURCE:-20000}"
RARE_ROLLOUT_STEPS="${RARE_ROLLOUT_STEPS:-12}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
MODEL_SIZE="${MODEL_SIZE:-large}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"

"$PYTHON" data/collect_editable_world_transitions.py \
  --output "$DATASET" \
  --games pong \
  --modes normal \
  --policies random heuristic mixed \
  --episodes "$EPISODES" \
  --steps-per-episode "$STEPS_PER_EPISODE" \
  --counterfactual \
  --rare-events \
  --rare-samples-per-source "$RARE_SAMPLES_PER_SOURCE" \
  --rare-rollout-steps "$RARE_ROLLOUT_STEPS" \
  --rare-counterfactual \
  --val-fraction "$VAL_FRACTION" \
  --chunk-size "$CHUNK_SIZE" \
  --seed "$SEED"

"$PYTHON" train_gns_shared_simulator.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --model-size "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --train-combos pong:normal

echo "Generated dataset: $DATASET"
echo "Trained model:     $OUTPUT"
