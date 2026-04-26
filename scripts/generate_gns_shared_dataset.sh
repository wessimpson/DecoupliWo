#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
OUTPUT="${OUTPUT:-data/transitions/gns_shared/pong_breakout_seed0}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-5000}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-900}"
RARE_SAMPLES_PER_SOURCE="${RARE_SAMPLES_PER_SOURCE:-20000}"
RARE_ROLLOUT_STEPS="${RARE_ROLLOUT_STEPS:-12}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"

"$PYTHON" data/collect_editable_world_transitions.py \
  --output "$OUTPUT" \
  --games pong breakout \
  --modes normal gravity teleport \
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

echo "Generated GNS shared dataset at $OUTPUT"
