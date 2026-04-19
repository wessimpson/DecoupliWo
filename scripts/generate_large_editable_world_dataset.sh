#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
OUTPUT="${OUTPUT:-data/transitions/editable_world/pong_breakout_large_seed0}"
SEED="${SEED:-0}"
GAMES="${GAMES:-pong breakout}"
MODES="${MODES:-normal gravity teleport}"
POLICIES="${POLICIES:-random heuristic mixed}"
EPISODES="${EPISODES:-5000}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-900}"
RARE_SAMPLES_PER_SOURCE="${RARE_SAMPLES_PER_SOURCE:-20000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
COUNTERFACTUAL="${COUNTERFACTUAL:-1}"

read -r -a GAME_ARGS <<< "$GAMES"
read -r -a MODE_ARGS <<< "$MODES"
read -r -a POLICY_ARGS <<< "$POLICIES"

COUNTERFACTUAL_ARG="--counterfactual"
if [[ "$COUNTERFACTUAL" == "0" || "$COUNTERFACTUAL" == "false" || "$COUNTERFACTUAL" == "False" ]]; then
  COUNTERFACTUAL_ARG="--no-counterfactual"
fi

"$PYTHON" data/collect_editable_world_transitions.py \
  --output "$OUTPUT" \
  --games "${GAME_ARGS[@]}" \
  --modes "${MODE_ARGS[@]}" \
  --policies "${POLICY_ARGS[@]}" \
  --episodes "$EPISODES" \
  --steps-per-episode "$STEPS_PER_EPISODE" \
  "$COUNTERFACTUAL_ARG" \
  --rare-events \
  --rare-samples-per-source "$RARE_SAMPLES_PER_SOURCE" \
  --rare-counterfactual \
  --val-fraction "$VAL_FRACTION" \
  --chunk-size "$CHUNK_SIZE" \
  --seed "$SEED"

"$PYTHON" data/inspect_pong_transitions.py "$OUTPUT" --split train --examples 5

echo "Large editable-world dataset generated."
echo "Output: $OUTPUT"
