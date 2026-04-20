#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python}"
OUTPUT="${OUTPUT:-data/transitions/editable_world/pong_all_rules_large_seed0}"
SEED="${SEED:-0}"
POLICIES="${POLICIES:-random heuristic mixed}"
EPISODES="${EPISODES:-10000}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-900}"
RARE_SAMPLES_PER_SOURCE="${RARE_SAMPLES_PER_SOURCE:-50000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"

read -r -a POLICY_ARGS <<< "$POLICIES"

"$PYTHON" data/collect_editable_world_transitions.py \
  --output "$OUTPUT" \
  --games pong \
  --modes normal gravity teleport \
  --policies "${POLICY_ARGS[@]}" \
  --episodes "$EPISODES" \
  --steps-per-episode "$STEPS_PER_EPISODE" \
  --counterfactual \
  --rare-events \
  --rare-samples-per-source "$RARE_SAMPLES_PER_SOURCE" \
  --rare-counterfactual \
  --val-fraction "$VAL_FRACTION" \
  --chunk-size "$CHUNK_SIZE" \
  --seed "$SEED"

"$PYTHON" data/inspect_pong_transitions.py "$OUTPUT" --split train --examples 5

echo "Large Pong all-rules dataset generated."
echo "Output: $OUTPUT"
echo "Rows estimate:"
echo "  rollout ~= EPISODES * STEPS_PER_EPISODE * 3 rules"
echo "  rare    ~= 8 rare_sources * RARE_SAMPLES_PER_SOURCE * 3 rules"
