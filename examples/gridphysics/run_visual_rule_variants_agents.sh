#!/usr/bin/env bash
# Run all visual shooting-game variants with MCTS and random agents.
# From repo root: requires out/ compiled and gson-2.6.2.jar

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

JAVA_BIN=""
for candidate in \
  "${JAVA_HOME:+$JAVA_HOME/bin/java}" \
  "/opt/homebrew/opt/openjdk@21/bin/java" \
  "/opt/homebrew/opt/openjdk@17/bin/java" \
  "$(command -v java 2>/dev/null)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    JAVA_BIN="$candidate"
    break
  fi
done

if [[ -z "$JAVA_BIN" ]]; then
  echo "No usable java found." >&2
  exit 1
fi

if [[ ! -f "$ROOT/out/core/competition/GVGExecutor.class" ]]; then
  echo "Compiled classes not found under out/. Compile the project first." >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
MCTS_AGENT="tracks.singlePlayer.advanced.sampleMCTS.Agent"
RANDOM_AGENT="tracks.singlePlayer.simple.sampleRandom.Agent"
REPETITIONS="${REPETITIONS:-1}"
VISUALS="${VISUALS:-0}"

VARIANTS=(
  mark_enemy
  two_stage_enemy
  reward_flash
  enemy_warning
  shielded_enemy
  target_rank
  impact_outline
  paint_cycle
  danger_tint
  victory_glow
)

run_one() {
  local agent_label="$1"
  local agent_class="$2"
  local game_base="$3"
  local level_base="$4"
  echo "=== ${agent_label}: ${game_base}.txt + ${level_base}.txt ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${game_base}.txt" \
    -l "examples/gridphysics/${level_base}.txt" \
    -ag "$agent_class" \
    -vis "$VISUALS" \
    -rep "$REPETITIONS"
}

for variant in "${VARIANTS[@]}"; do
  run_one "MCTS" "$MCTS_AGENT" "aliens_rules_visual_${variant}" aliens_lvl0
  run_one "Random" "$RANDOM_AGENT" "aliens_rules_visual_${variant}" aliens_lvl0
  run_one "MCTS" "$MCTS_AGENT" "waves_rules_visual_${variant}" waves_lvl0
  run_one "Random" "$RANDOM_AGENT" "waves_rules_visual_${variant}" waves_lvl0
  run_one "MCTS" "$MCTS_AGENT" "chopper_rules_visual_${variant}" chopper_lvl0
  run_one "Random" "$RANDOM_AGENT" "chopper_rules_visual_${variant}" chopper_lvl0
done

echo "All visual variant runs finished."
