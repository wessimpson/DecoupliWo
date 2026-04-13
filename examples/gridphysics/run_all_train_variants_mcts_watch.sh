#!/usr/bin/env bash
# Watch the built-in MCTS agent play every aliens / chopper / waves training & rules variant.
# Graphics on. Each run plays one episode; the next starts when the match ends.
# From repo root: needs out/ compiled and gson-2.6.2.jar

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
  echo "Compiled classes not found under out/." >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
AGENT="tracks.singlePlayer.advanced.sampleMCTS.Agent"

run_one() {
  local game_base="$1"
  local level_base="$2"
  echo ""
  echo "=== MCTS playing: ${game_base}.txt  (level ${level_base}.txt) ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${game_base}.txt" \
    -l "examples/gridphysics/${level_base}.txt" \
    -ag "$AGENT" \
    -vis 1
}

# Aliens family
run_one aliens_train_physics_a aliens_lvl0
run_one aliens_train_physics_b aliens_lvl0
run_one aliens_train_physics_c aliens_lvl0
run_one aliens_rules_shots_pass_bases aliens_lvl0
run_one aliens_rules_ricochet aliens_lvl0

# Chopper family
run_one chopper_rules_ricochet chopper_lvl0
run_one chopper_rules_shots_pass_clouds_satellite chopper_lvl0
run_one chopper_train_physics_a chopper_lvl0
run_one chopper_train_physics_b chopper_lvl0
run_one chopper_train_physics_c chopper_lvl0

# Waves family
run_one waves_rules_ricochet waves_lvl0
run_one waves_rules_shots_pass_obstacles waves_lvl0
run_one waves_train_physics_a waves_lvl0
run_one waves_train_physics_b waves_lvl0
run_one waves_train_physics_c waves_lvl0

echo "All variant runs finished."
