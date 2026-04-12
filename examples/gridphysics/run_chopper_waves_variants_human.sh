#!/usr/bin/env bash
# Human-play chopper and waves rule/physics variants (same families as aliens variants).
# Close each window to continue. Requires compiled out/ and gson-2.6.2.jar at repo root.

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
HUMAN="tracks.singlePlayer.tools.human.Agent"

run_game() {
  local G="$1"
  local L="$2"
  echo ""
  echo "=== ${G} + ${L} (close window when done) ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${G}.txt" \
    -l "examples/gridphysics/${L}.txt" \
    -ag "$HUMAN" \
    -vis 1
}

run_game chopper_rules_ricochet chopper_lvl0
run_game chopper_rules_shots_pass_clouds_satellite chopper_lvl0
run_game chopper_train_physics_a chopper_lvl0
run_game chopper_train_physics_b chopper_lvl0
run_game chopper_train_physics_c chopper_lvl0

run_game waves_rules_ricochet waves_lvl0
run_game waves_rules_shots_pass_obstacles waves_lvl0
run_game waves_train_physics_a waves_lvl0
run_game waves_train_physics_b waves_lvl0
run_game waves_train_physics_c waves_lvl0

echo "Chopper and waves variant runs finished."
