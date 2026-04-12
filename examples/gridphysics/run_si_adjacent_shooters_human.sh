#!/usr/bin/env bash
# Space Invaders–adjacent grid games (NOT the aliens.txt family): run each with human + graphics.
# Close each window to open the next.
#   ikaruga   — polarity shooter, portals + bombers (closest structure to SI among non-aliens games)
#   waves     — ship vs waves, shields / asteroids, timeout survival
#   eggomania — vertical shoot vs dropping threats (chickens / eggs)
#   missilecommand — protect cities from incoming (Missile Command style)

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
  echo "No usable java found. Set JAVA_HOME or install OpenJDK." >&2
  exit 1
fi

if [[ ! -f "$ROOT/out/core/competition/GVGExecutor.class" ]]; then
  echo "Compiled classes not found under out/. Compile the project first." >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
HUMAN="tracks.singlePlayer.tools.human.Agent"

# game_basename level_basename (level file = examples/gridphysics/${level}.txt)
GAMES=(
  "ikaruga:ikaruga_lvl0"
  "waves:waves_lvl0"
  "eggomania:eggomania_lvl0"
  "missilecommand:missilecommand_lvl0"
)

for pair in "${GAMES[@]}"; do
  G="${pair%%:*}"
  L="${pair##*:}"
  echo ""
  echo "=== ${G} (${L}) — close the window when done ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${G}.txt" \
    -l "examples/gridphysics/${L}.txt" \
    -ag "$HUMAN" \
    -vis 1
done

echo "All SI-adjacent shooters finished."
