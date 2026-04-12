#!/usr/bin/env bash
# Second batch: arcade-style neighbors of Space Invaders (NOT aliens.txt family).
# Close each window to continue.
#   solarfox    — Solar Fox–style: ship in a grid, collect blibs, dodge shots
#   deflection  — open space, satellites fire bombs, planets, black holes
#   chopper     — helicopter vs tanks / clouds (shoot up & down, portals)
#   seaquest    — submarine torpedoes vs fish (underwater Atari-style)

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

GAMES=(
  "solarfox:solarfox_lvl0"
  "deflection:deflection_lvl0"
  "chopper:chopper_lvl0"
  "seaquest:seaquest_lvl0"
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

echo "Batch 2 finished."
