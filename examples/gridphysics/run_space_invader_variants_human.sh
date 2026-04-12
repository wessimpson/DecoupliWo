#!/usr/bin/env bash
# Play Space Invaders (aliens) variants one after another — human agent, graphics on.
# Close each game window to advance to the next variant.
# From repo root, compile first if needed:
#   mkdir -p out && find src -name "*.java" > /tmp/gvgai_sources.txt && \
#   javac --release 8 -encoding UTF-8 -d out -cp gson-2.6.2.jar @/tmp/gvgai_sources.txt

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
  echo "No usable java found. Set JAVA_HOME or install OpenJDK (e.g. brew install openjdk@21)." >&2
  exit 1
fi

if [[ ! -f "$ROOT/out/core/competition/GVGExecutor.class" ]]; then
  echo "Compiled classes not found under out/. Compile the project first (see comments at top of this script)." >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
LEVEL="examples/gridphysics/aliens_lvl0.txt"
HUMAN="tracks.singlePlayer.tools.human.Agent"

VARIANTS=(
  aliens
  aliens_ggame
  aliens_train_physics_a
  aliens_train_physics_b
  aliens_train_physics_c
  aliens_rules_shots_pass_bases
  aliens_rules_ricochet
)

for G in "${VARIANTS[@]}"; do
  echo ""
  echo "=== Starting: examples/gridphysics/${G}.txt (close the window when done) ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${G}.txt" \
    -l "$LEVEL" \
    -ag "$HUMAN" \
    -vis 1
done

echo "All Space Invader variants finished."
