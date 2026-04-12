#!/usr/bin/env bash
# Play each Space Invaders training variant (human agent, graphics on).
# From repo root, compile first if needed:
#   mkdir -p out && find src -name "*.java" > /tmp/gvgai_sources.txt && \
#   javac --release 8 -encoding UTF-8 -d out -cp gson-2.6.2.jar @/tmp/gvgai_sources.txt

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

JAVA_BIN="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}/bin/java"
if [[ ! -x "$JAVA_BIN" ]]; then
  JAVA_BIN="$(command -v java)"
fi

if [[ ! -f "$ROOT/out/core/competition/GVGExecutor.class" ]]; then
  echo "Compiled classes not found under out/. Compile the project first (see comments at top of this script)." >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
LEVEL="examples/gridphysics/aliens_lvl0.txt"
HUMAN="tracks.singlePlayer.tools.human.Agent"

for G in aliens_train_physics_a aliens_train_physics_b aliens_train_physics_c; do
  echo ""
  echo "=== Starting: examples/gridphysics/${G}.txt (close the window when done) ==="
  "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
    -g "examples/gridphysics/${G}.txt" \
    -l "$LEVEL" \
    -ag "$HUMAN" \
    -vis 1
done

echo "All three variants finished."
