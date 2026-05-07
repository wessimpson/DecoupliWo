#!/usr/bin/env bash
# Run only Ghostbusters in examples/gameDesign with the built-in MCTS agent.

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
  echo "Compile first (or run a compile command once)." >&2
  exit 1
fi

if [[ ! -f "$ROOT/gson-2.6.2.jar" ]]; then
  echo "Missing gson jar at $ROOT/gson-2.6.2.jar" >&2
  exit 1
fi

LEVEL="${1:-0}"
GAME_PATH="examples/gameDesign/ghostbusters.txt"
LEVEL_PATH="examples/gameDesign/ghostbusters_lvl${LEVEL}.txt"
AGENT="tracks.singlePlayer.advanced.sampleMCTS.Agent"
CP="out:gson-2.6.2.jar"

if [[ ! -f "$GAME_PATH" ]]; then
  echo "Missing game file: $GAME_PATH" >&2
  exit 1
fi

if [[ ! -f "$LEVEL_PATH" ]]; then
  echo "Missing level file: $LEVEL_PATH" >&2
  echo "Use level index 0-4, e.g. ./run_mcts_ghostbusters.sh 4" >&2
  exit 1
fi

echo "=== MCTS playing Ghostbusters (level $LEVEL) ==="
"$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
  -g "$GAME_PATH" \
  -l "$LEVEL_PATH" \
  -ag "$AGENT" \
  -vis 1
