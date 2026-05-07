#!/usr/bin/env bash
# Watch MCTS agent play only zelda in examples/data_collection.
# Graphics on. Runs one visual episode.

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

if [[ ! -f "$ROOT/out/tracks/singlePlayer/RunDataCollectionAgent.class" ]]; then
  echo "Compiled classes not found under out/." >&2
  echo "Compile first (or run a *.ps1 data-collection script once)." >&2
  exit 1
fi

if [[ ! -f "$ROOT/gson-2.6.2.jar" ]]; then
  echo "Missing gson jar at $ROOT/gson-2.6.2.jar" >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
GAME="zelda"
AGENT="tracks.singlePlayer.advanced.olets.Agent"

echo ""
echo "=== MCTS playing: ${GAME} (agent: ${AGENT}) ==="
"$JAVA_BIN" -cp "$CP" tracks.singlePlayer.RunDataCollectionAgent \
  --game "$GAME" \
  --agent "$AGENT" \
  --visuals

echo ""
echo "Zelda visual run finished."
