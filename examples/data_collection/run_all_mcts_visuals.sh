#!/usr/bin/env bash
# Watch the built-in MCTS agent play every game in examples/data_collection.
# Graphics on. Each run plays one episode; the next starts when the match ends.

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
AGENT_DEFAULT="tracks.singlePlayer.advanced.sampleMCTS.Agent"
AGENT_STRONG="tracks.singlePlayer.advanced.olets.Agent"
GAME_DIR="examples/data_collection"

shopt -s nullglob
games=("$GAME_DIR"/*.txt)
shopt -u nullglob

if [[ ${#games[@]} -eq 0 ]]; then
  echo "No game .txt files found in $GAME_DIR." >&2
  exit 1
fi

for game_path in "${games[@]}"; do
  game_file="$(basename "$game_path")"
  game_stem="${game_file%.txt}"
  # Skip level map files if they exist in this folder.
  if [[ "$game_stem" == *_lvl* ]]; then
    continue
  fi

  agent="$AGENT_DEFAULT"
  if [[ "$game_stem" == "zelda" ]]; then
    agent="$AGENT_STRONG"
  fi
  echo ""
  echo "=== MCTS playing: ${game_stem} (agent: ${agent}) ==="
  "$JAVA_BIN" -cp "$CP" tracks.singlePlayer.RunDataCollectionAgent \
    --game "$game_stem" \
    --agent "$agent" \
    --visuals
done

echo ""
echo "All data_collection games finished."
