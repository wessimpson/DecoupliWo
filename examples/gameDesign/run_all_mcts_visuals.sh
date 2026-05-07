#!/usr/bin/env bash
# Watch the built-in MCTS agent play every base game in examples/gameDesign.
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

if [[ ! -f "$ROOT/out/core/competition/GVGExecutor.class" ]]; then
  echo "Compiled classes not found under out/." >&2
  echo "Compile first (or run a compile command once)." >&2
  exit 1
fi

if [[ ! -f "$ROOT/gson-2.6.2.jar" ]]; then
  echo "Missing gson jar at $ROOT/gson-2.6.2.jar" >&2
  exit 1
fi

CP="out:gson-2.6.2.jar"
AGENT="tracks.singlePlayer.advanced.sampleMCTS.Agent"
GAME_DIR="examples/gameDesign"

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

  # Skip level files and other non-base variants.
  if [[ "$game_stem" == *_lvl* ]]; then
    continue
  fi

  level_path="$GAME_DIR/${game_stem}_lvl0.txt"
  echo ""
  if [[ -f "$level_path" ]]; then
    echo "=== MCTS playing: ${game_stem} (level ${game_stem}_lvl0) ==="
    "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
      -g "$game_path" \
      -l "$level_path" \
      -ag "$AGENT" \
      -vis 1
  else
    echo "=== MCTS playing: ${game_stem} (no _lvl0 found) ==="
    "$JAVA_BIN" -cp "$CP" core.competition.GVGExecutor \
      -g "$game_path" \
      -ag "$AGENT" \
      -vis 1
  fi
done

echo ""
echo "All gameDesign games finished."
