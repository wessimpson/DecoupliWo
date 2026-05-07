#!/usr/bin/env bash
# Run one data_collection rule/game on macOS with MCTS.
# Example:
#   bash examples/data_collection/run_mcts_rule_mac.sh --rule aliens_rules_two_hit_color --visuals

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RULE=""
LEVEL=""
VISUALS=0
SEED=""
SCALE="0.5"
AGENT="tracks.singlePlayer.advanced.sampleMCTS.Agent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rule)
      RULE="${2:-}"
      shift 2
      ;;
    --level)
      LEVEL="${2:-}"
      shift 2
      ;;
    --agent)
      AGENT="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --scale)
      SCALE="${2:-}"
      shift 2
      ;;
    --visuals)
      VISUALS=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --rule <rule_stem_or_path> [--level <level_path>] [--agent <fqcn>] [--seed <int>] [--scale <float>] [--visuals]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$RULE" ]]; then
  echo "Missing required --rule argument." >&2
  exit 1
fi

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

if [[ ! -f "gson-2.6.2.jar" ]]; then
  echo "Missing gson-2.6.2.jar in repo root." >&2
  exit 1
fi

mkdir -p out
if [[ ! -f "out/tracks/singlePlayer/RunDataCollectionAgent.class" ]]; then
  echo "Compiling Java sources..."
  rg --files src -g "*.java" > sources_build.txt
  javac --release 8 -encoding UTF-8 -d out -cp gson-2.6.2.jar @sources_build.txt
fi

rule_arg="$RULE"
if [[ "$RULE" != *"/"* && "$RULE" != *.txt ]]; then
  rule_arg="examples/data_collection/${RULE}.txt"
elif [[ "$RULE" != *"/"* && "$RULE" == *.txt ]]; then
  rule_arg="examples/data_collection/${RULE}"
fi

cmd=( "$JAVA_BIN" -cp "out:gson-2.6.2.jar" tracks.singlePlayer.RunDataCollectionAgent --game "$rule_arg" --agent "$AGENT" --scale "$SCALE" )
if [[ -n "$LEVEL" ]]; then
  cmd+=( --level "$LEVEL" )
fi
if [[ -n "$SEED" ]]; then
  cmd+=( --seed "$SEED" )
fi
if [[ "$VISUALS" -eq 1 ]]; then
  cmd+=( --visuals )
else
  cmd+=( --no-visuals )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
