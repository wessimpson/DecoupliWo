#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
GVGAI_ROOT="$PACKAGE_ROOT/gvgai"
BUILD_DIR="$GVGAI_ROOT/gym_gvgai/envs/gvgai/GVGAI_Build"
CLASS="tracks.singlePlayer.RunDataCollectionAgent"

MODE="train"
SOURCE_ROOT="$GVGAI_ROOT/gym_gvgai/envs/games_world_model"
SPRITE_ROOT="$GVGAI_ROOT/gym_gvgai/envs/gvgai/sprites"
OUTPUT_BASE="/hdd2/soyuj/transition_data"
OUTPUT_ROOT=""
TOTAL_TIMESTEPS=100000
NUM_ENVS=1
SCALE=1.0
CHUNK_SIZE=1000
SEED=""
SKIP_BUILD=0
DRY_RUN=0
RESUME=0
BUDGET_SCOPE="stem"
LEVELS="all"
TRAIN_BASES="auto"
TEST_BASES="defender,jaws,zelda"
TEST_INCLUDE_VARIANTS=0

usage() {
  cat <<'EOF'
Usage: training_data_gvgai/scripts/collect_gvgai_variants.sh [options]

Collect DecoupliWo-compatible GVGAI transition shards from the pulled
games_world_model layout:
  <source-root>/<base>/<stem>.txt
  <source-root>/<base>/lvl0.txt ... lvlN.txt

Options:
  --mode train|test|all             Split/matrix to collect (default: train).
  --source-root DIR                 games_world_model root to discover.
  --sprite-root DIR                 GVGAI sprite image asset root.
  --output-base DIR                 Base output dir (default: /hdd2/soyuj/transition_data).
  --output-root DIR                 Override exact split output root. For --mode all, pass separate runs.
  --total-timesteps N               Frames per game/rule stem before profile weighting (default: 100000).
  --budget-scope stem|level         stem: split budget across levels; level: N frames per level (default: stem).
  --levels all|0,1,2                Levels to collect from each base game (default: all discovered lvl*.txt).
  --num-envs N                      Parallel envs per Java process (default: 1).
  --scale F                         Saved frame scale (default: 1.0, full 120x120 RGB).
  --chunk-size N                    Rows per shard (default: 1000).
  --seed N                          Base RNG seed.
  --train-bases auto|a,b,c          Train base-game dirs. auto means every discovered base dir.
  --test-bases a,b,c                Optional eval/holdout collection dirs (default: defender,jaws,zelda).
  --test-include-variants           Include *_rules_* files in test split too.
  --dry-run                         Print discovered jobs without running Java.
  --resume                          Skip completed stem/level/profile jobs already present in the output folder.
  --skip-build                      Do not run python build.py first.
  -h, --help                        Show this help.

Profiles for train are weighted:
  mcts_exploit=20%, mcts_balanced=40%, mcts_explore=20%, mcts_scout=10%, random=10%

The optional test mode uses mcts_balanced only by default. For the usual
world-model setup here, collect --mode train; unseen variant transfer is an
evaluation protocol, not a separate default data split.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --sprite-root) SPRITE_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --budget-scope) BUDGET_SCOPE="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-bases) TRAIN_BASES="$2"; shift 2 ;;
    --test-bases) TEST_BASES="$2"; shift 2 ;;
    --test-include-variants) TEST_INCLUDE_VARIANTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
SPRITE_ROOT="$(cd "$SPRITE_ROOT" && pwd)"
OUTPUT_BASE="$(mkdir -p "$OUTPUT_BASE" && cd "$OUTPUT_BASE" && pwd)"

case "$MODE" in
  train|test|all) ;;
  *) echo "Unknown mode: $MODE" >&2; usage; exit 1 ;;
esac

case "$BUDGET_SCOPE" in
  stem|level) ;;
  *) echo "Unknown --budget-scope: $BUDGET_SCOPE" >&2; usage; exit 1 ;;
esac

if [[ "$DRY_RUN" -eq 0 && "$SKIP_BUILD" -eq 0 ]]; then
  (cd "$GVGAI_ROOT" && python build.py)
fi

profiles=(mcts_exploit mcts_balanced mcts_explore mcts_scout random)
weights=(20 40 20 10 10)

split_csv() {
  local raw="$1"
  local item
  echo "$raw" | tr ',' '\n' | while IFS= read -r item; do
    item="$(echo "$item" | xargs)"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

contains_item() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

all_base_dirs() {
  find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
}

level_indices_for_base() {
  local base="$1"
  if [[ "$LEVELS" != "all" ]]; then
    echo "$LEVELS" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | sort -n
    return
  fi
  find "$SOURCE_ROOT/$base" -maxdepth 1 -type f -name 'lvl*.txt' -exec basename {} \; \
    | sed -n 's/^lvl\([0-9][0-9]*\)\.txt$/\1/p' \
    | sort -n
}

stems_for_base() {
  local base="$1"
  local include_variants="$2"
  if [[ "$include_variants" -eq 0 ]]; then
    [[ -f "$SOURCE_ROOT/$base/$base.txt" ]] && echo "$base"
    return
  fi
  find "$SOURCE_ROOT/$base" -maxdepth 1 -type f -name '*.txt' ! -name 'lvl*.txt' -exec basename {} \; \
    | sed 's/\.txt$//' \
    | sort
}

rule_tag_for_stem() {
  local base="$1"
  local stem="$2"
  if [[ "$stem" == "$base" ]]; then
    echo "base"
  elif [[ "$stem" == "$base"_rules_* ]]; then
    echo "${stem#${base}_rules_}"
  else
    echo "unknown"
  fi
}

default_output_root() {
  local split="$1"
  echo "$OUTPUT_BASE/$split"
}

stem_frame_count() {
  local env_dir="$1"
  python - "$env_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
frames = 0
if root.is_dir():
    for obs in root.glob("shard_*/obs.npy"):
        try:
            frames += int(np.load(obs, mmap_mode="r").shape[0])
        except Exception:
            pass
print(frames)
PY
}

job_frame_count() {
  local env_dir="$1"
  local profile="$2"
  local level="$3"
  python - "$env_dir" "$profile" "$level" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np

root = Path(sys.argv[1])
want_profile = sys.argv[2]
want_level = str(sys.argv[3])
frames = 0
if root.is_dir():
    for shard in root.glob("shard_*"):
        obs_path = shard / "obs.npy"
        meta_path = shard / "metadata.json"
        if not obs_path.is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("profile", "")) != want_profile:
                continue
            if str(meta.get("level_index", "")) != want_level:
                continue
            frames += int(np.load(obs_path, mmap_mode="r").shape[0])
        except Exception:
            pass
print(frames)
PY
}

stem_is_complete() {
  local split="$1"
  local stem="$2"
  local expected_frames="$3"
  local out_root="${OUTPUT_ROOT:-$(default_output_root "$split")}"
  local frames
  frames="$(stem_frame_count "$out_root/$stem")"
  [[ "$frames" -ge "$expected_frames" ]]
}

job_is_complete() {
  local split="$1"
  local stem="$2"
  local level="$3"
  local profile="$4"
  local expected_frames="$5"
  local out_root="${OUTPUT_ROOT:-$(default_output_root "$split")}"
  local frames
  frames="$(job_frame_count "$out_root/$stem" "$profile" "$level")"
  [[ "$frames" -ge "$expected_frames" ]]
}

frames_for_level() {
  local total="$1"
  local level_pos="$2"
  local level_count="$3"
  if [[ "$BUDGET_SCOPE" == "level" ]]; then
    echo "$total"
    return
  fi
  local base=$(( total / level_count ))
  local rem=$(( total % level_count ))
  if [[ "$level_pos" -lt "$rem" ]]; then
    echo $(( base + 1 ))
  else
    echo "$base"
  fi
}

run_one() {
  local split="$1"
  local base="$2"
  local stem="$3"
  local level="$4"
  local profile="$5"
  local frames="$6"
  local rule_tag
  rule_tag="$(rule_tag_for_stem "$base" "$stem")"
  local out_root="${OUTPUT_ROOT:-$(default_output_root "$split")}"
  local game_file="$SOURCE_ROOT/$base/$stem.txt"
  local level_file="$SOURCE_ROOT/$base/lvl${level}.txt"

  if [[ "$frames" -lt 1 ]]; then
    frames=1
  fi

  if [[ ! -f "$game_file" ]]; then
    echo "Missing game file: $game_file" >&2
    exit 1
  fi
  if [[ ! -f "$level_file" ]]; then
    echo "Missing level file: $level_file" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\t%s\t%s\tlvl%s\t%s\t%s frames\t%s\n' "$split" "$base" "$stem" "$level" "$profile" "$frames" "$out_root"
    return
  fi

  if [[ "$RESUME" -eq 1 ]] && job_is_complete "$split" "$stem" "$level" "$profile" "$frames"; then
    echo "==> $split $stem lvl$level $profile already has >= $frames frames; skipping due to --resume"
    return
  fi

  mkdir -p "$out_root"
  local args=(
    "$CLASS"
    --game "$game_file"
    --level "$level_file"
    --level-index "$level"
    --profile "$profile"
    --split "$split"
    --output-root "$out_root"
    --total-timesteps "$frames"
    --num-envs "$NUM_ENVS"
    --scale "$SCALE"
    --chunk-size "$CHUNK_SIZE"
    --source-root "$SOURCE_ROOT"
    --sprite-root "$SPRITE_ROOT"
    --source-base-game "$base"
    --source-rule-tag "$rule_tag"
  )
  if [[ -n "$SEED" ]]; then
    args+=(--seed "$SEED")
  fi
  echo "==> $split $stem lvl$level $profile $frames frames -> $out_root/$stem"
  (cd "$GVGAI_ROOT" && java -cp "$BUILD_DIR" "${args[@]}")
}

collect_base_train() {
  local base="$1"
  local levels=()
  while IFS= read -r level; do levels+=("$level"); done < <(level_indices_for_base "$base")
  local stems=()
  while IFS= read -r stem; do stems+=("$stem"); done < <(stems_for_base "$base" 1)
  if [[ "${#levels[@]}" -eq 0 || "${#stems[@]}" -eq 0 ]]; then
    echo "Skipping $base: no levels or game files discovered." >&2
    return
  fi
  local stem level_idx level profile_idx weighted_total frames expected_frames
  expected_frames="$TOTAL_TIMESTEPS"
  if [[ "$BUDGET_SCOPE" == "level" ]]; then
    expected_frames=$(( TOTAL_TIMESTEPS * ${#levels[@]} ))
  fi
  for stem in "${stems[@]}"; do
    for profile_idx in "${!profiles[@]}"; do
      weighted_total=$(( TOTAL_TIMESTEPS * weights[profile_idx] / 100 ))
      for level_idx in "${!levels[@]}"; do
        level="${levels[$level_idx]}"
        frames="$(frames_for_level "$weighted_total" "$level_idx" "${#levels[@]}")"
        run_one train "$base" "$stem" "$level" "${profiles[$profile_idx]}" "$frames"
      done
    done
  done
}

collect_base_test() {
  local base="$1"
  local levels=()
  while IFS= read -r level; do levels+=("$level"); done < <(level_indices_for_base "$base")
  local stems=()
  while IFS= read -r stem; do stems+=("$stem"); done < <(stems_for_base "$base" "$TEST_INCLUDE_VARIANTS")
  if [[ "${#levels[@]}" -eq 0 || "${#stems[@]}" -eq 0 ]]; then
    echo "Skipping $base: no levels or game files discovered." >&2
    return
  fi
  local stem level_idx level frames expected_frames
  expected_frames="$TOTAL_TIMESTEPS"
  if [[ "$BUDGET_SCOPE" == "level" ]]; then
    expected_frames=$(( TOTAL_TIMESTEPS * ${#levels[@]} ))
  fi
  for stem in "${stems[@]}"; do
    for level_idx in "${!levels[@]}"; do
      level="${levels[$level_idx]}"
      frames="$(frames_for_level "$TOTAL_TIMESTEPS" "$level_idx" "${#levels[@]}")"
      run_one test "$base" "$stem" "$level" mcts_balanced "$frames"
    done
  done
}

collect_train() {
  local train_bases=()
  local base
  if [[ "$TRAIN_BASES" == "auto" ]]; then
    while IFS= read -r base; do
      train_bases+=("$base")
    done < <(all_base_dirs)
  else
    while IFS= read -r base; do
      train_bases+=("$base")
    done < <(split_csv "$TRAIN_BASES")
  fi

  echo "Train bases: ${train_bases[*]}"
  echo "Source root: $SOURCE_ROOT"
  echo "Sprite root: $SPRITE_ROOT"
  echo "Output root: ${OUTPUT_ROOT:-$(default_output_root train)}"
  echo "Budget: $TOTAL_TIMESTEPS frames per stem, scope=$BUDGET_SCOPE"
  for base in "${train_bases[@]}"; do
    collect_base_train "$base"
  done
}

collect_test() {
  local test_bases=()
  local base
  while IFS= read -r base; do
    test_bases+=("$base")
  done < <(split_csv "$TEST_BASES")
  echo "Test bases: ${test_bases[*]}"
  echo "Source root: $SOURCE_ROOT"
  echo "Sprite root: $SPRITE_ROOT"
  echo "Output root: ${OUTPUT_ROOT:-$(default_output_root test)}"
  echo "Budget: $TOTAL_TIMESTEPS frames per stem, scope=$BUDGET_SCOPE"
  for base in "${test_bases[@]}"; do
    collect_base_test "$base"
  done
}

case "$MODE" in
  train) collect_train ;;
  test) collect_test ;;
  all) collect_train; collect_test ;;
esac
