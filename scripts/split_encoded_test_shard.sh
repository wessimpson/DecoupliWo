#!/bin/bash
set -euo pipefail

# Copy or move one shard from every encoded train env directory into encoded/test.
#
# Default source layout:
#   data/transitions/encoded/train/<env>/shard_00100/...
#
# Default destination layout:
#   data/transitions/encoded/test/<env>/shard_00100/...
#
# Examples:
#   bash scripts/split_encoded_test_shard.sh
#   SHARD_NAME=shard_00100 MODE=move bash scripts/split_encoded_test_shard.sh
#   TRANSITIONS_ROOT=/scratch/sjb8193/DecoupliWo/data/transitions bash scripts/split_encoded_test_shard.sh

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
TRANSITIONS_ROOT="${TRANSITIONS_ROOT:-$REPO_ROOT/data/transitions}"
ENCODED_SUBDIR="${ENCODED_SUBDIR:-encoded}"
SHARD_NAME="${SHARD_NAME:-shard_00100}"
MODE="${MODE:-copy}"  # copy or move

train_root="$TRANSITIONS_ROOT/$ENCODED_SUBDIR/train"
test_root="$TRANSITIONS_ROOT/$ENCODED_SUBDIR/test"

if [[ ! -d "$train_root" ]]; then
	echo "Missing train root: $train_root" >&2
	exit 2
fi

case "$MODE" in
	copy|move) ;;
	*)
		echo "MODE must be copy or move, got: $MODE" >&2
		exit 2
		;;
esac

mkdir -p "$test_root"

count=0
missing=0
skipped=0

for env_dir in "$train_root"/*; do
	[[ -d "$env_dir" ]] || continue
	env_name="$(basename "$env_dir")"
	src="$env_dir/$SHARD_NAME"
	dst="$test_root/$env_name/$SHARD_NAME"

	if [[ ! -d "$src" ]]; then
		echo "missing: $src"
		missing=$((missing + 1))
		continue
	fi
	if [[ -e "$dst" ]]; then
		echo "exists, skip: $dst"
		skipped=$((skipped + 1))
		continue
	fi

	mkdir -p "$(dirname "$dst")"
	if [[ "$MODE" == "move" ]]; then
		mv "$src" "$dst"
		echo "moved: $src -> $dst"
	else
		cp -a "$src" "$dst"
		echo "copied: $src -> $dst"
	fi
	count=$((count + 1))
done

echo "done: $MODE count=$count missing=$missing skipped=$skipped"
