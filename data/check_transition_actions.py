"""Scan transition shards and print which action ids appear (assault + space_invaders)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# Ale_py / Gymnasium Atari v5 minimal action set (same order as common wrappers)
ATARI_NAMES = (
	"NOOP",
	"FIRE",
	"UP",
	"RIGHT",
	"LEFT",
	"DOWN",
	"UPRIGHT",
	"UPLEFT",
	"DOWNRIGHT",
	"DOWNLEFT",
	"UPFIRE",
	"RIGHTFIRE",
	"LEFTFIRE",
	"DOWNFIRE",
	"UPRIGHTFIRE",
	"UPLEFTFIRE",
	"DOWNRIGHTFIRE",
	"DOWNLEFTFIRE",
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument(
		"--root",
		type=Path,
		default=Path(__file__).resolve().parent / "transitions",
		help="Directory containing train/ and test/ splits",
	)
	p.add_argument(
		"--envs",
		nargs="*",
		default=["assault", "space_invaders"],
		help="Game folder names under each split",
	)
	p.add_argument("--splits", nargs="*", default=["train", "test"])
	return p.parse_args()


def scan_env(split_dir: Path, env_name: str) -> tuple[int, Counter[int], list[Path]]:
	env_root = split_dir / env_name
	if not env_root.is_dir():
		return 0, Counter(), []
	shards = sorted(p for p in env_root.glob("shard_*") if (p / "action.npy").exists())
	counter: Counter[int] = Counter()
	total = 0
	for shard in shards:
		a = np.load(shard / "action.npy", mmap_mode="r")
		a = np.asarray(a).ravel().astype(np.int64, copy=False)
		total += a.size
		u, c = np.unique(a, return_counts=True)
		for ui, ci in zip(u.tolist(), c.tolist()):
			counter[int(ui)] += int(ci)
	return total, counter, shards


def main() -> None:
	args = parse_args()
	root: Path = args.root
	if not root.is_dir():
		print(f"No transitions root: {root}")
		return

	for split in args.splits:
		split_dir = root / split
		if not split_dir.is_dir():
			print(f"[{split}] (missing)")
			continue
		for env in args.envs:
			total, ctr, shards = scan_env(split_dir, env)
			if not shards:
				print(f"[{split}/{env}] no shards")
				continue
			ids = sorted(ctr)
			print(f"[{split}/{env}] shards={len(shards)}  action_rows={total}")
			print(f"  unique_ids ({len(ids)}): {ids}")
			for aid in ids:
				name = ATARI_NAMES[aid] if 0 <= aid < len(ATARI_NAMES) else "?"
				pct = 100.0 * ctr[aid] / total
				print(f"    {aid:3d}  {name:16s}  n={ctr[aid]:>10,}  ({pct:5.2f}%)")
			nact = shards[0] / "n_actions.npy"
			if nact.exists():
				print(f"  n_actions.npy (first shard): {int(np.load(nact))}")
		print()


if __name__ == "__main__":
	main()
