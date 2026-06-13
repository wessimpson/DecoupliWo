#!/usr/bin/env python3
"""Count frames (rows in obs.npy) per env folder under transition shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def npy_frame_count(path: Path) -> int:
	"""Read row count from .npy header only (no mmap of pixel data)."""
	with path.open("rb") as f:
		version = np.lib.format.read_magic(f)
		if version == (1, 0):
			shape, _, _ = np.lib.format.read_array_header_1_0(f)
		elif version == (2, 0):
			shape, _, _ = np.lib.format.read_array_header_2_0(f)
		else:
			raise ValueError(f"unsupported .npy version {version!r} in {path}")
	return int(shape[0])


def count_env_frames(env_dir: Path) -> tuple[int, int]:
	"""Return (frame_count, shard_count) for one env directory."""
	frames = 0
	shards = 0
	for shard in sorted(env_dir.glob("shard_*")):
		obs_path = shard / "obs.npy"
		if not obs_path.is_file():
			continue
		n = npy_frame_count(obs_path)
		if n <= 0:
			continue
		frames += n
		shards += 1
	return frames, shards


def count_split(split_dir: Path) -> list[tuple[str, int, int]]:
	rows: list[tuple[str, int, int]] = []
	for env_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
		frames, shards = count_env_frames(env_dir)
		if shards > 0:
			rows.append((env_dir.name, frames, shards))
	return rows


def main() -> None:
	p = argparse.ArgumentParser(description="Count transition frames per env folder.")
	p.add_argument(
		"--root",
		type=str,
		default=str(Path("data") / "transitions" / "train"),
		help="Split dir (e.g. data/transitions/train) or transitions root with train/test subdirs.",
	)
	p.add_argument("--csv", type=str, default="", help="Optional path to write per-env CSV.")
	args = p.parse_args()
	root = Path(args.root)

	if (root / "train").is_dir() or (root / "test").is_dir():
		splits = [name for name in ("train", "test", "val") if (root / name).is_dir()]
	else:
		splits = [root.name]

	grand_frames = 0
	grand_shards = 0
	grand_envs = 0

	for split in splits:
		split_dir = root if len(splits) == 1 and splits[0] == root.name else root / split
		if not split_dir.is_dir():
			continue
		rows = count_split(split_dir)
		if not rows:
			print(f"{split_dir}: no env folders with shard_*/obs.npy")
			continue

		split_frames = sum(r[1] for r in rows)
		split_shards = sum(r[2] for r in rows)
		grand_frames += split_frames
		grand_shards += split_shards
		grand_envs += len(rows)

		label = split if len(splits) > 1 else str(split_dir)
		print(f"\n=== {label} ({len(rows)} envs, {split_shards} shards, {split_frames:,} frames) ===")
		print(f"{'env':<48} {'shards':>8} {'frames':>12}")
		print("-" * 72)
		for env, frames, shards in rows:
			print(f"{env:<48} {shards:>8} {frames:>12,}")
		print("-" * 72)
		print(f"{'TOTAL':<48} {split_shards:>8} {split_frames:>12,}")

	if len(splits) > 1:
		print(f"\n=== ALL SPLITS ===")
		print(f"envs={grand_envs}  shards={grand_shards:,}  frames={grand_frames:,}")

	if args.csv:
		out = Path(args.csv)
		out.parent.mkdir(parents=True, exist_ok=True)
		with out.open("w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(["split", "env", "shards", "frames"])
			for split in splits:
				split_dir = root if len(splits) == 1 and splits[0] == root.name else root / split
				if not split_dir.is_dir():
					continue
				for env, frames, shards in count_split(split_dir):
					w.writerow([split, env, shards, frames])
		print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
	main()
