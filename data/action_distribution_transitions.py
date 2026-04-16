"""Print per-shard and pooled action counts + probabilities for transition shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _shard_dirs(env_dir: Path) -> list[Path]:
	dirs = sorted(
		p for p in env_dir.glob("shard_*")
		if p.is_dir() and (p / "action.npy").is_file()
	)
	return dirs


def _resolve_shard(env_dir: Path, shard_arg: str) -> Path:
	s = shard_arg.strip()
	if s.startswith("shard_"):
		p = env_dir / s
	else:
		try:
			idx = int(s, 10)
		except ValueError as e:
			raise SystemExit(f"Invalid --shard {shard_arg!r}: use e.g. shard_00012 or 12") from e
		p = env_dir / f"shard_{idx:05d}"
	if not p.is_dir():
		raise SystemExit(f"Shard directory not found: {p}")
	if not (p / "action.npy").is_file():
		raise SystemExit(f"No action.npy under {p}")
	return p


def _filter_shards(env_dir: Path, all_shards: list[Path], shard_arg: str | None) -> list[Path]:
	if shard_arg is None or not str(shard_arg).strip():
		return all_shards
	return [_resolve_shard(env_dir, str(shard_arg))]


def _count_actions(act: np.ndarray, num_actions: int) -> tuple[np.ndarray, int]:
	"""Returns (counts shape [num_actions], n_invalid)."""
	a = np.asarray(act).ravel().astype(np.int64, copy=False)
	valid = (a >= 0) & (a < num_actions)
	n_invalid = int((~valid).sum())
	if valid.any():
		c = np.bincount(a[valid], minlength=num_actions).astype(np.int64)
	else:
		c = np.zeros(num_actions, dtype=np.int64)
	return c, n_invalid


def _fmt_row(aid: int, count: int, n_valid: int) -> str:
	p = float(count) / float(n_valid) if n_valid > 0 else 0.0
	return f"  a{aid:2d}: {int(count):8d}   p={p:.6f}   ({100.0 * p:6.2f}% of valid)"


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Action distribution per shard and pooled (transitions train/test env folder).",
	)
	p.add_argument(
		"--transitions-root",
		type=str,
		default=str(Path("data") / "transitions" / "train"),
		help="Split root, e.g. data/transitions/train",
	)
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument(
		"--shard",
		type=str,
		default=None,
		help="If set, only this shard (e.g. shard_00007 or 7). Otherwise all shards.",
	)
	p.add_argument(
		"--num-actions",
		type=int,
		default=7,
		help="Number of discrete action bins (0..N-1). Default 19 → ids 0..18 (Atari-style).",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	env_dir = Path(args.transitions_root) / args.env
	if not env_dir.is_dir():
		raise SystemExit(f"Not a directory: {env_dir}")

	all_shards = _shard_dirs(env_dir)
	if not all_shards:
		raise SystemExit(f"No shard_*/action.npy under {env_dir}")

	shards = _filter_shards(env_dir, all_shards, args.shard)
	num_actions = int(args.num_actions)
	if num_actions < 1:
		raise SystemExit("--num-actions must be >= 1")
	total = np.zeros(num_actions, dtype=np.int64)
	total_invalid = 0

	print(f"env_dir={env_dir}")
	print(f"num_actions={num_actions} (ids 0..{num_actions - 1})")
	if args.shard:
		print(f"shard_filter={args.shard!r}  (processing {len(shards)} shard(s))")
	else:
		print(f"shards={len(shards)} (all)")
	print()

	for shard in shards:
		act = np.load(shard / "action.npy", mmap_mode="r")
		c, n_inv = _count_actions(act, num_actions)
		total += c
		total_invalid += n_inv
		n_valid = int(c.sum())
		n_all = int(act.size)

		print(f"--- {shard.name}  rows={n_all}  valid={n_valid}  invalid_or_oob={n_inv} ---")
		for aid in range(num_actions):
			print(_fmt_row(aid, int(c[aid]), n_valid))
		print()

	n_valid_all = int(total.sum())
	title = "SELECTED SHARD(S)" if args.shard else "ALL SHARDS"
	print(f"=== {title} (pooled over printed shards) ===")
	print(f"valid={n_valid_all}  invalid_or_oob={total_invalid}")
	for aid in range(num_actions):
		print(_fmt_row(aid, int(total[aid]), n_valid_all))


if __name__ == "__main__":
	main()
