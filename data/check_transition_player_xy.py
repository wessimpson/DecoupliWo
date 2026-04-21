"""
Scan transition shards for ``player_x.npy`` / ``player_y.npy`` (written by ``collect_transitions.py``).

Default: ``data/transitions/train/aliens`` — prints per-shard row counts, NaN counts, global min/max/mean,
and **inferred sprite size** (median positive step among sorted unique x and y; fallback 32).
Use that value as ``--sprite_size`` for ``train_position_transformer.py`` validation masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def infer_sprite_size(xy: np.ndarray) -> float:
	"""Square cell side from raw (x, y): median positive step on sorted uniques per axis."""
	if xy.size == 0:
		return 32.0

	def positive_steps(a: np.ndarray) -> np.ndarray:
		u = np.unique(np.asarray(a, dtype=np.float64))
		if u.size < 2:
			return np.array([], dtype=np.float64)
		d = np.diff(np.sort(u))
		return d[d > 1e-6]

	sx = positive_steps(xy[:, 0])
	sy = positive_steps(xy[:, 1])
	parts = [p for p in (sx, sy) if p.size > 0]
	if not parts:
		return 32.0
	return float(np.median(np.concatenate(parts)))


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument(
		"--root",
		type=Path,
		default=Path(__file__).resolve().parent / "transitions" / "train" / "aliens",
		help="Env directory containing shard_* (default: transitions/train/aliens).",
	)
	p.add_argument(
		"--split",
		type=str,
		default=None,
		help="If set, use <repo>/data/transitions/<split>/<env> (overrides --root when combined with --env).",
	)
	p.add_argument("--env", type=str, default="aliens", help="Used only with --split (default aliens).")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	if args.split is not None:
		base = Path(__file__).resolve().parent / "transitions" / args.split / args.env
		root = base
	else:
		root = Path(args.root)
	if not root.is_dir():
		raise FileNotFoundError(f"Not a directory: {root}")

	shards = sorted(p for p in root.glob("shard_*") if p.is_dir())
	if not shards:
		raise FileNotFoundError(f"No shard_* under {root}")

	all_px: list[np.ndarray] = []
	all_py: list[np.ndarray] = []
	all_act: list[np.ndarray] = []

	print(f"root: {root.resolve()}")
	print(f"shards: {len(shards)}")
	print()

	for shard in shards:
		px_p = shard / "player_x.npy"
		py_p = shard / "player_y.npy"
		act_p = shard / "action.npy"
		if not px_p.is_file() or not py_p.is_file():
			print(f"{shard.name}: missing player_x.npy or player_y.npy — skip")
			continue
		px = np.load(px_p, mmap_mode="r")
		py = np.load(py_p, mmap_mode="r")
		px = np.asarray(px).ravel()
		py = np.asarray(py).ravel()
		if px.shape != py.shape:
			print(f"{shard.name}: shape mismatch player_x {px.shape} vs player_y {py.shape}")
			continue
		n = px.size
		nan_x = int(np.isnan(px).sum())
		nan_y = int(np.isnan(py).sum())
		valid = np.isfinite(px) & np.isfinite(py)
		if act_p.is_file():
			act = np.load(act_p, mmap_mode="r")
			act = np.asarray(act).ravel()
			if act.size != n:
				print(f"{shard.name}: action len {act.size} != player len {n}")
			else:
				all_act.append(act)
		print(
			f"{shard.name}: n={n:,}  nan_x={nan_x} nan_y={nan_y}  "
			f"px[min,max]=({np.nanmin(px):.4g},{np.nanmax(px):.4g})  "
			f"py[min,max]=({np.nanmin(py):.4g},{np.nanmax(py):.4g})",
		)
		if valid.any():
			print(
				f"         finite: px mean={np.mean(px[valid]):.4g} std={np.std(px[valid]):.4g}  "
				f"py mean={np.mean(py[valid]):.4g} std={np.std(py[valid]):.4g}",
			)
		all_px.append(px)
		all_py.append(py)

	if not all_px:
		print("No player_x/player_y data loaded.")
		return

	px_cat = np.concatenate(all_px)
	py_cat = np.concatenate(all_py)
	n = px_cat.size
	print()
	print("--- pooled (all shards) ---")
	print(f"rows: {n:,}")
	print(f"nan: player_x={int(np.isnan(px_cat).sum())}  player_y={int(np.isnan(py_cat).sum())}")
	valid = np.isfinite(px_cat) & np.isfinite(py_cat)
	if valid.any():
		print(
			f"finite: px min/max/mean/std = {np.min(px_cat[valid]):.6g} / {np.max(px_cat[valid]):.6g} / "
			f"{np.mean(px_cat[valid]):.6g} / {np.std(px_cat[valid]):.6g}",
		)
		print(
			f"        py min/max/mean/std = {np.min(py_cat[valid]):.6g} / {np.max(py_cat[valid]):.6g} / "
			f"{np.mean(py_cat[valid]):.6g} / {np.std(py_cat[valid]):.6g}",
		)
		xy_fin = np.stack([px_cat[valid], py_cat[valid]], axis=1).astype(np.float64)
		ss = infer_sprite_size(xy_fin)
		print(f"inferred_sprite_size (for --sprite_size): {ss:.6g}")
	if all_act:
		a = np.concatenate(all_act)
		if a.size == n:
			print(f"action.npy: same length as player rows ({a.size:,})  unique actions: {sorted(np.unique(a).tolist())}")


if __name__ == "__main__":
	main()
