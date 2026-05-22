"""
Embedding cluster analysis: VAE latent ``z_t`` vs simple history-side latent features.

Samples windows from encoded transition shards across games/rules, computes:
  - ``z_t``: target-frame VAE latent (flattened)
  - ``h_frame``: last frame of history latents ``z_hist[:, -1]`` (flattened), same shape family as ``z_t``
  - ``h_state``: per-timestep spatial mean of ``z_hist`` then flattened (``K * C`` dims)

Writes PCA and UMAP 2-D scatter plots. **Game** and **rule** each get a dedicated
subfolder with a **two-panel** figure (``z`` left, ``h`` right) per reducer (PCA, UMAP).
Other labels get single-panel plots in the block root. Scene metadata uses heuristics
from raw ``obs.npy`` when available.

Reports silhouette scores per label type (categorical labels directly; continuous
labels binned into quantiles for silhouette only).

Example::

  python -m world_model.eval_embedding_clusters \\
    --checkpoint world_model/checkpoints/dit_encoded_rules_all_env_adv_cfg_normal/step_0250000 \\
    --split test --max_samples 8000 --output_dir runs/embedding_clusters
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from world_model.dataset import (
	MixedEncodedRolloutVideoDataset,
	encoded_dirs_all_under_split,
	encoded_dirs_with_rules,
	encoded_folder_base_game,
	preprocess_latent,
)
from world_model.inference import load_world_model
from world_model.train_dynamics import CONTEXT_LEN, DEFAULT_PRETRAINED_DYNAMICS

try:
	import umap
except ImportError:  # pragma: no cover
	umap = None  # type: ignore


# Dedicated two-panel (z | h) plots per reducer live under ``<block>/<label_key>/``.
SEPARATED_PAIR_LABELS: tuple[tuple[str, str], ...] = (
	("game_id", "game"),
	("rule_id", "rule"),
)

LABEL_SPECS: tuple[tuple[str, str, bool], ...] = (
	("action", "action", False),
	("object_count", "object_count", True),
	("bullet_speed", "bullet_speed", True),
	("player_x", "player_x", True),
	("player_y", "player_y", True),
	("enemy_x", "enemy_x", True),
	("enemy_y", "enemy_y", True),
)


@dataclass
class SampleMeta:
	game_folder: str
	base_game: str
	game_id: int
	rule_id: int
	shard_row: int
	action: int
	player_x: float
	player_y: float
	object_count: float
	bullet_speed: float
	enemy_x: float
	enemy_y: float


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded")
	p.add_argument("--split", type=str, choices=("train", "test"), default="test")
	p.add_argument("--env", type=str, default=None, help="Base game name; default = all encoded env folders.")
	p.add_argument("--checkpoint", type=str, default=DEFAULT_PRETRAINED_DYNAMICS)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
	p.add_argument("--max_samples", type=int, default=6000, help="Cap total windows (stratified across env dirs).")
	p.add_argument("--stride", type=int, default=8, help="Window stride when subsampling each env dataset.")
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--output_dir", type=str, default=str(Path("runs") / "embedding_clusters"))
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--umap_neighbors", type=int, default=30)
	p.add_argument("--umap_min_dist", type=float, default=0.1)
	p.add_argument("--silhouette_bins", type=int, default=12, help="Quantile bins for continuous label silhouette.")
	p.add_argument("--skip_umap", action="store_true", help="Only PCA (no umap-learn required).")
	p.add_argument("--dpi", type=int, default=120)
	p.add_argument(
		"--show",
		action="store_true",
		help="Pop up matplotlib windows for the game/rule z|h pair plots (blocks until closed).",
	)
	p.add_argument(
		"--open_browser",
		action="store_true",
		help="Open index.html in the default browser after writing plots.",
	)
	return p.parse_args()


def _raw_shard_path(
	transitions_root: Path,
	encoded_subdir: str,
	split: str,
	game_folder: str,
	shard_name: str,
) -> Path:
	return transitions_root / split / game_folder / shard_name


def _rgb_hwc(obs: np.ndarray) -> np.ndarray:
	a = np.asarray(obs)
	if a.ndim == 4:
		a = a[0]
	return a[..., -3:].astype(np.uint8)


def _heuristic_scene(obs: np.ndarray, player_x: float, player_y: float, prev_obs: np.ndarray | None) -> tuple[float, float, float, float]:
	"""Rough scene stats from RGB (object count, enemy xy, bullet speed proxy)."""
	rgb = _rgb_hwc(obs)
	gray = rgb.mean(axis=-1)
	bg = float(np.median(gray))
	fg = gray < (bg - 12.0)
	if not np.any(fg):
		return 0.0, math.nan, math.nan, 0.0

	try:
		from scipy import ndimage

		labeled, n_comp = ndimage.label(fg)
		if n_comp == 0:
			return 0.0, math.nan, math.nan, 0.0
		sizes = ndimage.sum(fg, labeled, index=np.arange(1, n_comp + 1))
		order = np.argsort(-sizes)
		centroids: list[tuple[float, float, float]] = []
		for lab in order[: min(12, n_comp)]:
			if sizes[lab - 1] < 8:
				continue
			ys, xs = np.nonzero(labeled == lab)
			centroids.append((float(sizes[lab - 1]), float(xs.mean()), float(ys.mean())))
		object_count = float(len(centroids))
	except Exception:
		ys, xs = np.nonzero(fg)
		object_count = 1.0
		centroids = [(float(len(xs)), float(xs.mean()), float(ys.mean()))]

	enemy_x, enemy_y = math.nan, math.nan
	if not (math.isnan(player_x) or math.isnan(player_y)):
		best_d = -1.0
		for _sz, cx, cy in centroids:
			d = (cx - player_x) ** 2 + (cy - player_y) ** 2
			if d > best_d:
				best_d = d
				enemy_x, enemy_y = cx, cy
	elif centroids:
		_sz, enemy_x, enemy_y = centroids[0]

	bullet_speed = 0.0
	if prev_obs is not None:
		p0 = _rgb_hwc(prev_obs).astype(np.int16)
		p1 = rgb.astype(np.int16)
		diff = np.abs(p1 - p0).mean(axis=-1)
		motion = diff > 28
		if np.any(motion):
			bullet_speed = float(motion.sum()) / max(1, motion.size)

	return object_count, enemy_x, enemy_y, bullet_speed


def _rule_tag_from_folder(game_folder: str) -> str:
	if "_rules_" not in game_folder:
		return "null"
	tag = game_folder.split("_rules_", 1)[1]
	from world_model.dataset import LEGACY_NULL_RULE_TAGS

	if tag in LEGACY_NULL_RULE_TAGS:
		return "null"
	return tag


def _rule_id_from_folder(game_folder: str) -> int:
	tag = _rule_tag_from_folder(game_folder)
	if tag == "null":
		return -1
	from world_model.dataset import RULE_TAG_TO_INDEX

	return int(RULE_TAG_TO_INDEX.get(tag, -2))


def _build_game_id_map(folders: list[str]) -> dict[str, int]:
	base_games = sorted({encoded_folder_base_game(f) for f in folders})
	return {g: i for i, g in enumerate(base_games)}


def _collect_indices(ds: MixedEncodedRolloutVideoDataset, max_samples: int, stride: int, rng: np.random.Generator) -> list[int]:
	n = len(ds)
	if n <= max_samples:
		return list(range(n))
	# Stratify: equal budget per encoded env folder
	by_folder: dict[str, list[int]] = {}
	for i in range(n):
		f = ds.window_game_folder(i)
		by_folder.setdefault(f, []).append(i)
	per = max(1, max_samples // max(1, len(by_folder)))
	chosen: list[int] = []
	for folder, idxs in sorted(by_folder.items()):
		sub = idxs[:: max(1, stride)]
		if len(sub) > per:
			sub = list(rng.choice(sub, size=per, replace=False))
		chosen.extend(sub)
	if len(chosen) > max_samples:
		chosen = list(rng.choice(np.array(chosen, dtype=np.int64), size=max_samples, replace=False))
	return sorted(int(i) for i in chosen)


def _attach_row_metadata(
	ds: MixedEncodedRolloutVideoDataset,
	indices: list[int],
	transitions_root: Path,
	encoded_subdir: str,
	split: str,
	game_id_map: dict[str, int],
) -> list[SampleMeta]:
	K = ds.seq_len - 1
	out: list[SampleMeta] = []
	for idx in tqdm(indices, desc="metadata", leave=False):
		meta, row = ds._resolve_index(idx)  # noqa: SLF001 — eval-only introspection
		game_folder = meta.game_name
		base = encoded_folder_base_game(game_folder)
		shard = ds._get_shard(meta)  # noqa: SLF001
		act = int(shard["action"][row + K - 1])

		px = py = math.nan
		obj_c = 0.0
		ex = ey = math.nan
		bspd = 0.0
		raw_dir = _raw_shard_path(transitions_root, encoded_subdir, split, game_folder, meta.path.name)
		prev_obs = None
		if raw_dir.is_dir():
			px_path, py_path = raw_dir / "player_x.npy", raw_dir / "player_y.npy"
			if px_path.is_file() and py_path.is_file():
				px_a = np.load(px_path, mmap_mode="r")
				py_a = np.load(py_path, mmap_mode="r")
				tgt = row + K
				if tgt < len(px_a):
					px, py = float(px_a[tgt]), float(py_a[tgt])
			obs_path = raw_dir / "obs.npy"
			if obs_path.is_file():
				obs_mm = np.load(obs_path, mmap_mode="r")
				if row + K < len(obs_mm):
					obs_t = obs_mm[row + K]
					if row + K > 0:
						prev_obs = obs_mm[row + K - 1]
					obj_c, ex, ey, bspd = _heuristic_scene(obs_t, px, py, prev_obs)

		out.append(
			SampleMeta(
				game_folder=game_folder,
				base_game=base,
				game_id=game_id_map[base],
				rule_id=_rule_id_from_folder(game_folder),
				shard_row=row + K,
				action=act,
				player_x=px,
				player_y=py,
				object_count=obj_c,
				bullet_speed=bspd,
				enemy_x=ex,
				enemy_y=ey,
			)
		)
	return out


@torch.no_grad()
def _encode_batches(
	_wm: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	z_list: list[np.ndarray] = []
	hf_list: list[np.ndarray] = []
	hs_list: list[np.ndarray] = []
	for batch in tqdm(loader, desc="encode"):
		z_hist = batch["history_latents"].to(device)
		z_tgt = batch["target_latent"].to(device)
		z_list.append(z_tgt.float().cpu().numpy().reshape(z_tgt.shape[0], -1))
		hf_list.append(z_hist[:, -1].float().cpu().numpy().reshape(z_hist.shape[0], -1))
		hm = z_hist.mean(dim=(3, 4)).float().cpu().numpy()
		hs_list.append(hm.reshape(hm.shape[0], -1))
	return np.concatenate(z_list, axis=0), np.concatenate(hf_list, axis=0), np.concatenate(hs_list, axis=0)


def _fit_pca(X: np.ndarray, n_components: int = 2, seed: int = 0) -> np.ndarray:
	pca = PCA(n_components=n_components, random_state=seed)
	return pca.fit_transform(X)


def _fit_umap(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
	if umap is None:
		raise RuntimeError("umap-learn is not installed; pip install umap-learn or pass --skip_umap")
	reducer = umap.UMAP(
		n_components=2,
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		metric="euclidean",
		random_state=seed,
	)
	return reducer.fit_transform(X)


def _labels_from_meta(meta: list[SampleMeta], key: str) -> np.ndarray:
	if key == "game_id":
		return np.array([m.game_id for m in meta], dtype=np.int64)
	if key == "rule_id":
		return np.array([m.rule_id for m in meta], dtype=np.int64)
	if key == "action":
		return np.array([m.action for m in meta], dtype=np.int64)
	if key == "object_count":
		return np.array([m.object_count for m in meta], dtype=np.float64)
	if key == "bullet_speed":
		return np.array([m.bullet_speed for m in meta], dtype=np.float64)
	if key == "player_x":
		return np.array([m.player_x for m in meta], dtype=np.float64)
	if key == "player_y":
		return np.array([m.player_y for m in meta], dtype=np.float64)
	if key == "enemy_x":
		return np.array([m.enemy_x for m in meta], dtype=np.float64)
	if key == "enemy_y":
		return np.array([m.enemy_y for m in meta], dtype=np.float64)
	raise KeyError(key)


def _viz_color_labels(meta: list[SampleMeta], key: str) -> np.ndarray:
	"""Human-readable categorical labels for game/rule scatter coloring."""
	if key == "game_id":
		return np.array([m.base_game for m in meta], dtype=object)
	if key == "rule_id":
		return np.array([_rule_tag_from_folder(m.game_folder) for m in meta], dtype=object)
	raise KeyError(key)


def _bin_labels(y: np.ndarray, n_bins: int) -> np.ndarray | None:
	mask = np.isfinite(y)
	if mask.sum() < n_bins * 2:
		return None
	vals = y[mask]
	qs = np.linspace(0, 1, n_bins + 1)
	edges = np.unique(np.quantile(vals, qs))
	if len(edges) < 3:
		return None
	binned = np.full(y.shape, -1, dtype=np.int64)
	binned[mask] = np.digitize(vals, edges[1:-1], right=False)
	return binned


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float | None:
	if labels is None:
		return None
	lab = np.asarray(labels)
	valid = lab >= 0
	if valid.sum() < 3:
		return None
	lab_v = lab[valid]
	n_classes = len(np.unique(lab_v))
	if n_classes < 2 or n_classes >= valid.sum():
		return None
	try:
		return float(silhouette_score(X[valid], lab_v, metric="euclidean"))
	except Exception:
		return None


def _color_map_for_labels(values: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
	uniq = sorted(np.unique(values), key=str)
	cmap = plt.colormaps["tab20"].resampled(max(len(uniq), 1))
	return {str(u): cmap(i % 20) for i, u in enumerate(uniq)}


def _scatter_categorical_on_ax(
	ax: plt.Axes,
	xy: np.ndarray,
	values: np.ndarray,
	title: str,
	*,
	legend: bool,
	color_map: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
	if color_map is None:
		color_map = _color_map_for_labels(values)
	for u in sorted(np.unique(values), key=str):
		m = values == u
		ax.scatter(
			xy[m, 0], xy[m, 1], s=4, alpha=0.55, label=str(u),
			c=[color_map[str(u)]],
		)
	if legend and len(color_map) <= 24:
		ax.legend(markerscale=3, fontsize=6, loc="best")
	ax.set_title(title)
	ax.set_xlabel("dim 0")
	ax.set_ylabel("dim 1")


def _scatter_save(
	path: Path,
	xy: np.ndarray,
	values: np.ndarray,
	title: str,
	*,
	categorical: bool,
	dpi: int,
) -> None:
	fig, ax = plt.subplots(figsize=(7, 6))
	if categorical:
		_scatter_categorical_on_ax(ax, xy, values, title, legend=True)
	else:
		m = np.isfinite(values)
		sc = ax.scatter(xy[m, 0], xy[m, 1], c=values[m], s=4, alpha=0.65, cmap="viridis")
		plt.colorbar(sc, ax=ax, fraction=0.046)
		ax.set_title(title)
		ax.set_xlabel("dim 0")
		ax.set_ylabel("dim 1")
	fig.tight_layout()
	fig.savefig(path, dpi=dpi)
	plt.close(fig)


def _scatter_pair_z_h_save(
	path: Path,
	xy_z: np.ndarray,
	xy_h: np.ndarray,
	values: np.ndarray,
	*,
	method: str,
	label_title: str,
	encoder_name: str,
	dpi: int,
) -> None:
	"""Side-by-side: VAE latent z (left) vs encoder h (right), shared color legend."""
	color_map = _color_map_for_labels(values)
	fig, (ax_z, ax_h) = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
	_scatter_categorical_on_ax(
		ax_z, xy_z, values, f"VAE latent z — {method.upper()}", legend=False, color_map=color_map,
	)
	_scatter_categorical_on_ax(
		ax_h, xy_h, values, f"{encoder_name} h — {method.upper()}", legend=False, color_map=color_map,
	)
	uniq = sorted(np.unique(values), key=str)
	if len(uniq) <= 24:
		handles = [
			plt.Line2D(
				[0], [0], marker="o", color="w", markerfacecolor=color_map[str(u)],
				markersize=6, label=str(u),
			)
			for u in uniq
		]
		fig.legend(handles=handles, loc="lower center", ncol=min(len(uniq), 8), fontsize=7, markerscale=2)
	fig.suptitle(f"Colored by {label_title}", fontsize=11, y=1.02)
	fig.tight_layout()
	fig.savefig(path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)



def _html_cards(out_dir: Path, pngs: list[Path]) -> str:
	chunks: list[str] = []
	for png in pngs:
		rel = png.relative_to(out_dir).as_posix()
		chunks.append(
			f'<div class="card"><div class="path">{rel}</div>'
			f'<a href="{rel}"><img src="{rel}" loading="lazy"></a></div>'
		)
	return "".join(chunks)


def _write_html_index(out_dir: Path) -> Path:
	"""Build a browsable gallery of every PNG under ``out_dir``."""
	pngs = sorted(out_dir.rglob("*.png"))
	index_path = out_dir / "index.html"
	priority_keys = ("game_id", "rule_id", "z_vs_h")
	main = [p for p in pngs if any(k in p.as_posix() for k in priority_keys)]
	rest = [p for p in pngs if p not in main]
	html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Embedding cluster plots</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#111;color:#eee}}
h1,h2{{color:#fff}} a{{color:#8cf}}
.grid{{display:flex;flex-wrap:wrap;gap:1rem}}
.card{{background:#222;border-radius:8px;padding:.5rem;max-width:720px}}
.card img{{max-width:100%;height:auto;display:block}}
.path{{font-size:.75rem;color:#aaa;word-break:break-all;margin:.25rem 0 .5rem}}
</style></head><body>
<h1>Embedding clusters</h1>
<p>{len(pngs)} PNG(s) - <code>{out_dir.resolve()}</code></p>
<h2>Main - game and rule (z | h)</h2>
<div class="grid">{_html_cards(out_dir, main)}</div>
<h2>All plots</h2>
<div class="grid">{_html_cards(out_dir, rest)}</div>
</body></html>"""
	index_path.write_text(html, encoding="utf-8")
	return index_path


def _print_plot_summary(out_dir: Path) -> None:
	pngs = sorted(out_dir.rglob("*.png"))
	print(f"\n=== Visualization: {len(pngs)} PNG files ===")
	print(f"  Gallery:  {out_dir.resolve() / 'index.html'}")
	highlights = [
		out_dir / "h_frame" / "game_id" / "pca_z_vs_h.png",
		out_dir / "h_frame" / "rule_id" / "pca_z_vs_h.png",
		out_dir / "h_frame" / "game_id" / "umap_z_vs_h.png",
		out_dir / "h_frame" / "rule_id" / "umap_z_vs_h.png",
	]
	for p in highlights:
		if p.is_file():
			print(f"  {p.relative_to(out_dir)}")
	if not pngs:
		print("  (no PNG files found — plotting may have failed silently)")


_BLOCK_H_LABEL = {
	"h_frame": "last history z",
	"h_state": "spatially pooled history z",
}


def _run_embedding_block(
	name: str,
	X_z: np.ndarray,
	X_h: np.ndarray,
	meta: list[SampleMeta],
	out_dir: Path,
	seed: int,
	umap_neighbors: int,
	umap_min_dist: float,
	silhouette_bins: int,
	skip_umap: bool,
	dpi: int,
) -> dict[str, Any]:
	h_label = _BLOCK_H_LABEL.get(name, name)
	block_dir = out_dir / name
	block_dir.mkdir(parents=True, exist_ok=True)
	reducers: dict[str, np.ndarray] = {"pca_z": _fit_pca(X_z, seed=seed), "pca_h": _fit_pca(X_h, seed=seed)}
	if not skip_umap:
		reducers["umap_z"] = _fit_umap(X_z, seed, umap_neighbors, umap_min_dist)
		reducers["umap_h"] = _fit_umap(X_h, seed, umap_neighbors, umap_min_dist)

	scores: dict[str, Any] = {}
	for red_name, xy_z in reducers.items():
		if not red_name.endswith("_z"):
			continue
		xy_h = reducers[red_name.replace("_z", "_h")]
		method = red_name.split("_", 1)[0]

		for label_key, label_title in SEPARATED_PAIR_LABELS:
			y_sil = _labels_from_meta(meta, label_key)
			y_viz = _viz_color_labels(meta, label_key)
			lab_sil = y_sil
			sz = _silhouette_safe(xy_z, lab_sil)
			sh = _silhouette_safe(xy_h, lab_sil)
			scores.setdefault(method, {}).setdefault(label_key, {})
			scores[method][label_key]["z"] = sz
			scores[method][label_key]["h"] = sh
			pair_dir = block_dir / label_key
			pair_dir.mkdir(parents=True, exist_ok=True)
			_scatter_pair_z_h_save(
				pair_dir / f"{method}_z_vs_h.png",
				xy_z,
				xy_h,
				y_viz,
				method=method,
				label_title=label_title,
				encoder_name=h_label,
				dpi=dpi,
			)

		for label_key, label_title, is_cont in LABEL_SPECS:
			y = _labels_from_meta(meta, label_key)
			lab_sil = _bin_labels(y, silhouette_bins) if is_cont else y
			sz = _silhouette_safe(xy_z, lab_sil)
			sh = _silhouette_safe(xy_h, lab_sil)
			scores.setdefault(method, {}).setdefault(label_key, {})
			scores[method][label_key]["z"] = sz
			scores[method][label_key]["h"] = sh

			for emb_tag, xy, emb_name in (("z", xy_z, "VAE latent z"), ("h", xy_h, h_label)):
				sub = f"{method}_{emb_tag}"
				cat = not is_cont
				title = f"{emb_name} — {method.upper()} — colored by {label_title}"
				fname = block_dir / f"{sub}_by_{label_key}.png"
				_scatter_save(fname, xy, y, title, categorical=cat, dpi=dpi)

	np.savez_compressed(block_dir / "coords.npz", **reducers)
	return scores


def main() -> None:
	args = parse_args()
	rng = np.random.default_rng(args.seed)
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	transitions_root = Path(args.transitions_root)
	encoded_root = transitions_root / args.encoded_subdir / args.split
	if args.env is None:
		pairs = encoded_dirs_all_under_split(encoded_root)
	else:
		pairs = encoded_dirs_with_rules(encoded_root, args.env)

	folders = [p.name for p, _ in pairs]
	game_id_map = _build_game_id_map(folders)
	K = int(args.context_len)
	seq_len = K + 1

	ds = MixedEncodedRolloutVideoDataset(
		pairs, seq_len=seq_len, stride=1, num_actions=args.num_actions,
	).with_transform(partial(preprocess_latent, history_len=K))

	indices = _collect_indices(ds, args.max_samples, args.stride, rng)
	print(f"Selected {len(indices):,} / {len(ds):,} windows from {len(pairs)} encoded env folders")
	meta = _attach_row_metadata(ds, indices, transitions_root, args.encoded_subdir, args.split, game_id_map)

	subset = Subset(ds, indices)
	loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(args.checkpoint),
		num_actions=args.num_actions,
		history_len=K,
	)
	wm.eval()
	wm.to(device)

	X_z, X_h_frame, X_h_state = _encode_batches(wm, loader, device)
	np.savez_compressed(
		out_dir / "embeddings.npz",
		z=X_z,
		h_frame=X_h_frame,
		h_state=X_h_state,
		game_id=np.array([m.game_id for m in meta]),
		rule_id=np.array([m.rule_id for m in meta]),
		action=np.array([m.action for m in meta]),
	)

	summary: dict[str, Any] = {
		"n_samples": len(indices),
		"split": args.split,
		"checkpoint": args.checkpoint,
		"context_len": K,
		"frame_encoder": {},
		"state_token": {},
	}
	summary["frame_encoder"] = _run_embedding_block(
		"h_frame",
		X_z,
		X_h_frame,
		meta,
		out_dir,
		args.seed,
		args.umap_neighbors,
		args.umap_min_dist,
		args.silhouette_bins,
		args.skip_umap,
		args.dpi,
	)
	summary["state_token"] = _run_embedding_block(
		"h_state",
		X_z,
		X_h_state,
		meta,
		out_dir,
		args.seed,
		args.umap_neighbors,
		args.umap_min_dist,
		args.silhouette_bins,
		args.skip_umap,
		args.dpi,
	)

	# Compact console report: rule silhouette z vs history-side features
	print("\nSilhouette (rule_id) — z vs history-derived h panels:")
	for block_name, block_scores in (("frame_encoder", summary["frame_encoder"]), ("state_token", summary["state_token"])):
		for method, per_label in block_scores.items():
			r = per_label.get("rule_id", {})
			print(f"  [{block_name}] {method}: z={r.get('z')}  h={r.get('h')}")

	report_path = out_dir / "silhouette_scores.json"
	with report_path.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	index_path = _write_html_index(out_dir)
	_print_plot_summary(out_dir)
	print(f"Metrics: {report_path.resolve()}")
	if args.open_browser:
		import webbrowser

		webbrowser.open(index_path.resolve().as_uri())


if __name__ == "__main__":
	main()
