"""Interactive inference: user-key actions + toggleable rule composition → generated frames."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from matplotlib.patches import Rectangle

from world_model.util.checkpoint import load_world_model
from world_model.util.rules import rule_onehot_from_tags, rule_panel_tags

__all__ = ["load_world_model"]


def _tensor_to_imshow01(t: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] → [H,W,3] float32 [0,1]."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype(np.float32)


def _rule_square_caption(tag: str) -> str:
	"""Two-line text inside a rule tile."""
	if tag == "null":
		return "clear\n(null)"
	if "_" in tag:
		a, _, b = tag.partition("_")
		return f"{a}\n{b}"
	return tag


def _list_encoded_shards(root: Path) -> list[Path]:
	shards = sorted(
		p for p in root.glob("shard_*")
		if (p / "latent.npy").is_file() and (p / "action.npy").is_file()
	)
	if not shards:
		raise FileNotFoundError(f"No shard_* with latent/action under {root.resolve()}")
	return shards


def _resolve_bootstrap_encoded_dir(env: str, shard_dir: Optional[str | Path]) -> Path:
	"""Where to read ``latent.npy`` / ``action.npy`` for bootstrap context.

	If ``shard_dir`` is set: use that path if it already contains ``latent.npy``, else treat it as an
	encoded env directory with ``shard_*`` children.
	Otherwise: ``data/transitions/encoded/train/<env>/`` (same layout as dynamics training).
	"""
	if shard_dir is None or str(shard_dir).strip() == "":
		root = Path("data") / "transitions" / "encoded" / "train" / str(env)
		if not root.is_dir():
			raise FileNotFoundError(f"Missing encoded env dir {root.resolve()}")
		_list_encoded_shards(root)
		return root

	p = Path(shard_dir).expanduser().resolve()
	if (p / "latent.npy").is_file() and (p / "action.npy").is_file():
		return p
	if p.is_dir():
		_list_encoded_shards(p)
		return p
	raise FileNotFoundError(
		f"No usable encoded bootstrap path: {p} is not a shard dir (missing latent.npy/action.npy) "
		f"and has no shard_* children with both.",
	)


def _bootstrap_encoded_slice(root: Path, start: int, K: int) -> tuple[np.ndarray, np.ndarray]:
	"""Load ``K`` contiguous latent/action rows from one shard or a linear index across sorted shards."""
	if (root / "latent.npy").is_file() and (root / "action.npy").is_file():
		lat = np.load(root / "latent.npy", mmap_mode="r")
		acts = np.load(root / "action.npy", mmap_mode="r")
		n = int(lat.shape[0])
		if start + K > n:
			raise ValueError(f"Need at least start+K <= {n}; got start={start}, K={K}")
		return lat[start : start + K], acts[start : start + K]

	shards = _list_encoded_shards(root)
	lengths = [int(np.load(s / "latent.npy", mmap_mode="r").shape[0]) for s in shards]
	total = sum(lengths)
	if start + K > total:
		raise ValueError(f"Need at least start+K <= {total}; got start={start}, K={K}")

	g = int(start)
	lat_parts: list[np.ndarray] = []
	act_parts: list[np.ndarray] = []
	need = K
	for shard, length in zip(shards, lengths):
		if g >= length:
			g -= length
			continue
		lat = np.load(shard / "latent.npy", mmap_mode="r")
		acts = np.load(shard / "action.npy", mmap_mode="r")
		take = min(need, length - g)
		lat_parts.append(lat[g : g + take])
		act_parts.append(acts[g : g + take])
		need -= take
		g = 0
		if need == 0:
			break
	return np.concatenate(lat_parts, axis=0), np.concatenate(act_parts, axis=0)


def run_autoregressive(
	ckpt_dir: str,
	env: str = "aliens",
	num_actions: int = 7,
	history_len: int | None = None,
	num_inference_steps: int = 30,
	bootstrap_start_idx: int = 0,
	vae_checkpoint: Optional[str] = None,
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
	shard_dir: Optional[str | Path] = None,
) -> None:
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	selected_tags: set[str] = set()
	rule_oh = rule_onehot_from_tags(selected_tags, device=device)
	wm = load_world_model(
		ckpt_dir,
		num_actions=num_actions,
		history_len=history_len,
		vae_checkpoint=vae_checkpoint,
		cfg_scale_action=cfg_scale_action,
		cfg_scale_rule=cfg_scale_rule,
	)
	K = wm.history_len
	wm.eval()

	start = int(bootstrap_start_idx)
	if start < 0:
		raise ValueError(f"--bootstrap_start_idx must be >= 0, got {start}")

	boot_root = _resolve_bootstrap_encoded_dir(env, shard_dir)
	print(f"Bootstrap encoded: {boot_root}")
	lat_slice, act_slice = _bootstrap_encoded_slice(boot_root, start, K)

	# Bootstrap: K pre-encoded latents, decode to pixels so history matches autoregressive outputs.
	with torch.no_grad():
		z_boot = torch.from_numpy(np.asarray(lat_slice, dtype=np.float32)).unsqueeze(0).to(device)
		boot_pixels = wm.decode_video(z_boot)[0].cpu()
	gen_hist: deque[torch.Tensor] = deque([boot_pixels[i].clone() for i in range(K)], maxlen=K)
	action_hist: deque[int] = deque([int(a) for a in act_slice], maxlen=K)
	data_pos = start

	fig = plt.figure(figsize=(7.6, 11.5))
	gs = gridspec.GridSpec(4, 1, height_ratios=[0.68, 1.0, 0.05, 0.22], hspace=0.015, figure=fig)
	rule_ax = fig.add_subplot(gs[0])
	ax_top = fig.add_subplot(gs[1])
	step_ax = fig.add_subplot(gs[2])
	text_ax = fig.add_subplot(gs[3])
	for ax in (rule_ax, ax_top, step_ax, text_ax):
		ax.axis("off")
	rule_ax.set_xlim(0, 1)
	rule_ax.set_ylim(0, 1)
	step_ax.set_xlim(0, 1)
	step_ax.set_ylim(0, 1)
	text_ax.set_xlim(0, 1)
	text_ax.set_ylim(0, 1)
	last_boot = boot_pixels[-1]
	h, w = int(last_boot.shape[-2]), int(last_boot.shape[-1])
	img_top = ax_top.imshow(_tensor_to_imshow01(last_boot), vmin=0, vmax=1, interpolation="nearest")
	status = step_ax.text(
		0.5, 0.5, f"step={data_pos}",
		ha="center", va="center", fontsize=13, color="#212529",
	)

	# ── Rule tiles: clear + atomic tags (toggle multi-select → multi-hot) ────────
	panel_tiles = ["null"] + rule_panel_tags()
	tag_key_to_label = {str(i + 1): tag for i, tag in enumerate(rule_panel_tags())}
	for digit_key, label in list(tag_key_to_label.items()):
		tag_key_to_label["k" + digit_key] = label
	ncols_r = 4
	nrows_r = (len(panel_tiles) + ncols_r - 1) // ncols_r
	rx0, ry0, rx1, ry1 = 0.02, 0.02, 0.98, 0.98
	padx_r, pady_r = 0.014, 0.012
	cell_rw = (rx1 - rx0 - padx_r * (ncols_r + 1)) / ncols_r
	cell_rh = (ry1 - ry0 - pady_r * (nrows_r + 1)) / nrows_r
	_rule_gray = "#E9ECEF"
	_rule_gray_edge = "#6C757D"
	_rule_hi = "#4C6EF5"
	_rule_hi_edge = "#364FC7"
	rule_tile_rects: list[tuple[Rectangle, str]] = []
	rule_tile_bounds: list[tuple[float, float, float, float, str]] = []
	for i, tile_id in enumerate(panel_tiles):
		row = i // ncols_r
		col = i % ncols_r
		x = rx0 + padx_r + col * (cell_rw + padx_r)
		y_top = ry1 - pady_r - row * (cell_rh + pady_r)
		y = y_top - cell_rh
		rect = Rectangle(
			(x, y), cell_rw, cell_rh, linewidth=1.8, edgecolor=_rule_gray_edge, facecolor=_rule_gray,
		)
		rule_ax.add_patch(rect)
		rule_tile_rects.append((rect, tile_id))
		rule_tile_bounds.append((x, y, cell_rw, cell_rh, tile_id))
		rule_ax.text(
			x + cell_rw / 2.0, y + cell_rh * 0.5, _rule_square_caption(tile_id),
			ha="center", va="center", fontsize=10, color="#212529",
		)

	def _sync_rule_oh() -> None:
		nonlocal rule_oh
		rule_oh = rule_onehot_from_tags(selected_tags, device=device)

	def _paint_rule_hud() -> None:
		for rect, tile_id in rule_tile_rects:
			active = (tile_id == "null" and not selected_tags) or (tile_id in selected_tags)
			if active:
				rect.set_facecolor(_rule_hi)
				rect.set_edgecolor(_rule_hi_edge)
			else:
				rect.set_facecolor(_rule_gray)
				rect.set_edgecolor(_rule_gray_edge)

	def _clear_rules() -> None:
		selected_tags.clear()
		_sync_rule_oh()
		_paint_rule_hud()

	def _toggle_rule_tag(tag: str) -> None:
		if tag in selected_tags:
			selected_tags.remove(tag)
		else:
			selected_tags.add(tag)
		_sync_rule_oh()
		_paint_rule_hud()

	_paint_rule_hud()

	key_to_action = {
		"up": 1,
		"left": 2,
		"down": 3,
		"right": 4,
		" ": 5,
		"space": 5,
	}
	key_w, key_h = 0.20, 0.22
	bottom_row_y = 0.04
	top_row_y = bottom_row_y + key_h
	left_col_x = 0.06
	down_col_x = left_col_x + key_w
	right_col_x = down_col_x + key_w
	fire_col_x = right_col_x + key_w + 0.08
	key_layout = (
		("UP", "up", (down_col_x, top_row_y)),
		("LEFT", "left", (left_col_x, bottom_row_y)),
		("DOWN", "down", (down_col_x, bottom_row_y)),
		("RIGHT", "right", (right_col_x, bottom_row_y)),
		("FIRE", "fire", (fire_col_x, bottom_row_y)),
	)
	action_to_name = {v: k for k, v in key_to_action.items() if k != " "}
	action_to_name[5] = "fire"
	key_boxes: dict[str, Rectangle] = {}
	for label, key_name, (x, y) in key_layout:
		rect = Rectangle((x, y), key_w, key_h, linewidth=1.6, edgecolor="#6C757D", facecolor="#E9ECEF")
		text_ax.add_patch(rect)
		text_ax.text(x + key_w / 2.0, y + key_h / 2.0, label, ha="center", va="center", fontsize=10, color="#212529")
		key_boxes[key_name] = rect


	def _paint_action_hud(action_idx: int | None) -> None:
		for rect in key_boxes.values():
			rect.set_facecolor("#E9ECEF")
			rect.set_edgecolor("#6C757D")
		if action_idx is None:
			return
		key_name = action_to_name.get(action_idx)
		if key_name is None:
			return
		rect = key_boxes.get(key_name)
		if rect is not None:
			rect.set_facecolor("#4C6EF5")
			rect.set_edgecolor("#364FC7")

	_paint_action_hud(None)

	def on_click(event):
		if event.inaxes != rule_ax or event.button != 1:
			return
		xd, yd = event.xdata, event.ydata
		if xd is None or yd is None:
			return
		for x, y, rw, rh, tile_id in rule_tile_bounds:
			if x <= xd <= x + rw and y <= yd <= y + rh:
				if tile_id == "null":
					_clear_rules()
				else:
					_toggle_rule_tag(tile_id)
				fig.canvas.draw_idle()
				return

	def on_key(event):
		nonlocal data_pos
		key = event.key or ""
		if key in {"0", "n"}:
			_clear_rules()
			fig.canvas.draw_idle()
			return
		tag = tag_key_to_label.get(key)
		if tag is not None:
			_toggle_rule_tag(tag)
			fig.canvas.draw_idle()
			return

		a_step = key_to_action.get(event.key)
		if a_step is None:
			return
		if a_step >= num_actions:
			status.set_text(f"Action {a_step} out of range for num_actions={num_actions}")
			fig.canvas.draw_idle()
			return

		ha = torch.tensor(list(action_hist), dtype=torch.long, device=device).view(1, K)
		fa = torch.tensor([a_step], dtype=torch.long, device=device)

		ctx = torch.stack(list(gen_hist), dim=0).unsqueeze(0).to(device)
		with torch.no_grad():
			z_hist = wm.encode_video(ctx)
			z_next = wm.generate_next_frame(
				z_hist, ha, fa, num_inference_steps=num_inference_steps,
				rule_onehot=rule_oh.expand(z_hist.shape[0], -1),
			)
			gen = wm.decode_video(z_next)[0, 0].cpu()

		img_top.set_data(_tensor_to_imshow01(gen))
		_paint_action_hud(a_step)

		gen_hist.append(gen.clone())
		action_hist.append(a_step)
		data_pos += 1
		status.set_text(
			f"step={data_pos}",
		)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("key_press_event", on_key)
	fig.canvas.mpl_connect("button_press_event", on_click)
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--ckpt_dir", type=str, default=str(
		Path("world_model") / "checkpoints" / "dit_encoded_rules_all_env" / "20260528_193324" / "step_1187800"
	))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="world_model/checkpoints/vae/vae.pt",
		help="Path to vae.pt (empty lets trainer_state decide).",
	)
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument(
		"--shard_dir",
		type=str,
		default=None,
		help="Bootstrap from encoded transitions instead of data/transitions/encoded/train/<env>/. "
		"Pass a shard folder (…/shard_00000) or an encoded env dir (e.g. data/transitions/encoded/test/aliens).",
	)
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=None, help="History length K; default reads context_len from checkpoint.")
	p.add_argument(
		"--bootstrap_start_idx",
		type=int,
		default=100,
		help="Initial frame index used for bootstrap context. 0 keeps current behavior; 30 starts from frame 30.",
	)
	p.add_argument(
		"--cfg_scale_action",
		type=float,
		default=None,
		help="Override action CFG scale (default: trainer_state.cfg_scale_action or legacy cfg_scale, else 1.5).",
	)
	p.add_argument(
		"--cfg_scale_rule",
		type=float,
		default=None,
		help="Override rule CFG scale (default: trainer_state.cfg_scale_rule or legacy cfg_scale, else 1.5).",
	)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		vae_checkpoint=args.vae_checkpoint.strip() or None,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		history_len=args.context_len,
		bootstrap_start_idx=args.bootstrap_start_idx,
		cfg_scale_action=args.cfg_scale_action,
		cfg_scale_rule=args.cfg_scale_rule,
		shard_dir=args.shard_dir,
	)


if __name__ == "__main__":
	main()
