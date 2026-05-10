"""Interactive inference with user-key actions and generated frames only."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from matplotlib.patches import Rectangle

from world_model.dataset import (
	IMG_TRANSFORMS,
	LEGACY_NULL_RULE_TAGS,
	NUM_RULE_TYPES,
	RULE_TAG_TO_INDEX,
	RULE_TAGS,
	crop_hw_div8,
)
from world_model.model.world_model import WorldModel


def _read_trainer_args(ckpt_dir: Path) -> dict[str, Any]:
	p = ckpt_dir / "trainer_state.pt"
	if not p.is_file():
		return {}
	blob = torch.load(p, map_location="cpu", weights_only=False)
	return dict(blob.get("args") or {})


def _coalesce(meta: dict[str, Any], key: str, override: Any, fallback: Any) -> Any:
	if override is not None:
		return override
	if key in meta and meta[key] is not None:
		return meta[key]
	return fallback


def _cfg_scale_from_meta(meta: dict[str, Any], key_new: str, key_legacy: str, override: float | None, default: float) -> float:
	if override is not None:
		return float(override)
	if key_new in meta and meta[key_new] is not None:
		return float(meta[key_new])
	if key_legacy in meta and meta[key_legacy] is not None:
		return float(meta[key_legacy])
	return float(default)


def load_world_model(
	ckpt_dir: Path,
	num_actions: int,
	history_len: int,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
) -> WorldModel:
	"""Load weights from ``ckpt_dir`` (same layout as :meth:`WorldModel.save_diffuser`).

	If ``trainer_state.pt`` exists, ``vae_checkpoint`` defaults from saved args when omitted.
	"""
	ckpt_dir = Path(ckpt_dir)
	meta = _read_trainer_args(ckpt_dir)

	vae_eff = _coalesce(meta, "vae_checkpoint", vae_checkpoint, None)
	c_sa = _cfg_scale_from_meta(meta, "cfg_scale_action", "cfg_scale", cfg_scale_action, 1.5)
	c_sr = _cfg_scale_from_meta(meta, "cfg_scale_rule", "cfg_scale", cfg_scale_rule, 1.5)
	wm = WorldModel(
		num_actions=num_actions,
		cross_attention_dim=768,
		vae_checkpoint=vae_eff,
		prediction_type="v_prediction",
		history_len=history_len,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		cfg_scale_action=c_sa,
		cfg_scale_rule=c_sr,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = wm.to(device)
	wm.load_diffuser_checkpoint(ckpt_dir, device)
	return wm


def _tensor_to_imshow01(t: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] → [H,W,3] float32 [0,1]."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype(np.float32)


def _rule_vec(name: str) -> torch.Tensor:
	"""[1, NUM_RULE_TYPES] float: NULL (zeros) for base games, else multi-hot from tag name or preset."""
	n0 = str(name).lower().strip()
	v = torch.zeros(1, NUM_RULE_TYPES, dtype=torch.float32)
	if n0 in {"", "normal", "null", "base", "zeros"}:
		return v
	if n0 in LEGACY_NULL_RULE_TAGS or n0 == "rules_fast":
		return v
	# Legacy / alias names
	legacy = {"rule1": "null", "rule2": "null"}
	n = legacy.get(n0, n0)
	if n == "null" or n == "normal":
		return v
	if n in {"multishot+ricochet", "rule3+rule4", "combo34"}:
		v[0, RULE_TAG_TO_INDEX["multishot"]] = 1.0
		v[0, RULE_TAG_TO_INDEX["ricochet"]] = 1.0
		return v
	if n in {"multishot+shoot_walls", "combo35", "rule3+rule5"}:
		v[0, RULE_TAG_TO_INDEX["multishot"]] = 1.0
		v[0, RULE_TAG_TO_INDEX["shoot_walls"]] = 1.0
		return v
	if n in RULE_TAG_TO_INDEX:
		v[0, RULE_TAG_TO_INDEX[n]] = 1.0
		return v
	if "_rules_" in n:
		tag = n.split("_rules_", 1)[1]
		if tag in LEGACY_NULL_RULE_TAGS:
			return v
		if tag in RULE_TAG_TO_INDEX:
			v[0, RULE_TAG_TO_INDEX[tag]] = 1.0
			return v
	tags_known = ", ".join(RULE_TAGS)
	raise ValueError(f"Unknown rule preset {name!r}. Try null/normal, multishot+ricochet, or a RULE_TAGS entry: {tags_known}")


def _is_valid_rule_name(n: str) -> bool:
	s = str(n).lower().strip()
	if s in {"", "normal", "null", "base", "zeros", "multishot+ricochet", "multishot+shoot_walls"} or s in RULE_TAG_TO_INDEX:
		return True
	if s in LEGACY_NULL_RULE_TAGS or s in {"rules_fast", "rule1", "rule2"}:
		return True
	if "_rules_" in s:
		tag = s.split("_rules_", 1)[1]
		return tag in LEGACY_NULL_RULE_TAGS or tag in RULE_TAG_TO_INDEX
	return False


def _rule_menu_key_to_label() -> dict[str, str]:
	"""Interactive menu: 1=null, 2..(1+len(RULE_TAGS))=each tag in order, last key=multishot+ricochet combo."""
	out: dict[str, str] = {}
	out["1"] = "null"
	key_i = 2
	for tag in RULE_TAGS:
		out[str(key_i)] = tag
		key_i += 1
	out[str(key_i)] = "multishot+ricochet"
	for digit_key, label in list(out.items()):
		out["k" + digit_key] = label
	return out


RULE_KEY_TO_LABEL = _rule_menu_key_to_label()

_RULE_MENU_MAX_KEY = 1 + len(RULE_TAGS) + 1  # null + each tag + combo


def _rule_panel_vec_labels_ordered() -> list[str]:
	"""Rule HUD / digit order: null, each ``RULE_TAGS``, then multishot+ricochet."""
	return ["null"] + list(RULE_TAGS) + ["multishot+ricochet", "multishot+shoot_walls"]


def _panel_rule_id(rule_name: str) -> str:
	"""Canonical label matching :func:`_rule_panel_vec_labels_ordered` entries."""
	s = str(rule_name).lower().strip()
	if s in {"", "normal", "null", "base", "zeros"}:
		return "null"
	if s in LEGACY_NULL_RULE_TAGS or s == "rules_fast":
		return "null"
	legacy = {"rule1": "null", "rule2": "null"}
	s = legacy.get(s, s)
	if s == "normal":
		return "null"
	if s in {"multishot+ricochet", "rule3+rule4", "combo34"}:
		return "multishot+ricochet"
	if s in {"multishot+shoot_walls", "rule3+rule5", "combo35"}:
		return "multishot+shoot_walls"
	if s in RULE_TAG_TO_INDEX:
		return s
	if "_rules_" in s:
		tag = s.split("_rules_", 1)[1]
		if tag in LEGACY_NULL_RULE_TAGS:
			return "null"
		if tag in RULE_TAG_TO_INDEX:
			return tag
	return "null"


def _rule_square_caption(vec_lab: str) -> str:
	"""Two-line text inside a rule tile (digit hint added separately)."""
	if vec_lab == "null":
		return "null"
	if vec_lab == "multishot+ricochet":
		return "multishot\n+ ricochet"
	if vec_lab == "multishot+shoot_walls":
		return "multishot\n+ shoot_walls"
	if "_" in vec_lab:
		a, _, b = vec_lab.partition("_")
		return f"{a}\n{b}"
	return vec_lab


def _rule_menu_help_lines() -> list[str]:
	lines = ["  1  null (base game / no rule tag — all-zero conditioning)"]
	i = 2
	for tag in RULE_TAGS:
		lines.append(f"  {i}  {tag}")
		i += 1
	lines.append(f"  {i}  multishot+ricochet (multi-hot combo)")
	return lines


def run_autoregressive(
	ckpt_dir: str,
	env: str = "aliens",
	num_actions: int = 7,
	history_len: int = 2,
	num_inference_steps: int = 30,
	bootstrap_start_idx: int = 0,
	vae_checkpoint: Optional[str] = None,
	rule: str = "null",
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
) -> None:
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	current_rule = str(rule).lower().strip()
	if not _is_valid_rule_name(current_rule):
		current_rule = "null"
	rule_oh = _rule_vec(current_rule).to(device)
	wm = load_world_model(
		Path(ckpt_dir), num_actions, K,
		vae_checkpoint=vae_checkpoint,
		cfg_scale_action=cfg_scale_action,
		cfg_scale_rule=cfg_scale_rule,
	)
	wm.eval()

	def tx(frame: np.ndarray) -> torch.Tensor:
		rgb = np.asarray(frame)[..., -3:]
		h, w = crop_hw_div8(*rgb.shape[:2])
		return IMG_TRANSFORMS(rgb[:h, :w])

	transitions_root = Path("data") / "transitions" / "train" / str(env)
	shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists())
	assert shards, f"No shards under {transitions_root}"
	obs = np.load(shards[0] / "obs.npy", mmap_mode="r")
	acts = np.load(shards[0] / "action.npy", mmap_mode="r")
	n = int(obs.shape[0])
	start = int(bootstrap_start_idx)
	if start < 0:
		raise ValueError(f"--bootstrap_start_idx must be >= 0, got {start}")
	if start + K > n:
		raise ValueError(f"Need at least start+K <= {n}; got start={start}, K={K}")

	# Bootstrap: K real frames starting from ``start`` (same preprocessing as training).
	init = torch.stack([tx(np.asarray(obs[start + i])[..., -3:]) for i in range(K)], dim=0)
	gen_hist: deque[torch.Tensor] = deque([init[i].clone() for i in range(K)], maxlen=K)
	action_hist: deque[int] = deque([int(a) for a in acts[start : start + K]], maxlen=K)
	data_pos = start

	fig = plt.figure(figsize=(7.6, 8.0))
	gs = gridspec.GridSpec(4, 1, height_ratios=[0.34, 1.0, 0.05, 0.34], hspace=0.015, figure=fig)
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
	preview = tx(np.asarray(obs[start])[..., -3:])
	h, w = int(preview.shape[-2]), int(preview.shape[-1])
	img_top = ax_top.imshow(np.zeros((h, w, 3), dtype=np.float32), vmin=0, vmax=1, interpolation="nearest")
	status = step_ax.text(
		0.5, 0.5, f"step={data_pos}",
		ha="center", va="center", fontsize=13, color="#212529",
	)

	# ── Rule tiles (top of HUD): same order / digits as CLI menu ─────────────────
	panel_rules = _rule_panel_vec_labels_ordered()
	n_rule_tiles = len(panel_rules)
	ncols_r = 3
	nrows_r = (n_rule_tiles + ncols_r - 1) // ncols_r
	rx0, ry0, rx1, ry1 = 0.02, 0.08, 0.98, 0.94
	padx_r, pady_r = 0.016, 0.03
	cell_rw = (rx1 - rx0 - padx_r * (ncols_r + 1)) / ncols_r
	cell_rh = (ry1 - ry0 - pady_r * (nrows_r + 1)) / nrows_r
	_rule_gray = "#E9ECEF"
	_rule_gray_edge = "#6C757D"
	_rule_hi = "#4C6EF5"
	_rule_hi_edge = "#364FC7"
	rule_tile_rects: list[tuple[Rectangle, str]] = []
	rule_tile_bounds: list[tuple[float, float, float, float, str]] = []
	for i, vec_lab in enumerate(panel_rules):
		row = i // ncols_r
		col = i % ncols_r
		x = rx0 + padx_r + col * (cell_rw + padx_r)
		y_top = ry1 - pady_r - row * (cell_rh + pady_r)
		y = y_top - cell_rh
		rect = Rectangle(
			(x, y), cell_rw, cell_rh, linewidth=1.8, edgecolor=_rule_gray_edge, facecolor=_rule_gray,
		)
		rule_ax.add_patch(rect)
		rule_tile_rects.append((rect, vec_lab))
		rule_tile_bounds.append((x, y, cell_rw, cell_rh, vec_lab))
		rule_ax.text(
			x + cell_rw / 2.0, y + cell_rh * 0.38, _rule_square_caption(vec_lab),
			ha="center", va="center", fontsize=8, color="#212529",
		)

	def _paint_rule_hud(selected_id: str) -> None:
		for rect, vec_lab in rule_tile_rects:
			if vec_lab == selected_id:
				rect.set_facecolor(_rule_hi)
				rect.set_edgecolor(_rule_hi_edge)
			else:
				rect.set_facecolor(_rule_gray)
				rect.set_edgecolor(_rule_gray_edge)

	def _apply_rule_vec_lab(vec_lab: str) -> None:
		nonlocal current_rule, rule_oh
		current_rule = vec_lab
		rule_oh = _rule_vec(current_rule).to(device)
		_paint_rule_hud(_panel_rule_id(current_rule))

	_paint_rule_hud(_panel_rule_id(current_rule))

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
		for x, y, rw, rh, vec_lab in rule_tile_bounds:
			if x <= xd <= x + rw and y <= yd <= y + rh:
				_apply_rule_vec_lab(vec_lab)
				fig.canvas.draw_idle()
				return

	def on_key(event):
		nonlocal data_pos
		rule_next = RULE_KEY_TO_LABEL.get(event.key or "")
		if rule_next is not None:
			_apply_rule_vec_lab(rule_next)
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
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit_encoded_rules_all_env_adv_cfg_normal" / "step_0240000"))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="world_model/checkpoints/vae/vae.pt",
		help="Path to vae.pt (empty lets trainer_state decide).",
	)
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=4, help="History length K (same as training).")
	p.add_argument(
		"--bootstrap_start_idx",
		type=int,
		default=10,
		help="Initial frame index used for bootstrap context. 0 keeps current behavior; 30 starts from frame 30.",
	)
	p.add_argument(
		"--rule",
		type=str,
		default="null",
		help="Rule multi-hot preset: null/normal/base (zeros) or any RULE_TAGS name (see world_model/dataset.py), "
		"or multishot+ricochet. Folder-style `game_rules_tag` also works.",
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
	rule_name = str(args.rule).strip().lower()
	if not _is_valid_rule_name(rule_name):
		rule_name = "null"
	print("Rule preset (must match training RULE_TAGS / folder suffixes). Pick one:")
	for line in _rule_menu_help_lines():
		print(line)
	print(f"(During the matplotlib window, digit keys 1–{_RULE_MENU_MAX_KEY} switch rule the same way.)")
	choice = input(f"Rule [1-{_RULE_MENU_MAX_KEY}]: ").strip()
	if choice not in RULE_KEY_TO_LABEL:
		raise ValueError(f"Choose an integer 1–{_RULE_MENU_MAX_KEY} (see list above).")
	rule_name = RULE_KEY_TO_LABEL[choice]
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		vae_checkpoint=args.vae_checkpoint.strip() or None,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		history_len=args.context_len,
		bootstrap_start_idx=args.bootstrap_start_idx,
		rule=rule_name,
		cfg_scale_action=args.cfg_scale_action,
		cfg_scale_rule=args.cfg_scale_rule,
	)


if __name__ == "__main__":
	main()
