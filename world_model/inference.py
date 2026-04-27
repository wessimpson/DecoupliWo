"""Interactive inference with user-key actions and generated frames only."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from matplotlib.patches import Rectangle

from world_model.dataset import IMG_TRANSFORMS, crop_hw_div8
from world_model.dataset import NUM_RULE_TYPES
from world_model.model.world_model import WorldModel


def _load_sd(path: Path, device: torch.device) -> dict:
	return torch.load(path, map_location=device, weights_only=True)


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

	unet_path = ckpt_dir / "unet.pt"
	if not unet_path.is_file():
		raise FileNotFoundError(f"Missing UNet weights: {unet_path}")
	try:
		wm.diffuser.unet.load_state_dict(_load_sd(unet_path, device), strict=True)
	except RuntimeError as e:
		raise RuntimeError(
			f"UNet load failed from {unet_path}. "
			f"Ensure unet.pt matches this architecture (full SD1.4 UNet2D + widened conv_in)."
		) from e
	emb_path = ckpt_dir / "action_embedding.pt"
	if not emb_path.exists():
		emb_path = ckpt_dir / "future_action_embedding.pt"
	if emb_path.exists():
		wm.diffuser.action_embedding.load_state_dict(_load_sd(emb_path, device))
	rule_path = ckpt_dir / "rule_projection.pt"
	if rule_path.is_file():
		wm.diffuser.rule_projection.load_state_dict(_load_sd(rule_path, device))
	else:
		with torch.no_grad():
			wm.diffuser.rule_projection.weight.zero_()
	frame_state_path = ckpt_dir / "frame_state_encoder.pt"
	if frame_state_path.is_file():
		wm.diffuser.frame_state_encoder.load_state_dict(_load_sd(frame_state_path, device))
	state_proj_path = ckpt_dir / "state_token_projection.pt"
	if state_proj_path.is_file():
		wm.diffuser.state_token_projection.load_state_dict(_load_sd(state_proj_path, device))
	state_enc_path = ckpt_dir / "state_encoder.pt"
	if state_enc_path.is_file():
		# Legacy alias (same module as frame_state_encoder).
		wm.state_encoder.load_state_dict(_load_sd(state_enc_path, device))
	rule_adv_path = ckpt_dir / "rule_adversary.pt"
	if rule_adv_path.is_file():
		wm.rule_adversary.load_state_dict(_load_sd(rule_adv_path, device))
	for legacy_name in ("action_mlp.pt", "future_action_mlp.pt"):
		legacy = ckpt_dir / legacy_name
		if legacy.is_file():
			raise RuntimeError(
				f"Checkpoint has legacy {legacy_name} (MLP action head). Retrain with the current embedding-only diffuser."
			)
	return wm


def _tensor_to_imshow01(t: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] → [H,W,3] float32 [0,1]."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype(np.float32)


def _rule_vec(name: str) -> torch.Tensor:
	"""[1, NUM_RULE_TYPES] float rule vector: one-hot or combined presets."""
	n = str(name).lower().strip()
	v = torch.zeros(1, NUM_RULE_TYPES, dtype=torch.float32)
	if n in {"multishot+ricochet", "rule3+rule4", "combo34", "5"}:
		v[0, 2] = 1.0
		v[0, 3] = 1.0
		return v
	idx = {"normal": 0, "fast": 1, "rules_fast": 1, "multishot": 2, "ricochet": 3}.get(n, 0)
	v[0, idx] = 1.0
	return v


RULE_LABELS = ("normal", "fast", "multishot", "ricochet", "multishot+ricochet")
RULE_KEY_TO_LABEL = {
	"1": "normal",
	"2": "fast",
	"3": "multishot",
	"4": "ricochet",
	"5": "multishot+ricochet",
}


def run_autoregressive(
	ckpt_dir: str,
	env: str = "aliens",
	num_actions: int = 7,
	history_len: int = 2,
	num_inference_steps: int = 30,
	bootstrap_start_idx: int = 0,
	vae_checkpoint: Optional[str] = None,
	rule: str = "normal",
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
) -> None:
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	current_rule = str(rule).lower().strip()
	if current_rule not in RULE_LABELS:
		current_rule = "normal"
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

	fig = plt.figure(figsize=(7.2, 6.0))
	gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.08, 0.30], hspace=0.01, figure=fig)
	ax_top = fig.add_subplot(gs[0])
	step_ax = fig.add_subplot(gs[1])
	text_ax = fig.add_subplot(gs[2])
	for ax in (ax_top, step_ax, text_ax):
		ax.axis("off")
	step_ax.set_xlim(0, 1)
	step_ax.set_ylim(0, 1)
	text_ax.set_xlim(0, 1)
	text_ax.set_ylim(0, 1)
	preview = tx(np.asarray(obs[start])[..., -3:])
	h, w = int(preview.shape[-2]), int(preview.shape[-1])
	img_top = ax_top.imshow(np.zeros((h, w, 3), dtype=np.float32), vmin=0, vmax=1, interpolation="nearest")
	ax_top.set_title(f"Rule: {current_rule}", fontsize=14, pad=8)
	status = step_ax.text(
		0.5, 0.5, f"step={data_pos}",
		ha="center", va="center", fontsize=14, color="#212529",
	)
	key_to_action = {
		"up": 1,
		"left": 2,
		"down": 3,
		"right": 4,
		" ": 5,
		"space": 5,
	}
	key_w, key_h = 0.20, 0.30
	bottom_row_y = 0.10
	top_row_y = bottom_row_y + key_h  # zero vertical margin to bottom row
	left_col_x = 0.06
	down_col_x = left_col_x + key_w  # zero horizontal margin
	right_col_x = down_col_x + key_w  # zero horizontal margin
	fire_col_x = right_col_x + key_w + 0.08  # add left margin before FIRE
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

	def on_key(event):
		nonlocal data_pos, current_rule, rule_oh
		rule_next = RULE_KEY_TO_LABEL.get(event.key or "")
		if rule_next is not None:
			current_rule = rule_next
			rule_oh = _rule_vec(current_rule).to(device)
			ax_top.set_title(f"Rule: {current_rule}", fontsize=14, pad=8)
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
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit_encoded_rules_all_env_adv" / "step_0150000"))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="world_model/checkpoints/vae/vae.pt",
		help="Path to vae.pt (empty = default world_model/checkpoints/vae/vae.pt)",
	)
	p.add_argument("--env", type=str, default="defender")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=4, help="History length K (same as training).")
	p.add_argument(
		"--bootstrap_start_idx",
		type=int,
		default=0,
		help="Initial frame index used for bootstrap context. 0 keeps current behavior; 30 starts from frame 30.",
	)
	p.add_argument(
		"--rule",
		type=str,
		default="normal",
		help="Rule conditioning preset: normal | fast | multishot | ricochet | multishot+ricochet.",
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
	if rule_name not in RULE_LABELS:
		rule_name = "normal"
	print("Select initial rule condition: 1=normal, 2=fast, 3=multishot, 4=ricochet, 5=rule3+rule4")
	choice = input("Rule [1-5]: ").strip()
	if choice not in RULE_KEY_TO_LABEL:
		raise ValueError("please select proper rule id")
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
