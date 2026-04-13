"""Interactive chunk-based autoregressive inference with keyboard control."""

from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

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


def load_world_model(
	ckpt_dir: Path,
	num_actions: int,
	history_len: int,
	chunk_len: int,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
	trainable_parts: Optional[str] = None,
	unet_top_n_blocks: Optional[int] = None,
	lora_rank: Optional[int] = None,
	lora_alpha: Optional[float] = None,
	lora_include_motion: Optional[bool] = None,
) -> WorldModel:
	"""Load weights from ``ckpt_dir`` (same layout as :meth:`WorldModel.save_diffuser`).

	If ``trainer_state.pt`` exists (written by ``train_world_model``), finetuning options
	(``trainable_parts``, LoRA, ``unet_top_n_blocks``) default from saved args so the
	module graph matches ``unet.pt``. Pass explicit kwargs to override.
	"""
	ckpt_dir = Path(ckpt_dir)
	meta = _read_trainer_args(ckpt_dir)

	vae_eff = _coalesce(meta, "vae_checkpoint", vae_checkpoint, None)
	tp = _coalesce(meta, "trainable_parts", trainable_parts, "full")
	ut = _coalesce(meta, "unet_top_n_blocks", unet_top_n_blocks, 2)
	lr = _coalesce(meta, "lora_rank", lora_rank, 8)
	la = _coalesce(meta, "lora_alpha", lora_alpha, 8.0)
	lm = _coalesce(meta, "lora_include_motion", lora_include_motion, False)

	wm = WorldModel(
		num_actions=num_actions,
		cross_attention_dim=768,
		vae_checkpoint=vae_eff,
		prediction_type="epsilon",
		history_len=history_len,
		chunk_len=chunk_len,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		trainable_parts=tp,
		unet_top_n_blocks=int(ut),
		lora_rank=int(lr),
		lora_alpha=float(la),
		lora_include_motion=bool(lm),
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
			f"UNet load failed (trainable_parts={tp!r}, lora_rank={lr}). "
			f"If this checkpoint used a different finetuning mode, pass matching "
			f"trainable_parts / lora_* to load_world_model, or ensure trainer_state.pt is beside unet.pt."
		) from e
	emb_path = ckpt_dir / "action_embedding.pt"
	if not emb_path.exists():
		emb_path = ckpt_dir / "future_action_embedding.pt"
	if emb_path.exists():
		wm.diffuser.action_embedding.load_state_dict(_load_sd(emb_path, device))
	mlp_path = ckpt_dir / "action_mlp.pt"
	if not mlp_path.exists():
		mlp_path = ckpt_dir / "future_action_mlp.pt"
	if mlp_path.exists():
		wm.diffuser.mlp.load_state_dict(_load_sd(mlp_path, device))
	return wm


def run_autoregressive(
	ckpt_dir: str,
	env: str = "space_invaders",
	num_actions: int = 18,
	history_len: int = 4,
	chunk_len: int = 4,
	num_inference_steps: int = 30,
	image_size: tuple[int, int] = (210, 160),
	vae_checkpoint: Optional[str] = None,
	trainable_parts: Optional[str] = None,
) -> None:
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(ckpt_dir), num_actions, history_len, chunk_len,
		vae_checkpoint=vae_checkpoint,
		trainable_parts=trainable_parts,
	)
	wm.eval()

	h, w = image_size
	transitions_root = Path("data") / "transitions" / "train" / str(env)
	shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists())
	assert shards, f"No shards under {transitions_root}"
	obs = np.load(shards[0] / "obs.npy", mmap_mode="r")
	acts = np.load(shards[0] / "action.npy", mmap_mode="r")
	assert obs.shape[0] >= history_len

	frames_np = np.array(obs[:history_len], copy=True)
	frames = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 127.5 - 1.0  # [K,3,H,W]
	if (frames.shape[-2], frames.shape[-1]) != (h, w):
		frames = F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)
	context = deque(list(frames.unbind(0)), maxlen=history_len)
	action_context = deque(acts[:history_len].astype(np.int64).tolist(), maxlen=history_len)
	pending_actions: list[int] = []

	fig = plt.figure(figsize=(4, 4.8))
	gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1, figure=fig)
	ax = fig.add_subplot(gs[0])
	ax.axis("off")
	text_ax = fig.add_subplot(gs[1])
	text_ax.axis("off")
	img_disp = ax.imshow(np.zeros((h, w, 3), dtype=np.float32), interpolation="nearest")
	status = text_ax.text(0.5, 0.5, "Press keys to queue actions", ha="center", va="center", fontsize=9)

	key_to_action: Dict[str, int] = {
		"up": 2, "down": 5, "left": 4, "right": 3,
		" ": 0, " z": 1, "x": 6, "c": 7,
	}
	step_idx = {"t": 0}

	def generate_chunk(action_ids: list[int]):
		"""Generate chunk_len frames, return list of [3,H,W] tensors."""
		ctx = torch.stack(list(context)).unsqueeze(0).to(device)     # [1,K,3,H,W]
		ha = torch.tensor(list(action_context), dtype=torch.long).unsqueeze(0).to(device)  # [1,K]
		fa = torch.tensor(action_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1,N]
		with torch.no_grad():
			z_hist = wm.encode_video(ctx)
			z_chunk = wm.generate_next_chunk(
				z_hist, ha, fa, num_inference_steps=num_inference_steps,
			)
			pixels = wm.decode_video(z_chunk)  # [1,N,3,H,W]
		return pixels[0]  # [N,3,H,W]

	def on_key(event):
		key = event.key
		a = key_to_action.get(key, 0)
		pending_actions.append(a)
		status.set_text(f"Queued: {len(pending_actions)}/{chunk_len}")
		fig.canvas.draw_idle()

		if len(pending_actions) >= chunk_len:
			ids = pending_actions[:chunk_len]
			chunk = generate_chunk(ids)
			for i, frame in enumerate(chunk.unbind(0)):
				frame_np = ((frame.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy()
				context.append(frame.cpu())
				action_context.append(ids[i])
				if i == len(chunk) - 1:
					img_disp.set_data(frame_np)
			step_idx["t"] += chunk_len
			pending_actions.clear()
			status.set_text(f"t={step_idx['t']}  (last chunk generated)")
			fig.canvas.draw_idle()

	fig.canvas.mpl_connect("key_press_event", on_key)
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit" / "step_0050000"))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="world_model/checkpoints/vae/vae.pt",
		help="Path to vae.pt (empty = default world_model/checkpoints/vae/vae.pt)",
	)
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--history_len", type=int, default=8)
	p.add_argument("--chunk_len", type=int, default=3)
	p.add_argument("--height", type=int, default=208)
	p.add_argument("--width", type=int, default=160)
	p.add_argument(
		"--trainable_parts",
		type=str,
		default=None,
		help="Override finetuning layout; omit to use trainer_state.pt next to ckpt_dir (if saved by train_world_model).",
	)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		vae_checkpoint=args.vae_checkpoint.strip() or None,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		history_len=args.history_len,
		chunk_len=args.chunk_len,
		image_size=(args.height, args.width),
		trainable_parts=args.trainable_parts,
	)


if __name__ == "__main__":
	main()
