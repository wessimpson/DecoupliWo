"""Replay inference: dataset actions only, Space advances one timestep.

Top: model prediction. Bottom: ground-truth next frame from the same rollout.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torchvision import transforms

from world_model.dataset import IMG_TRANSFORMS, _resize_hw_divisible_by_8
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
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
) -> WorldModel:
	"""Load weights from ``ckpt_dir`` (same layout as :meth:`WorldModel.save_diffuser`).

	If ``trainer_state.pt`` exists, ``vae_checkpoint`` defaults from saved args when omitted.
	"""
	ckpt_dir = Path(ckpt_dir)
	meta = _read_trainer_args(ckpt_dir)

	vae_eff = _coalesce(meta, "vae_checkpoint", vae_checkpoint, None)
	wm = WorldModel(
		num_actions=num_actions,
		cross_attention_dim=768,
		vae_checkpoint=vae_eff,
		prediction_type="epsilon",
		history_len=history_len,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
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
	ctx_path = ckpt_dir / "action_context.pt"
	if ctx_path.exists():
		wm.diffuser.action_context.load_state_dict(_load_sd(ctx_path, device))
	else:
		legacy = ckpt_dir / "action_mlp.pt"
		if not legacy.exists():
			legacy = ckpt_dir / "future_action_mlp.pt"
		if legacy.exists():
			raise RuntimeError(
				f"Checkpoint has legacy {legacy.name} (MLP action head); this build expects action_context.pt "
				f"(lightweight attention). Retrain or migrate weights."
			)
	return wm


def _tensor_to_imshow01(t: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] → [H,W,3] float32 [0,1]."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype(np.float32)


def run_autoregressive(
	ckpt_dir: str,
	env: str = "space_invaders",
	num_actions: int = 18,
	history_len: int = 8,
	num_inference_steps: int = 30,
	image_size: tuple[int, int] = (208, 160),
	vae_checkpoint: Optional[str] = None,
) -> None:
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(ckpt_dir), num_actions, K,
		vae_checkpoint=vae_checkpoint,
	)
	wm.eval()

	h, w = _resize_hw_divisible_by_8((int(image_size[0]), int(image_size[1])))
	tx = transforms.Compose([
		IMG_TRANSFORMS,
		transforms.Resize((h, w), antialias=True),
	])

	transitions_root = Path("data") / "transitions" / "train" / str(env)
	shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists())
	assert shards, f"No shards under {transitions_root}"
	obs = np.load(shards[0] / "obs.npy", mmap_mode="r")
	acts = np.load(shards[0] / "action.npy", mmap_mode="r")
	n = int(obs.shape[0])
	assert n >= K + 1, f"Need at least K+1={K+1} frames, got {n}"

	# Bootstrap: first K real frames (same preprocessing as training).
	init = torch.stack([tx(np.asarray(obs[i])[..., -3:]) for i in range(K)], dim=0)
	gen_hist: deque[torch.Tensor] = deque([init[i].clone() for i in range(K)], maxlen=K)
	data_pos = 0

	fig = plt.figure(figsize=(6, 7))
	gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.12], hspace=0.2, figure=fig)
	ax_top = fig.add_subplot(gs[0])
	ax_bot = fig.add_subplot(gs[1])
	text_ax = fig.add_subplot(gs[2])
	for ax in (ax_top, ax_bot, text_ax):
		ax.axis("off")
	img_top = ax_top.imshow(np.zeros((h, w, 3), dtype=np.float32), vmin=0, vmax=1, interpolation="nearest")
	img_bot = ax_bot.imshow(np.zeros((h, w, 3), dtype=np.float32), vmin=0, vmax=1, interpolation="nearest")
	ax_top.set_title("generated (model)", fontsize=10)
	ax_bot.set_title("ground truth (dataset)", fontsize=10)
	status = text_ax.text(
		0.5, 0.5, "Press Space to advance one timestep", ha="center", va="center", fontsize=10,
	)

	def _is_space(key: str | None) -> bool:
		return key in (" ", "space")

	def on_key(event):
		nonlocal data_pos
		if not _is_space(event.key):
			return
		if data_pos + K >= n:
			status.set_text("End of trajectory (no more frames).")
			fig.canvas.draw_idle()
			return

		ha = torch.from_numpy(acts[data_pos : data_pos + K].astype(np.int64)).view(1, K).to(device)
		a_step = int(acts[data_pos + K - 1])
		fa = torch.tensor([a_step], dtype=torch.long, device=device)

		ctx = torch.stack(list(gen_hist), dim=0).unsqueeze(0).to(device)
		gt_idx = data_pos + K
		gt_frame = tx(np.asarray(obs[gt_idx])[..., -3:]).cpu()

		with torch.no_grad():
			z_hist = wm.encode_video(ctx)
			z_next = wm.generate_next_frame(
				z_hist, ha, fa, num_inference_steps=num_inference_steps,
			)
			gen = wm.decode_video(z_next)[0, 0].cpu()

		img_top.set_data(_tensor_to_imshow01(gen))
		img_bot.set_data(_tensor_to_imshow01(gt_frame))

		gen_hist.append(gen.clone())
		data_pos += 1
		status.set_text(
			f"step={data_pos}  window_start={data_pos - 1}  dataset_action={a_step}  "
			f"frames_left={n - data_pos - K}",
		)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("key_press_event", on_key)
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit" / "step_0100000"))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="world_model/checkpoints/vae/vae.pt",
		help="Path to vae.pt (empty = default world_model/checkpoints/vae/vae.pt)",
	)
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--context_len", type=int, default=8, help="History length K (same as training).")
	p.add_argument("--height", type=int, default=208)
	p.add_argument("--width", type=int, default=160)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		vae_checkpoint=args.vae_checkpoint.strip() or None,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		history_len=args.context_len,
		image_size=(args.height, args.width),
	)


if __name__ == "__main__":
	main()
