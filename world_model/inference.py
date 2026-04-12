"""Interactive chunk-based autoregressive inference with keyboard control."""

from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from world_model.model.world_model import WorldModel


def _load_sd(path: Path, device: torch.device) -> dict:
	return torch.load(path, map_location=device, weights_only=True)


def load_world_model(
	ckpt_dir: Path,
	num_actions: int,
	history_len: int,
	chunk_len: int,
	vae_pretrained: str | Path | None = None,
	pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
) -> WorldModel:
	ckpt_dir = Path(ckpt_dir)
	local_vae = Path("world_model") / "checkpoints" / "vae"
	if vae_pretrained is None:
		vae_pretrained = str(local_vae) if (local_vae / "config.json").exists() else "stabilityai/sd-vae-ft-mse"

	wm = WorldModel(
		num_actions=num_actions,
		vae_pretrained=vae_pretrained,
		cross_attention_dim=768,
		prediction_type="epsilon",
		history_len=history_len,
		chunk_len=chunk_len,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = wm.to(device)

	if (ckpt_dir / "unet.pt").exists():
		wm.diffuser.unet.load_state_dict(_load_sd(ckpt_dir / "unet.pt", device))
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
	vae_pretrained: Optional[str] = None,
) -> None:
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(ckpt_dir), num_actions, history_len, chunk_len,
		vae_pretrained=vae_pretrained,
	)
	wm.eval()

	h, w = image_size
	transitions_root = Path("data") / "transitions" / str(env)
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
		" ": 0, "z": 1, "x": 6, "c": 7,
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
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit" / "step_0100000"))
	p.add_argument("--vae_pretrained", type=str, default="")
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--num_inference_steps", type=int, default=30)
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--history_len", type=int, default=4)
	p.add_argument("--chunk_len", type=int, default=4)
	p.add_argument("--height", type=int, default=210)
	p.add_argument("--width", type=int, default=160)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		vae_pretrained=args.vae_pretrained.strip() or None,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		history_len=args.history_len,
		chunk_len=args.chunk_len,
		image_size=(args.height, args.width),
	)


if __name__ == "__main__":
	main()
