from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from world_model.model.world_model import WorldModel


def load_world_model(ckpt_dir: Path, num_actions: int, buffer_size: int, model_size: str, num_train_timesteps: int, context_noise_max: float = 0.7, context_noise_buckets: int = 10) -> WorldModel:
	ckpt_dir = Path(ckpt_dir)
	wm = WorldModel(
		action_embedding_dim=num_actions,
		wan_vae_dir=Path("world_model") / "checkpoints" / "vae",
		latent_channels=16,
		buffer_size=buffer_size,
		cross_attention_dim=768,
		num_train_timesteps=num_train_timesteps,
		prediction_type="v_prediction",
		model_size=model_size,
		gradient_checkpointing=False,
		context_noise_max=context_noise_max,
		context_noise_buckets=context_noise_buckets,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = wm.to(device)
	# Load diffuser weights (unet.pt, action_embedding.pt)
	unet_path = ckpt_dir / "unet.pt"
	act_path = ckpt_dir / "action_embedding.pt"
	if unet_path.exists():
		wm.diffuser.unet.load_state_dict(torch.load(unet_path, map_location=device))
	if act_path.exists():
		wm.diffuser.action_embedding.load_state_dict(torch.load(act_path, map_location=device))
	return wm


def decode_to_image(world_model: WorldModel, z_frame: torch.Tensor, device: torch.device) -> np.ndarray:
	with torch.no_grad():
		img = world_model.decode_frame(z_frame.unsqueeze(0), device=device)[0]  # [3,H,W]
		img = ((img.clamp(-1, 1) + 1.0) * 0.5).cpu().permute(1, 2, 0).numpy()  # [H,W,3] in [0,1]
	return img


def run_autoregressive(
	ckpt_dir: str,
	num_actions: int = 18,
	buffer_size: int = 6,
	model_size: str = "base",
	num_train_timesteps: int = 1000,
	initial_context: Optional[np.ndarray] = None,
	image_size: tuple[int, int] = (210, 160),
	context_noise_max: float = 0.7,
	context_noise_buckets: int = 10,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	world_model = load_world_model(Path(ckpt_dir), num_actions, buffer_size, model_size, num_train_timesteps, context_noise_max, context_noise_buckets)
	world_model.eval()

	# Initialize context frames: use black frames if none provided
	h, w = image_size
	if initial_context is None:
		init = np.zeros((buffer_size, 3, h, w), dtype=np.float32)  # [-1,1] domain
	else:
		init = initial_context.astype(np.float32)  # assume [T,3,H,W] in [-1,1]
		if init.shape[0] < buffer_size:
			pad = np.zeros((buffer_size - init.shape[0], 3, h, w), dtype=np.float32)
			init = np.concatenate([pad, init], axis=0)
	context = deque([torch.from_numpy(f) for f in init], maxlen=buffer_size)
	actions_hist = deque([0 for _ in range(buffer_size)], maxlen=buffer_size)

	# UI with matplotlib
	fig, ax = plt.subplots(figsize=(4, 4))
	ax.axis("off")
	img_disp = ax.imshow(np.zeros((h, w, 3), dtype=np.float32), interpolation="nearest")

	# Action keyboard mapping (customize as needed)
	key_to_action: Dict[str, int] = {
		"up": 2,
		"down": 5,
		"left": 4,
		"right": 3,
		" ": 0,  # noop on space
		"z": 1,  # fire
		"x": 6,  # fire+right (example)
		"c": 7,  # fire+left (example)
	}
	current_action = {"a": 0}

	def on_key(event):
		key = event.key
		if key in key_to_action:
			current_action["a"] = key_to_action[key]
		else:
			current_action["a"] = 0

	fig.canvas.mpl_connect("key_press_event", on_key)

	def gen_next_frame():
		# Prepare context batch [1, BUF, 3, H, W] in [-1,1]
		ctx = torch.stack(list(context), dim=0)  # [BUF,3,H,W]
		ctx = ctx.unsqueeze(0)  # [1,BUF,3,H,W]
		acts = torch.tensor(list(actions_hist), dtype=torch.long).unsqueeze(0)  # [1,BUF]
		with torch.no_grad():
			# Encode context to latents
			z_ctx_btchw = world_model.encode_video(ctx, device=device)  # [1,16,BUF,h',w']
			z_ctx = z_ctx_btchw.permute(0, 2, 1, 3, 4).contiguous().squeeze(0)  # [BUF,16,h',w']
		# Target latent initialization
		z_tgt = torch.zeros_like(z_ctx[-1], device=device)  # [16,h',w']
		# One-step denoising-style prediction for demo
		with torch.no_grad():
			pred_noise, _ = world_model.diffusion_forward(z_ctx.unsqueeze(0), z_tgt.unsqueeze(0), acts.to(device))
		z_next = pred_noise  # [1,16,h',w']
		frame_np = decode_to_image(world_model, z_next.squeeze(0), device)
		# Update context with decoded frame
		frame_t = torch.from_numpy((frame_np * 2.0 - 1.0).astype(np.float32)).permute(2, 0, 1)  # [3,H,W]
		context.append(frame_t)
		actions_hist.append(current_action["a"])
		return frame_np

	def update(_):
		frame = gen_next_frame()
		img_disp.set_data(frame)
		return (img_disp,)

	anim = animation.FuncAnimation(fig, update, interval=50, blit=True)
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser(description="Autoregressive inference with keyboard-controlled actions.")
	p.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint dir containing unet.pt, action_embedding.pt")
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--buffer_size", type=int, default=6)
	p.add_argument("--model_size", type=str, choices=["small", "base", "large"], default="base")
	p.add_argument("--num_train_timesteps", type=int, default=1000)
	p.add_argument("--height", type=int, default=210)
	p.add_argument("--width", type=int, default=160)
	p.add_argument("--context_noise_max", type=float, default=0.7)
	p.add_argument("--context_noise_buckets", type=int, default=10)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		num_actions=args.num_actions,
		buffer_size=args.buffer_size,
		model_size=args.model_size,
		num_train_timesteps=args.num_train_timesteps,
		initial_context=None,
		image_size=(args.height, args.width),
		context_noise_max=args.context_noise_max,
		context_noise_buckets=args.context_noise_buckets,
	)


if __name__ == "__main__":
	main()
