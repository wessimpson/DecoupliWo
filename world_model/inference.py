from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation

from world_model.model.world_model import WorldModel


def _load_state_dict(path: Path, device: torch.device) -> dict:
	return torch.load(path, map_location=device, weights_only=True)


def load_world_model(ckpt_dir: Path, num_actions: int, buffer_size: int, model_size: str, num_train_timesteps: int, context_noise_max: float = 0.7, context_noise_buckets: int = 10, wan_vae_dir: Path | None = None) -> WorldModel:
	ckpt_dir = Path(ckpt_dir)
	if wan_vae_dir is None:
		wan_vae_dir = Path("world_model") / "checkpoints" / "vae"
	wm = WorldModel(
		action_embedding_dim=num_actions,
		wan_vae_dir=wan_vae_dir,
		latent_channels=16,
		buffer_size=buffer_size,
		cross_attention_dim=768,
		num_train_timesteps=num_train_timesteps,
		prediction_type="epsilon",
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
	noise_level_path = ckpt_dir / "noise_level_embedding.pt"
	sched_dir = ckpt_dir / "noise_scheduler"
	if unet_path.exists():
		wm.diffuser.unet.load_state_dict(_load_state_dict(unet_path, device))
	if act_path.exists():
		wm.diffuser.action_embedding.load_state_dict(_load_state_dict(act_path, device))
	if noise_level_path.exists():
		wm.diffuser.noise_level_embedding.load_state_dict(_load_state_dict(noise_level_path, device))
	else:
		# Older checkpoints did not save this embedding; zero it so inference is not driven by random conditioning.
		wm.diffuser.noise_level_embedding.weight.data.zero_()
	if sched_dir.exists():
		# Load scheduler config if present
		wm.diffuser.noise_scheduler = wm.diffuser.noise_scheduler.from_pretrained(str(sched_dir))
		wm.diffuser.noise_scheduler.register_to_config(prediction_type="epsilon")
	return wm


def decode_to_image(
	world_model: WorldModel,
	z_frame: torch.Tensor,
	device: torch.device,
	image_size: tuple[int, int],
) -> np.ndarray:
	with torch.no_grad():
		img = world_model.decode_frame(z_frame.unsqueeze(0), device=device)[0]  # [3,H,W]
		if tuple(img.shape[-2:]) != tuple(image_size):
			img = F.interpolate(img.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False).squeeze(0)
		img = ((img.clamp(-1, 1) + 1.0) * 0.5).cpu().permute(1, 2, 0).numpy()  # [H,W,3] in [0,1]
	return img


def run_autoregressive(
	ckpt_dir: str,
	env: str = "space_invaders",
	num_actions: int = 18,
	buffer_size: int = 8,
	model_size: str = "base",
	num_train_timesteps: int = 1000,
	num_inference_steps: int = 50,
	initial_context: Optional[np.ndarray] = None,
	image_size: tuple[int, int] = (210, 160),
	context_noise_max: float = 0.7,
	context_noise_buckets: int = 10,
	context_noise_alpha: float = 0.0,
	wan_vae_dir: Optional[str] = None,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	world_model = load_world_model(
		Path(ckpt_dir),
		num_actions,
		buffer_size,
		model_size,
		num_train_timesteps,
		context_noise_max,
		context_noise_buckets,
		wan_vae_dir=Path(wan_vae_dir) if wan_vae_dir is not None else None,
	)
	world_model.eval()
	# Configure scheduler for inference steps (separate from training timesteps)
	world_model.diffuser.noise_scheduler.set_timesteps(num_inference_steps)
	world_model.num_train_timesteps = int(num_inference_steps)

	# Initialize context frames from first shard of env transitions
	h, w = image_size
	try:
		transitions_root = Path("data") / "transitions" / str(env)
		shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists() and (p / "action.npy").exists())
		if not shards:
			raise FileNotFoundError(f"No shards under {transitions_root}")
		first = shards[0]
		obs = np.load(first / "obs.npy", mmap_mode="r")
		acts = np.load(first / "action.npy", mmap_mode="r")
		if obs.shape[0] < buffer_size:
			raise ValueError(f"Not enough frames in first shard: {obs.shape[0]} < {buffer_size}")
		if acts.shape[0] < buffer_size:
			raise ValueError(f"Not enough actions in first shard: {acts.shape[0]} < {buffer_size}")
		frames_np = np.array(obs[:buffer_size], copy=True)
		frames = torch.from_numpy(frames_np).float()  # [T,H,W,3], 0..255
		frames = frames.permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
		frames = frames / 127.5 - 1.0  # [-1,1]
		if (frames.shape[-2], frames.shape[-1]) != (h, w):
			frames = F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)
		context = deque([frames[t] for t in range(buffer_size)], maxlen=buffer_size)
	except FileNotFoundError:
		raise FileNotFoundError(f"No transitions found for env {env}")
	# Seed action history aligned with context frames: a_t transitions f_t -> f_{t+1}
	# Here we take the first buffer_size actions for the first buffer_size frames context
	actions_hist = deque([int(a) for a in acts[:buffer_size]], maxlen=buffer_size)

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
		raw_actions = list(actions_hist)  # length = buffer_size
		with torch.no_grad():
			# Encode context to latents
			z_ctx_btchw = world_model.encode_video(ctx, device=device)  # [1,16,BUF,h',w']
			z_ctx = z_ctx_btchw.permute(0, 2, 1, 3, 4).contiguous().squeeze(0)  # [T_eff,16,h',w']
		# Downsample action history to match effective temporal length of VAE latents
		T_eff = z_ctx.shape[0]
		if T_eff <= 0:
			raise RuntimeError("Empty context latents after VAE encoding")
		if T_eff == len(raw_actions):
			acts = torch.tensor(raw_actions, dtype=torch.long).unsqueeze(0).to(device)  # [1,T_eff]
		else:
			# Chunk actions into T_eff nearly-equal groups and take the last action from each group
			acts_eff = []
			n = len(raw_actions)
			for i in range(T_eff):
				start = int(round(i * n / T_eff))
				end = int(round((i + 1) * n / T_eff))
				end = max(end, start + 1)
				acts_eff.append(raw_actions[end - 1])
			acts = torch.tensor(acts_eff, dtype=torch.long).unsqueeze(0).to(device)  # [1,T_eff]

		# Proper diffusion denoising loop to sample next latent
		with torch.no_grad():
			B = 1
			Tbuf, C, Hh, Ww = z_ctx.shape
			# Inference should default to clean context; older checkpoints are especially sensitive to random noise tokens.
			alpha = torch.full((B,), float(context_noise_alpha), device=device, dtype=z_ctx.dtype)
			alpha = alpha.clamp(0.0, float(world_model.context_noise_max))
			alpha_b = alpha.view(B, 1, 1, 1, 1)
			ctx_eps = torch.randn_like(z_ctx.unsqueeze(0), dtype=z_ctx.dtype)  # [1,BUF,16,h',w']
			z_ctx_noisy = z_ctx.unsqueeze(0) + alpha_b * ctx_eps  # [1,BUF,16,h',w']

			# Initial target latent noise scaled by scheduler sigma
			latent_init = torch.randn((B, C, Hh, Ww), device=device, dtype=world_model.diffuser.unet.dtype)
			latent_init = latent_init * world_model.diffuser.noise_scheduler.init_noise_sigma

			# Match training conditioning: one noise token followed by action tokens.
			den = max(world_model.context_noise_max, 1e-8)
			bucket_idx = (alpha / den * world_model.context_noise_buckets).clamp(0, world_model.context_noise_buckets - 1).long()  # [B]
			act_cond = world_model.diffuser.action_embedding(acts).to(world_model.diffuser.unet.dtype)  # [1,T_eff,dim]
			noise_token = world_model.diffuser.noise_level_embedding(bucket_idx).unsqueeze(1).to(world_model.diffuser.unet.dtype)  # [B,1,dim]
			enc_states = torch.cat([noise_token, act_cond], dim=1)  # [B,T_eff+1,dim]

			# Prepare frames tensor [B, T+1, C, H, W]
			frames = torch.cat([z_ctx_noisy, latent_init.unsqueeze(1)], dim=1).to(world_model.diffuser.unet.dtype)  # [1,BUF+1,16,h',w']

			# Denoising loop updates only the last frame
			sched = world_model.diffuser.noise_scheduler
			for t in sched.timesteps:
				latents_in = frames.view(B, (Tbuf + 1) * C, Hh, Ww).contiguous()
				latents_in = sched.scale_model_input(latents_in, t)
				noise_pred = world_model.diffuser.unet(
					latents_in, t, encoder_hidden_states=enc_states, return_dict=False
				)[0]  # [B,C,H,W] target-frame prediction only
				last = frames[:, -1]  # [B,C,H,W]
				last = sched.step(noise_pred, t, last, return_dict=False)[0]
				frames[:, -1] = last  # keep context frames fixed

			z_next = frames[:, -1]  # [B,16,h',w']

		frame_np = decode_to_image(world_model, z_next.squeeze(0), device, image_size=(h, w))
		# Update context with decoded frame
		frame_t = torch.from_numpy((frame_np * 2.0 - 1.0).astype(np.float32)).permute(2, 0, 1)  # [3,H,W]
		context.append(frame_t)
		actions_hist.append(current_action["a"])
		return frame_np

	def update(_):
		frame = gen_next_frame()
		img_disp.set_data(frame)
		return (img_disp,)

	anim = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
	plt.tight_layout()
	plt.show()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser(description="Autoregressive inference with keyboard-controlled actions.")
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit" / "step_0003000"))
	p.add_argument("--wan_vae_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--num_inference_steps", type=int, default=50)

	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--buffer_size", type=int, default=8)
	p.add_argument("--model_size", type=str, choices=["small", "base", "large"], default="base")
	p.add_argument("--num_train_timesteps", type=int, default=1000)
	p.add_argument("--height", type=int, default=210)
	p.add_argument("--width", type=int, default=160)
	p.add_argument("--context_noise_max", type=float, default=0.7)
	p.add_argument("--context_noise_buckets", type=int, default=10)
	p.add_argument("--context_noise_alpha", type=float, default=0.0)
	args = p.parse_args()
	run_autoregressive(
		ckpt_dir=args.ckpt_dir,
		env=args.env,
		wan_vae_dir=args.wan_vae_dir,
		num_actions=args.num_actions,
		num_inference_steps=args.num_inference_steps,
		
		buffer_size=args.buffer_size,
		model_size=args.model_size,
		num_train_timesteps=args.num_train_timesteps,
		initial_context=None,
		image_size=(args.height, args.width),
		context_noise_max=args.context_noise_max,
		context_noise_buckets=args.context_noise_buckets,
		context_noise_alpha=args.context_noise_alpha,
	)


if __name__ == "__main__":
	main()
