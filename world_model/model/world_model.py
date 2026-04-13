"""
Chunk-based temporal world model.

Architecture: frozen SD VAE + temporalised SD1.5 UNet (AnimateDiff motion modules).
Training:    denoise a future latent chunk conditioned on history latents + past and future actions.
Inference:   generate a future chunk via iterative denoising.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from world_model.model.net import Diffuser, VAE
from world_model.model.net.vae import DEFAULT_VAE_PT


class WorldModel(nn.Module):
	def __init__(
		self,
		num_actions: int,
		cross_attention_dim: int,
		vae_checkpoint: str | Path | None = None,
		prediction_type: str = "epsilon",
		history_len: int = 4,
		chunk_len: int = 4,
		gradient_checkpointing: bool = False,
		pretrained_model_name_or_path: str | None = None,
		trainable_parts: str = "full",
		unet_top_n_blocks: int = 2,
		lora_rank: int = 8,
		lora_alpha: float = 8.0,
		lora_include_motion: bool = False,
	) -> None:
		super().__init__()
		self.history_len = history_len
		self.chunk_len = chunk_len

		pt = Path(DEFAULT_VAE_PT if vae_checkpoint is None else vae_checkpoint)
		self.vae = VAE(checkpoint=pt)
		self.vae.freeze()
		self.latent_channels = self.vae.latent_channels

		self.diffuser = Diffuser(
			num_actions=num_actions,
			latent_channels=self.latent_channels,
			cross_attention_dim=cross_attention_dim,
			prediction_type=prediction_type,
			pretrained_model_name_or_path=pretrained_model_name_or_path,
		)
		self.diffuser.configure_trainable(
			trainable_parts,
			unet_top_n_blocks=unet_top_n_blocks,
			lora_rank=lora_rank,
			lora_alpha=lora_alpha,
			lora_include_motion=lora_include_motion,
		)
		if gradient_checkpointing:
			self.diffuser.unet.enable_gradient_checkpointing()

		self.num_train_timesteps = int(self.diffuser.noise_scheduler.config.num_train_timesteps)

	def trainable_parameters(self):
		for p in self.diffuser.parameters():
			if p.requires_grad:
				yield p

	def enable_gradient_checkpointing(self) -> None:
		self.diffuser.unet.enable_gradient_checkpointing()

	# ── VAE helpers ───────────────────────────────────────────────

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,T,3,H,W] → [B,T,C,h,w] scaled latents (no grad)."""
		return self.vae.encode_video(pixels.to(next(self.vae.parameters()).device))

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,T,C,h,w] → [B,T,3,H,W] (no grad)."""
		return self.vae.decode_video(latents.to(next(self.vae.parameters()).device))

	def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,3,H,W] → [B,C,h,w] scaled latents (no grad)."""
		return self.vae.encode_pixels(pixels.to(next(self.vae.parameters()).device))

	def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,C,h,w] → [B,3,H,W] (no grad)."""
		return self.vae.decode_latents(latents.to(next(self.vae.parameters()).device))

	# ── Training forward ─────────────────────────────────────────

	def diffusion_forward(
		self,
		z_hist: torch.Tensor,
		z_tgt: torch.Tensor,
		history_actions: torch.Tensor,
		future_actions: torch.Tensor,
		timesteps: torch.Tensor,
		noise: torch.Tensor,
		delta_hist: torch.Tensor | None = None,
		gamma: float = 0.0,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		z_hist:          [B, K, C, h, w] clean history latents
		z_tgt:           [B, N, C, h, w] clean target latents
		history_actions: [B, K]  actions aligned with history frames
		future_actions:  [B, N]  actions for the target chunk
		timesteps:       [B]
		noise:           [B, N, C, h, w]
		delta_hist:      [B, K, C, h, w] | None
		gamma:           corruption scale

		Returns (model_pred, noise) both [B, N, C, h, w].
		"""
		B, K, C, h, w = z_hist.shape
		N = z_tgt.shape[1]
		device = z_tgt.device

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)

		t_flat = timesteps.repeat_interleave(N)  # [B*N]
		noisy_tgt = self.diffuser.noise_scheduler.add_noise(
			z_tgt.reshape(B * N, C, h, w),
			noise.reshape(B * N, C, h, w),
			t_flat,
		).reshape(B, N, C, h, w)

		x = torch.cat([z_hist, noisy_tgt], dim=1)  # [B, K+N, C, h, w]
		all_actions = torch.cat(
			[history_actions.to(device), future_actions.to(device)], dim=1,
		)  # [B, K+N]

		out = self.diffuser(x, timesteps, all_actions)
		model_pred = out[:, K:]
		return model_pred, noise

	# ── Inference ─────────────────────────────────────────────────

	@torch.no_grad()
	def generate_next_chunk(
		self,
		z_hist: torch.Tensor,
		history_actions: torch.Tensor,
		future_actions: torch.Tensor,
		num_inference_steps: int = 30,
		delta_hist: torch.Tensor | None = None,
		gamma: float = 0.0,
	) -> torch.Tensor:
		"""Generate future latent chunk [B, N, C, h, w] from history."""
		B, K, C, h, w = z_hist.shape
		N = future_actions.shape[1]
		device = z_hist.device
		dtype = self.diffuser.unet.dtype

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)

		z_hist = z_hist.to(dtype=dtype)

		all_actions = torch.cat(
			[history_actions.to(device), future_actions.to(device)], dim=1,
		)  # [B, K+N]

		latents = torch.randn(B, N, C, h, w, device=device, dtype=dtype)
		sched = self.diffuser.noise_scheduler
		latents = latents * sched.init_noise_sigma

		sched.set_timesteps(num_inference_steps)
		# diffusers keeps timesteps on CPU by default; UNet + latents are on `device`.
		ts = sched.timesteps
		if isinstance(ts, torch.Tensor):
			ts = ts.to(device=device)
		for t in ts:
			x = torch.cat([z_hist, latents], dim=1)  # [B, K+N, C, h, w]
			t_batch = t.unsqueeze(0).expand(B).contiguous()
			out = self.diffuser(x, t_batch, all_actions)
			pred = out[:, K:]  # [B, N, C, h, w]
			latents = sched.step(
				pred.reshape(B * N, C, h, w),
				t,
				latents.reshape(B * N, C, h, w),
				return_dict=False,
			)[0].reshape(B, N, C, h, w)

		return latents

	# ── TODO: future extension points ────────────────────────────
	# - memory_retrieve(z_hist, memory_bank) → memory_context
	# - multi_segment_rollout(z_hist, actions_seq, segment_boundaries)

	# ── Save / load ──────────────────────────────────────────────

	def save_diffuser(self, out_dir: Path) -> None:
		"""Persist full UNet weights (including LoRA tensors if injected), action head, and DDIM config."""
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		torch.save(self.diffuser.unet.state_dict(), out_dir / "unet.pt")
		torch.save(self.diffuser.action_embedding.state_dict(), out_dir / "action_embedding.pt")
		torch.save(self.diffuser.mlp.state_dict(), out_dir / "action_mlp.pt")
		sched_dir = out_dir / "noise_scheduler"
		sched_dir.mkdir(parents=True, exist_ok=True)
		self.diffuser.noise_scheduler.save_pretrained(str(sched_dir))



"""test"""

if __name__ == "__main__":
	import numpy as np
	from torchvision import transforms

	K, N = 4, 4
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# ── Load a few frames from first test shard ──────────────────
	test_root = Path("data") / "transitions" / "test"
	shard = None
	for env_dir in sorted(test_root.iterdir()):
		for s in sorted(env_dir.glob("shard_*")):
			if (s / "obs.npy").exists():
				shard = s
				break
		if shard:
			break
	assert shard is not None, f"no shard under {test_root}"
	print(f"shard: {shard}")

	obs = np.load(shard / "obs.npy", mmap_mode="r")
	act = np.load(shard / "action.npy", mmap_mode="r")
	seq = K + N
	assert obs.shape[0] >= seq, f"need >= {seq} frames, got {obs.shape[0]}"

	tx = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0),
		transforms.Resize((208, 160), antialias=True),
	])
	frames = torch.stack([tx(np.asarray(obs[i])[..., -3:]) for i in range(seq)])  # [K+N, 3, H, W]
	actions = torch.from_numpy(act[:seq].astype(np.int64))

	history = frames[:K].unsqueeze(0).to(device)          # [1, K, 3, H, W]
	target  = frames[K:K + N].unsqueeze(0).to(device)     # [1, N, 3, H, W]
	hist_act = actions[:K].unsqueeze(0).to(device)          # [1, K]
	fut_act = actions[K:K + N].unsqueeze(0).to(device)      # [1, N]

	# ── Build model ──────────────────────────────────────────────
	print("loading WorldModel ...")
	wm = WorldModel(
		num_actions=18,
		cross_attention_dim=768,
		vae_checkpoint=DEFAULT_VAE_PT,
		prediction_type="epsilon",
		history_len=K,
		chunk_len=N,
		pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
	).to(device)
	print(f"  latent_channels={wm.latent_channels}  num_train_timesteps={wm.num_train_timesteps}")

	# ── VAE encode ───────────────────────────────────────────────
	with torch.no_grad():
		z_hist = wm.encode_video(history)   # [1, K, C, h, w]
		z_tgt  = wm.encode_video(target)    # [1, N, C, h, w]
	print(f"  z_hist={tuple(z_hist.shape)}  z_tgt={tuple(z_tgt.shape)}")

	# ── Diffusion forward ────────────────────────────────────────
	B = 1
	t = torch.randint(0, wm.num_train_timesteps, (B,), device=device)
	noise = torch.randn_like(z_tgt)

	print("running diffusion_forward ...")
	pred, tgt_noise = wm.diffusion_forward(z_hist, z_tgt, hist_act, fut_act, t, noise)
	print(f"  model_pred={tuple(pred.shape)}  noise={tuple(tgt_noise.shape)}")
	loss = torch.nn.functional.mse_loss(pred.float(), tgt_noise.float())
	print(f"  mse_loss={loss.item():.4f}")
	print("OK")

