from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.model.net.vae import WanVAE


class AtariVAEEncoder(nn.Module):
	"""
	Env-facing wrapper around WanVAE that:
	- accepts frames [B, T, 3, H, W] in [-1, 1]
	- resizes to WanVAE input
	- returns per-frame latents [B, T, latent_dim]
	"""

	def __init__(
		self,
		pretrained_dir: str | Path,
		image_size: int = 256,
		pool_size: int = 4,
		z_dim: int = 16,
	) -> None:
		super().__init__()
		pretrained_dir = Path(pretrained_dir)
		ckpt = pretrained_dir / "Wan2.1_VAE.pth"
		if not ckpt.is_file():
			raise FileNotFoundError(
				f"Missing {ckpt}. Place ``Wan2.1_VAE.pth`` in ``world_model/checkpoint/vae/`` "
				"(default) or set a directory containing the file."
			)
		if image_size % 8 != 0:
			raise ValueError("image_size must be divisible by 8 for Wan VAE.")
		self.image_size = image_size
		self.pool_size = pool_size
		self.z_dim = z_dim
		self.wan = WanVAE(pretrained_path=str(ckpt), z_dim=z_dim)
		self.wan.eval()
		self.wan.requires_grad_(False)

	@property
	def latent_dim(self) -> int:
		return self.z_dim * self.pool_size * self.pool_size

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.wan.to(*args, **kwargs)
		return self

	@torch.no_grad()
	def encode_frames(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
		b, t, c, h, w = x.shape
		if c != 3:
			raise ValueError(f"expected 3 RGB channels, got {c}")
		frames = x.view(b * t, c, h, w)
		frames = F.interpolate(
			frames,
			size=(self.image_size, self.image_size),
			mode="bilinear",
			align_corners=False,
		)
		videos = frames.unsqueeze(2)  # [N, 3, 1, H, W]
		dtype = next(self.wan.model.parameters()).dtype
		videos = videos.to(device=device, dtype=dtype)
		z = self.wan.single_encode(videos, device=device)  # [N, 16, 1, h', w']
		if z.dim() == 5 and z.size(2) == 1:
			z = z.squeeze(2)  # [N, 16, h', w']
		z = F.adaptive_avg_pool2d(z, (self.pool_size, self.pool_size))
		z = z.flatten(1).to(device=device)  # [N, latent_dim]
		return z.view(b, t, -1)

def _load_frames_from_transitions(shard_path: Path, max_frames: int) -> torch.Tensor:
	import numpy as np
	data = np.load(str(shard_path))
	if "obs" not in data:
		raise KeyError(f"'obs' not found in {shard_path}")
	obs = data["obs"]  # [N, H, W, 3] uint8
	if obs.ndim != 4 or obs.shape[-1] != 3:
		raise ValueError(f"Expected obs with shape [N,H,W,3], got {obs.shape}")
	n = min(max_frames, obs.shape[0])
	obs = obs[:n].astype(np.float32) / 127.5 - 1.0  # to [-1, 1]
	obs = np.transpose(obs, (0, 3, 1, 2))  # [N, 3, H, W]
	frames = torch.from_numpy(obs)[None, ...]  # [1, N, 3, H, W]
	return frames

def main() -> None:
	import numpy as np
	import matplotlib.pyplot as plt
	# Fixed configuration
	transitions_dir = Path("data") / "transitions" / "space_invaders"
	wan_vae_dir = Path("world_model") / "checkpoints" / "vae"
	num_frames = 16
	image_size = 256
	pool_size = 4
	z_dim = 16

	if not transitions_dir.is_dir():
		raise FileNotFoundError(f"Transitions directory not found: {transitions_dir}")
	shard_paths = sorted(transitions_dir.glob("shard_*.npz"))
	if not shard_paths:
		raise FileNotFoundError(f"No shard_*.npz files found in {transitions_dir}")
	shard_path = shard_paths[0]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = AtariVAEEncoder(
		pretrained_dir=wan_vae_dir,
		image_size=image_size,
		pool_size=pool_size,
		z_dim=z_dim,
	).to(device)

	frames = _load_frames_from_transitions(shard_path, max_frames=num_frames)  # [1, T, 3, H, W]
	with torch.no_grad():
		z = encoder.encode_frames(frames, device=device)  # [1, T, latent_dim]
		print("frames shape:", tuple(frames.shape))
		print("latents shape:", tuple(z.shape))
		# Reconstruct a single frame (first)
		first = frames[:, 0]  # [1, 3, H, W]
		first_resized = torch.nn.functional.interpolate(
			first, size=(image_size, image_size), mode="bilinear", align_corners=False
		)
		video_1f = first_resized.unsqueeze(2)  # [1, 3, 1, H, W]
		dtype = next(encoder.wan.model.parameters()).dtype
		video_1f = video_1f.to(device=device, dtype=dtype)
		latent_1f = encoder.wan.single_encode(video_1f, device=device)  # [1, 16, 1, h', w']
		recon_1f = encoder.wan.single_decode(latent_1f, device=device)  # [1, 3, 1, H, W] in [-1, 1]
		recon_img = recon_1f[0, :, 0].detach().cpu().float()  # [3, H, W]
		orig_img = video_1f[0, :, 0].detach().cpu().float()   # [3, H, W]
		to_disp = lambda t: ((t + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
		orig_disp = to_disp(orig_img)
		recon_disp = to_disp(recon_img)

	# Show original vs reconstructed (left/right), no saving
	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	axes[0].imshow(orig_disp)
	axes[0].set_title("Original")
	axes[0].axis("off")
	axes[1].imshow(recon_disp)
	axes[1].set_title("Reconstructed")
	axes[1].axis("off")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()

