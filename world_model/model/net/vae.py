from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

DEFAULT_VAE_PT = Path("world_model") / "checkpoints" / "vae" / "vae.pt"
_HUB = "stabilityai/sd-vae-ft-mse"


class VAE(nn.Module):
	"""Frozen VAE for 120×120 GVGAI frames (15×15 grid, 8px cells) → 4×15×15 latents."""

	pixel_hw = (120, 120)
	latent_hw = (15, 15)
	latent_channels = 4
	scaling_factor = 0.18215

	def __init__(self, checkpoint: Union[str, Path, None] = None) -> None:
		super().__init__()
		if checkpoint is None:
			self.autoencoder = AutoencoderKL.from_config(AutoencoderKL.load_config(_HUB))
		else:
			pt = Path(checkpoint)
			assert pt.is_file(), f"missing VAE checkpoint: {pt}"
			self.autoencoder = AutoencoderKL.from_pretrained(_HUB)
			self.autoencoder.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))

	def freeze(self) -> None:
		self.autoencoder.eval()
		self.autoencoder.requires_grad_(False)

	def _dtype(self) -> torch.dtype:
		return next(self.autoencoder.parameters()).dtype

	def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
		x = pixels.to(dtype=self._dtype())
		with torch.no_grad():
			return self.autoencoder.encode(x).latent_dist.mode() * self.scaling_factor

	def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
		z = latents.to(dtype=self._dtype()) / self.scaling_factor
		with torch.no_grad():
			return self.autoencoder.decode(z).sample

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		B, T = pixels.shape[:2]
		z = self.encode_pixels(pixels.reshape(B * T, *pixels.shape[2:]))
		return z.reshape(B, T, *z.shape[1:])

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		B, T = latents.shape[:2]
		px = self.decode_latents(latents.reshape(B * T, *latents.shape[2:]))
		return px.reshape(B, T, *px.shape[1:])


def _to_uint8(t: torch.Tensor):
	return t.detach().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).numpy()


def main() -> None:
	import matplotlib.pyplot as plt
	import numpy as np
	from world_model.dataset import obs_array_to_pixels

	root = Path("data") / "transitions" / "test"
	shard = next(
		s
		for env in sorted(root.iterdir())
		if env.is_dir()
		for s in sorted(env.glob("shard_*"))
		if (s / "obs.npy").is_file()
	)
	obs = np.load(shard / "obs.npy", mmap_mode="r")
	T = min(6, obs.shape[0])
	px = obs_array_to_pixels(np.copy(np.asarray(obs[:T])), resize_to=None)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not DEFAULT_VAE_PT.is_file():
		print(f"No checkpoint at {DEFAULT_VAE_PT}; using scratch VAE for smoke test.")
	vae = VAE(DEFAULT_VAE_PT if DEFAULT_VAE_PT.is_file() else None).eval().to(device)
	vae.freeze()
	z = vae.encode_pixels(px.to(device))
	recon = vae.decode_latents(z).cpu()
	print(f"shard={shard} pixel={vae.pixel_hw} latent={tuple(z.shape[-3:])} mse={(px - recon).pow(2).mean():.5f}")

	fig, ax = plt.subplots(2, T, figsize=(1.5 * T, 3))
	for t in range(T):
		ax[0, t].imshow(_to_uint8(px[t]))
		ax[0, t].axis("off")
		ax[1, t].imshow(_to_uint8(recon[t]))
		ax[1, t].axis("off")
	ax[0, 0].set_ylabel("orig")
	ax[1, 0].set_ylabel("recon")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
