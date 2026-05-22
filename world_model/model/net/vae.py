from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

DEFAULT_VAE_PT = Path("world_model") / "checkpoints" / "vae" / "vae.pt"

LATENT_CHANNELS = 4
VAE_DOWNSAMPLE = 8
LATENT_H = 11
LATENT_W = 30
PIXEL_H = LATENT_H * VAE_DOWNSAMPLE
PIXEL_W = LATENT_W * VAE_DOWNSAMPLE
SCALING_FACTOR = 0.18215


def vae_latent_hw() -> tuple[int, int]:
	return LATENT_H, LATENT_W


def vae_pixel_hw() -> tuple[int, int]:
	return PIXEL_H, PIXEL_W


def build_autoencoder_kl() -> AutoencoderKL:
	"""Randomly initialized KL-VAE (SD-style 8× downsample, 4 latent channels)."""
	return AutoencoderKL(
		in_channels=3,
		out_channels=3,
		down_block_types=(
			"DownEncoderBlock2D",
			"DownEncoderBlock2D",
			"DownEncoderBlock2D",
			"DownEncoderBlock2D",
		),
		up_block_types=(
			"UpDecoderBlock2D",
			"UpDecoderBlock2D",
			"UpDecoderBlock2D",
			"UpDecoderBlock2D",
		),
		block_out_channels=(128, 256, 512, 512),
		layers_per_block=2,
		latent_channels=LATENT_CHANNELS,
		sample_size=max(PIXEL_H, PIXEL_W),
	)


class VAE(nn.Module):
	"""Frozen KL-VAE: scratch architecture + weights from a single ``vae.pt`` file."""

	def __init__(self, checkpoint: Union[str, Path]) -> None:
		super().__init__()
		pt = Path(checkpoint)
		assert pt.is_file(), f"VAE checkpoint must be an existing .pt file: {pt}"
		self.autoencoder = build_autoencoder_kl()
		self.autoencoder.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
		print(f"[VAE] scratch AutoencoderKL latent={LATENT_CHANNELS}x{LATENT_H}x{LATENT_W} + {pt}")

	@property
	def latent_channels(self) -> int:
		return LATENT_CHANNELS

	@property
	def scaling_factor(self) -> float:
		return SCALING_FACTOR

	def freeze(self) -> None:
		self.autoencoder.eval()
		self.autoencoder.requires_grad_(False)

	def _dtype(self) -> torch.dtype:
		return next(self.autoencoder.parameters()).dtype

	def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[N,3,H,W] in [-1,1] → scaled latents [N,C,h,w]. Expects H={PIXEL_H}, W={PIXEL_W}."""
		x = pixels.to(dtype=self._dtype())
		with torch.no_grad():
			return self.autoencoder.encode(x).latent_dist.mode() * self.scaling_factor

	def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
		"""Scaled latents [N,C,h,w] → pixels [N,3,H,W] in [-1,1]."""
		z = latents.to(dtype=self._dtype()) / self.scaling_factor
		with torch.no_grad():
			return self.autoencoder.decode(z).sample

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,T,3,H,W] → [B,T,C,h,w] scaled latents."""
		B, T = pixels.shape[:2]
		z = self.encode_pixels(pixels.reshape(B * T, *pixels.shape[2:]))
		return z.reshape(B, T, *z.shape[1:])

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,T,C,h,w] → [B,T,3,H,W]."""
		B, T = latents.shape[:2]
		px = self.decode_latents(latents.reshape(B * T, *latents.shape[2:]))
		return px.reshape(B, T, *px.shape[1:])


"""test"""

def _to_uint8(t: torch.Tensor):
	"""[-1,1] float CHW → uint8 HWC numpy."""
	import numpy as np
	return t.detach().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).numpy()


def main() -> None:
	import matplotlib.pyplot as plt
	import numpy as np

	from world_model.dataset import obs_array_to_pixels

	# ── locate first shard ──
	test_root = Path("data") / "transitions" / "test"
	shard = None
	for env in sorted(test_root.iterdir()):
		if not env.is_dir():
			continue
		for s in sorted(env.glob("shard_*")):
			if (s / "obs.npy").exists():
				shard = s
				break
		if shard:
			break
	assert shard is not None, f"no shard with obs.npy under {test_root}"
	print(f"shard: {shard}")

	obs = np.load(shard / "obs.npy", mmap_mode="r")
	T = min(8, obs.shape[0])
	frames = obs_array_to_pixels(obs[:T], resize_to=vae_pixel_hw())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vae = VAE(checkpoint=DEFAULT_VAE_PT)
	vae.freeze()
	vae.to(device)

	inp = frames[:1].to(device)
	z = vae.encode_pixels(inp)
	recon = vae.decode_latents(z)
	mse = (inp.float() - recon.float()).pow(2).mean().item()
	psnr = 10.0 * np.log10(1.0 / mse) if mse else float("inf")
	print(f"[frame] latent={tuple(z.shape)} PSNR={psnr:.2f} dB")

	fig, axes = plt.subplots(1, 2)
	axes[0].imshow(_to_uint8(inp[0])); axes[0].set_title("original")
	axes[1].imshow(_to_uint8(recon[0])); axes[1].set_title(f"recon ({psnr:.1f} dB)")
	for ax in axes:
		ax.axis("off")
	fig.suptitle("Single-frame reconstruction")
	plt.tight_layout()

	vid = frames.unsqueeze(0).to(device)
	z_vid = vae.encode_video(vid)
	recon_vid = vae.decode_video(z_vid)
	print(f"[video] latent={tuple(z_vid.shape)}")

	fig2, axes2 = plt.subplots(2, T, figsize=(2 * T, 4))
	for t in range(T):
		axes2[0, t].imshow(_to_uint8(vid[0, t])); axes2[0, t].axis("off")
		axes2[1, t].imshow(_to_uint8(recon_vid[0, t])); axes2[1, t].axis("off")
	axes2[0, 0].set_ylabel("orig"); axes2[1, 0].set_ylabel("recon")
	fig2.suptitle("Video reconstruction")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
