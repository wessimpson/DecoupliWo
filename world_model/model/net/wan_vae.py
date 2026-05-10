from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

__all__ = ["WanVAE", "DEFAULT_WAN_VAE_REPO"]

# Diffusers-format checkpoint on the Hub (same ``subfolder="vae"`` as ``WanPipeline.from_pretrained``).
DEFAULT_WAN_VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class WanVAE(nn.Module):
	"""Wan 3D-VAE used **single-frame**; API matches ``world_model.model.net.vae.VAE``.

	Weights load from Hugging Face via ``AutoencoderKLWan.from_pretrained(..., subfolder="vae")`` (no local ``.pth``).
	Latents match the Wan training convention: ``(mu - mean) * (1/std)`` per channel (``scaling_factor`` is ``1``).
	"""

	def __init__(
		self,
		pretrained_model_id: str,
		*,
		z_dim: int = 16,
		torch_dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()
		self._z_dim = int(z_dim)
		dt = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
		self.vae = AutoencoderKLWan.from_pretrained(
			pretrained_model_id,
			subfolder="vae",
			torch_dtype=dt,
		)
		if int(self.vae.config.z_dim) != self._z_dim:
			raise ValueError(f"z_dim={z_dim} but Hub VAE has z_dim={self.vae.config.z_dim}")
		self.vae.requires_grad_(False)
		self.vae.eval()
		print(f"[WanVAE] AutoencoderKLWan from {pretrained_model_id!r} latent_channels={self._z_dim}")

	def freeze(self) -> None:
		self.eval()
		self.vae.eval()
		self.vae.requires_grad_(False)

	def _dtype(self) -> torch.dtype:
		return next(self.vae.parameters()).dtype

	@property
	def latent_channels(self) -> int:
		return self._z_dim

	@property
	def scaling_factor(self) -> float:
		return 1.0

	def _mean_invstd_views(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
		cfg = self.vae.config
		z = self._z_dim
		mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, z, 1, 1, 1)
		inv_std = (1.0 / torch.tensor(cfg.latents_std, device=device, dtype=dtype)).view(1, z, 1, 1, 1)
		return mean, inv_std

	def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[N,3,H,W] in [-1,1] → normalized latent [N,C,h,w]."""
		dt = self._dtype()
		x = pixels.to(dtype=dt).unsqueeze(2)
		with torch.no_grad():
			mu = self.vae.encode(x).latent_dist.mode()
		mean, inv_std = self._mean_invstd_views(mu.device, mu.dtype)
		z = (mu - mean) * inv_std
		if z.dim() == 5:
			z = z[:, :, 0, :, :].contiguous()
		return z

	def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
		"""[N,C,h,w] normalized → [N,3,H,W] in [-1,1]."""
		dt = self._dtype()
		mean, inv_std = self._mean_invstd_views(latents.device, dt)
		z = latents.to(dtype=dt).unsqueeze(2)
		z_raw = z / inv_std + mean
		with torch.no_grad():
			dec = self.vae.decode(z_raw, return_dict=True).sample
		if dec.dim() == 5:
			dec = dec[:, :, 0, :, :].contiguous()
		return dec.clamp_(-1, 1)

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,T,3,H,W] → [B,T,C,h,w]."""
		B, T = pixels.shape[:2]
		z = self.encode_pixels(pixels.reshape(B * T, *pixels.shape[2:]))
		return z.reshape(B, T, *z.shape[1:])

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,T,C,h,w] → [B,T,3,H,W]."""
		B, T = latents.shape[:2]
		px = self.decode_latents(latents.reshape(B * T, *latents.shape[2:]))
		return px.reshape(B, T, *px.shape[1:])


def _to_uint8_tb(t: torch.Tensor):
	"""[-1,1] float CHW → uint8 HWC numpy."""
	import numpy as np

	return (
		t.detach().cpu().clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).numpy()
	)


def main() -> None:
	"""Smoke test on ``aliens`` (``data/transitions/train/aliens`` or ``.../test/aliens``)."""
	import numpy as np
	import matplotlib.pyplot as plt

	from world_model.dataset import obs_array_to_pixels

	aliens_train = Path("data") / "transitions" / "train" / "aliens"
	shard = None
	for base in (aliens_train, Path("data") / "transitions" / "test" / "aliens"):
		if not base.is_dir():
			continue
		for s in sorted(base.glob("shard_*")):
			if (s / "obs.npy").is_file():
				shard = s
				break
		if shard is not None:
			break
	assert shard is not None, "need aliens shard_*/obs.npy under data/transitions/train or test"
	print(f"shard: {shard}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vae = WanVAE(DEFAULT_WAN_VAE_REPO, z_dim=16)
	vae.freeze()
	vae.to(device)

	obs = np.load(shard / "obs.npy", mmap_mode="r")
	take = min(8, obs.shape[0])
	pixels = obs_array_to_pixels(np.asarray(obs[:take]))

	inp = pixels[:1].to(device=device, dtype=vae._dtype())
	z = vae.encode_pixels(inp)
	recon = vae.decode_latents(z)
	mse = (inp.float() - recon.float()).pow(2).mean().item()
	psnr = 10.0 * np.log10(1.0 / mse) if mse else float("inf")
	print(f"[frame] latent={tuple(z.shape)} PSNR={psnr:.2f} dB")

	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	axes[0].imshow(_to_uint8_tb(inp[0].cpu()))
	axes[0].set_title("original")
	axes[1].imshow(_to_uint8_tb(recon[0].cpu()))
	axes[1].set_title(f"wan recon ({psnr:.1f} dB)")
	for ax in axes:
		ax.axis("off")
	fig.suptitle("Aliens · WanVAE (single-frame path)")
	plt.tight_layout()

	vid = pixels.unsqueeze(0).to(device=device, dtype=vae._dtype())
	z_vid = vae.encode_video(vid)
	recon_vid = vae.decode_video(z_vid)
	print(f"[mini-video] latent={tuple(z_vid.shape)} recon={tuple(recon_vid.shape)}")
	plt.show()


if __name__ == "__main__":
	main()
