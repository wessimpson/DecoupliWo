from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


DEFAULT_CHECKPOINT = Path("world_model") / "checkpoints" / "vae"
DEFAULT_PRETRAINED = "stabilityai/sd-vae-ft-mse"


class VAE(nn.Module):
	"""Frozen SD VAE used as frame tokenizer (default: stabilityai/sd-vae-ft-mse)."""

	def __init__(
		self,
		checkpoint: Optional[Union[str, Path]] = DEFAULT_CHECKPOINT,
		pretrained: Optional[str] = DEFAULT_PRETRAINED,
	) -> None:
		super().__init__()
		ckpt = Path(checkpoint) if checkpoint else None
		ckpt_file = ckpt / "vae.pt" if ckpt else None
		if ckpt_file and ckpt_file.exists():
			pid = pretrained or DEFAULT_PRETRAINED
			self.autoencoder = AutoencoderKL.from_pretrained(pid)
			self.autoencoder.load_state_dict(torch.load(ckpt_file, map_location="cpu", weights_only=True))
			print(f"[VAE] loaded checkpoint: {ckpt_file}")
		else:
			pid = pretrained or DEFAULT_PRETRAINED
			self.autoencoder = AutoencoderKL.from_pretrained(pid)
			print(f"[VAE] loaded pretrained: {pid}")

	@property
	def latent_channels(self) -> int:
		return int(self.autoencoder.config.latent_channels)

	@property
	def scaling_factor(self) -> float:
		return float(self.autoencoder.config.scaling_factor)

	def freeze(self) -> None:
		self.autoencoder.eval()
		self.autoencoder.requires_grad_(False)

	def _dtype(self) -> torch.dtype:
		return next(self.autoencoder.parameters()).dtype

	def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[N,3,H,W] in [-1,1] → scaled latents [N,C,h,w]."""
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
	import numpy as np
	from torchvision import transforms
	import matplotlib.pyplot as plt

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

	# ── load frames ──
	obs = np.load(shard / "obs.npy", mmap_mode="r")
	h, w = (max(8, (v // 8) * 8) for v in (208, 160))
	tx = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0),
		transforms.Resize((h, w), antialias=True),
	])
	T = min(8, obs.shape[0])
	frames = []
	for i in range(T):
		f = np.asarray(obs[i])
		if f.shape[-1] > 3:
			f = f[..., -3:]
		if f.dtype != np.uint8:
			f = np.clip(f, 0, 255).astype(np.uint8)
		frames.append(tx(f))
	frames = torch.stack(frames)  # [T,3,H,W]

	# ── model ──
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vae = VAE()
	vae.freeze()
	vae.to(device)

	# ── single-frame roundtrip ──
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

	# ── video roundtrip ──
	vid = frames.unsqueeze(0).to(device)  # [1,T,3,H,W]
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
