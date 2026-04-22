"""Categorical VAE over ASCII tile grids.

Inputs are ``Long[B, H, W]`` per-cell byte ids (0..255). The encoder produces
continuous latents ``[B, C, H, W]`` with the same spatial size as the input
(no downsampling - the grid is already coarse). The decoder produces per-cell
logits ``[B, V, H, W]`` so training uses token cross-entropy, not MSE.

The public interface mirrors :class:`world_model.model.net.vae.VAE` so
``WorldModel`` can swap between pixel and ASCII modalities without touching
downstream code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.ascii.constants import VOCAB_SIZE

DEFAULT_ASCII_VAE_PT = Path("world_model") / "checkpoints" / "ascii_vae" / "vae.pt"

EMBED_DIM: int = 64
HIDDEN_DIM: int = 128
LATENT_CHANNELS: int = 4
GROUP_NORM_GROUPS: int = 32


def _conv_block(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
		nn.GroupNorm(GROUP_NORM_GROUPS, out_ch),
		nn.SiLU(),
	)


class _Encoder(nn.Module):
	def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, latent_channels: int) -> None:
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.body = nn.Sequential(
			_conv_block(embed_dim, hidden_dim),
			_conv_block(hidden_dim, hidden_dim),
		)
		self.to_mu = nn.Conv2d(hidden_dim, latent_channels, kernel_size=1)
		self.to_logvar = nn.Conv2d(hidden_dim, latent_channels, kernel_size=1)

	def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = self.embed(ids).permute(0, 3, 1, 2).contiguous()
		h = self.body(x)
		return self.to_mu(h), self.to_logvar(h)


class _Decoder(nn.Module):
	def __init__(self, vocab_size: int, hidden_dim: int, latent_channels: int) -> None:
		super().__init__()
		self.body = nn.Sequential(
			_conv_block(latent_channels, hidden_dim),
			_conv_block(hidden_dim, hidden_dim),
		)
		self.to_logits = nn.Conv2d(hidden_dim, vocab_size, kernel_size=1)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		return self.to_logits(self.body(z))


class ASCIIVAE(nn.Module):
	"""Categorical VAE over ``Long[B, H, W]`` ASCII grids."""

	def __init__(
		self,
		vocab_size: int = VOCAB_SIZE,
		embed_dim: int = EMBED_DIM,
		hidden_dim: int = HIDDEN_DIM,
		latent_channels: int = LATENT_CHANNELS,
	) -> None:
		super().__init__()
		self.vocab_size = int(vocab_size)
		self._latent_channels = int(latent_channels)
		self.encoder = _Encoder(self.vocab_size, embed_dim, hidden_dim, self._latent_channels)
		self.decoder = _Decoder(self.vocab_size, hidden_dim, self._latent_channels)
		self.register_buffer("scaling_factor_buf", torch.tensor(1.0, dtype=torch.float32))

	@property
	def latent_channels(self) -> int:
		return self._latent_channels

	@property
	def scaling_factor(self) -> float:
		return float(self.scaling_factor_buf.item())

	def set_scaling_factor(self, value: float) -> None:
		"""Set the post-training latent std used to normalize diffusion inputs."""
		assert value > 0, f"scaling_factor must be positive, got {value}"
		self.scaling_factor_buf = torch.tensor(float(value), dtype=torch.float32)

	def freeze(self) -> None:
		self.train(False)
		self.requires_grad_(False)

	@staticmethod
	def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		return mu + std * torch.randn_like(std)

	def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Training forward: returns ``(logits [B,V,H,W], mu, logvar)`` before scaling."""
		assert ids.dtype == torch.long, f"ids must be long, got {ids.dtype}"
		assert ids.ndim == 3, f"ids must be [B,H,W], got shape {ids.shape}"
		mu, logvar = self.encoder(ids)
		z = self._reparameterize(mu, logvar) if self.training else mu
		logits = self.decoder(z)
		return logits, mu, logvar

	@staticmethod
	def elbo_loss(
		logits: torch.Tensor,
		mu: torch.Tensor,
		logvar: torch.Tensor,
		targets: torch.Tensor,
		kl_beta: float,
	) -> tuple[torch.Tensor, dict[str, float]]:
		"""Token-CE reconstruction plus kl_beta*KL(N(mu,sigma^2) || N(0,1)); per-cell means."""
		ce = F.cross_entropy(logits, targets, reduction="mean")
		kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
		total = ce + kl_beta * kl
		return total, {"ce": ce.detach().item(), "kl": kl.detach().item(), "total": total.detach().item()}

	def encode_ids(self, ids: torch.Tensor) -> torch.Tensor:
		"""[B,H,W] long -> scaled latents [B,C,H,W] using the posterior mean (no grad)."""
		assert ids.dtype == torch.long, f"ids must be long, got {ids.dtype}"
		assert ids.ndim == 3, f"ids must be [B,H,W], got shape {ids.shape}"
		with torch.no_grad():
			mu, _ = self.encoder(ids)
			return mu * self.scaling_factor

	def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,C,H,W] scaled latents -> logits [B,V,H,W] (no grad)."""
		assert latents.ndim == 4, f"latents must be [B,C,H,W], got shape {latents.shape}"
		with torch.no_grad():
			z = latents / self.scaling_factor
			return self.decoder(z)

	def encode_video(self, ids: torch.Tensor) -> torch.Tensor:
		"""[B,T,H,W] long -> [B,T,C,h,w] scaled latents (no grad)."""
		assert ids.ndim == 4, f"ids must be [B,T,H,W], got shape {ids.shape}"
		b, t = ids.shape[:2]
		z = self.encode_ids(ids.reshape(b * t, *ids.shape[2:]))
		return z.reshape(b, t, *z.shape[1:])

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,T,C,h,w] scaled latents -> logits [B,T,V,H,W] (no grad)."""
		assert latents.ndim == 5, f"latents must be [B,T,C,h,w], got shape {latents.shape}"
		b, t = latents.shape[:2]
		logits = self.decode_latents(latents.reshape(b * t, *latents.shape[2:]))
		return logits.reshape(b, t, *logits.shape[1:])

	@staticmethod
	def logits_to_ids(logits: torch.Tensor) -> torch.Tensor:
		"""Greedy decode: ``[..., V, H, W] -> [..., H, W]`` long."""
		return logits.argmax(dim=-3)


def load_ascii_vae(checkpoint: Union[str, Path]) -> ASCIIVAE:
	"""Load an :class:`ASCIIVAE` from a ``.pt`` state dict (raises if missing)."""
	pt = Path(checkpoint)
	assert pt.is_file(), f"ASCIIVAE checkpoint must be an existing .pt file: {pt}"
	model = ASCIIVAE()
	state = torch.load(pt, map_location="cpu", weights_only=True)
	model.load_state_dict(state)
	return model
