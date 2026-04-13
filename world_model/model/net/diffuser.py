"""SD v1.4 UNet2D denoiser: stacked history + noisy next frame; action slot K repeats a[K-1] (causal into target)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel


class _LightweightActionContext(nn.Module):
	"""Single pre-norm self-attention over F action tokens (residual), replaces MLP mix."""

	def __init__(self, dim: int, num_heads: int) -> None:
		super().__init__()
		if dim % num_heads != 0:
			raise ValueError(f"cross_attention_dim={dim} must be divisible by num_heads={num_heads}")
		self.norm = nn.LayerNorm(dim)
		self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, F, D]
		h = self.norm(x)
		y, _ = self.attn(h, h, h, need_weights=False)
		return x + y


class Diffuser(nn.Module):
	"""UNet2DConditionModel (SD 1.4): latents ``[B, K+1, C, H, W]`` folded to ``[B, (K+1)*C, H, W]``."""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		cross_attention_dim: int,
		history_len: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
		pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
		num_action_attn_heads: int = 4,
	) -> None:
		super().__init__()
		self.latent_channels = latent_channels
		self.cross_attention_dim = cross_attention_dim
		self.history_len = int(history_len)
		self.num_latent_frames = self.history_len + 1
		stacked_in = self.num_latent_frames * latent_channels

		unet = UNet2DConditionModel.from_pretrained(
			pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False,
		)
		assert latent_channels == unet.config.in_channels, (
			f"VAE latent_channels={latent_channels} != pretrained UNet in_channels={unet.config.in_channels}"
		)
		old_conv = unet.conv_in
		new_conv = nn.Conv2d(
			stacked_in,
			old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
		)
		with torch.no_grad():
			new_conv.bias.copy_(old_conv.bias)
			for i in range(self.num_latent_frames):
				sl = slice(i * latent_channels, (i + 1) * latent_channels)
				new_conv.weight[:, sl, :, :].copy_(old_conv.weight)
		unet.conv_in = new_conv
		unet.register_to_config(in_channels=stacked_in)

		self.unet = unet

		self.action_embedding = nn.Embedding(num_actions, cross_attention_dim)
		self.action_context = _LightweightActionContext(cross_attention_dim, num_action_attn_heads)
		nn.init.normal_(self.action_embedding.weight, std=0.02)

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

	def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
		"""[B, F] → [B, F, D] cross-attention context (F = K+1)."""
		return self.action_context(self.action_embedding(actions))

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		actions: torch.Tensor,
	) -> torch.Tensor:
		"""Stacked-frame denoise: predict noise / v for the next frame (4-channel UNet output).

		Args:
			noisy_latents: [B, F, C, H, W] with F = history_len + 1 (history clean + noisy target)
			timesteps:     [B]
			actions:       [B, F]  per-frame action ids

		Returns:
			[B, C, H, W]  model output aligned with target latent channels
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		x = noisy_latents.reshape(B, F * C, H, W).contiguous()
		enc = self.embed_actions(actions)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
		return out
