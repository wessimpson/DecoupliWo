"""SD v1.4 UNet2D denoiser: stacked history + noisy next frame; action slot K repeats a[K-1] (causal into target).

Supports two latent shapes:
	* native SD latents (``latent_channels=4``): pretrained ``conv_in`` weights are tiled across
	  the ``K+1`` history slots, matching the original behaviour for pixel modalities.
	* ASCII tile-grid latents (``latent_channels!=4`` and/or small spatial dims): ``conv_in``/``conv_out``
	  are re-initialized from scratch, and an optional ``latent_upsample`` round-trip keeps the
	  UNet's internal resolution in a range where the pretrained blocks still carry useful priors.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel

SD_DEFAULT_IN_CHANNELS: int = 4


class Diffuser(nn.Module):
	"""UNet2DConditionModel (compact): latents ``[B, K+1, C, H, W]`` folded to ``[B, (K+1)*C, H, W]``."""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		cross_attention_dim: int,
		history_len: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"] = "v_prediction",
		pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
		num_action_attn_heads: int = 4,
		latent_upsample: int = 1,
	) -> None:
		super().__init__()
		self.latent_channels = int(latent_channels)
		self.cross_attention_dim = cross_attention_dim
		self.history_len = int(history_len)
		self.num_latent_frames = self.history_len + 1
		self.latent_upsample = int(latent_upsample)
		assert self.latent_upsample >= 1, f"latent_upsample must be >= 1, got {latent_upsample}"
		stacked_in = self.num_latent_frames * self.latent_channels

		unet = UNet2DConditionModel(
			sample_size=None,
			in_channels=stacked_in,
			out_channels=latent_channels,
			cross_attention_dim=cross_attention_dim,
			layers_per_block=1,
			block_out_channels=(160, 320, 320),
			down_block_types=(
				"CrossAttnDownBlock2D",
				"CrossAttnDownBlock2D",
				"DownBlock2D",
			),
			up_block_types=(
				"UpBlock2D",
				"CrossAttnUpBlock2D",
				"CrossAttnUpBlock2D",
			),
			attention_head_dim=(8, 8, 8),
			norm_num_groups=32,
		)

		old_conv = unet.conv_in
		new_conv = nn.Conv2d(
			stacked_in,
			old_conv.out_channels,
			kernel_size=old_conv.kernel_size,
			stride=old_conv.stride,
			padding=old_conv.padding,
		)
		if self.latent_channels == SD_DEFAULT_IN_CHANNELS:
			with torch.no_grad():
				new_conv.bias.copy_(old_conv.bias)
				for i in range(self.num_latent_frames):
					sl = slice(i * self.latent_channels, (i + 1) * self.latent_channels)
					new_conv.weight[:, sl, :, :].copy_(old_conv.weight)
		unet.conv_in = new_conv
		unet.register_to_config(in_channels=stacked_in)

		if self.latent_channels != SD_DEFAULT_IN_CHANNELS:
			old_out = unet.conv_out
			new_out = nn.Conv2d(
				old_out.in_channels,
				self.latent_channels,
				kernel_size=old_out.kernel_size,
				stride=old_out.stride,
				padding=old_out.padding,
			)
			unet.conv_out = new_out
			unet.register_to_config(out_channels=self.latent_channels)

		self.unet = unet

		if self.latent_upsample > 1:
			self.pre_upsample = nn.ConvTranspose2d(
				stacked_in, stacked_in,
				kernel_size=self.latent_upsample, stride=self.latent_upsample,
			)
			self.post_downsample = nn.Conv2d(
				self.latent_channels, self.latent_channels,
				kernel_size=self.latent_upsample, stride=self.latent_upsample,
			)
			with torch.no_grad():
				self.pre_upsample.weight.zero_()
				for c in range(stacked_in):
					self.pre_upsample.weight[c, c].fill_(1.0 / (self.latent_upsample ** 2))
				self.pre_upsample.bias.zero_()
				self.post_downsample.weight.zero_()
				for c in range(self.latent_channels):
					self.post_downsample.weight[c, c].fill_(1.0)
				self.post_downsample.bias.zero_()
		else:
			self.pre_upsample = None
			self.post_downsample = None

		self.action_embedding = nn.Embedding(num_actions, cross_attention_dim)
		self.action_context = _LightweightActionContext(cross_attention_dim, num_action_attn_heads)
		nn.init.normal_(self.action_embedding.weight, std=0.02)
		with torch.no_grad():
			self.action_embedding.weight[self.null_action_index].zero_()

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

	def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
		"""[B, F] -> [B, F, D] cross-attention context (F = K+1)."""
		return self.action_context(self.action_embedding(actions))

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		action: torch.Tensor,
	) -> torch.Tensor:
		"""Stacked-frame denoise: predict noise / v for the next frame.

		Args:
			noisy_latents: [B, F, C, H, W] with F = history_len + 1 (history clean + noisy target)
			timesteps:     [B]
			actions:       [B, F]  per-frame action ids

		Returns:
			[B, C, H, W]  model output aligned with target latent channels at the INPUT spatial size
		"""
		b, f, c, h, w = noisy_latents.shape
		assert f == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {f}"
		assert c == self.latent_channels, f"Expected C={self.latent_channels}, got {c}"

		x = noisy_latents.reshape(b, f * c, h, w).contiguous()
		if self.pre_upsample is not None:
			x = self.pre_upsample(x)
		enc = self.embed_actions(actions)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]

		if self.post_downsample is not None:
			out = self.post_downsample(out)
		return out
