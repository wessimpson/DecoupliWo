"""Compact UNet2D denoiser: stacked history + noisy next frame; cross-attn conditioned on single action a_t."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel


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
		cond_drop_prob: float = 0.1,
		cfg_scale: float = 1.5,
	) -> None:
		super().__init__()
		self.cond_drop_prob = float(cond_drop_prob)
		self.cfg_scale = float(cfg_scale)
		self.latent_channels = latent_channels
		self.cross_attention_dim = cross_attention_dim
		self.history_len = int(history_len)
		self.num_latent_frames = self.history_len + 1
		stacked_in = self.num_latent_frames * latent_channels

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

		self.unet = unet

		self.num_actions = int(num_actions)
		self.null_action_index = self.num_actions
		self.action_embedding = nn.Embedding(self.num_actions + 1, cross_attention_dim)
		nn.init.normal_(self.action_embedding.weight, std=0.02)
		with torch.no_grad():
			self.action_embedding.weight[self.null_action_index].zero_()

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		action: torch.Tensor,
	) -> torch.Tensor:
		"""Stacked-frame denoise: predict noise / v for the next frame (4-channel UNet output).

		``action`` indices are env actions ``0 .. num_actions-1``, or ``null_action_index`` (= ``num_actions``)
		for unconditional / CFG. Training: ``cond_drop_prob`` replaces some rows with the null index.
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		x = noisy_latents.reshape(B, F * C, H, W).contiguous()
		a = action
		p = self.cond_drop_prob
		if p > 0 and self.training:
			drop = torch.rand(B, device=a.device, dtype=torch.float32) < p
			a = torch.where(drop, torch.full_like(a, self.null_action_index), a)
		enc = self.action_embedding(a).unsqueeze(1)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
		return out
