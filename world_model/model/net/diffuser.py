"""Temporal denoiser core: SD1.5 spatial backbone + pretrained AnimateDiff motion modules.

Conditions on full per-frame actions [B, F] aligned with latents [B, F, C, H, W]
(history + future when WorldModel concatenates along F).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, MotionAdapter, UNet2DConditionModel
from diffusers.models.unets.unet_motion_model import UNetMotionModel

MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


class Diffuser(nn.Module):
	"""Temporal denoiser: SD1.5 spatial + AnimateDiff temporal + action cross-attn."""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		cross_attention_dim: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
		pretrained_model_name_or_path: Optional[str] = None,
		motion_adapter_id: str = MOTION_ADAPTER_ID,
	) -> None:
		super().__init__()
		self.latent_channels = latent_channels
		self.cross_attention_dim = cross_attention_dim

		unet2d = UNet2DConditionModel.from_pretrained(
			pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False,
		)
		assert latent_channels == unet2d.config.in_channels, (
			f"VAE latent_channels={latent_channels} != UNet in_channels={unet2d.config.in_channels}"
		)
		adapter = MotionAdapter.from_pretrained(motion_adapter_id)
		self.unet: UNetMotionModel = UNetMotionModel.from_unet2d(unet2d, motion_adapter=adapter)

		self.action_embedding = nn.Embedding(num_actions, cross_attention_dim)
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(cross_attention_dim, cross_attention_dim),
		)
		nn.init.normal_(self.action_embedding.weight, std=0.02)

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)
		self._attn_lora_injected = False

	def configure_trainable(
		self,
		policy: str,
		*,
		unet_top_n_blocks: int = 2,
		lora_rank: int = 8,
		lora_alpha: float = 8.0,
		lora_include_motion: bool = False,
	) -> None:
		from world_model.model.net.trainable_parts import apply_diffuser_train_policy

		apply_diffuser_train_policy(
			self,
			policy,
			unet_top_n_blocks=unet_top_n_blocks,
			lora_rank=lora_rank,
			lora_alpha=lora_alpha,
			lora_include_motion=lora_include_motion,
		)

	def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
		"""[B, F] → [B, F, D] cross-attention context."""
		return self.mlp(self.action_embedding(actions))

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		actions: torch.Tensor,
	) -> torch.Tensor:
		"""Denoise latent frames with temporal attention.

		Args:
			noisy_latents: [B, F, C, H, W]
			timesteps:     [B]
			actions:       [B, F]  per-frame action ids (past + future aligned with F)

		Returns:
			[B, F, C, H, W]  model output (epsilon / v / sample per scheduler config)
		"""
		B, F, C, H, W = noisy_latents.shape
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		enc = self.embed_actions(actions)  # [B, F, D]
		enc = enc.repeat_interleave(F, dim=0)  # [B*F, F, D]

		x = noisy_latents.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]  # [B, C, F, H, W]

		return out.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W]
