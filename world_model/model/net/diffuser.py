"""Compact UNet2D denoiser over raw VAE latents."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel


class Diffuser(nn.Module):
	"""UNet denoiser: `[B,K+1,C,H,W]` latents folded to `[B,(K+1)*C,H,W]`.

	The base/original model uses only an action token. The residual model also
	uses a rule-correction token whose zero vector means "no correction".
	"""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		cross_attention_dim: int,
		history_len: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"] = "v_prediction",
		pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
		num_rules: int = 0,
		cfg_both_drop_prob: float = 0.10,
		cfg_action_drop_prob: float = 0.05,
		cfg_rule_drop_prob: float = 0.05,
		cfg_scale_action: float = 1.5,
		cfg_scale_rule: float = 1.5,
		zero_init_output: bool = False,
	) -> None:
		super().__init__()
		self.cfg_both_drop_prob = float(cfg_both_drop_prob)
		self.cfg_action_drop_prob = float(cfg_action_drop_prob)
		self.cfg_rule_drop_prob = float(cfg_rule_drop_prob)
		p_sum = self.cfg_both_drop_prob + self.cfg_action_drop_prob + self.cfg_rule_drop_prob
		if p_sum > 1.0:
			raise ValueError(f"cfg_*_drop_prob sum is {p_sum}, must be <= 1.0")
		self.cfg_scale_action = float(cfg_scale_action)
		self.cfg_scale_rule = float(cfg_scale_rule)
		self.latent_channels = int(latent_channels)
		self.cross_attention_dim = int(cross_attention_dim)
		self.history_len = int(history_len)
		self.num_latent_frames = self.history_len + 1
		self.num_rules = int(num_rules)
		if self.num_rules < 0:
			raise ValueError(f"num_rules must be >= 0, got {self.num_rules}")
		stacked_in = self.num_latent_frames * self.latent_channels

		self.unet = UNet2DConditionModel(
			sample_size=None,
			in_channels=stacked_in,
			out_channels=self.latent_channels,
			cross_attention_dim=self.cross_attention_dim,
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

		self.num_actions = int(num_actions)
		self.null_action_index = self.num_actions
		self.action_embedding = nn.Embedding(self.num_actions + 1, self.cross_attention_dim)
		nn.init.normal_(self.action_embedding.weight, std=0.02)
		with torch.no_grad():
			self.action_embedding.weight[self.null_action_index].zero_()

		self.rule_projection: nn.Linear | None
		if self.num_rules > 0:
			self.rule_projection = nn.Linear(self.num_rules, self.cross_attention_dim, bias=False)
			nn.init.normal_(self.rule_projection.weight, std=0.02)
		else:
			self.rule_projection = None

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

		if zero_init_output:
			self.zero_initialize_output()

	def zero_initialize_output(self) -> None:
		"""Make the denoiser predict exact zeros before residual training."""
		nn.init.zeros_(self.unet.conv_out.weight)
		if self.unet.conv_out.bias is not None:
			nn.init.zeros_(self.unet.conv_out.bias)

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		action: torch.Tensor,
		rule_onehot: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Predict noise / v / sample for the next latent frame.

		``action`` indices are env actions ``0 .. num_actions-1``, or
		``null_action_index`` for CFG. If ``num_rules > 0``, ``rule_onehot`` is
		a `[B,num_rules]` correction vector; all zeros means original behavior.
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		x = noisy_latents.reshape(B, F * C, H, W).contiguous()
		a = action.to(device=noisy_latents.device, dtype=torch.long)
		dev, dt = a.device, self.action_embedding.weight.dtype

		roh: torch.Tensor | None = None
		if self.num_rules > 0:
			if rule_onehot is None:
				roh = torch.zeros(B, self.num_rules, device=dev, dtype=dt)
			else:
				roh = rule_onehot.to(device=dev, dtype=dt)
				if roh.shape != (B, self.num_rules):
					raise ValueError(f"rule_onehot expected [B,{self.num_rules}], got {tuple(roh.shape)}")
		elif rule_onehot is not None and rule_onehot.numel() > 0:
			raise ValueError("rule_onehot was provided, but this Diffuser was built with num_rules=0")

		if self.training:
			if self.num_rules > 0:
				p0 = self.cfg_both_drop_prob
				p1 = p0 + self.cfg_action_drop_prob
				p2 = p1 + self.cfg_rule_drop_prob
				if p2 > 0:
					u = torch.rand(B, device=dev, dtype=torch.float32)
					drop_both = u < p0
					drop_action_only = (u >= p0) & (u < p1)
					drop_rule_only = (u >= p1) & (u < p2)
					null_a = torch.full_like(a, self.null_action_index)
					a = torch.where(drop_both | drop_action_only, null_a, a)
					assert roh is not None
					roh = torch.where((drop_both | drop_rule_only).unsqueeze(1), torch.zeros_like(roh), roh)
			else:
				p_action = self.cfg_both_drop_prob + self.cfg_action_drop_prob
				if p_action > 0:
					drop_action = torch.rand(B, device=dev, dtype=torch.float32) < p_action
					a = torch.where(drop_action, torch.full_like(a, self.null_action_index), a)

		action_enc = self.action_embedding(a)
		tokens = [action_enc]
		if self.rule_projection is not None:
			assert roh is not None
			tokens.append(self.rule_projection(roh))
		enc = torch.stack(tokens, dim=1)

		return self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
