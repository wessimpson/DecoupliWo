"""UNet2D denoiser (base scale): stacked history + noisy next frame; cross-attn on action + rule tokens."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel

from world_model.dataset import NUM_RULE_TYPES

BASE_UNET_KWARGS: dict = {
	"layers_per_block": 1,
	"block_out_channels": (160, 320, 320),
	"down_block_types": (
		"CrossAttnDownBlock2D",
		"CrossAttnDownBlock2D",
		"DownBlock2D",
	),
	"up_block_types": (
		"UpBlock2D",
		"CrossAttnUpBlock2D",
		"CrossAttnUpBlock2D",
	),
	"attention_head_dim": (8, 8, 8),
	"norm_num_groups": 32,
}


class Diffuser(nn.Module):
	"""UNet2DConditionModel (base scale): latents ``[B, K+1, C, H, W]`` → ``[B, (K+1)*C, H, W]``."""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		cross_attention_dim: int,
		history_len: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"] = "v_prediction",
		pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
		cfg_both_drop_prob: float = 0.10,
		cfg_action_drop_prob: float = 0.05,
		cfg_rule_drop_prob: float = 0.05,
		cfg_scale_action: float = 1.5,
		cfg_scale_rule: float = 1.5,
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
		self.latent_channels = latent_channels
		self.cross_attention_dim = cross_attention_dim
		self.history_len = int(history_len)
		self.num_latent_frames = self.history_len + 1
		stacked_in = self.num_latent_frames * latent_channels

		self.unet = UNet2DConditionModel(
			sample_size=None,
			in_channels=stacked_in,
			out_channels=latent_channels,
			cross_attention_dim=cross_attention_dim,
			**BASE_UNET_KWARGS,
		)

		self.num_actions = int(num_actions)
		self.null_action_index = self.num_actions
		self.action_embedding = nn.Embedding(self.num_actions + 1, cross_attention_dim)
		nn.init.normal_(self.action_embedding.weight, std=0.02)
		with torch.no_grad():
			self.action_embedding.weight[self.null_action_index].zero_()

		self.num_rules = int(NUM_RULE_TYPES)
		self.rule_embedding = nn.Embedding(self.num_rules, cross_attention_dim)
		nn.init.normal_(self.rule_embedding.weight, std=0.02)
		self.null_rule_embedding = nn.Parameter(torch.zeros(cross_attention_dim))
		nn.init.normal_(self.null_rule_embedding, std=0.02)

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

	def _rule_condition(self, roh: torch.Tensor, rule_uc: torch.Tensor) -> torch.Tensor:
		"""``[B, 1+R, D]``: null-rule slot + atomic rule slots (zeros when inactive or CFG-dropped)."""
		B, D = roh.shape[0], self.cross_attention_dim
		dev, dt = roh.device, self.rule_embedding.weight.dtype
		uc = rule_uc.view(B, 1, 1)
		null_slot = self.null_rule_embedding.view(1, 1, -1).expand(B, 1, -1)
		null_slot = torch.where(uc, null_slot, torch.zeros(B, 1, D, device=dev, dtype=dt))
		atomic = self.rule_embedding.weight.unsqueeze(0) * roh.unsqueeze(-1)
		atomic = torch.where(uc, torch.zeros_like(atomic), atomic)
		return torch.cat([null_slot, atomic], dim=1)

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		action: torch.Tensor,
		rule_onehot: torch.Tensor | None = None,
		rule_uncond: bool = False,
	) -> torch.Tensor:
		"""Predict noise / v for the next frame.

		Cross-attention: ``[action, null_rule, atomic_1, …, atomic_R]`` (length ``2 + R``).
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		x = noisy_latents.reshape(B, F * C, H, W).contiguous()
		a = action
		dev, dt = a.device, self.action_embedding.weight.dtype
		if rule_onehot is None:
			roh = torch.zeros(B, self.num_rules, device=dev, dtype=dt)
		else:
			roh = rule_onehot.to(device=dev, dtype=dt)
			if roh.shape != (B, self.num_rules):
				raise ValueError(f"rule_onehot expected [B,{self.num_rules}], got {tuple(roh.shape)}")

		rule_uc = torch.full((B,), rule_uncond, device=dev, dtype=torch.bool)
		p0 = self.cfg_both_drop_prob
		p1 = p0 + self.cfg_action_drop_prob
		p2 = p1 + self.cfg_rule_drop_prob
		if self.training and p2 > 0:
			u = torch.rand(B, device=a.device, dtype=torch.float32)
			drop_both = u < p0
			drop_action_only = (u >= p0) & (u < p1)
			drop_rule_only = (u >= p1) & (u < p2)
			null_a = torch.full_like(a, self.null_action_index)
			a = torch.where(drop_both | drop_action_only, null_a, a)
			rule_uc = rule_uc | drop_both | drop_rule_only

		action_enc = self.action_embedding(a)
		enc = torch.cat([action_enc.unsqueeze(1), self._rule_condition(roh, rule_uc)], dim=1)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
		return out
