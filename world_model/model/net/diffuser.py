"""Compact UNet2D denoiser: history-encoded stack + encoded noisy target; cross-attn on action + rule."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel

from world_model.dataset import NUM_RULE_TYPES


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
		# History encoder processes all history frames jointly as one state block.
		hist_ch = self.history_len * latent_channels
		self.history_state_encoder = nn.Sequential(
			nn.Conv2d(hist_ch, hist_ch, kernel_size=3, padding=1),
			nn.SiLU(),
			nn.Conv2d(hist_ch, hist_ch, kernel_size=3, padding=1),
			nn.SiLU(),
		)
		# Noisy target frame encoder (single frame branch).
		self.frame_state_encoder = nn.Sequential(
			nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
			nn.SiLU(),
			nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
			nn.SiLU(),
		)
		self.state_token_projection = nn.Linear(hist_ch, cross_attention_dim)

		# Smaller SD-style UNet: trim width and cross-attn capacity for faster/lower-VRAM training.
		unet = UNet2DConditionModel(
			sample_size=None,
			in_channels=stacked_in,
			out_channels=latent_channels,
			cross_attention_dim=cross_attention_dim,
			layers_per_block=1,
			block_out_channels=(128, 256, 256),
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

		self.num_rules = int(NUM_RULE_TYPES)
		# [B, R] multi-hot → rule token [B, D]; bias=False so zeros → NULL / CFG dropout.
		self.rule_projection = nn.Linear(self.num_rules, cross_attention_dim, bias=False)
		nn.init.normal_(self.rule_projection.weight, std=0.02)

		self.noise_scheduler = DDIMScheduler.from_pretrained(
			pretrained_model_name_or_path, subfolder="scheduler",
		)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)

	def forward(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		action: torch.Tensor,
		rule_onehot: torch.Tensor | None = None,
		state_token: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Stacked-frame denoise: predict noise / v for the next frame (4-channel UNet output).

		``action`` indices are env actions ``0 .. num_actions-1``, or ``null_action_index`` (= ``num_actions``)
		for unconditional / CFG.

		Training: one ``u ~ U(0,1)`` per row sets dropout mode (non-overlapping intervals):
		``cfg_both_drop_prob`` → null action + zero rule; ``cfg_action_drop_prob`` → null action, keep rule;
		``cfg_rule_drop_prob`` → zero rule, keep action.

		``rule_onehot`` [B, R] float multi-hot (``world_model.dataset.RULE_TAGS``, length R); ``None`` → **NULL / base**
		(all zeros), same as a folder without ``_rules_*``.

		``state_token`` is accepted for API compatibility but intentionally unused here.
		Cross-attention uses two tokens:
		``[..., 0, :]`` = action embedding, ``[..., 1, :]`` = rule projection.
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		# History branch: process x1..xK jointly.
		z_hist = noisy_latents[:, :-1]
		h_hist = self.encode_history_stack(z_hist)  # [B, K*C, H, W]
		# Noisy target branch: process x_{K+1}_noisy separately.
		z_noisy = noisy_latents[:, -1]              # [B, C, H, W]
		h_noisy = self.frame_state_encoder(z_noisy)
		# UNet input uses encoded history + encoded noisy target, never raw x.
		x = torch.cat([h_hist, h_noisy], dim=1).contiguous()
		a = action
		dev, dt = a.device, self.action_embedding.weight.dtype
		if rule_onehot is None:
			roh = torch.zeros(B, self.num_rules, device=dev, dtype=dt)
		else:
			roh = rule_onehot.to(device=dev, dtype=dt)
			if roh.shape != (B, self.num_rules):
				raise ValueError(f"rule_onehot expected [B,{self.num_rules}], got {tuple(roh.shape)}")

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
			roh = torch.where((drop_both | drop_rule_only).unsqueeze(1), torch.zeros_like(roh), roh)
		action_enc = self.action_embedding(a)
		rule_enc = self.rule_projection(roh)
		_ = state_token  # accepted for API compatibility; conditioning stays action+rule only.
		enc = torch.stack([action_enc, rule_enc], dim=1)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
		return out

	def encode_history_stack(self, z_hist: torch.Tensor) -> torch.Tensor:
		"""Jointly encode history latents `[B,K,C,H,W]` -> `[B,K*C,H,W]`."""
		B, K, C, H, W = z_hist.shape
		if K != self.history_len:
			raise ValueError(f"Expected history_len={self.history_len}, got {K}")
		if C != self.latent_channels:
			raise ValueError(f"Expected latent C={self.latent_channels}, got {C}")
		x = z_hist.reshape(B, K * C, H, W).contiguous()
		return self.history_state_encoder(x)

	def state_token_from_history(self, z_hist: torch.Tensor) -> torch.Tensor:
		"""Joint history hidden state `[B,D]` from all history frames at once."""
		h_hist = self.encode_history_stack(z_hist)
		v = h_hist.mean(dim=(2, 3))
		return self.state_token_projection(v)
