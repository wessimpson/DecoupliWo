"""Compact UNet2D denoiser: stacked history + noisy next frame; cross-attn on action + rule + state tokens (seq len 3)."""

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
		# Shared frame-wise encoder applied to every latent frame (history + noisy target)
		# before concatenation into UNet input channels.
		self.frame_state_encoder = nn.Sequential(
			nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
			nn.SiLU(),
			nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
			nn.SiLU(),
		)
		self.state_token_projection = nn.Linear(latent_channels, cross_attention_dim)

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

		``rule_onehot`` [B, R] float multi-hot (``world_model.dataset.RULE_TAGS``, length R).
		If ``None``, uses **NULL**: all-zero vector (same as base-game folders without ``_rules_*`` and same as
		CFG rule-dropout target).

		``state_token`` [B, D] optional state feature projected upstream; if absent, derived from encoded history
		in the noisy stack. Cross-attention sees three tokens:
		``[..., 0, :]`` = action embedding, ``[..., 1, :]`` = rule projection, ``[..., 2, :]`` = state token.
		"""
		B, F, C, H, W = noisy_latents.shape
		assert F == self.num_latent_frames, f"Expected F={self.num_latent_frames}, got {F}"
		assert C == self.latent_channels, f"Expected C={self.latent_channels}, got {C}"

		# Shared state encoder over each frame: x1..xK and x_{K+1}_noisy.
		h_frames = self.encode_frame_stack(noisy_latents)
		x = h_frames.reshape(B, F * C, H, W).contiguous()
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
		if state_token is None:
			state_enc = self.state_token_from_encoded_stack(h_frames).to(device=dev, dtype=dt)
		else:
			state_enc = state_token.to(device=dev, dtype=dt)
			if state_enc.shape != (B, self.cross_attention_dim):
				raise ValueError(
					f"state_token expected [B,{self.cross_attention_dim}], got {tuple(state_enc.shape)}"
				)
		enc = torch.stack([action_enc, rule_enc, state_enc], dim=1)

		out = self.unet(
			x, timesteps, encoder_hidden_states=enc, return_dict=False,
		)[0]
		return out

	def encode_frame_stack(self, latents: torch.Tensor) -> torch.Tensor:
		"""Apply shared frame encoder to `[B,F,C,H,W]` -> encoded `[B,F,C,H,W]`."""
		B, F, C, H, W = latents.shape
		if C != self.latent_channels:
			raise ValueError(f"Expected latent C={self.latent_channels}, got {C}")
		z = latents.reshape(B * F, C, H, W).contiguous()
		h = self.frame_state_encoder(z)
		return h.reshape(B, F, C, H, W).contiguous()

	def state_token_from_encoded_stack(self, encoded_stack: torch.Tensor) -> torch.Tensor:
		"""Make `[B,D]` state token from encoded history frames in `[B,F,C,H,W]`."""
		if encoded_stack.shape[1] < 2:
			raise ValueError("encoded_stack needs at least one history frame and one target/noisy frame")
		h_hist = encoded_stack[:, :-1]  # history only
		v = h_hist.mean(dim=(1, 3, 4))
		return self.state_token_projection(v)

	def state_token_from_history(self, z_hist: torch.Tensor) -> torch.Tensor:
		"""Encode `[B,K,C,H,W]` history and project to a `[B,D]` state token."""
		if z_hist.shape[1] != self.history_len:
			raise ValueError(f"Expected history_len={self.history_len}, got {z_hist.shape[1]}")
		enc = self.encode_frame_stack(z_hist)
		v = enc.mean(dim=(1, 3, 4))
		return self.state_token_projection(v)
