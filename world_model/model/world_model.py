"""
Next-frame temporal world model.

Architecture: frozen VAE + base-scale UNet2D (stacked history + noisy next in channel dim).
Cross-attn: action + null-rule slot + one token per atomic RULE_TAGS slot.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from world_model.model.net import Diffuser, VAE
from world_model.model.net.vae import DEFAULT_VAE_PT


class WorldModel(nn.Module):
	def __init__(
		self,
		num_actions: int,
		cross_attention_dim: int,
		vae_checkpoint: str | Path | None = None,
		prediction_type: str = "v_prediction",
		history_len: int = 2,
		gradient_checkpointing: bool = False,
		pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
		cfg_both_drop_prob: float = 0.10,
		cfg_action_drop_prob: float = 0.05,
		cfg_rule_drop_prob: float = 0.05,
		cfg_scale_action: float = 1.5,
		cfg_scale_rule: float = 1.5,
	) -> None:
		super().__init__()
		self.history_len = history_len

		vc = vae_checkpoint
		if vc is not None and str(vc).strip() == "":
			vc = None
		pt = Path(DEFAULT_VAE_PT if vc is None else vc)
		self.vae = VAE(checkpoint=pt)
		self.vae.freeze()
		self.latent_channels = self.vae.latent_channels

		self.diffuser = Diffuser(
			num_actions=num_actions,
			latent_channels=self.latent_channels,
			cross_attention_dim=cross_attention_dim,
			history_len=history_len,
			prediction_type=prediction_type,
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			cfg_both_drop_prob=cfg_both_drop_prob,
			cfg_action_drop_prob=cfg_action_drop_prob,
			cfg_rule_drop_prob=cfg_rule_drop_prob,
			cfg_scale_action=cfg_scale_action,
			cfg_scale_rule=cfg_scale_rule,
		)
		if gradient_checkpointing:
			self.diffuser.unet.enable_gradient_checkpointing()

		self.num_train_timesteps = int(self.diffuser.noise_scheduler.config.num_train_timesteps)

	def trainable_parameters(self):
		yield from self.diffuser.parameters()

	def enable_gradient_checkpointing(self) -> None:
		self.diffuser.unet.enable_gradient_checkpointing()

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		return self.vae.encode_video(pixels.to(next(self.vae.parameters()).device))

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		return self.vae.decode_video(latents.to(next(self.vae.parameters()).device))

	def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
		return self.vae.encode_pixels(pixels.to(next(self.vae.parameters()).device))

	def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
		return self.vae.decode_latents(latents.to(next(self.vae.parameters()).device))

	def diffusion_forward(
		self,
		z_hist: torch.Tensor,
		z_tgt: torch.Tensor,
		history_actions: torch.Tensor,
		timesteps: torch.Tensor,
		noise: torch.Tensor,
		delta_hist: torch.Tensor | None = None,
		gamma: float = 0.0,
		rule_onehot: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		B, K, C, h, w = z_hist.shape
		device = z_tgt.device

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)

		sched = self.diffuser.noise_scheduler
		noisy_tgt = sched.add_noise(z_tgt, noise, timesteps)

		x = torch.cat([z_hist, noisy_tgt.unsqueeze(1)], dim=1)
		a_t = history_actions[:, -1].to(device)
		roh = None if rule_onehot is None else rule_onehot.to(
			device=device, dtype=self.diffuser.action_embedding.weight.dtype,
		)

		model_pred = self.diffuser(x, timesteps, a_t, roh)

		pt = sched.config.prediction_type
		if pt == "v_prediction":
			target = sched.get_velocity(z_tgt, noise, timesteps)
		elif pt == "sample":
			target = z_tgt
		else:
			target = noise
		return model_pred, target

	@torch.no_grad()
	def generate_next_frame(
		self,
		z_hist: torch.Tensor,
		history_actions: torch.Tensor,
		transition_action: torch.Tensor,
		num_inference_steps: int = 30,
		delta_hist: torch.Tensor | None = None,
		gamma: float = 0.0,
		rule_onehot: torch.Tensor | None = None,
	) -> torch.Tensor:
		B, K, C, h, w = z_hist.shape
		device = z_hist.device
		dtype = self.diffuser.unet.dtype

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)

		z_hist = z_hist.to(dtype=dtype)
		a_t = transition_action.to(device=device, dtype=torch.long)
		dt_rule = self.diffuser.action_embedding.weight.dtype
		if rule_onehot is None:
			roh = torch.zeros(B, self.diffuser.num_rules, device=device, dtype=dt_rule)
		else:
			roh = rule_onehot.to(device=device, dtype=dt_rule)

		latents = torch.randn(B, C, h, w, device=device, dtype=dtype)
		sched = self.diffuser.noise_scheduler
		latents = latents * sched.init_noise_sigma

		sched.set_timesteps(num_inference_steps)
		ts = sched.timesteps
		if isinstance(ts, torch.Tensor):
			ts = ts.to(device=device)
		sc_a = self.diffuser.cfg_scale_action
		sc_r = self.diffuser.cfg_scale_rule
		for t in ts:
			x = torch.cat([z_hist, latents.unsqueeze(1)], dim=1)
			t_batch = t.unsqueeze(0).expand(B).contiguous()
			null_a = torch.full_like(a_t, self.diffuser.null_action_index)
			if math.isclose(sc_a, 0.0, rel_tol=0.0, abs_tol=1e-6) and math.isclose(sc_r, 0.0, rel_tol=0.0, abs_tol=1e-6):
				pred = self.diffuser(x, t_batch, null_a, roh, rule_uncond=True)
			elif math.isclose(sc_a, 1.0, rel_tol=0.0, abs_tol=1e-6) and math.isclose(sc_r, 1.0, rel_tol=0.0, abs_tol=1e-6):
				pred = self.diffuser(x, t_batch, a_t, roh)
			else:
				pred_aa = self.diffuser(x, t_batch, a_t, roh)
				pred_0a = self.diffuser(x, t_batch, null_a, roh)
				pred_a0 = self.diffuser(x, t_batch, a_t, roh, rule_uncond=True)
				pred_00 = self.diffuser(x, t_batch, null_a, roh, rule_uncond=True)
				pred = pred_00 + sc_a * (pred_aa - pred_0a) + sc_r * (pred_aa - pred_a0)
			latents = sched.step(pred, t, latents, return_dict=False)[0]

		return latents.unsqueeze(1)

	def save_diffuser(self, out_dir: Path) -> None:
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		torch.save(self.diffuser.unet.state_dict(), out_dir / "unet.pt")
		torch.save(self.diffuser.action_embedding.state_dict(), out_dir / "action_embedding.pt")
		torch.save(self.diffuser.rule_embedding.state_dict(), out_dir / "rule_embedding.pt")
		torch.save(self.diffuser.null_rule_embedding.detach().cpu(), out_dir / "null_rule_embedding.pt")
		sched_dir = out_dir / "noise_scheduler"
		sched_dir.mkdir(parents=True, exist_ok=True)
		self.diffuser.noise_scheduler.save_pretrained(str(sched_dir))

	def load_diffuser_checkpoint(self, ckpt_dir: Path | str, device: torch.device) -> None:
		from diffusers import DDIMScheduler

		ckpt_dir = Path(ckpt_dir)

		def _sd(p: Path) -> dict:
			return torch.load(p, map_location=device, weights_only=True)

		unet_path = ckpt_dir / "unet.pt"
		if not unet_path.is_file():
			raise FileNotFoundError(f"Missing UNet weights: {unet_path}")
		try:
			self.diffuser.unet.load_state_dict(_sd(unet_path), strict=True)
		except RuntimeError as e:
			raise RuntimeError(
				f"UNet load failed from {unet_path}. "
				f"Ensure unet.pt matches this architecture (base UNet2D + widened conv_in)."
			) from e
		emb_path = ckpt_dir / "action_embedding.pt"
		if not emb_path.exists():
			emb_path = ckpt_dir / "future_action_embedding.pt"
		if emb_path.is_file():
			self.diffuser.action_embedding.load_state_dict(_sd(emb_path))
		rule_emb_path = ckpt_dir / "rule_embedding.pt"
		if rule_emb_path.is_file():
			self.diffuser.rule_embedding.load_state_dict(_sd(rule_emb_path))
		else:
			rule_proj_path = ckpt_dir / "rule_projection.pt"
			if rule_proj_path.is_file():
				w = _sd(rule_proj_path)["weight"]
				with torch.no_grad():
					self.diffuser.rule_embedding.weight.copy_(w.T.to(self.diffuser.rule_embedding.weight))
		null_rule_path = ckpt_dir / "null_rule_embedding.pt"
		if null_rule_path.is_file():
			null_t = torch.load(null_rule_path, map_location=device, weights_only=True)
			with torch.no_grad():
				self.diffuser.null_rule_embedding.copy_(null_t.to(self.diffuser.null_rule_embedding))
		for legacy_name in ("action_mlp.pt", "future_action_mlp.pt"):
			legacy = ckpt_dir / legacy_name
			if legacy.is_file():
				raise RuntimeError(
					f"Checkpoint has legacy {legacy_name} (MLP action head). Retrain with the current embedding-only diffuser."
				)
		sched_dir = ckpt_dir / "noise_scheduler"
		if sched_dir.is_dir():
			self.diffuser.noise_scheduler = DDIMScheduler.from_pretrained(str(sched_dir))
			pt = self.diffuser.noise_scheduler.config.prediction_type
			self.diffuser.noise_scheduler.register_to_config(prediction_type=pt)
		self.num_train_timesteps = int(self.diffuser.noise_scheduler.config.num_train_timesteps)
