"""
Next-frame temporal world models.

Base model: frozen VAE + action-conditioned UNet denoiser trained on original variants.
Residual model: frozen base model + rule-conditioned UNet denoiser predicting a correction in v-space.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from world_model.model.net import Diffuser, VAE
from world_model.model.net.vae import DEFAULT_VAE_PT


def diffusion_target(
	sched,
	z_tgt: torch.Tensor,
	noise: torch.Tensor,
	timesteps: torch.Tensor,
) -> torch.Tensor:
	"""Target matching the scheduler prediction type."""
	pt = sched.config.prediction_type
	if pt == "v_prediction":
		return sched.get_velocity(z_tgt, noise, timesteps)
	if pt == "sample":
		return z_tgt
	return noise


class WorldModel(nn.Module):
	"""Frozen VAE plus a single diffusion denoiser over raw VAE latents."""

	def __init__(
		self,
		num_actions: int,
		cross_attention_dim: int,
		vae_checkpoint: str | Path | None = None,
		prediction_type: str = "v_prediction",
		history_len: int = 2,
		gradient_checkpointing: bool = False,
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
		self.history_len = int(history_len)

		pt = Path(DEFAULT_VAE_PT if vae_checkpoint is None else vae_checkpoint)
		self.vae = VAE(checkpoint=pt)
		self.vae.freeze()
		self.latent_channels = self.vae.latent_channels

		self.diffuser = Diffuser(
			num_actions=num_actions,
			latent_channels=self.latent_channels,
			cross_attention_dim=cross_attention_dim,
			history_len=self.history_len,
			prediction_type=prediction_type,
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			num_rules=num_rules,
			cfg_both_drop_prob=cfg_both_drop_prob,
			cfg_action_drop_prob=cfg_action_drop_prob,
			cfg_rule_drop_prob=cfg_rule_drop_prob,
			cfg_scale_action=cfg_scale_action,
			cfg_scale_rule=cfg_scale_rule,
			zero_init_output=zero_init_output,
		)
		if gradient_checkpointing:
			self.diffuser.unet.enable_gradient_checkpointing()

		self.num_train_timesteps = int(self.diffuser.noise_scheduler.config.num_train_timesteps)

	def trainable_parameters(self):
		yield from self.diffuser.parameters()

	def enable_gradient_checkpointing(self) -> None:
		self.diffuser.unet.enable_gradient_checkpointing()

	# VAE helpers

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,T,3,H,W] -> [B,T,C,h,w] scaled latents."""
		return self.vae.encode_video(pixels.to(next(self.vae.parameters()).device))

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,T,C,h,w] -> [B,T,3,H,W]."""
		return self.vae.decode_video(latents.to(next(self.vae.parameters()).device))

	def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
		"""[B,3,H,W] -> [B,C,h,w] scaled latents."""
		return self.vae.encode_pixels(pixels.to(next(self.vae.parameters()).device))

	def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
		"""[B,C,h,w] -> [B,3,H,W]."""
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
		"""Training forward for one denoiser.

		Returns `(model_pred, target)` where target is noise / v / sample according
		to the scheduler. For the residual model, the caller converts this target
		to a delta target by subtracting the frozen base prediction.
		"""
		B, _, _, _, _ = z_hist.shape
		device = z_tgt.device

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)

		sched = self.diffuser.noise_scheduler
		noisy_tgt = sched.add_noise(z_tgt, noise, timesteps)
		x = torch.cat([z_hist, noisy_tgt.unsqueeze(1)], dim=1)
		a_t = history_actions[:, -1].to(device)
		roh = None
		if rule_onehot is not None:
			roh = rule_onehot.to(device=device, dtype=self.diffuser.action_embedding.weight.dtype)
		model_pred = self.diffuser(x, timesteps, a_t, roh)
		target = diffusion_target(sched, z_tgt, noise, timesteps)
		assert model_pred.shape[0] == B
		return model_pred, target

	def predict_from_noisy(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		transition_action: torch.Tensor,
		rule_onehot: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Raw denoiser prediction without CFG expansion."""
		device = noisy_latents.device
		a_t = transition_action.to(device=device, dtype=torch.long)
		roh = None
		if rule_onehot is not None:
			roh = rule_onehot.to(device=device, dtype=self.diffuser.action_embedding.weight.dtype)
		return self.diffuser(noisy_latents, timesteps, a_t, roh)

	def guided_prediction(
		self,
		noisy_latents: torch.Tensor,
		timesteps: torch.Tensor,
		transition_action: torch.Tensor,
		rule_onehot: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""CFG prediction for either action-only base or action+rule residual denoisers."""
		B = int(noisy_latents.shape[0])
		device = noisy_latents.device
		a_t = transition_action.to(device=device, dtype=torch.long)
		null_a = torch.full_like(a_t, self.diffuser.null_action_index)
		sc_a = float(self.diffuser.cfg_scale_action)
		sc_r = float(self.diffuser.cfg_scale_rule)

		if self.diffuser.num_rules <= 0:
			if math.isclose(sc_a, 0.0, rel_tol=0.0, abs_tol=1e-6):
				return self.diffuser(noisy_latents, timesteps, null_a, None)
			if math.isclose(sc_a, 1.0, rel_tol=0.0, abs_tol=1e-6):
				return self.diffuser(noisy_latents, timesteps, a_t, None)
			pred_a = self.diffuser(noisy_latents, timesteps, a_t, None)
			pred_0 = self.diffuser(noisy_latents, timesteps, null_a, None)
			return pred_0 + sc_a * (pred_a - pred_0)

		dt_rule = self.diffuser.action_embedding.weight.dtype
		if rule_onehot is None:
			roh = torch.zeros(B, self.diffuser.num_rules, device=device, dtype=dt_rule)
		else:
			roh = rule_onehot.to(device=device, dtype=dt_rule)
		roh_u = torch.zeros(B, self.diffuser.num_rules, device=device, dtype=dt_rule)

		if math.isclose(sc_a, 0.0, rel_tol=0.0, abs_tol=1e-6) and math.isclose(sc_r, 0.0, rel_tol=0.0, abs_tol=1e-6):
			return self.diffuser(noisy_latents, timesteps, null_a, roh_u)
		if math.isclose(sc_a, 1.0, rel_tol=0.0, abs_tol=1e-6) and math.isclose(sc_r, 1.0, rel_tol=0.0, abs_tol=1e-6):
			return self.diffuser(noisy_latents, timesteps, a_t, roh)
		pred_aa = self.diffuser(noisy_latents, timesteps, a_t, roh)
		pred_0a = self.diffuser(noisy_latents, timesteps, null_a, roh)
		pred_a0 = self.diffuser(noisy_latents, timesteps, a_t, roh_u)
		pred_00 = self.diffuser(noisy_latents, timesteps, null_a, roh_u)
		return pred_00 + sc_a * (pred_aa - pred_0a) + sc_r * (pred_aa - pred_a0)

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
		"""Generate next latent frame `[B,1,C,h,w]` with this denoiser."""
		B, _, C, h, w = z_hist.shape
		device = z_hist.device
		dtype = self.diffuser.unet.dtype

		if delta_hist is not None and gamma > 0:
			z_hist = z_hist + gamma * delta_hist
		elif gamma > 0:
			z_hist = z_hist + gamma * torch.randn_like(z_hist)
		z_hist = z_hist.to(dtype=dtype)
		a_t = transition_action.to(device=device, dtype=torch.long)

		latents = torch.randn(B, C, h, w, device=device, dtype=dtype)
		sched = self.diffuser.noise_scheduler
		latents = latents * sched.init_noise_sigma
		sched.set_timesteps(num_inference_steps)
		ts = sched.timesteps
		if isinstance(ts, torch.Tensor):
			ts = ts.to(device=device)
		for t in ts:
			x = torch.cat([z_hist, latents.unsqueeze(1)], dim=1)
			t_batch = t.unsqueeze(0).expand(B).contiguous()
			pred = self.guided_prediction(x, t_batch, a_t, rule_onehot=rule_onehot)
			latents = sched.step(pred, t, latents, return_dict=False)[0]
		return latents.unsqueeze(1)

	def save_diffuser(self, out_dir: Path) -> None:
		"""Persist denoiser weights and scheduler config."""
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		torch.save(self.diffuser.unet.state_dict(), out_dir / "unet.pt")
		torch.save(self.diffuser.action_embedding.state_dict(), out_dir / "action_embedding.pt")
		if self.diffuser.rule_projection is not None:
			torch.save(self.diffuser.rule_projection.state_dict(), out_dir / "rule_projection.pt")
		sched_dir = out_dir / "noise_scheduler"
		sched_dir.mkdir(parents=True, exist_ok=True)
		self.diffuser.noise_scheduler.save_pretrained(str(sched_dir))


class ResidualWorldModel(nn.Module):
	"""Frozen base predictor plus trainable residual denoiser."""

	def __init__(self, base_model: WorldModel, residual_model: WorldModel) -> None:
		super().__init__()
		if base_model.diffuser.num_rules != 0:
			raise ValueError("base_model must be action-only (num_rules=0)")
		if residual_model.diffuser.num_rules <= 0:
			raise ValueError("residual_model must have rule correction dimensions")
		if base_model.history_len != residual_model.history_len:
			raise ValueError("base and residual history_len must match")
		if base_model.latent_channels != residual_model.latent_channels:
			raise ValueError("base and residual latent channels must match")
		self.base_model = base_model
		self.residual_model = residual_model
		self.history_len = base_model.history_len
		self.latent_channels = base_model.latent_channels
		self.num_train_timesteps = residual_model.num_train_timesteps
		self.vae = base_model.vae
		self.freeze_base()

	def freeze_base(self) -> None:
		self.base_model.eval()
		for p in self.base_model.parameters():
			p.requires_grad_(False)

	def trainable_parameters(self):
		yield from self.residual_model.trainable_parameters()

	def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
		return self.base_model.decode_video(latents)

	def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
		return self.base_model.decode_frames(latents)

	def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
		return self.base_model.encode_video(pixels)

	def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
		return self.base_model.encode_frames(pixels)

	def residual_forward(
		self,
		z_hist: torch.Tensor,
		z_tgt: torch.Tensor,
		history_actions: torch.Tensor,
		timesteps: torch.Tensor,
		noise: torch.Tensor,
		rule_onehot: torch.Tensor,
		normal_anchor_mask: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return residual pred, residual target, frozen base pred, full diffusion target."""
		device = z_tgt.device
		sched = self.residual_model.diffuser.noise_scheduler
		noisy_tgt = sched.add_noise(z_tgt, noise, timesteps)
		x = torch.cat([z_hist, noisy_tgt.unsqueeze(1)], dim=1)
		a_t = history_actions[:, -1].to(device)
		with torch.no_grad():
			base_pred = self.base_model.guided_prediction(x, timesteps, a_t)
		delta_pred = self.residual_model.predict_from_noisy(x, timesteps, a_t, rule_onehot=rule_onehot)
		full_target = diffusion_target(sched, z_tgt, noise, timesteps)
		delta_target = full_target - base_pred.detach()
		if normal_anchor_mask is not None:
			mask = normal_anchor_mask.to(device=device, dtype=torch.bool).view(-1, 1, 1, 1)
			delta_target = torch.where(mask, torch.zeros_like(delta_target), delta_target)
		return delta_pred, delta_target, base_pred, full_target

	@torch.no_grad()
	def generate_next_frame(
		self,
		z_hist: torch.Tensor,
		history_actions: torch.Tensor,
		transition_action: torch.Tensor,
		num_inference_steps: int = 30,
		rule_onehot: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Generate next frame with `v_total = v_base + delta_v` at each denoising step."""
		B, _, C, h, w = z_hist.shape
		device = z_hist.device
		dtype = self.base_model.diffuser.unet.dtype
		z_hist = z_hist.to(dtype=dtype)
		a_t = transition_action.to(device=device, dtype=torch.long)

		latents = torch.randn(B, C, h, w, device=device, dtype=dtype)
		sched = self.residual_model.diffuser.noise_scheduler
		latents = latents * sched.init_noise_sigma
		sched.set_timesteps(num_inference_steps)
		ts = sched.timesteps
		if isinstance(ts, torch.Tensor):
			ts = ts.to(device=device)
		for t in ts:
			x = torch.cat([z_hist, latents.unsqueeze(1)], dim=1)
			t_batch = t.unsqueeze(0).expand(B).contiguous()
			base_pred = self.base_model.guided_prediction(x, t_batch, a_t)
			delta_pred = self.residual_model.guided_prediction(x, t_batch, a_t, rule_onehot=rule_onehot)
			latents = sched.step(base_pred + delta_pred, t, latents, return_dict=False)[0]
		return latents.unsqueeze(1)

	def save_residual(self, out_dir: Path) -> None:
		self.residual_model.save_diffuser(out_dir)


"""test"""

if __name__ == "__main__":
	import numpy as np
	from torchvision import transforms

	K = 2
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	test_root = Path("data") / "transitions" / "test"
	shard = None
	for env_dir in sorted(test_root.iterdir()):
		for s in sorted(env_dir.glob("shard_*")):
			if (s / "obs.npy").exists():
				shard = s
				break
		if shard:
			break
	assert shard is not None, f"no shard under {test_root}"
	print(f"shard: {shard}")

	obs = np.load(shard / "obs.npy", mmap_mode="r")
	act = np.load(shard / "action.npy", mmap_mode="r")
	seq = K + 1
	assert obs.shape[0] >= seq, f"need >= {seq} frames, got {obs.shape[0]}"

	tx = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0),
		transforms.Resize((208, 160), antialias=True),
	])
	frames = torch.stack([tx(np.asarray(obs[i])[..., -3:]) for i in range(seq)])
	actions = torch.from_numpy(act[:seq].astype(np.int64))

	history = frames[:K].unsqueeze(0).to(device)
	target = frames[K].unsqueeze(0).to(device)
	hist_act = actions[:K].unsqueeze(0).to(device)

	print("loading WorldModel ...")
	wm = WorldModel(
		num_actions=18,
		cross_attention_dim=768,
		vae_checkpoint=DEFAULT_VAE_PT,
		prediction_type="v_prediction",
		history_len=K,
		pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
	).to(device)
	print(f"  latent_channels={wm.latent_channels}  num_train_timesteps={wm.num_train_timesteps}")

	with torch.no_grad():
		z_hist = wm.encode_video(history)
		z_tgt = wm.encode_frames(target)
	print(f"  z_hist={tuple(z_hist.shape)}  z_tgt={tuple(z_tgt.shape)}")

	B = 1
	t = torch.randint(0, wm.num_train_timesteps, (B,), device=device)
	noise = torch.randn_like(z_tgt)

	print("running diffusion_forward ...")
	pred, target_v = wm.diffusion_forward(z_hist, z_tgt, hist_act, t, noise)
	print(f"  model_pred={tuple(pred.shape)}  target={tuple(target_v.shape)}")
	loss = torch.nn.functional.mse_loss(pred.float(), target_v.float())
	print(f"  mse_loss={loss.item():.4f}")
	print("OK")
