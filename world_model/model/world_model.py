from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.model.net import WanVAE, Diffuser

class WorldModel(nn.Module):
	"""
	World Model that combines a VAE and a Diffuser.
	Input video 
		-> VAE
		-> Diffuser
			-> UNet
			-> Action embedding
			-> Noise scheduler
		-> Output video
	"""
	def __init__(
		self,
		action_embedding_dim: int,
		wan_vae_dir: str | Path,
		latent_channels: int,
		buffer_size: int,
		cross_attention_dim: int,
		num_train_timesteps: int,
		prediction_type: str,
		model_size: str = "small",
		gradient_checkpointing: bool = False,
		context_noise_max: float = 0.7,
		context_noise_buckets: int = 10,
	) -> None:
		super().__init__()
		self.buffer_size = buffer_size
		# GameNGen-style context noise configuration
		self.context_noise_max = context_noise_max
		self.context_noise_buckets = context_noise_buckets

		self.diffuser = Diffuser(
			num_actions=action_embedding_dim,
			latent_channels=latent_channels,
			buffer_size=buffer_size,
			cross_attention_dim=cross_attention_dim,
			num_train_timesteps=num_train_timesteps,
			prediction_type=prediction_type,
			model_size=model_size,  # choose width/depth preset
		)
		if gradient_checkpointing:
			self.diffuser.unet.enable_gradient_checkpointing()
		self.num_train_timesteps = num_train_timesteps

		wan_vae_dir = Path(wan_vae_dir)
		self.vae = WanVAE(pretrained_path=str(wan_vae_dir / "Wan2.1_VAE.pth"))
		self.vae.eval()
		self.vae.requires_grad_(False)

	def enable_gradient_checkpointing(self) -> None:
		self.diffuser.unet.enable_gradient_checkpointing()

	def trainable_parameters(self):
		return self.diffuser.parameters()


	def encode_video(self, videos: torch.Tensor, device: torch.device) -> torch.Tensor:
		"""
		Encode frames with WanVAE.
		input: [B, 3, BUF, H, W]
		output: [B, 16, T, h', w']
		"""
		video_bcthw = videos.permute(0, 2, 1, 3, 4).contiguous()
		return self.vae.single_encode(video_bcthw.to(device, non_blocking=True), device=device)

	def encode_frame(self, frame: torch.Tensor, device: torch.device) -> torch.Tensor:
		"""
		Encode a single frame with WanVAE.
		input: [B, 3, H, W]
		output: [B, 16, h', w']
		"""
		video = frame.unsqueeze(2)  # [B,3,1,H,W]
		with torch.no_grad():
			z = self.vae.single_encode(video.to(device, non_blocking=True), device=device)  # [B,16,1,h',w']
		return z.squeeze(2).contiguous()

	def decode_frame(self, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
		"""
		Decode a batch of latent frames to images.
		latents: [B, 16, h', w']
		returns: [B, 3, 256, 256]
		"""
		video = latents.unsqueeze(2) # [B, 16, 1, h', w']
		with torch.no_grad():
			video = self.vae.single_decode(video, device=device)  # [B,3,1,256,256]
			return video.squeeze(2).contiguous()

	def diffusion_forward(
		self,
		z_ctx: torch.Tensor,  # [B,T,16,h',w']
		z_tgt: torch.Tensor,  # [B,16,h',w']
		context_actions: torch.Tensor,  # [B,T] int64
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Single-step diffusion training forward pass.
		Inputs:
			z_ctx: [B, T, 16, h', w']
			z_tgt:    [B, 16, h', w']
			context_actions: [B,T]
		Returns:
			model_pred: [B, 16, h', w']
			noise:      [B, 16, h', w']
		"""
		device = z_tgt.device
		B = z_tgt.shape[0]
		# 1) Diffusion noise on target latent
		num_train_timesteps = self.num_train_timesteps
		timesteps = torch.randint(0, num_train_timesteps, (B,), device=device).long()
		noise = torch.randn_like(z_tgt, dtype=self.diffuser.unet.dtype)
		noisy_last = self.diffuser.noise_scheduler.add_noise(z_tgt, noise, timesteps)
		
		# 2) Context noise augmentation on history latents
		B, Tbuf, C, latent_h, latent_w = z_ctx.shape
		# Sample per-sample alpha ~ Uniform(0, self.context_noise_max); keep alpha as shape [B]
		alpha = torch.rand((B,), device=z_ctx.device, dtype=z_ctx.dtype) * float(self.context_noise_max)
		alpha_broadcast = alpha.view(B, 1, 1, 1, 1)
		ctx_eps = torch.randn_like(z_ctx, dtype=z_ctx.dtype)
		z_ctx_noisy = z_ctx + alpha_broadcast * ctx_eps

		# 3) Build model input by folding time into channels
		concatenated = torch.cat([z_ctx_noisy, noisy_last.unsqueeze(1)], dim=1)  # [B, Tbuf+1, C, h', w']
		latents_in = concatenated.view(B, (Tbuf + 1) * C, latent_h, latent_w).contiguous()
		latent_scaled = self.diffuser.noise_scheduler.scale_model_input(latents_in, timesteps)
		
		# 4) Build conditioning from action sequence + context-noise bucket
		den = max(self.context_noise_max, 1e-8)
		bucket_idx = (alpha / den * self.context_noise_buckets).clamp(0, self.context_noise_buckets - 1).long()
		# context_actions: [B,T] -> [B,T,dim]
		act_ids = context_actions.to(device, non_blocking=True).long()
		act_cond = self.diffuser.action_embedding(act_ids)  # [B,T,dim]
		noise_token = self.diffuser.noise_level_embedding(bucket_idx).unsqueeze(1)  # [B,1,dim]
		enc_states = torch.cat([noise_token, act_cond], dim=1)  # [B,T+1,dim]
		model_pred = self.diffuser.unet(latent_scaled, timesteps, encoder_hidden_states=enc_states, return_dict=False)[0]
 		
		return model_pred, noise

	def save_diffuser(self, out_dir: Path) -> None:
		"""
		Save diffuser components (UNet, action embedding, noise scheduler config) to out_dir.
		"""
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		torch.save(self.diffuser.unet.state_dict(), out_dir / "unet.pt")
		torch.save(self.diffuser.action_embedding.state_dict(), out_dir / "action_embedding.pt")
		torch.save(self.diffuser.noise_level_embedding.state_dict(), out_dir / "noise_level_embedding.pt")
		sched_dir = out_dir / "noise_scheduler"
		sched_dir.mkdir(parents=True, exist_ok=True)
		self.diffuser.noise_scheduler.save_pretrained(str(sched_dir))



"""
This is a test script to check if the model is working correctly.
"""
def main() -> None:
	NUM_TRAIN_TIMESTEPS = 1000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Build a small model for a quick IO test
	world_model = WorldModel(
		action_embedding_dim=18,
		wan_vae_dir=Path("world_model") / "checkpoints" / "vae",
		latent_channels=16,
		buffer_size=16,
		cross_attention_dim=768,
		num_train_timesteps=1000,
		prediction_type="epsilon",
	).to(device)

	# Create dummy inputs
	B, BUF, H, W = 2, 16, 256, 256
	context = torch.rand(B, BUF, 3, H, W) * 2 - 1  # [-1,1], CPU
	target = torch.rand(B, 3, H, W) * 2 - 1        # [-1,1], CPU
	last_action = torch.zeros(B, dtype=torch.long)  # dummy action ids

	# Encode via VAE; function handles layout normalization internally
	with torch.no_grad():
		z_ctx_btchw = world_model.encode_video(context, device=device)  # [B,16,T,h',w']
		z_ctx = z_ctx_btchw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,16,h',w']
		z_tgt = world_model.encode_frame(target, device=device)       # [B,16,h',w']

	# Diffusion forward (single step noise prediction) via model.forward
	with torch.no_grad():
		model_pred, noise = world_model.diffusion_forward(z_ctx, z_tgt, last_action)

	# Decode a target latent back to image to complete IO loop
	with torch.no_grad():
		recon_video = world_model.vae.single_decode(z_tgt.unsqueeze(2), device=device)  # [B,3,1,256,256]
	print("Shapes:",
		  f"original={tuple(target.shape)}",
	      f"z_ctx={tuple(z_ctx.shape)}",
	      f"z_tgt={tuple(z_tgt.shape)}",
	      f"pred={tuple(model_pred.shape)}",
	      f"recon={tuple(recon_video.shape)}")


if __name__ == "__main__":
	main()
