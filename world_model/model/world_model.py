from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel

from world_model.model.net.vae import WanVAE

# Mirror the sample structure but use our own VAE (WanVAE) and latent channel math.
BUFFER_SIZE = 8  # number of conditioning frames


def get_model(
	action_embedding_dim: int,
	wan_vae_dir: str | Path,
	latent_channels: int = 16,
	skip_image_conditioning: bool = False,
) -> tuple[UNet2DConditionModel, WanVAE, nn.Embedding, DDIMScheduler]:
	"""
	Returns:
	- unet: diffusion U-Net that operates in latent space
	- vae: our WanVAE wrapper (decoder used; encoder used externally in helpers)
	- action_embedding: Embedding(num_actions+1, 768) for action conditioning
	- noise_scheduler: DDIMScheduler configured for v_prediction
	"""
	# Action embedding like sample (dim=768 to match UNet conditioning width)
	action_embedding = nn.Embedding(num_embeddings=action_embedding_dim + 1, embedding_dim=768)
	nn.init.normal_(action_embedding.weight, mean=0.0, std=0.02)

	# Scheduler (v_prediction like sample)
	noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
	noise_scheduler.register_to_config(prediction_type="v_prediction")

	# Base UNet (create minimal UNet; users can replace with pretrained if desired)
	unet = UNet2DConditionModel(
		sample_size=None,  # flex spatial
		in_channels=latent_channels * (BUFFER_SIZE + 1),  # fold frames into channels
		out_channels=latent_channels,  # predict noise for last frame channels
		down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
		up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
		block_out_channels=(320, 640, 1280, 1280),
		layers_per_block=2,
		cross_attention_dim=768,  # to match action embedding
		num_class_embeds=None,
	)
	# Ensure conv_in channels are correct if we skip/enable image conditioning
	if not skip_image_conditioning:
		new_in_channels = latent_channels * (BUFFER_SIZE + 1)
		if unet.config.in_channels != new_in_channels:
			unet.conv_in = nn.Conv2d(new_in_channels, 320, kernel_size=3, stride=1, padding=1)
			nn.init.xavier_uniform_(unet.conv_in.weight)
			nn.init.zeros_(unet.conv_in.bias)
			unet.config["in_channels"] = new_in_channels

	# Our own VAE (WanVAE)
	wan_vae_dir = Path(wan_vae_dir)
	vae = WanVAE(pretrained_path=str(wan_vae_dir / "Wan2.1_VAE.pth"))
	vae.eval()
	vae.requires_grad_(False)

	return unet, vae, action_embedding, noise_scheduler


def _count_params(module: nn.Module) -> tuple[int, int]:
	total = sum(p.numel() for p in module.parameters())
	trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
	return total, trainable


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	if torch.cuda.is_available():
		print(f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()

	unet, vae, action_embedding, noise_scheduler = get_model(
		action_embedding_dim=18,
		wan_vae_dir=Path("world_model") / "checkpoints" / "vae",
		latent_channels=16,
		skip_image_conditioning=False,
	)
	unet = unet.to(device)
	vae = vae.to(device)
	action_embedding = action_embedding.to(device)

	unet_total, unet_trainable = _count_params(unet)
	vae_total, vae_trainable = _count_params(vae)
	act_total, act_trainable = _count_params(action_embedding)
	print(f"UNet params: total={unet_total:,}, trainable={unet_trainable:,}")
	print(f"WanVAE params: total={vae_total:,}, trainable={vae_trainable:,}")
	print(f"Action embedding params: total={act_total:,}, trainable={act_trainable:,}")
	print(f"Scheduler prediction type: {noise_scheduler.config.prediction_type}")

	if torch.cuda.is_available():
		allocated = torch.cuda.memory_allocated() / (1024**3)
		reserved = torch.cuda.memory_reserved() / (1024**3)
		peak = torch.cuda.max_memory_allocated() / (1024**3)
		print(f"GPU memory allocated: {allocated:.3f} GB")
		print(f"GPU memory reserved:  {reserved:.3f} GB")
		print(f"GPU memory peak:      {peak:.3f} GB")
	else:
		print("CUDA not available; GPU memory stats are unavailable.")


if __name__ == "__main__":
	main()


