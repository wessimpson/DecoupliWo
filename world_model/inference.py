from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from torch.amp import autocast

from data.dataset import RolloutClipDataset
from world_model.model.world_model import get_model, BUFFER_SIZE


def tensor_to_image(x: torch.Tensor) -> Image.Image:
	x = ((x.clamp(-1, 1) + 1.0) * 0.5).cpu().permute(1, 2, 0).numpy()
	x = (x * 255.0).round().astype(np.uint8)
	return Image.fromarray(x)


def main() -> None:
	parser = argparse.ArgumentParser(description="World model inference (diffusion-style) using WanVAE.")
	parser.add_argument("--env", type=str, default="space_invaders")
	parser.add_argument("--transitions_root", type=str, default=str(Path("data") / "Transitions"))
	parser.add_argument("--wan_vae_dir", type=str, default=str(Path("world_model") / "checkpoint" / "vae"))
	parser.add_argument("--num_actions", type=int, default=18)
	parser.add_argument("--seq_len", type=int, default=BUFFER_SIZE)
	parser.add_argument("--steps", type=int, default=16)
	parser.add_argument("--denoise_steps", type=int, default=30)
	parser.add_argument("--out", type=str, default="wm_rollout.gif")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds = RolloutClipDataset(Path(args.transitions_root) / args.env, seq_len=args.seq_len, image_size=84, stride=1)
	sample = ds[0]
	obs_btchw = sample["obs"].unsqueeze(0).to(device)  # [1, T, 3, H, W] in [-1,1]
	actions_t = sample["actions"].to(device)

	unet, vae, action_embedding, noise_scheduler = get_model(
		action_embedding_dim=args.num_actions, wan_vae_dir=args.wan_vae_dir, latent_channels=16, skip_image_conditioning=False
	)
	unet = unet.to(device)
	vae = vae.to(device)
	action_embedding = action_embedding.to(device)
	image_processor = VaeImageProcessor(vae_scale_factor=8)
	unet.eval()
	vae.eval()

	def encode_context(images_btchw: torch.Tensor) -> torch.Tensor:
		b, t, c, h, w = images_btchw.shape
		assert t == BUFFER_SIZE
		imgs = torch.nn.functional.interpolate(images_btchw.view(b * t, c, h, w), size=(256, 256), mode="bilinear", align_corners=False)
		videos = imgs.unsqueeze(2)  # [B*T, 3, 1, H, W]
		z = vae.single_encode(videos.to(device), device=device)  # [B*T, 16, 1, h', w']
		if z.dim() == 5:
			z = z.squeeze(2)
		latent = z.view(b, t, 16, z.shape[-2], z.shape[-1])
		return latent

	with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
		context_latents = encode_context(obs_btchw[:, :BUFFER_SIZE])  # [1,B,16,h',w']
		batch_size = 1
		latent_h, latent_w = context_latents.shape[-2], context_latents.shape[-1]
		num_channels_latents = 16
		images = []
		a_t = actions_t[-1:].view(1).to(device)
		enc_states = action_embedding(a_t)  # [1,768]
		for _ in range(args.steps):
			target = randn_tensor((batch_size, num_channels_latents, latent_h, latent_w), generator=None, device=device, dtype=unet.dtype)
			noise_scheduler.set_timesteps(args.denoise_steps, device=device)
			timesteps = noise_scheduler.timesteps
			latents = torch.cat([context_latents, target.unsqueeze(1)], dim=1).view(batch_size, -1, latent_h, latent_w)
			for t in timesteps:
				latent_in = noise_scheduler.scale_model_input(latents, t)
				noise_pred = unet(
					latent_in, t, encoder_hidden_states=enc_states.unsqueeze(0), class_labels=torch.zeros(batch_size, dtype=torch.long, device=device), return_dict=False
				)[0]
				reshaped = latents.view(batch_size, BUFFER_SIZE + 1, num_channels_latents, latent_h, latent_w)
				last = reshaped[:, -1]
				denoised_last = noise_scheduler.step(noise_pred, t, last, return_dict=False)[0]
				reshaped[:, -1] = denoised_last
				latents = reshaped.view(batch_size, -1, latent_h, latent_w)
			last_latent = latents.view(batch_size, BUFFER_SIZE + 1, num_channels_latents, latent_h, latent_w)[:, -1].unsqueeze(2)
			img = vae.single_decode(last_latent, device=device)[:, :, 0]  # [B,3,H,W]
			images.append(tensor_to_image(img[0]))
			# slide window
			context_latents = torch.cat([context_latents[:, 1:], last_latent.squeeze(2).unsqueeze(1)], dim=1)

	if images:
		images[0].save(args.out, save_all=True, append_images=images[1:], duration=100, loop=0)
		print("Saved:", args.out)
	else:
		print("No images generated.")


if __name__ == "__main__":
	main()

