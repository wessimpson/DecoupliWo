from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from torch.amp import autocast

from world_model.dataset import RolloutClipDataset, preprocess
from world_model.model.world_model import WorldModel


BUFFER_SIZE = 8
NUM_ACTIONS = 18
LATENT_CHANNELS = 16
CROSS_ATTENTION_DIM = 768
NUM_TRAIN_TIMESTEPS = 1000
PREDICTION_TYPE = "v_prediction"


def tensor_to_image(x: torch.Tensor) -> Image.Image:
	x = ((x.clamp(-1, 1) + 1.0) * 0.5).cpu().permute(1, 2, 0).numpy()
	x = (x * 255.0).round().astype(np.uint8)
	return Image.fromarray(x)


def main() -> None:
	parser = argparse.ArgumentParser(description="World model inference (diffusion-style) using WanVAE.")
	parser.add_argument("--env", type=str, default="space_invaders")
	parser.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	parser.add_argument("--preprocessed_root", type=str, default=str(Path("world_model") / "preprocessed"))
	parser.add_argument("--wan_vae_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	parser.add_argument("--seq_len", type=int, default=BUFFER_SIZE)
	parser.add_argument("--image_size", type=int, default=256)
	parser.add_argument("--steps", type=int, default=16)
	parser.add_argument("--denoise_steps", type=int, default=30)
	parser.add_argument("--out", type=str, default="wm_rollout.gif")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds = RolloutClipDataset(
		Path(args.preprocessed_root) / args.env / f"{args.image_size}px",
		seq_len=args.seq_len + 1,
		stride=1,
		num_actions=NUM_ACTIONS,
	)
	ds = ds.with_transform(partial(preprocess, buffer_size=BUFFER_SIZE))
	sample = ds[0]
	obs_btchw = sample["context_frames"].unsqueeze(0).to(device)  # [1, T, 3, H, W] in [-1,1]
	actions_t = sample["last_action"].view(1).to(device)

	world_model = WorldModel(
		action_embedding_dim=NUM_ACTIONS,
		wan_vae_dir=args.wan_vae_dir,
		latent_channels=LATENT_CHANNELS,
		buffer_size=BUFFER_SIZE,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		num_train_timesteps=NUM_TRAIN_TIMESTEPS,
		prediction_type=PREDICTION_TYPE,
	)
	world_model = world_model.to(device)
	image_processor = VaeImageProcessor(vae_scale_factor=8)
	world_model.eval()

	def encode_context(images_btchw: torch.Tensor) -> torch.Tensor:
		b, t, c, h, w = images_btchw.shape
		assert t == BUFFER_SIZE
		imgs = torch.nn.functional.interpolate(images_btchw.view(b * t, c, h, w), size=(256, 256), mode="bilinear", align_corners=False)
		videos = imgs.unsqueeze(2)  # [B*T, 3, 1, H, W]
		z = world_model.encode_video(videos, device=device)  # [B*T, 16, 1, h', w']
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
		a_t = actions_t.view(1).to(device)
		enc_states = world_model.encode_actions(a_t)  # [1,1,768]
		for _ in range(args.steps):
			target = randn_tensor((batch_size, num_channels_latents, latent_h, latent_w), generator=None, device=device, dtype=world_model.model_dtype())
			world_model.diffuser.noise_scheduler.set_timesteps(args.denoise_steps, device=device)
			timesteps = world_model.diffuser.noise_scheduler.timesteps
			latents = torch.cat([context_latents, target.unsqueeze(1)], dim=1).view(batch_size, -1, latent_h, latent_w)
			for t in timesteps:
				latent_in = world_model.scale_model_input(latents, t)
				noise_pred = world_model.predict_noise(latent_in, t, encoder_hidden_states=enc_states)
				reshaped = latents.view(batch_size, BUFFER_SIZE + 1, num_channels_latents, latent_h, latent_w)
				last = reshaped[:, -1]
				denoised_last = world_model.diffuser.noise_scheduler.step(noise_pred, t, last, return_dict=False)[0]
				reshaped[:, -1] = denoised_last
				latents = reshaped.view(batch_size, -1, latent_h, latent_w)
			last_latent = latents.view(batch_size, BUFFER_SIZE + 1, num_channels_latents, latent_h, latent_w)[:, -1].unsqueeze(2)
			img = world_model.vae.single_decode(last_latent, device=device)[:, :, 0]  # [B,3,H,W]
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
