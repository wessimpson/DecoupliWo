from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDIMScheduler
from tqdm.auto import tqdm

from data.dataset import RolloutClipDataset
from world_model.model.world_model import get_model, BUFFER_SIZE


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train diffusion-style world model (UNet + WanVAE) with TensorBoard logging.")
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--wan_vae_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--seq_len", type=int, default=BUFFER_SIZE + 1, help="BUFFER_SIZE context + 1 target")
	p.add_argument("--image_size", type=int, default=256)
	p.add_argument("--batch_size", type=int, default=1)
	p.add_argument("--steps", type=int, default=20000)
	p.add_argument("--lr", type=float, default=2e-4)
	p.add_argument("--log_dir", type=str, default="runs/world_model")
	p.add_argument("--denoise_steps", type=int, default=30)
	p.add_argument("--checkpoint_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit"))
	p.add_argument("--save_every", type=int, default=10000)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Report device/GPU info
	print(f"Device: {device}")
	if torch.cuda.is_available():
		try:
			print(f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
		except Exception:
			pass
	else:
		print("CUDA not available - running on CPU build of PyTorch.")
	ds = RolloutClipDataset(Path(args.transitions_root) / args.env, seq_len=args.seq_len, image_size=args.image_size, stride=1)
	pin = torch.cuda.is_available()
	# With num_workers=0, do not set prefetch_factor
	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin, persistent_workers=False)

	unet, vae, action_embedding, noise_scheduler = get_model(
		action_embedding_dim=args.num_actions, wan_vae_dir=args.wan_vae_dir, latent_channels=16, skip_image_conditioning=False
	)
	unet = unet.to(device)
	vae = vae.to(device)
	action_embedding = action_embedding.to(device)
	vae.eval()
	for p in vae.parameters():
		p.requires_grad_(False)

	optimizer = torch.optim.AdamW(list(unet.parameters()) + list(action_embedding.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
	writer = SummaryWriter(log_dir=args.log_dir)
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)

	def save_checkpoint(step: int) -> None:
		out_dir = ckpt_root / f"step_{step:07d}"
		out_dir.mkdir(parents=True, exist_ok=True)
		torch.save(unet.state_dict(), out_dir / "unet.pt")
		torch.save(action_embedding.state_dict(), out_dir / "action_embedding.pt")
		noise_scheduler.save_pretrained(str(out_dir / "noise_scheduler"))
		torch.save(
			{
				"step": step,
				"optimizer": optimizer.state_dict(),
				"args": vars(args),
			},
			out_dir / "trainer_state.pt",
		)

	global_step = 0
	pbar = tqdm(total=args.steps, desc="Training", unit="step")
	def encode_context_chunked(context_btchw: torch.Tensor, chunk_frames: int = 64) -> torch.Tensor:
		"""
		Encode context frames with WanVAE in small chunks to reduce peak memory.
		context_btchw: [B, BUFFER_SIZE, 3, H, W] in [-1, 1]
		returns: [B, BUFFER_SIZE, 16, h', w']
		"""
		b, tbuf, c, h, w = context_btchw.shape
		assert tbuf == BUFFER_SIZE
		total = b * tbuf
		latents: list[torch.Tensor] = []
		idx = 0
		while idx < total:
			end = min(idx + chunk_frames, total)
			slice_btchw = context_btchw.reshape(total, c, h, w)[idx:end]  # [N,3,H,W]
			# resize to WanVAE input (256) in smaller batches
			resized = torch.nn.functional.interpolate(slice_btchw, size=(256, 256), mode="bilinear", align_corners=False)
			videos = resized.unsqueeze(2)  # [N,3,1,256,256]
			with torch.no_grad():
				z = vae.single_encode(videos.to(device), device=device)  # [N,16,1,h',w']
				if z.dim() == 5:
					z = z.squeeze(2)
			latents.append(z)
			idx = end
		z_ctx_cat = torch.cat(latents, dim=0)  # [B*BUF,16,h',w']
		latent_h, latent_w = z_ctx_cat.shape[-2], z_ctx_cat.shape[-1]
		return z_ctx_cat.view(b, tbuf, 16, latent_h, latent_w)

	while global_step < args.steps:
		for batch in loader:
			if global_step >= args.steps:
				break
			# move small tensors with non_blocking when pin_memory is enabled
			obs = batch["obs"].to(device, non_blocking=True)  # [B, T, 3, H, W] in [-1,1]
			acts = batch["actions"].to(device, non_blocking=True)  # [B, T]
			b, t, c, h, w = obs.shape
			assert t == args.seq_len and t >= BUFFER_SIZE + 1
			context = obs[:, :BUFFER_SIZE]  # [B, BUF, 3,H,W]
			# encode context frames with WanVAE in chunks to avoid high peak memory
			z_ctx = encode_context_chunked(context, chunk_frames=64)  # [B,BUF,16,h',w']
			latent_h, latent_w = z_ctx.shape[-2], z_ctx.shape[-1]

			# Encode target frame (the (BUFFER_SIZE)-th frame in the clip)
			target_frame = obs[:, BUFFER_SIZE]  # [B, 3, H, W]
			resized_tgt = torch.nn.functional.interpolate(target_frame, size=(256, 256), mode="bilinear", align_corners=False)
			videos_tgt = resized_tgt.unsqueeze(2)  # [B,3,1,256,256]
			with torch.no_grad():
				z_tgt = vae.single_encode(videos_tgt, device=device)  # [B,16,1,h',w']
				if z_tgt.dim() == 5:
					z_tgt = z_tgt.squeeze(2)

			# Sample per-sample timestep and noise for diffusion training (predict-noise objective)
			timesteps = torch.randint(0, DDIMScheduler.from_config(noise_scheduler.config).config.num_train_timesteps, (b,), device=device).long()
			noise = torch.randn_like(z_tgt, dtype=unet.dtype)
			noisy_last = noise_scheduler.add_noise(z_tgt, noise, timesteps)

			# Concatenate context latents and noisy target latent along "frame" dim then fold into channels
			concatenated = torch.cat([z_ctx, noisy_last.unsqueeze(1)], dim=1)  # [B,BUF+1,16,h',w']
			latents_in = concatenated.reshape(b, (BUFFER_SIZE + 1) * 16, latent_h, latent_w).contiguous()

			# Action conditioning on last observed action
			a_last = acts[:, BUFFER_SIZE - 1].contiguous().view(-1)
			enc_states = action_embedding(a_last).unsqueeze(1)  # [B,1,768]

			optimizer.zero_grad(set_to_none=True)
			latent_scaled = noise_scheduler.scale_model_input(latents_in, timesteps)
			model_pred = unet(latent_scaled, timesteps, encoder_hidden_states=enc_states, return_dict=False)[0]  # [B,16,h',w']
			# Predict epsilon for the target (last) frame only
			loss = F.mse_loss(model_pred.float(), noise.float())
			loss.backward()
			optimizer.step()

			if global_step % 20 == 0:
				writer.add_scalar("loss/train", loss.item(), global_step)
				# log learning rate (first param group)
				writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
			# update progress bar postfix
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})
			pbar.update(1)
			global_step += 1
			if args.save_every > 0 and global_step % args.save_every == 0:
				save_checkpoint(global_step)
		# continue until reaching args.steps

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Training finished. TensorBoard logs at:", args.log_dir)
	print("Checkpoints saved to:", ckpt_root)


if __name__ == "__main__":
	main()

