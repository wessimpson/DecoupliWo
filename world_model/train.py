from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

from functools import partial
from world_model.dataset import RolloutVideoDataset, preprocess
from world_model.model.world_model import WorldModel


BUFFER_SIZE = 8
LATENT_CHANNELS = 16
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "v_prediction"


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train diffusion-style world model (UNet + WanVAE) with TensorBoard logging.")
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--wan_vae_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--seq_len", type=int, default=BUFFER_SIZE + 1, help="BUFFER_SIZE context + 1 target")
	p.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=(210, 160), help="Resize frames to (H W)")
	p.add_argument("--batch_size", type=int, default=16)
	p.add_argument("--num_train_epochs", type=int, default=10, help="Optional number of epochs to train instead of fixed steps")
	p.add_argument("--max_train_steps", type=int, default=20000, help="Optional cap on total optimizer update steps")
	p.add_argument("--lr", type=float, default=2e-4)
	p.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup steps for LR scheduler")
	p.add_argument("--log_dir", type=str, default="runs/world_model")
	p.add_argument("--denoise_steps", type=int, default=30)
	p.add_argument("--num_train_timesteps", type=int, default=1000, help="Diffusion training horizon (noise scheduler).")
	p.add_argument("--model_size", type=str, choices=["small", "base", "large"], default="base", help="UNet size preset")
	p.add_argument("--gradient_checkpointing", action="store_true", help="Enable UNet gradient checkpointing to save VRAM")
	p.add_argument("--checkpoint_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit"))
	p.add_argument("--save_every", type=int, default=10000)
	p.add_argument(
		"--num_workers",
		type=int,
		default=0,
		help="DataLoader workers. Keep 0 for large .npz shards (worker RAM blowups are common).",
	)
	p.add_argument("--prefetch_factor", type=int, default=2)
	p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps for larger effective batch size.")
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	return p.parse_args()

def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Report device/GPU info
	print(f"Device: {device}")
	if torch.cuda.is_available():
		print(f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
	else:
		print("CUDA not available - running on CPU build of PyTorch.")

	ds = RolloutVideoDataset(
		Path(args.transitions_root) / args.env,
		seq_len=args.seq_len,
		stride=1,
		num_actions=args.num_actions,
	)
	# Attach preprocessing transform: normalize and resize per args.resize
	ds = ds.with_transform(partial(preprocess, buffer_size=BUFFER_SIZE, resize_to=args.resize))

	print(f"Dataset windows: {len(ds):,}")
	# Configure DataLoader workers
	num_workers = int(args.num_workers)
	pin_memory = torch.cuda.is_available()

	loader_args = dict(
		dataset=ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	if num_workers > 0:
		loader_args["persistent_workers"] = True
		if args.prefetch_factor is not None:
			loader_args["prefetch_factor"] = int(args.prefetch_factor)

	loader = DataLoader(**loader_args)

	world_model = WorldModel(
		action_embedding_dim=args.num_actions,
		wan_vae_dir=args.wan_vae_dir,
		latent_channels=LATENT_CHANNELS,
		buffer_size=BUFFER_SIZE,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		num_train_timesteps=int(args.num_train_timesteps),
		prediction_type=PREDICTION_TYPE,
		model_size=str(args.model_size),
		gradient_checkpointing=bool(args.gradient_checkpointing),
	)
	world_model = world_model.to(device)

	use_amp = device.type == "cuda" and args.mixed_precision != "no"
	amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
	scaler = GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))

	optimizer = torch.optim.AdamW(world_model.trainable_parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

	# Compute scheduler steps similar to Accelerator logic (single-process)
	import math
	world_size = 1
	num_warmup_steps_for_scheduler = int(args.lr_warmup_steps) * world_size
	len_train_dataloader_after_sharding = math.ceil(len(loader) / world_size)
	num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / max(1, int(args.grad_accum_steps)))
	num_training_steps_for_scheduler = int(args.num_train_epochs) * num_update_steps_per_epoch * world_size
	if args.max_train_steps is not None:
		num_training_steps_for_scheduler = min(num_training_steps_for_scheduler, int(args.max_train_steps))

	# Linear warmup then cosine decay
	def lr_lambda(current_step: int) -> float:
		if num_training_steps_for_scheduler <= 0:
			return 1.0
		if current_step < num_warmup_steps_for_scheduler and num_warmup_steps_for_scheduler > 0:
			return float(current_step) / float(max(1, num_warmup_steps_for_scheduler))
		# cosine from warmup_end -> total_steps
		progress = float(current_step - num_warmup_steps_for_scheduler) / float(
			max(1, num_training_steps_for_scheduler - num_warmup_steps_for_scheduler)
		)
		import math as _m
		return 0.5 * (1.0 + _m.cos(_m.pi * progress))

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	writer = SummaryWriter(log_dir=args.log_dir)
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)
	accum_steps = max(1, int(args.grad_accum_steps))

	def save_checkpoint(step: int) -> None:
		out_dir = ckpt_root / f"step_{step:07d}"
		out_dir.mkdir(parents=True, exist_ok=True)
		world_model.save_diffuser(out_dir)
		torch.save(
			{
				"step": step,
				"optimizer": optimizer.state_dict(),
				"args": vars(args),
			},
			out_dir / "trainer_state.pt",
		)

	global_step = 0
	pbar = tqdm(total=num_training_steps_for_scheduler, desc="Training", unit="step")
	while global_step < num_training_steps_for_scheduler:
		consumed_batches = 0
		for batch in loader:
			if global_step >= num_training_steps_for_scheduler:
				break
			consumed_batches += 1
			context = batch["context_frames"]  # [B, BUF, 3, 256, 256], CPU
			target_frame = batch["target_frame"]  # [B, 3, 256, 256], CPU
			last_action = batch["last_action"]  # [B], CPU
			b, tbuf, c, h, w = context.shape
			
			# Encode context frames
			with torch.no_grad():
				z_ctx_btchw = world_model.encode_video(context, device=device)  # [B,16,BUF,h',w']
				z_ctx = z_ctx_btchw.permute(0, 2, 1, 3, 4).contiguous()  # [B,BUF,16,h',w']

			with torch.no_grad():
				z_tgt = world_model.encode_frame(target_frame, device=device)       # [B,16,h',w']

			# Diffusion forward in latent space
			with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				model_pred, noise = world_model.diffusion_forward(z_ctx, z_tgt, last_action)

			if (consumed_batches % accum_steps) == 0:
				optimizer.zero_grad(set_to_none=True)

			with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				loss = F.mse_loss(model_pred.float(), noise.float())
				loss_to_backprop = loss / accum_steps
			if scaler.is_enabled():
				scaler.scale(loss_to_backprop).backward()
			else:
				loss_to_backprop.backward()

			should_step = ((consumed_batches % accum_steps) == 0) or (global_step + 1 == num_training_steps_for_scheduler)
			if should_step:
				if scaler.is_enabled():
					scaler.step(optimizer)
					scaler.update()
				else:
					optimizer.step()
				scheduler.step()
				global_step += 1
				pbar.update(1)

			if global_step % 20 == 0:
				writer.add_scalar("loss/train", loss.item(), global_step)
				# log learning rate (first param group)
				writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
			# update progress bar postfix
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})
			if args.save_every > 0 and global_step % args.save_every == 0:
				save_checkpoint(global_step)
		if consumed_batches == 0:
			raise RuntimeError("DataLoader yielded 0 batches. Check dataset path/length and seq_len settings.")
		# continue until reaching num_training_steps_for_scheduler

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Training finished. TensorBoard logs at:", args.log_dir)
	print("Checkpoints saved to:", ckpt_root)


if __name__ == "__main__":
	main()
