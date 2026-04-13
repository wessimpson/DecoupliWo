"""
Train chunk-based temporal world model.

Architecture: frozen SD VAE + temporalised SD1.5 (AnimateDiff motion modules).
Loss:         epsilon MSE on future chunk only.
History:      corrupted with Matrix-3.0-style error-buffer residuals.

Val autoregressive metrics: one rollout per eval (``num_inference_steps`` per chunk). PSNR and preview images
are logged per horizon under ``val/psnr_ar/hNN`` and ``val/ar_rollout/hNN/{generated,target}`` for TensorBoard toggling.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from world_model.dataset import RolloutVideoDataset, preprocess
from world_model.model.error_buffer import ErrorBuffer
from world_model.model.net.trainable_parts import TRAINABLE_PARTS_CHOICES, count_trainable_params
from world_model.model.world_model import WorldModel

HISTORY_LEN = 8
CHUNK_LEN = 3
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "epsilon"
PRETRAINED_MODEL_NAME_OR_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def psnr_neg1_to_01(pred: torch.Tensor, tgt: torch.Tensor) -> float:
	"""Mean PSNR (dB) in [0,1] space; pred/tgt in [-1,1]."""
	p = ((pred.clamp(-1, 1) + 1) * 0.5).float()
	t = ((tgt.clamp(-1, 1) + 1) * 0.5).float()
	mse = (p - t).pow(2).mean().item()
	if mse <= 0:
		return float("inf")
	return 10.0 * math.log10(1.0 / mse)


def future_residuals_as_history_block(delta_bn: torch.Tensor, K: int) -> torch.Tensor:
	"""Map target-chunk residuals [B, N, C, h, w] to history length K for error-buffer storage."""
	B, N, C, h, w = delta_bn.shape
	if N == K:
		return delta_bn
	if N > K:
		return delta_bn[:, :K]
	pad = delta_bn[:, -1:].expand(B, K - N, C, h, w)
	return torch.cat([delta_bn, pad], dim=1)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train chunk-based temporal world model.")
	p.add_argument("--env", type=str, default="space_invaders")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--vae_checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"), help="Path to vae.pt (hub architecture + this state dict)")
	p.add_argument("--num_actions", type=int, default=18)
	p.add_argument("--history_len", type=int, default=HISTORY_LEN)
	p.add_argument("--chunk_len", type=int, default=CHUNK_LEN)
	p.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=(208, 160))
	p.add_argument("--batch_size", type=int, default=2)
	p.add_argument("--num_train_epochs", type=int, default=10)
	p.add_argument("--max_train_steps", type=int, default=100_000)
	p.add_argument("--lr", type=float, default=5e-5)
	p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
	p.add_argument("--lr_warmup_steps", type=int, default=500)
	p.add_argument("--log_dir", type=str, default="runs/world_model")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--gradient_checkpointing", action="store_true")
	p.add_argument("--gamma", type=float, default=0.1, help="History corruption scale (after warmup)")
	p.add_argument("--gamma_warmup_steps", type=int, default=500, help="Linearly ramp corruption 0→gamma over this many optimizer steps")
	p.add_argument("--error_buffer_cap", type=int, default=5_000)
	p.add_argument("--validation_every", type=int, default=10_000)
	p.add_argument("--checkpoint_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit"))
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument("--val_samples", type=int, default=4)
	p.add_argument("--num_workers", type=int, default=0)
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	p.add_argument(
		"--trainable_parts",
		type=str,
		default="full",
		choices=TRAINABLE_PARTS_CHOICES,
		help="What to train in Diffuser: full UNet+action head, motion-only, LoRA on spatial attn, last up-blocks, etc.",
	)
	p.add_argument(
		"--unet_top_n_blocks",
		type=int,
		default=2,
		help="With trainable_parts=unet_top: train last N up_blocks plus conv_out.",
	)
	p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for lora_attn / action_motion_lora.")
	p.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha scaling.")
	p.add_argument(
		"--lora_include_motion",
		action="store_true",
		help="Also wrap attention Linears inside motion_modules with LoRA (many more adapters).",
	)
	p.add_argument(
		"--val_ar_horizons",
		type=str,
		default="1,10,30",
		help="Comma-separated chunk horizons for val PSNR (autoregressive rollout, same num_inference_steps each chunk).",
	)
	return p.parse_args()


def _parse_int_list(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in s.split(",") if p.strip()]
	if not parts:
		raise ValueError("expected at least one integer")
	return tuple(int(x) for x in parts)


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	K, N = args.history_len, args.chunk_len
	seq_len = K + N

	# ── Data ──────────────────────────────────────────────────────
	trans_root = Path(args.transitions_root)
	mk_ds = lambda d: RolloutVideoDataset(d, seq_len=seq_len, stride=1, num_actions=args.num_actions).with_transform(
		partial(preprocess, history_len=K, chunk_len=N, resize_to=tuple(args.resize)),
	)
	ds_train = mk_ds(trans_root / "train" / args.env)
	ds_test = mk_ds(trans_root / "test" / args.env)
	print(f"Dataset windows: train={len(ds_train):,} test={len(ds_test):,}")

	loader = DataLoader(
		ds_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)

	# ── Model ─────────────────────────────────────────────────────
	world_model = WorldModel(
		num_actions=args.num_actions,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		vae_checkpoint=args.vae_checkpoint,
		prediction_type=PREDICTION_TYPE,
		history_len=K,
		chunk_len=N,
		gradient_checkpointing=args.gradient_checkpointing,
		pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
		trainable_parts=args.trainable_parts,
		unet_top_n_blocks=args.unet_top_n_blocks,
		lora_rank=args.lora_rank,
		lora_alpha=args.lora_alpha,
		lora_include_motion=args.lora_include_motion,
	).to(device)

	n_tr, n_tot = count_trainable_params(world_model.diffuser)
	if n_tr == 0:
		raise RuntimeError("No trainable parameters; check --trainable_parts and related flags.")
	print(f"Trainable (diffuser scalars): {n_tr:,} / {n_tot:,}  (policy={args.trainable_parts})")

	use_amp = device.type == "cuda" and args.mixed_precision != "no"
	amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
	scaler = GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))
	optimizer = torch.optim.AdamW(world_model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)

	steps_per_epoch = len(loader)
	total_steps = min(args.num_train_epochs * steps_per_epoch, args.max_train_steps)

	scheduler = get_scheduler(
		args.lr_scheduler, optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps,
		num_training_steps=total_steps,
	)

	# ── Logging / checkpoints ─────────────────────────────────────
	run_name = f"{args.env}_K{K}_N{N}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)

	# ── Error buffer ──────────────────────────────────────────────
	error_buffer = ErrorBuffer(capacity=args.error_buffer_cap)

	# ── Fixed validation samples ──────────────────────────────────
	val_ar_horizons = _parse_int_list(args.val_ar_horizons)
	if any(h < 1 for h in val_ar_horizons):
		raise ValueError("--val_ar_horizons must be positive integers")
	val_ar_max = max(val_ar_horizons)

	with torch.no_grad():
		val_items = [ds_test[i % len(ds_test)] for i in range(args.val_samples)]
		val_hist = torch.stack([s["history_frames"] for s in val_items])       # [Bv,K,3,H,W]
		val_tgt = torch.stack([s["target_frames"] for s in val_items])         # [Bv,N,3,H,W]
		val_hist_act = torch.stack([s["history_actions"] for s in val_items]).to(device)  # [Bv,K]
		val_fut_act = torch.stack([s["future_actions"] for s in val_items]).to(device)   # [Bv,N]
		z_hist_val = world_model.encode_video(val_hist)                       # [Bv,K,C,h,w]
		z_tgt_val = world_model.encode_video(val_tgt)                         # [Bv,N,C,h,w]

		val_ar_items: list[dict] = []
		for i in range(args.val_samples):
			idx = i % len(ds_test)
			row = ds_test.try_contiguous_rollout(idx, K, N, val_ar_max, tuple(args.resize))
			if row is not None:
				val_ar_items.append(row)
		if not val_ar_items:
			print(
				f"Warning: no val windows have K+N*{val_ar_max} contiguous rows; "
				f"skipping autoregressive val PSNR (shards too short at sampled indices)."
			)
		val_ar_hist = torch.stack([s["history_frames"] for s in val_ar_items]) if val_ar_items else None
		val_ar_gt = torch.stack([s["gt_chunks"] for s in val_ar_items]) if val_ar_items else None
		val_ar_hist_act = torch.stack([s["history_actions"] for s in val_ar_items]) if val_ar_items else None
		val_ar_fut = torch.stack([s["future_action_chunks"] for s in val_ar_items]) if val_ar_items else None

	def save_checkpoint(step: int) -> None:
		d = ckpt_root / f"step_{step:07d}"
		world_model.save_diffuser(d)
		torch.save({"step": step, "optimizer": optimizer.state_dict(), "args": vars(args)}, d / "trainer_state.pt")

	# ── Training loop ─────────────────────────────────────────────
	global_step = 0
	last_gamma_eff = 0.0
	pbar = tqdm(total=total_steps, desc="Training", unit="step")

	while global_step < total_steps:
		for batch in loader:
			if global_step >= total_steps:
				break
			optimizer.zero_grad(set_to_none=True)

			hist_frames = batch["history_frames"]    # [B, K, 3, H, W]
			tgt_frames = batch["target_frames"]      # [B, N, 3, H, W]
			hist_actions = batch["history_actions"]   # [B, K]
			fut_actions = batch["future_actions"]     # [B, N]
			B = hist_frames.shape[0]

			with torch.no_grad():
				z_hist = world_model.encode_video(hist_frames)   # [B, K, C, h, w]
				z_tgt = world_model.encode_video(tgt_frames)     # [B, N, C, h, w]

			# Error-buffer history corruption: fixed target gamma, linear warmup in global_step
			Wg = args.gamma_warmup_steps
			if Wg <= 0:
				gamma_eff = float(args.gamma)
			else:
				gamma_eff = float(args.gamma) * min(1.0, global_step / float(Wg))
			last_gamma_eff = gamma_eff

			delta_hist = None
			if error_buffer.ready():
				delta_hist = error_buffer.sample_like(z_hist)

			timesteps = torch.randint(0, world_model.num_train_timesteps, (B,), device=device).long()
			noise = torch.randn_like(z_tgt, dtype=world_model.diffuser.unet.dtype)

			with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				model_pred, target_noise = world_model.diffusion_forward(
					z_hist, z_tgt, hist_actions, fut_actions, timesteps, noise,
					delta_hist=delta_hist, gamma=gamma_eff,
				)
				loss = F.mse_loss(model_pred.float(), target_noise.float())

			if scaler.is_enabled():
				scaler.scale(loss).backward()
			else:
				loss.backward()

			# Update error buffer: store [B, K, C, h, w] blocks (mapped from future latent residuals)
			with torch.no_grad():
				alpha_bar = world_model.diffuser.noise_scheduler.alphas_cumprod.to(device)[timesteps]
				sqrt_a = alpha_bar.sqrt().view(B, 1, 1, 1, 1)
				sqrt_1ma = (1 - alpha_bar).sqrt().view(B, 1, 1, 1, 1)
				noisy_tgt = (sqrt_a * z_tgt + sqrt_1ma * noise).to(model_pred.dtype)
				z_hat = (noisy_tgt - sqrt_1ma * model_pred) / sqrt_a.clamp(min=1e-8)
				delta_fut = z_hat - z_tgt.to(z_hat.dtype)  # [B, N, C, h, w]
				error_buffer.push(future_residuals_as_history_block(delta_fut, K))

			if scaler.is_enabled():
				scaler.unscale_(optimizer)
			if args.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(world_model.trainable_parameters(), args.max_grad_norm)
			if scaler.is_enabled():
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			scheduler.step()
			global_step += 1
			pbar.update(1)
			pbar.set_postfix(
				loss=f"{loss.item():.4f}",
				gamma=f"{last_gamma_eff:.4f}",
				buf=len(error_buffer),
			)

			# Logging
			if global_step > 0 and global_step % 20 == 0:
				writer.add_scalar("train/loss", loss.item(), global_step)
				writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

			# Validation
			if global_step > 0 and args.validation_every > 0 and global_step % args.validation_every == 0:
				world_model.eval()
				with torch.no_grad():
					Bv = z_hist_val.shape[0]
					ts = torch.randint(0, world_model.num_train_timesteps, (Bv,), device=device).long()
					ns = torch.randn_like(z_tgt_val, dtype=world_model.diffuser.unet.dtype)
					pred_v, _ = world_model.diffusion_forward(
						z_hist_val, z_tgt_val, val_hist_act, val_fut_act, ts, ns,
					)
					val_mse = F.mse_loss(pred_v.float(), ns.float())
					writer.add_scalar("val/mse", val_mse.item(), global_step)

					# Autoregressive val PSNR: same num_inference_steps every chunk; horizons = chunk counts.
					if val_ar_hist is not None:
						z_ar = world_model.encode_video(val_ar_hist.to(device))
						h_act = val_ar_hist_act.to(device)
						assert val_ar_gt is not None and val_ar_fut is not None
						tgt_chunks = val_ar_gt.to(device)
						fut_dev = val_ar_fut.to(device)
						decoded_chunks: list[torch.Tensor] = []
						for s in range(val_ar_max):
							fa = fut_dev[:, s]
							chunk_lat = world_model.generate_next_chunk(
								z_ar, h_act, fa,
								num_inference_steps=args.num_inference_steps,
							)
							decoded_chunks.append(world_model.decode_video(chunk_lat))
							z_ar = torch.cat([z_ar[:, N:], chunk_lat], dim=1)
							h_act = torch.cat([h_act[:, N:], fa], dim=1)
						for h in val_ar_horizons:
							pred_h = torch.cat(decoded_chunks[:h], dim=1)
							tgt_h = tgt_chunks[:, :h].reshape(tgt_chunks.shape[0], h * N, *tgt_chunks.shape[3:])
							hid = f"h{h:02d}"
							writer.add_scalar(
								f"val/psnr_ar/{hid}_chunks",
								psnr_neg1_to_01(pred_h, tgt_h),
								global_step,
							)
							# First frame of the h-th predicted chunk vs same GT (toggle per horizon in Images tab).
							gen_f0 = (decoded_chunks[h - 1][:1, 0].clamp(-1, 1) + 1) * 0.5
							tgt_f0 = (tgt_chunks[:1, h - 1, 0].clamp(-1, 1) + 1) * 0.5
							writer.add_images(f"val/ar_rollout/{hid}/generated", gen_f0.cpu(), global_step)
							writer.add_images(f"val/ar_rollout/{hid}/target", tgt_f0.cpu(), global_step)
					else:
						tgt_dev = val_tgt.to(device)
						chunk_lat = world_model.generate_next_chunk(
							z_hist_val, val_hist_act, val_fut_act,
							num_inference_steps=args.num_inference_steps,
						)
						dec1 = world_model.decode_video(chunk_lat)
						img01 = (dec1[:1, 0].clamp(-1, 1) + 1) * 0.5
						tgt01 = (tgt_dev[:1, 0].clamp(-1, 1) + 1) * 0.5
						writer.add_images("val/generated_f0", img01.cpu(), global_step)
						writer.add_images("val/target_f0", tgt01.cpu(), global_step)
				world_model.train()

			if global_step > 0 and args.save_every > 0 and global_step % args.save_every == 0:
				save_checkpoint(global_step)

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Training finished.")


if __name__ == "__main__":
	main()
