"""
Train dynamics (diffusion UNet) on pre-VAE-encoded transition shards.

Same loop as ``train_world_model.py``, but batches load ``history_latents`` / ``target_latent``
from ``data/transitions/encoded/...`` (see ``encode_transition.py``). VAE is still loaded for
decode-only validation (PSNR / previews).

VRAM tips: lower ``--batch_size``; pass ``--gradient_checkpointing``; use ``--skip_autoregressive_val``
or shorter ``--val_ar_horizons``; lower ``--val_num_inference_steps`` vs training; raise
``--validation_every`` to val less often.
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

from world_model.dataset import EncodedRolloutVideoDataset, preprocess_latent
from world_model.model.error_buffer import ErrorBuffer
from world_model.model.world_model import WorldModel

CONTEXT_LEN = 2
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "v_prediction"
PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"


def psnr_neg1_to_01(pred: torch.Tensor, tgt: torch.Tensor) -> float:
	p = ((pred.clamp(-1, 1) + 1) * 0.5).float()
	t = ((tgt.clamp(-1, 1) + 1) * 0.5).float()
	mse = (p - t).pow(2).mean().item()
	if mse <= 0:
		return float("inf")
	return 10.0 * math.log10(1.0 / mse)


def future_residuals_as_history_block(delta_bn: torch.Tensor, K: int) -> torch.Tensor:
	if delta_bn.dim() == 4:
		delta_bn = delta_bn.unsqueeze(1)
	B, N, C, h, w = delta_bn.shape
	if N == K:
		return delta_bn
	if N > K:
		return delta_bn[:, :K]
	pad = delta_bn[:, -1:].expand(B, K - N, C, h, w)
	return torch.cat([delta_bn, pad], dim=1)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train dynamics on encoded transition latents.")
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded", help="Under transitions_root, same as encode script.")
	p.add_argument("--vae_checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"))
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--num_train_epochs", type=int, default=5)
	p.add_argument("--max_train_steps", type=int, default=500_000)
	p.add_argument("--lr", type=float, default=5e-5)
	p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
	p.add_argument("--lr_warmup_steps", type=int, default=500)
	p.add_argument("--log_dir", type=str, default="runs/world_model_dynamics")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--gradient_checkpointing", action="store_true")
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--gamma_warmup_steps", type=int, default=500)
	p.add_argument("--error_buffer_cap", type=int, default=5_000)
	p.add_argument("--validation_every", type=int, default=1_000)
	p.add_argument("--checkpoint_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit_encoded"))
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument("--val_samples", type=int, default=8)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	p.add_argument("--val_ar_horizons", type=str, default="1,10,30")
	p.add_argument(
		"--skip_autoregressive_val",
		action="store_true",
		help="Skip AR val (no multi-step generate+decode loop); keeps val MSE + single-frame preview only. Saves a lot of VRAM at validation.",
	)
	p.add_argument(
		"--val_num_inference_steps",
		type=int,
		default=None,
		help="DDIM steps during validation only (default: same as --num_inference_steps). Lower = less VRAM/time during val.",
	)
	p.add_argument(
		"--val_gt_decode_chunk",
		type=int,
		default=4,
		help="When building val GT RGB from latents, decode this many frames per VAE forward (lower = lower VRAM spike at startup).",
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

	K = args.context_len
	seq_len = K + 1
	encoded_root = Path(args.transitions_root) / args.encoded_subdir

	mk_ds = lambda d: EncodedRolloutVideoDataset(d, seq_len=seq_len, stride=1, num_actions=args.num_actions).with_transform(
		partial(preprocess_latent, history_len=K),
	)
	ds_train = mk_ds(encoded_root / "train" / args.env)
	ds_test = mk_ds(encoded_root / "test" / args.env)
	C = int(ds_train[0]["history_latents"].shape[1])
	print(f"Dataset windows: train={len(ds_train):,} test={len(ds_test):,}  latent_C={C}")
	if args.skip_autoregressive_val:
		print("AR validation disabled (--skip_autoregressive_val); val uses fewer GPU buffers.")

	loader = DataLoader(
		ds_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)

	world_model = WorldModel(
		num_actions=args.num_actions,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		vae_checkpoint=args.vae_checkpoint,
		prediction_type=PREDICTION_TYPE,
		history_len=K,
		gradient_checkpointing=args.gradient_checkpointing,
		pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
	).to(device)
	if world_model.latent_channels != C:
		raise ValueError(f"encoded latent C={C} != model latent_channels={world_model.latent_channels}")

	n_diff = sum(p.numel() for p in world_model.diffuser.parameters())
	print(f"Diffuser parameters: {n_diff:,}")

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

	run_name = f"{args.env}_K{K}_encoded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)
	error_buffer = ErrorBuffer(capacity=args.error_buffer_cap)

	val_ar_horizons = _parse_int_list(args.val_ar_horizons)
	if any(h < 1 for h in val_ar_horizons):
		raise ValueError("--val_ar_horizons must be positive integers")
	val_ar_max = max(val_ar_horizons)
	val_inf_steps = int(args.val_num_inference_steps) if args.val_num_inference_steps is not None else int(args.num_inference_steps)

	with torch.no_grad():
		val_items = [ds_test[i % len(ds_test)] for i in range(args.val_samples)]
		# Keep val caches on CPU (like train_world_model pixel buffers) to free VRAM during training.
		val_hist_z = torch.stack([s["history_latents"] for s in val_items])
		val_tgt_z = torch.stack([s["target_latent"] for s in val_items])
		val_hist_act = torch.stack([s["history_actions"] for s in val_items]).long()

		val_ar_items: list[dict] = []
		if not args.skip_autoregressive_val:
			for i in range(args.val_samples):
				idx = i % len(ds_test)
				row = ds_test.try_contiguous_ar(idx, K, val_ar_max)
				if row is not None:
					val_ar_items.append(row)
			if not val_ar_items:
				print(
					f"Warning: no val windows have K+{val_ar_max} contiguous rows; "
					f"skipping autoregressive val PSNR."
				)
		val_ar_hist_z = torch.stack([s["history_latents"] for s in val_ar_items]) if val_ar_items else None
		val_ar_hist_act = torch.stack([s["history_actions"] for s in val_ar_items]).long() if val_ar_items else None
		val_ar_fut = torch.stack([s["future_action_frames"] for s in val_ar_items]).long() if val_ar_items else None
		# Decode GT future latents → RGB on CPU (skipped if --skip_autoregressive_val).
		val_ar_gt_px: torch.Tensor | None = None
		if val_ar_items:
			gt_z = torch.stack([s["gt_future_latents"] for s in val_ar_items])
			dc = max(1, int(args.val_gt_decode_chunk))
			parts: list[torch.Tensor] = []
			for t0 in range(0, val_ar_max, dc):
				t1 = min(t0 + dc, val_ar_max)
				parts.append(world_model.decode_video(gt_z[:, t0:t1].to(device)).cpu())
			val_ar_gt_px = torch.cat(parts, dim=1)

	def save_checkpoint(step: int) -> None:
		d = ckpt_root / f"step_{step:07d}"
		world_model.save_diffuser(d)
		torch.save({"step": step, "optimizer": optimizer.state_dict(), "args": vars(args)}, d / "trainer_state.pt")

	global_step = 0
	last_gamma_eff = 0.0
	last_loss: float | None = None
	pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

	while global_step < total_steps:
		for batch in loader:
			if global_step >= total_steps:
				break

			if args.validation_every > 0 and global_step % args.validation_every == 0:
				world_model.eval()
				with torch.no_grad():
					vh = val_hist_z.to(device)
					vt = val_tgt_z.to(device)
					va = val_hist_act.to(device)
					Bv = vh.shape[0]
					ts = torch.randint(0, world_model.num_train_timesteps, (Bv,), device=device).long()
					ns = torch.randn_like(vt, dtype=world_model.diffuser.unet.dtype)
					pred_v, tgt_v = world_model.diffusion_forward(
						vh, vt, va, ts, ns,
					)
					val_mse = F.mse_loss(pred_v.float(), tgt_v.float())
					writer.add_scalar("val/mse", val_mse.item(), global_step)
					pbar.set_postfix(
						loss=("—" if last_loss is None else f"{last_loss:.4f}"),
						val_mse=f"{val_mse.item():.4f}",
						gamma=f"{last_gamma_eff:.4f}",
						buf=len(error_buffer),
					)

					if (
						not args.skip_autoregressive_val
						and val_ar_hist_z is not None
						and val_ar_gt_px is not None
					):
						z_ar = val_ar_hist_z.to(device)
						h_act = val_ar_hist_act.to(device)
						fut_dev = val_ar_fut.to(device)
						decoded_frames: list[torch.Tensor] = []
						for s in tqdm(range(val_ar_max), desc="val AR", leave=False, dynamic_ncols=True):
							fa = h_act[:, -1] if s == 0 else fut_dev[:, s - 1]
							z_next = world_model.generate_next_frame(
								z_ar, h_act, fa,
								num_inference_steps=val_inf_steps,
							)
							decoded_frames.append(world_model.decode_video(z_next))
							z_ar = torch.cat([z_ar[:, 1:], z_next], dim=1)
							h_act = torch.cat([h_act[:, 1:], fa.unsqueeze(1)], dim=1)
						for h in val_ar_horizons:
							pred_h = torch.cat(decoded_frames[:h], dim=1)
							tgt_h = val_ar_gt_px[:, :h].to(device)
							hid = f"h{h:02d}"
							writer.add_scalar(f"val/psnr_ar/{hid}", psnr_neg1_to_01(pred_h, tgt_h), global_step)
							gen_f0 = (decoded_frames[h - 1][:1, 0].clamp(-1, 1) + 1) * 0.5
							tgt_f0 = (tgt_h[:1, h - 1].clamp(-1, 1) + 1) * 0.5
							writer.add_images(f"val/ar_rollout/{hid}/generated", gen_f0.cpu(), global_step)
							writer.add_images(f"val/ar_rollout/{hid}/target", tgt_f0.cpu(), global_step)
						del decoded_frames
					else:
						chunk_lat = world_model.generate_next_frame(
							vh, va, va[:, -1],
							num_inference_steps=val_inf_steps,
						)
						dec1 = world_model.decode_video(chunk_lat)
						tgt01 = (world_model.decode_frames(vt)[:1, 0].clamp(-1, 1) + 1) * 0.5
						img01 = (dec1[:1, 0].clamp(-1, 1) + 1) * 0.5
						writer.add_images("val/generated_f0", img01.cpu(), global_step)
						writer.add_images("val/target_f0", tgt01.cpu(), global_step)
					del vh, vt, va
				world_model.train()

			optimizer.zero_grad(set_to_none=True)
			z_hist = batch["history_latents"].to(device)
			z_tgt = batch["target_latent"].to(device)
			hist_actions = batch["history_actions"].to(device)
			B = z_hist.shape[0]

			Wg = args.gamma_warmup_steps
			gamma_eff = float(args.gamma) if Wg <= 0 else float(args.gamma) * min(1.0, global_step / float(Wg))
			last_gamma_eff = gamma_eff

			delta_hist = error_buffer.sample_like(z_hist) if error_buffer.ready() else None
			timesteps = torch.randint(0, world_model.num_train_timesteps, (B,), device=device).long()
			noise = torch.randn_like(z_tgt, dtype=world_model.diffuser.unet.dtype)

			with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				model_pred, target = world_model.diffusion_forward(
					z_hist, z_tgt, hist_actions, timesteps, noise,
					delta_hist=delta_hist, gamma=gamma_eff,
				)
				loss = F.mse_loss(model_pred.float(), target.float())
			last_loss = loss.item()

			if scaler.is_enabled():
				scaler.scale(loss).backward()
			else:
				loss.backward()

			with torch.no_grad():
				alpha_bar = world_model.diffuser.noise_scheduler.alphas_cumprod.to(device)[timesteps]
				sqrt_a = alpha_bar.sqrt().view(B, 1, 1, 1)
				sqrt_1ma = (1 - alpha_bar).sqrt().view(B, 1, 1, 1)
				noisy_tgt = (sqrt_a * z_tgt + sqrt_1ma * noise).to(model_pred.dtype)
				pt = world_model.diffuser.noise_scheduler.config.prediction_type
				if pt == "v_prediction":
					z_hat = sqrt_a * noisy_tgt - sqrt_1ma * model_pred
				elif pt == "sample":
					z_hat = model_pred
				else:
					z_hat = (noisy_tgt - sqrt_1ma * model_pred) / sqrt_a.clamp(min=1e-8)
				delta_fut = z_hat - z_tgt.to(z_hat.dtype)
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
			pbar.set_postfix(loss=f"{loss.item():.4f}", gamma=f"{last_gamma_eff:.4f}", buf=len(error_buffer))

			if global_step > 0 and global_step % 20 == 0:
				writer.add_scalar("train/loss", loss.item(), global_step)
				writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

			if global_step > 0 and args.save_every > 0 and global_step % args.save_every == 0:
				save_checkpoint(global_step)

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Training finished.")


if __name__ == "__main__":
	main()
