"""Pretrain the original-mode action-conditioned dynamics model on encoded latents."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from functools import partial
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from world_model.dataset import (
	MixedEncodedRolloutVideoDataset,
	encoded_original_dirs_under_split,
	preprocess_latent,
)
from world_model.model.world_model import WorldModel

CONTEXT_LEN = 4
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "v_prediction"
PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"
DEFAULT_CHECKPOINT_DIR = Path("world_model") / "checkpoints" / "original_dynamics"


def psnr_neg1_to_01(pred: torch.Tensor, tgt: torch.Tensor) -> float:
	p = ((pred.clamp(-1, 1) + 1) * 0.5).float()
	t = ((tgt.clamp(-1, 1) + 1) * 0.5).float()
	mse = (p - t).pow(2).mean().item()
	if mse <= 0:
		return float("inf")
	return 10.0 * math.log10(1.0 / mse)


def _parse_int_list(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in s.split(",") if p.strip()]
	if not parts:
		raise ValueError("expected at least one integer")
	return tuple(int(x) for x in parts)


def _batched_ranges(n: int, chunk: int) -> list[tuple[int, int]]:
	c = max(1, int(chunk))
	return [(s, min(s + c, n)) for s in range(0, n, c)]


def _strip_preview_01(chunks: list[torch.Tensor], gap_px: int = 6) -> torch.Tensor:
	if not chunks:
		raise ValueError("need at least one image")
	if len(chunks) == 1:
		return chunks[0]
	ref = chunks[0]
	sep = torch.ones(1, 3, int(ref.shape[-2]), gap_px, device=ref.device, dtype=ref.dtype)
	parts = [chunks[0]]
	for ch in chunks[1:]:
		parts.extend([sep, ch])
	return torch.cat(parts, dim=-1)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--env", type=str, default=None, help="Base game folder. Omit to train on all original folders.")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded")
	p.add_argument("--vae_checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"))
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--num_train_epochs", type=int, default=5)
	p.add_argument("--max_train_steps", type=int, default=50000_000)
	p.add_argument("--lr", type=float, default=5e-5)
	p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
	p.add_argument("--lr_warmup_steps", type=int, default=500)
	p.add_argument("--log_dir", type=str, default="runs/world_model_original")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--gradient_checkpointing", action="store_true")
	p.add_argument("--gamma", type=float, default=0.1, help="Gaussian history-latent augmentation scale.")
	p.add_argument("--gamma_warmup_steps", type=int, default=500)
	p.add_argument("--validation_every", type=int, default=10_000)
	p.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument("--val_samples", type=int, default=8)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	p.add_argument("--val_ar_horizons", type=str, default="1,10,30")
	p.add_argument("--cfg_both_drop_prob", type=float, default=0.10)
	p.add_argument("--cfg_action_drop_prob", type=float, default=0.05)
	p.add_argument("--cfg_scale_action", type=float, default=1.5)
	return p.parse_args()


def _make_dataset(encoded_root: Path, split: str, env: str | None, seq_len: int, num_actions: int):
	dirs = encoded_original_dirs_under_split(encoded_root / split, env=env)
	pairs = [(p, tuple()) for p in dirs]
	ds = MixedEncodedRolloutVideoDataset(
		pairs, seq_len=seq_len, stride=1, num_actions=num_actions,
	).with_transform(partial(preprocess_latent, history_len=seq_len - 1))
	return ds, dirs


def main() -> None:
	args = parse_args()
	if args.env is not None:
		args.env = str(args.env).strip() or None
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	K = int(args.context_len)
	seq_len = K + 1
	encoded_root = Path(args.transitions_root) / args.encoded_subdir
	ds_train, train_dirs = _make_dataset(encoded_root, "train", args.env, seq_len, args.num_actions)
	ds_test = None
	test_dirs: list[Path] = []
	if args.validation_every > 0:
		try:
			ds_test, test_dirs = _make_dataset(encoded_root, "test", args.env, seq_len, args.num_actions)
		except FileNotFoundError as e:
			print(f"Warning: validation disabled because encoded test data was not found: {e}")
			args.validation_every = 0
	C = int(ds_train[0]["history_latents"].shape[1])
	test_n = 0 if ds_test is None else len(ds_test)
	print(f"Dataset windows: train={len(ds_train):,} test={test_n:,} latent_C={C}")
	for p in train_dirs:
		print(f"  train original dir: {p.name}")

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
		num_rules=0,
		cfg_both_drop_prob=args.cfg_both_drop_prob,
		cfg_action_drop_prob=args.cfg_action_drop_prob,
		cfg_rule_drop_prob=0.0,
		cfg_scale_action=args.cfg_scale_action,
		cfg_scale_rule=0.0,
	).to(device)
	if world_model.latent_channels != C:
		raise ValueError(f"encoded latent C={C} != model latent_channels={world_model.latent_channels}")
	print(f"Base diffuser parameters: {sum(p.numel() for p in world_model.diffuser.parameters()):,}")
	try:
		import lpips

		lpips_val = lpips.LPIPS(net="alex").to(device)
		lpips_val.eval()
		for p_lp in lpips_val.parameters():
			p_lp.requires_grad_(False)
	except Exception as e:
		lpips_val = None
		print(f"Warning: LPIPS validation disabled: {e}")

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

	run_name = f"{'all_originals' if args.env is None else args.env}_K{K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)
	val_ar_horizons = _parse_int_list(args.val_ar_horizons)
	val_ar_max = max(val_ar_horizons)
	val_ar_horizons_set = frozenset(val_ar_horizons)

	if ds_test is not None:
		S = min(int(args.val_samples), len(ds_test))
		val_items = [ds_test[i] for i in range(S)]
		val_hist_z = torch.stack([s["history_latents"] for s in val_items])
		val_tgt_z = torch.stack([s["target_latent"] for s in val_items])
		val_hist_act = torch.stack([s["history_actions"] for s in val_items]).long()
		val_ar_items: list[dict] = []
		val_ar_names: list[str] = []
		for folder in [p.name for p in test_dirs]:
			row = None
			for idx in range(len(ds_test)):
				if ds_test.window_game_folder(idx) != folder:
					continue
				row = ds_test.try_contiguous_ar(idx, K, val_ar_max)
				if row is not None:
					break
			if row is not None:
				val_ar_items.append(row)
				val_ar_names.append(folder)
		val_ar_hist_z = torch.stack([s["history_latents"] for s in val_ar_items]) if val_ar_items else None
		val_ar_hist_act = torch.stack([s["history_actions"] for s in val_ar_items]).long() if val_ar_items else None
		val_ar_fut = torch.stack([s["future_action_frames"] for s in val_ar_items]).long() if val_ar_items else None
		val_ar_gt_z = torch.stack([s["gt_future_latents"] for s in val_ar_items]) if val_ar_items else None
	else:
		val_hist_z = val_tgt_z = val_hist_act = None
		val_ar_hist_z = val_ar_hist_act = val_ar_fut = val_ar_gt_z = None
		val_ar_names = []

	def save_checkpoint(step: int) -> None:
		d = ckpt_root / f"step_{step:07d}"
		world_model.save_diffuser(d)
		blob_args = dict(vars(args))
		blob_args.update({
			"num_rules": 0,
			"prediction_type": PREDICTION_TYPE,
			"pretrained_model_name_or_path": PRETRAINED_MODEL_NAME_OR_PATH,
		})
		torch.save({"step": step, "optimizer": optimizer.state_dict(), "args": blob_args}, d / "trainer_state.pt")

	def validate(global_step: int, last_loss: float | None, gamma_eff: float) -> None:
		world_model.eval()
		with torch.no_grad():
			vh = val_hist_z.to(device)
			vt = val_tgt_z.to(device)
			va = val_hist_act.to(device)
			Bv = int(vh.shape[0])
			vb = max(1, int(args.batch_size))
			ts = torch.randint(0, world_model.num_train_timesteps, (Bv,), device=device).long()
			ns = torch.randn_like(vt, dtype=world_model.diffuser.unet.dtype)
			mse_num = 0.0
			mse_den = 0
			for s, e in _batched_ranges(Bv, vb):
				pred_b, tgt_b = world_model.diffusion_forward(vh[s:e], vt[s:e], va[s:e], ts[s:e], ns[s:e])
				d = (pred_b.float() - tgt_b.float()).pow(2)
				mse_num += d.sum().item()
				mse_den += d.numel()
			val_mse = mse_num / mse_den
			writer.add_scalar("val/mse", val_mse, global_step)

			gen_parts = []
			for s, e in _batched_ranges(Bv, vb):
				gen_parts.append(
					world_model.generate_next_frame(
						vh[s:e], va[s:e], va[s:e, -1],
						num_inference_steps=int(args.num_inference_steps),
					)
				)
			gen_lat = torch.cat(gen_parts, dim=0)
			gen_rgb = torch.cat([world_model.decode_video(gen_lat[s:e]) for s, e in _batched_ranges(Bv, vb)], dim=0)
			tgt_rgb = torch.cat([world_model.decode_frames(vt[s:e]) for s, e in _batched_ranges(Bv, vb)], dim=0)
			writer.add_scalar("val/psnr_f0", psnr_neg1_to_01(gen_rgb[:, 0], tgt_rgb), global_step)
			if lpips_val is not None:
				lp = lpips_val(gen_rgb[:, 0].float().clamp(-1, 1), tgt_rgb.float().clamp(-1, 1)).mean()
				writer.add_scalar("val/lpips_f0", float(lp.item()), global_step)
			writer.add_images(
				"val/generated_f0",
				_strip_preview_01([((gen_rgb[i : i + 1, 0].clamp(-1, 1) + 1) * 0.5).cpu() for i in range(min(4, Bv))]),
				global_step,
			)
			writer.add_images(
				"val/target_f0",
				_strip_preview_01([((tgt_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5).cpu() for i in range(min(4, Bv))]),
				global_step,
			)

			if val_ar_hist_z is not None and val_ar_gt_z is not None:
				z_ar = val_ar_hist_z.to(device)
				h_act = val_ar_hist_act.to(device)
				fut_dev = val_ar_fut.to(device)
				gt_z = val_ar_gt_z.to(device)
				Bar = int(z_ar.shape[0])
				pred_lat_h: dict[int, torch.Tensor] = {}
				for s in tqdm(range(val_ar_max), desc="val AR", leave=False, dynamic_ncols=True):
					fa = h_act[:, -1] if s == 0 else fut_dev[:, s - 1]
					zn_parts = [
						world_model.generate_next_frame(
							z_ar[s0:e0], h_act[s0:e0], fa[s0:e0],
							num_inference_steps=int(args.num_inference_steps),
						)
						for s0, e0 in _batched_ranges(Bar, vb)
					]
					z_next = torch.cat(zn_parts, dim=0)
					if s + 1 in val_ar_horizons_set:
						pred_lat_h[s + 1] = z_next.detach()
					z_ar = torch.cat([z_ar[:, 1:], z_next], dim=1)
					h_act = torch.cat([h_act[:, 1:], fa.unsqueeze(1)], dim=1)
				for h in val_ar_horizons:
					pred_rgb = torch.cat(
						[world_model.decode_video(pred_lat_h[h][s:e]) for s, e in _batched_ranges(Bar, vb)],
						dim=0,
					)[:, 0]
					tgt_rgb_h = torch.cat(
						[world_model.decode_video(gt_z[s:e, h - 1 : h]) for s, e in _batched_ranges(Bar, vb)],
						dim=0,
					)[:, 0]
					hid = f"h{h:02d}"
					writer.add_scalar(f"val/psnr_ar/{hid}", psnr_neg1_to_01(pred_rgb, tgt_rgb_h), global_step)
					if lpips_val is not None:
						lp_h = lpips_val(pred_rgb.float().clamp(-1, 1), tgt_rgb_h.float().clamp(-1, 1)).mean()
						writer.add_scalar(f"val/lpips_ar/{hid}", float(lp_h.item()), global_step)
					for i, folder in enumerate(val_ar_names):
						card = torch.cat([
							((tgt_rgb_h[i : i + 1].clamp(-1, 1) + 1) * 0.5),
							torch.ones(1, 3, 4, tgt_rgb_h.shape[-1], device=device),
							((pred_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5),
						], dim=-2)
						writer.add_images(f"val/ar_rollout/{hid}/{folder}", card.cpu(), global_step)

			print(f"step={global_step} loss={last_loss} val_mse={val_mse:.5f} gamma={gamma_eff:.4f}")
		world_model.train()

	global_step = 0
	last_loss: float | None = None
	pbar = tqdm(total=total_steps, desc="Original dynamics", unit="step", dynamic_ncols=True)
	for batch in cycle(loader):
		if global_step >= total_steps:
			break
		if args.validation_every > 0 and global_step % args.validation_every == 0:
			validate(global_step, last_loss, 0.0)

		optimizer.zero_grad(set_to_none=True)
		z_hist = batch["history_latents"].to(device)
		z_tgt = batch["target_latent"].to(device)
		hist_actions = batch["history_actions"].to(device)
		B = int(z_hist.shape[0])
		Wg = int(args.gamma_warmup_steps)
		gamma_eff = float(args.gamma) if Wg <= 0 else float(args.gamma) * min(1.0, global_step / float(Wg))
		timesteps = torch.randint(0, world_model.num_train_timesteps, (B,), device=device).long()
		noise = torch.randn_like(z_tgt, dtype=world_model.diffuser.unet.dtype)
		with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
			model_pred, target = world_model.diffusion_forward(
				z_hist, z_tgt, hist_actions, timesteps, noise, gamma=gamma_eff,
			)
			loss = F.mse_loss(model_pred.float(), target.float())
		last_loss = float(loss.item())
		if scaler.is_enabled():
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
		else:
			loss.backward()
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
		pbar.set_postfix(loss=f"{last_loss:.4f}", gamma=f"{gamma_eff:.4f}")

		if global_step % 20 == 0:
			writer.add_scalar("train/loss", last_loss, global_step)
			writer.add_scalar("train/gamma_eff", gamma_eff, global_step)
			writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
		if global_step > 0 and args.save_every > 0 and global_step % args.save_every == 0:
			save_checkpoint(global_step)

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Original dynamics training finished.")


if __name__ == "__main__":
	main()
