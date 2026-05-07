"""Train rule-correction residual dynamics on top of a frozen original-mode model."""

from __future__ import annotations

import argparse
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
	NUM_CORRECTION_RULE_TYPES,
	correction_rule_tuple_from_env_name,
	encoded_original_dirs_under_split,
	encoded_variant_dirs_under_split,
	preprocess_latent,
)
from world_model.model.checkpoint import load_world_model_checkpoint
from world_model.model.world_model import ResidualWorldModel, WorldModel

CONTEXT_LEN = 4
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "v_prediction"
PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"
DEFAULT_CHECKPOINT_DIR = Path("world_model") / "checkpoints" / "residual_dynamics"


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
	p.add_argument("--base_ckpt_dir", type=str, required=True)
	p.add_argument("--env", type=str, default=None, help="Base game or explicit variant. Omit to train on all variants.")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded")
	p.add_argument("--vae_checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"))
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--normal_anchor_ratio", type=float, default=0.25)
	p.add_argument("--num_train_epochs", type=int, default=5)
	p.add_argument("--max_train_steps", type=int, default=50000_000)
	p.add_argument("--lr", type=float, default=5e-5)
	p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
	p.add_argument("--lr_warmup_steps", type=int, default=500)
	p.add_argument("--log_dir", type=str, default="runs/world_model_residual")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--gradient_checkpointing", action="store_true")
	p.add_argument("--validation_every", type=int, default=10_000)
	p.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument("--val_samples", type=int, default=8)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	p.add_argument("--cfg_both_drop_prob", type=float, default=0.10)
	p.add_argument("--cfg_action_drop_prob", type=float, default=0.05)
	p.add_argument("--cfg_rule_drop_prob", type=float, default=0.05)
	p.add_argument("--cfg_scale_action", type=float, default=1.5)
	p.add_argument("--cfg_scale_rule", type=float, default=1.5)
	return p.parse_args()


def _make_mixed_dataset(pairs, seq_len: int, num_actions: int):
	return MixedEncodedRolloutVideoDataset(
		pairs, seq_len=seq_len, stride=1, num_actions=num_actions,
	).with_transform(partial(preprocess_latent, history_len=seq_len - 1))


def _concat_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
	keys = batches[0].keys()
	out: dict[str, torch.Tensor] = {}
	for k in keys:
		vals = [b[k] for b in batches if k in b]
		if torch.is_tensor(vals[0]):
			out[k] = torch.cat(vals, dim=0)
	return out


def _next_batch(it):
	return next(it)


def main() -> None:
	args = parse_args()
	if args.env is not None:
		args.env = str(args.env).strip() or None
	if not (0.0 <= float(args.normal_anchor_ratio) < 1.0):
		raise ValueError("--normal_anchor_ratio must be in [0, 1)")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	K = int(args.context_len)
	seq_len = K + 1
	encoded_root = Path(args.transitions_root) / args.encoded_subdir

	variant_train_pairs = encoded_variant_dirs_under_split(encoded_root / "train", env=args.env)
	variant_test_pairs = None
	if args.validation_every > 0:
		try:
			variant_test_pairs = encoded_variant_dirs_under_split(encoded_root / "test", env=args.env)
		except FileNotFoundError as e:
			print(f"Warning: validation disabled because encoded test variants were not found: {e}")
			args.validation_every = 0
	try:
		normal_train_dirs = encoded_original_dirs_under_split(encoded_root / "train", env=args.env)
	except FileNotFoundError:
		normal_train_dirs = []
	normal_test_dirs = []
	if args.validation_every > 0:
		try:
			normal_test_dirs = encoded_original_dirs_under_split(encoded_root / "test", env=args.env)
		except FileNotFoundError:
			normal_test_dirs = []
	normal_pairs = [(p, correction_rule_tuple_from_env_name(p.name)) for p in normal_train_dirs]
	normal_test_pairs = [(p, correction_rule_tuple_from_env_name(p.name)) for p in normal_test_dirs]

	anchor_ratio = float(args.normal_anchor_ratio)
	n_anchor = int(round(int(args.batch_size) * anchor_ratio))
	if anchor_ratio > 0 and n_anchor == 0:
		n_anchor = 1
	if n_anchor > 0 and not normal_pairs:
		raise FileNotFoundError("normal anchors requested, but no original folders were found")
	n_variant = int(args.batch_size) - n_anchor
	if n_variant < 1:
		raise ValueError("--normal_anchor_ratio leaves no variant samples in a batch")

	ds_variant = _make_mixed_dataset(variant_train_pairs, seq_len, args.num_actions)
	ds_variant_test = _make_mixed_dataset(variant_test_pairs, seq_len, args.num_actions) if variant_test_pairs is not None else None
	ds_normal = _make_mixed_dataset(normal_pairs, seq_len, args.num_actions) if n_anchor > 0 else None
	ds_normal_test = _make_mixed_dataset(normal_test_pairs, seq_len, args.num_actions) if normal_test_pairs and args.validation_every > 0 else None
	C = int(ds_variant[0]["history_latents"].shape[1])
	variant_test_n = 0 if ds_variant_test is None else len(ds_variant_test)
	print(
		f"Dataset windows: variants_train={len(ds_variant):,} variants_test={variant_test_n:,} "
		f"latent_C={C} anchor_batch={n_anchor}/{args.batch_size}",
	)
	for p, rv in variant_train_pairs:
		print(f"  train variant dir: {p.name} correction={list(rv)}")
	for p, rv in normal_pairs:
		print(f"  normal anchor dir: {p.name} correction={list(rv)}")

	variant_loader = DataLoader(
		ds_variant,
		batch_size=n_variant,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)
	normal_loader = None
	if ds_normal is not None:
		normal_loader = DataLoader(
			ds_normal,
			batch_size=n_anchor,
			shuffle=True,
			num_workers=args.num_workers,
			pin_memory=torch.cuda.is_available(),
			persistent_workers=args.num_workers > 0,
		)

	base_model = load_world_model_checkpoint(
		args.base_ckpt_dir,
		num_actions=args.num_actions,
		history_len=K,
		vae_checkpoint=args.vae_checkpoint,
		num_rules=0,
		cfg_scale_action=args.cfg_scale_action,
		device=device,
	)
	residual_model = WorldModel(
		num_actions=args.num_actions,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		vae_checkpoint=args.vae_checkpoint,
		prediction_type=PREDICTION_TYPE,
		history_len=K,
		gradient_checkpointing=args.gradient_checkpointing,
		pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
		num_rules=NUM_CORRECTION_RULE_TYPES,
		cfg_both_drop_prob=args.cfg_both_drop_prob,
		cfg_action_drop_prob=args.cfg_action_drop_prob,
		cfg_rule_drop_prob=args.cfg_rule_drop_prob,
		cfg_scale_action=args.cfg_scale_action,
		cfg_scale_rule=args.cfg_scale_rule,
		zero_init_output=True,
	).to(device)
	residual_model.vae = base_model.vae
	model = ResidualWorldModel(base_model, residual_model).to(device)
	if model.latent_channels != C:
		raise ValueError(f"encoded latent C={C} != model latent_channels={model.latent_channels}")
	print(f"Residual diffuser parameters: {sum(p.numel() for p in residual_model.diffuser.parameters()):,}")

	use_amp = device.type == "cuda" and args.mixed_precision != "no"
	amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
	scaler = GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))
	optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)
	steps_per_epoch = len(variant_loader)
	total_steps = min(args.num_train_epochs * steps_per_epoch, args.max_train_steps)
	scheduler = get_scheduler(
		args.lr_scheduler, optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps,
		num_training_steps=total_steps,
	)

	run_name = f"{'all_variants' if args.env is None else args.env}_K{K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)

	if ds_variant_test is not None:
		Sv = min(int(args.val_samples), len(ds_variant_test))
		val_items = [ds_variant_test[i] for i in range(Sv)]
		if ds_normal_test is not None and n_anchor > 0:
			Sn = min(max(1, int(args.val_samples // 2)), len(ds_normal_test))
			val_items.extend(ds_normal_test[i] for i in range(Sn))
		val_hist_z = torch.stack([s["history_latents"] for s in val_items])
		val_tgt_z = torch.stack([s["target_latent"] for s in val_items])
		val_hist_act = torch.stack([s["history_actions"] for s in val_items]).long()
		val_rule_oh = torch.stack([s["rule_onehot"] for s in val_items]).float()
	else:
		val_hist_z = val_tgt_z = val_hist_act = val_rule_oh = None

	def save_checkpoint(step: int) -> None:
		d = ckpt_root / f"step_{step:07d}"
		model.save_residual(d)
		blob_args = dict(vars(args))
		blob_args.update({
			"base_ckpt_dir": str(Path(args.base_ckpt_dir)),
			"num_rules": NUM_CORRECTION_RULE_TYPES,
			"prediction_type": PREDICTION_TYPE,
			"pretrained_model_name_or_path": PRETRAINED_MODEL_NAME_OR_PATH,
		})
		torch.save({"step": step, "optimizer": optimizer.state_dict(), "args": blob_args}, d / "trainer_state.pt")

	def _metrics(
		delta_pred: torch.Tensor,
		delta_target: torch.Tensor,
		base_pred: torch.Tensor,
		full_target: torch.Tensor,
		rule_oh: torch.Tensor,
	) -> dict[str, float]:
		variant_mask = rule_oh.abs().sum(dim=1) > 0
		normal_mask = ~variant_mask
		out: dict[str, float] = {
			"residual_mse": float(F.mse_loss(delta_pred.float(), delta_target.float()).item()),
			"residual_target_norm": float(delta_target.float().pow(2).mean().sqrt().item()),
			"residual_pred_norm": float(delta_pred.float().pow(2).mean().sqrt().item()),
		}
		if variant_mask.any():
			vm = variant_mask.view(-1, 1, 1, 1)
			out["base_mse_variant"] = float(F.mse_loss(base_pred[vm.expand_as(base_pred)].float(), full_target[vm.expand_as(full_target)].float()).item())
			combined = base_pred + delta_pred
			out["combined_mse_variant"] = float(F.mse_loss(combined[vm.expand_as(combined)].float(), full_target[vm.expand_as(full_target)].float()).item())
		if normal_mask.any():
			nm = normal_mask.view(-1, 1, 1, 1)
			out["normal_anchor_residual_norm"] = float(delta_pred[nm.expand_as(delta_pred)].float().pow(2).mean().sqrt().item())
		for ri, name in enumerate(("fast", "multishot", "ricochet")):
			rm = rule_oh[:, ri] > 0.5
			if rm.any():
				out[f"residual_pred_norm_{name}"] = float(delta_pred[rm].float().pow(2).mean().sqrt().item())
		return out

	def validate(global_step: int, last_loss: float | None) -> None:
		model.eval()
		with torch.no_grad():
			vh = val_hist_z.to(device)
			vt = val_tgt_z.to(device)
			va = val_hist_act.to(device)
			vr = val_rule_oh.to(device)
			Bv = int(vh.shape[0])
			vb = max(1, int(args.batch_size))
			ts = torch.randint(0, model.num_train_timesteps, (Bv,), device=device).long()
			ns = torch.randn_like(vt, dtype=model.residual_model.diffuser.unet.dtype)
			parts = []
			for s, e in _batched_ranges(Bv, vb):
				mask = vr[s:e].abs().sum(dim=1) == 0
				parts.append(model.residual_forward(vh[s:e], vt[s:e], va[s:e], ts[s:e], ns[s:e], vr[s:e], mask))
			delta_pred = torch.cat([p[0] for p in parts], dim=0)
			delta_target = torch.cat([p[1] for p in parts], dim=0)
			base_pred = torch.cat([p[2] for p in parts], dim=0)
			full_target = torch.cat([p[3] for p in parts], dim=0)
			metrics = _metrics(delta_pred, delta_target, base_pred, full_target, vr)
			for k, v in metrics.items():
				writer.add_scalar(f"val/{k}", v, global_step)

			variant_rows = torch.nonzero(vr.abs().sum(dim=1) > 0, as_tuple=False).flatten()
			if variant_rows.numel() > 0:
				rows = variant_rows[: min(4, int(variant_rows.numel()))]
				vh_p = vh[rows]
				va_p = va[rows]
				vr_p = vr[rows]
				vt_p = vt[rows]
				base_lat = model.base_model.generate_next_frame(
					vh_p, va_p, va_p[:, -1], num_inference_steps=int(args.num_inference_steps),
				)
				combined_lat = model.generate_next_frame(
					vh_p, va_p, va_p[:, -1],
					num_inference_steps=int(args.num_inference_steps),
					rule_onehot=vr_p,
				)
				base_rgb = model.decode_video(base_lat)[:, 0]
				combined_rgb = model.decode_video(combined_lat)[:, 0]
				tgt_rgb = model.decode_frames(vt_p)
				writer.add_images(
					"val/base_only_generated",
					_strip_preview_01([((base_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5).cpu() for i in range(len(rows))]),
					global_step,
				)
				writer.add_images(
					"val/residual_generated",
					_strip_preview_01([((combined_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5).cpu() for i in range(len(rows))]),
					global_step,
				)
				writer.add_images(
					"val/target",
					_strip_preview_01([((tgt_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5).cpu() for i in range(len(rows))]),
					global_step,
				)
			print(f"step={global_step} loss={last_loss} val={metrics}")
		model.train()
		model.freeze_base()

	global_step = 0
	last_loss: float | None = None
	pbar = tqdm(total=total_steps, desc="Residual dynamics", unit="step", dynamic_ncols=True)
	variant_iter = cycle(variant_loader)
	normal_iter = cycle(normal_loader) if normal_loader is not None else None
	while global_step < total_steps:
		if args.validation_every > 0 and global_step % args.validation_every == 0:
			validate(global_step, last_loss)

		batches = [_next_batch(variant_iter)]
		if normal_iter is not None:
			batches.append(_next_batch(normal_iter))
		batch = _concat_batches(batches)

		optimizer.zero_grad(set_to_none=True)
		z_hist = batch["history_latents"].to(device)
		z_tgt = batch["target_latent"].to(device)
		hist_actions = batch["history_actions"].to(device)
		rule_oh = batch["rule_onehot"].to(device).float()
		normal_mask = rule_oh.abs().sum(dim=1) == 0
		B = int(z_hist.shape[0])
		timesteps = torch.randint(0, model.num_train_timesteps, (B,), device=device).long()
		noise = torch.randn_like(z_tgt, dtype=model.residual_model.diffuser.unet.dtype)

		with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
			delta_pred, delta_target, base_pred, full_target = model.residual_forward(
				z_hist, z_tgt, hist_actions, timesteps, noise, rule_oh, normal_mask,
			)
			loss = F.mse_loss(delta_pred.float(), delta_target.float())
		last_loss = float(loss.item())
		if scaler.is_enabled():
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
		else:
			loss.backward()
		if args.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
		if scaler.is_enabled():
			scaler.step(optimizer)
			scaler.update()
		else:
			optimizer.step()
		scheduler.step()

		with torch.no_grad():
			metrics = _metrics(delta_pred.detach(), delta_target.detach(), base_pred.detach(), full_target.detach(), rule_oh)
		global_step += 1
		pbar.update(1)
		pbar.set_postfix(loss=f"{last_loss:.4f}", base=f"{metrics.get('base_mse_variant', 0.0):.4f}", comb=f"{metrics.get('combined_mse_variant', 0.0):.4f}")
		if global_step % 20 == 0:
			writer.add_scalar("train/loss", last_loss, global_step)
			writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
			for k, v in metrics.items():
				writer.add_scalar(f"train/{k}", v, global_step)
		if global_step > 0 and args.save_every > 0 and global_step % args.save_every == 0:
			save_checkpoint(global_step)

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Residual dynamics training finished.")


if __name__ == "__main__":
	main()
