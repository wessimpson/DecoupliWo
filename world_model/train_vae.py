"""
Train KL-VAE on data/transitions/train; validate on test.
Loss: MSE + 0.1*LPIPS + kl_weight*KL. 120×120 pixels → 4×15×15 latents. CUDA only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from world_model.model.net.vae import VAE
from world_model.util.data import FrameDataset
from world_model.util.evaluation import amp_autocast, eval_vae, log_scalars, vae_train_loss

LPIPS_W = 0.1
DEVICE = torch.device("cuda")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--train_dir", type=str, default=str(Path("data") / "transitions" / "train"))
	p.add_argument("--val_dir", type=str, default=str(Path("data") / "transitions" / "test"))
	p.add_argument("--init_checkpoint", type=str, default="", help="Optional vae.pt; empty = scratch.")
	p.add_argument("--output_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--epochs", type=int, default=1)
	p.add_argument("--max_train_steps", type=int, default=5_000_000)
	p.add_argument("--lr", type=float, default=1e-4)
	p.add_argument("--weight_decay", type=float, default=1e-5)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--log_dir", type=str, default=str(Path("runs") / "vae"))
	p.add_argument("--log_every", type=int, default=20)
	p.add_argument("--validation_every", type=int, default=10_000)
	p.add_argument("--val_batch_size", type=int, default=8)
	p.add_argument("--kl_weight", type=float, default=1e-6)
	p.add_argument("--warmup_steps", type=int, default=500)
	p.add_argument("--mixed_precision", type=str, choices=("no", "fp16", "bf16"), default="bf16")
	return p.parse_args()


def main() -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("train_vae requires CUDA")
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	train_ds = FrameDataset(Path(args.train_dir))
	val_ds = FrameDataset(Path(args.val_dir))
	loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)
	rng = np.random.default_rng(args.seed)
	k = min(args.val_batch_size, len(val_ds))
	idx = rng.choice(len(val_ds), size=k, replace=False) if k < len(val_ds) else np.arange(len(val_ds))
	val_x = torch.stack([val_ds[int(i)] for i in idx])

	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")))
	init_ckpt = str(args.init_checkpoint).strip() or None
	vae = VAE(init_ckpt).to(DEVICE)
	ae = vae.autoencoder
	ae.train()
	if init_ckpt:
		print(f"Loaded {init_ckpt}")

	lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()
	for p in lpips_fn.parameters():
		p.requires_grad_(False)
	opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
	mp = args.mixed_precision
	scaler = torch.amp.GradScaler("cuda", enabled=mp == "fp16")
	ph, pw = vae.pixel_hw
	lh, lw = vae.latent_hw
	print(f"train={len(train_ds):,} val={len(val_ds):,} pixel={ph}x{pw} latent={lh}x{lw}")

	step = 0
	eval_vae(ae, val_x, DEVICE, lpips_fn, args.kl_weight, mp, writer, step, lpips_w=LPIPS_W)
	total = min(args.epochs * len(loader), args.max_train_steps) if args.max_train_steps > 0 else args.epochs * len(loader)
	pbar = tqdm(total=total)
	while step < total:
		for batch in loader:
			ae.train()
			x = batch.to(device=DEVICE, dtype=torch.float32)
			with amp_autocast(DEVICE, mp):
				loss, parts = vae_train_loss(ae, x, lpips_fn, args.kl_weight, LPIPS_W)
			opt.zero_grad(set_to_none=True)
			if scaler.is_enabled():
				scaler.scale(loss).backward()
				if args.max_grad_norm > 0:
					scaler.unscale_(opt)
					torch.nn.utils.clip_grad_norm_(ae.parameters(), args.max_grad_norm)
				scaler.step(opt)
				scaler.update()
			else:
				loss.backward()
				if args.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(ae.parameters(), args.max_grad_norm)
				opt.step()

			step += 1
			lr = args.lr * min(1.0, step / args.warmup_steps) if args.warmup_steps > 0 else args.lr
			opt.param_groups[0]["lr"] = lr
			pbar.update(1)
			pbar.set_postfix(loss=float(loss.item()), lr=lr)

			if args.log_every <= 0 or step % args.log_every == 0:
				log_scalars(writer, "train", {"loss": loss.item(), **{k: v.item() for k, v in parts.items()}, "lr": lr}, step)
			if args.validation_every > 0 and step % args.validation_every == 0:
				eval_vae(ae, val_x, DEVICE, lpips_fn, args.kl_weight, mp, writer, step, lpips_w=LPIPS_W)
			if args.save_every > 0 and step % args.save_every == 0:
				Path(args.output_dir).mkdir(parents=True, exist_ok=True)
				torch.save(ae.state_dict(), Path(args.output_dir) / "vae.pt")
			if step >= total:
				break
	pbar.close()

	eval_vae(ae, val_x, DEVICE, lpips_fn, args.kl_weight, mp, writer, step, lpips_w=LPIPS_W)
	out = Path(args.output_dir) / "vae.pt"
	out.parent.mkdir(parents=True, exist_ok=True)
	torch.save(ae.state_dict(), out)
	writer.close()
	print(f"Saved {out.resolve()}")


if __name__ == "__main__":
	main()
