"""
Fine-tune stabilityai/sd-vae-ft-mse.
Train: data/transitions/train/**/shard_*/obs.npy  |  Val: data/transitions/test/**/shard_*/obs.npy
Loss: MSE + 0.1*LPIPS + kl_weight*KL.  TensorBoard: runs/vae/<timestamp>/

RGB frames are cropped to multiples of 8, then optionally bilinear-downscaled by ``--down_scale``
(integer factor; 1 = full resolution after crop).

CUDA: ``--mixed_precision`` (``no`` / ``fp16`` / ``bf16``, default ``bf16``) wraps VAE (+ LPIPS) in
autocast; ``fp16`` also enables ``GradScaler``. On CPU, mixed precision is disabled.
"""

from __future__ import annotations

import argparse
import bisect
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

LPIPS_W = 0.1


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--train_dir", type=str, default=str(Path("data") / "transitions" / "train"))
	p.add_argument("--val_dir", type=str, default=str(Path("data") / "transitions" / "test"))
	p.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/sd-vae-ft-mse")
	p.add_argument("--output_dir", type=str, default=str(Path("world_model") / "checkpoints" / "vae"))
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--epochs", type=int, default=1, help="stop after this many epochs (0 = unlimited, use --max_train_steps)")
	p.add_argument("--max_train_steps", type=int, default=500_000, help="stop after this many optimizer steps (0 = unlimited, use --epochs)")
	p.add_argument("--lr", type=float, default=1e-4)
	p.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
	p.add_argument("--max_grad_norm", type=float, default=1.0, help="clip global grad norm (L2)")
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--save_every", type=int, default=10_000, help="0 = save only at end")
	p.add_argument("--device", type=str, default=None)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--log_dir", type=str, default=str(Path("runs") / "vae"))
	p.add_argument("--log_every", type=int, default=20, help="0 = every step")
	p.add_argument("--validation_every", type=int, default=10_000, help="0 = no mid-run val")
	p.add_argument("--val_batch_size", type=int, default=8, help="max frames per val eval")
	p.add_argument("--kl_weight", type=float, default=1e-6, help="KL term multiplier (small)")
	p.add_argument("--warmup_steps", type=int, default=500, help="linear LR warmup to --lr over this many optimizer steps")
	p.add_argument(
		"--down_scale",
		type=int,
		default=1,
		help="After div-8 crop, divide H/W by this integer (bilinear). 1 = no extra resize.",
	)
	p.add_argument(
		"--mixed_precision",
		type=str,
		choices=("no", "fp16", "bf16"),
		default="bf16",
		help="CUDA autocast dtype for VAE (+ LPIPS) forward; fp16 uses GradScaler. Ignored on CPU.",
	)
	return p.parse_args()


def discover_shards(root: Path) -> list[Path]:
	out = sorted(
		p
		for p in root.rglob("shard_*")
		if p.is_dir() and (p / "obs.npy").exists() and (p / "action.npy").exists()
	)
	assert out, f"no shards under {root}"
	return out


def crop_hw_div8(h: int, w: int) -> tuple[int, int]:
	H, W = (h // 8) * 8, (w // 8) * 8
	assert H > 0 and W > 0, (h, w)
	return H, W


def amp_autocast(device: torch.device, mixed_precision: str):
	"""Autocast context for VAE training/val on CUDA; no-op on CPU or when ``mixed_precision`` is ``no``."""
	if mixed_precision == "no" or device.type != "cuda":
		return nullcontext()
	dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
	return torch.autocast(device_type="cuda", dtype=dtype)


def downscaled_hw_div8(h: int, w: int, down_scale: int) -> tuple[int, int]:
	"""Target (H, W) after shrinking by ``down_scale``, still multiples of 8 (for VAE)."""
	assert down_scale >= 1
	if down_scale <= 1:
		return h, w
	H = max(8, (h // down_scale // 8) * 8)
	W = max(8, (w // down_scale // 8) * 8)
	return H, W


def psnr_batch(x: torch.Tensor, y: torch.Tensor) -> float:
	x01, y01 = (x.clamp(-1, 1) + 1.0) * 0.5, (y.clamp(-1, 1) + 1.0) * 0.5
	mse = (x01 - y01).pow(2).flatten(1).mean(dim=1)
	return float((10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))).mean().item())


def recon_loss(
	vae: AutoencoderKL, x: torch.Tensor, lpips_fn: torch.nn.Module, kl_w: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	post = vae.encode(x).latent_dist
	recon = vae.decode(post.sample()).sample
	xf, rf = x.float(), recon.float()
	mse = F.mse_loss(rf, xf)
	lp = lpips_fn(xf, rf).mean()
	kl = post.kl().mean()
	loss = mse + LPIPS_W * lp + kl_w * kl
	return loss, {"mse": mse, "lpips": lp, "kl": kl}


@torch.no_grad()
def eval_val(
	vae: AutoencoderKL,
	pixels: torch.Tensor,
	device: torch.device,
	writer: SummaryWriter | None,
	step: int,
	lpips_fn: torch.nn.Module,
	kl_w: float,
	mixed_precision: str,
	max_img: int = 8,
) -> tuple[float, float, float]:
	vae.eval()
	x = pixels.to(device=device, dtype=torch.float32)
	with amp_autocast(device, mixed_precision):
		post = vae.encode(x).latent_dist
		recon = vae.decode(post.mode()).sample
	xf, rf = x.float(), recon.float()
	mse = F.mse_loss(rf, xf)
	lp = lpips_fn(xf, rf).mean()
	kl = post.kl().mean()
	total = float((mse + LPIPS_W * lp + kl_w * kl).item())
	if writer is not None:
		writer.add_scalar("val/loss", total, step)
		writer.add_scalar("val/mse", mse.item(), step)
		writer.add_scalar("val/lpips", lp.item(), step)
		writer.add_scalar("val/kl", kl.item(), step)
		writer.add_scalar("val/psnr", psnr_batch(x, recon), step)
		n = min(max_img, x.shape[0])
		writer.add_images("val/input", (x[:n].float().cpu().clamp(-1, 1) + 1.0) * 0.5, step)
		writer.add_images("val/reconstruction", (recon[:n].float().cpu().clamp(-1, 1) + 1.0) * 0.5, step)
	return total, mse.item(), psnr_batch(x, recon)


class AllFramesDataset(Dataset):
	def __init__(self, root: Path, down_scale: int = 1):
		super().__init__()
		self.down_scale = max(1, int(down_scale))
		self.paths: list[Path] = []
		self.ends: list[int] = []
		o = 0
		for p in discover_shards(root):
			n = int(np.load(p / "obs.npy", mmap_mode="r").shape[0])
			if n <= 0:
				continue
			self.paths.append(p)
			o += n
			self.ends.append(o)
		assert self.ends, "empty dataset"
		self.n = self.ends[-1]

	def __len__(self) -> int:
		return self.n

	def _loc(self, i: int) -> tuple[Path, int]:
		si = bisect.bisect_right(self.ends, i)
		return self.paths[si], i - (self.ends[si - 1] if si else 0)

	def __getitem__(self, i: int) -> torch.Tensor:
		p, r = self._loc(i)
		f = np.asarray(np.load(p / "obs.npy", mmap_mode="r")[r])[..., -3:]
		H, W = crop_hw_div8(*f.shape[:2])
		f = f[:H, :W].astype(np.float32) / 127.5 - 1.0
		x = torch.from_numpy(f).permute(2, 0, 1).contiguous()
		if self.down_scale > 1:
			Ht, Wt = downscaled_hw_div8(H, W, self.down_scale)
			x = F.interpolate(
				x.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False
			).squeeze(0)
		return x


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	mixed_precision = args.mixed_precision
	if mixed_precision != "no" and device.type != "cuda":
		print("Warning: --mixed_precision requires CUDA; using no mixed precision on CPU.")
		mixed_precision = "no"

	train_ds = AllFramesDataset(Path(args.train_dir), down_scale=args.down_scale)
	val_ds = AllFramesDataset(Path(args.val_dir), down_scale=args.down_scale)
	loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
		persistent_workers=args.num_workers > 0,
	)
	rng = np.random.default_rng(args.seed)
	k = min(args.val_batch_size, len(val_ds))
	idx = rng.choice(len(val_ds), size=k, replace=False) if k < len(val_ds) else np.arange(len(val_ds))
	val_x = torch.stack([val_ds[int(i)] for i in idx])

	log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir.parent.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path).to(device)
	vae.train()
	lpips_fn = lpips.LPIPS(net="alex").to(device)
	lpips_fn.eval()
	for p in lpips_fn.parameters():
		p.requires_grad_(False)

	opt = torch.optim.AdamW(
		vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
	)
	use_fp16_scaler = mixed_precision == "fp16" and device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
	print(
		f"train={len(train_ds):,}  val={len(val_ds):,}  down_scale={args.down_scale}  "
		f"mixed_precision={mixed_precision}  device={device}\nTensorBoard: {log_dir.resolve()}"
	)

	step = 0
	eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight, mixed_precision)

	ep = 0
	steps_per_epoch = len(loader)
	total_steps = args.epochs * steps_per_epoch
	if args.max_train_steps > 0:
		total_steps = min(total_steps, args.max_train_steps)

	pbar = tqdm(total=total_steps)
	while step < total_steps:
		ep += 1
		pbar.set_description(f"epoch {ep}")
		for batch in loader:
			vae.train()
			x = batch.to(device=device, dtype=torch.float32)
			with amp_autocast(device, mixed_precision):
				loss, parts = recon_loss(vae, x, lpips_fn, args.kl_weight)
			opt.zero_grad(set_to_none=True)
			if scaler.is_enabled():
				scaler.scale(loss).backward()
				if args.max_grad_norm > 0:
					scaler.unscale_(opt)
					torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
			else:
				loss.backward()
				if args.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
			s = step + 1
			scale = min(1.0, s / args.warmup_steps) if args.warmup_steps > 0 else 1.0
			for pg in opt.param_groups:
				pg["lr"] = args.lr * scale
			if scaler.is_enabled():
				scaler.step(opt)
				scaler.update()
			else:
				opt.step()
			step += 1
			pbar.update(1)
			pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

			if args.log_every <= 0 or step % args.log_every == 0:
				writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
				writer.add_scalar("train/loss", loss.item(), step)
				writer.add_scalar("train/mse", parts["mse"].item(), step)
				writer.add_scalar("train/lpips", parts["lpips"].item(), step)
				writer.add_scalar("train/kl", parts["kl"].item(), step)

			if args.validation_every > 0 and step % args.validation_every == 0:
				eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight, mixed_precision)
				vae.train()

			if args.save_every > 0 and step % args.save_every == 0:
				vae.eval()
				Path(args.output_dir).mkdir(parents=True, exist_ok=True)
				torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
				vae.train()

			if step >= total_steps:
				break
	pbar.close()

	eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight, mixed_precision)
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	vae.eval()
	torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
	writer.close()
	print(f"Saved VAE -> {(Path(args.output_dir) / 'vae.pt').resolve()}")


if __name__ == "__main__":
	main()