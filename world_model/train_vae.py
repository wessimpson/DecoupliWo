"""
Fine-tune stabilityai/sd-vae-ft-mse.
Train: data/transitions/train/**/shard_*/obs.npy  |  Val: data/transitions/test/**/shard_*/obs.npy
Loss: MSE + 0.1*LPIPS + kl_weight*KL.  TensorBoard: runs/vae/<timestamp>/
"""

from __future__ import annotations

import argparse
import bisect
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
	p.add_argument("--batch_size", type=int, default=2)
	p.add_argument("--max_train_steps", type=int, default=300_000, help="stop after this many optimizer steps")
	p.add_argument("--lr", type=float, default=1e-4)
	p.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
	p.add_argument("--max_grad_norm", type=float, default=1.0, help="clip global grad norm (L2)")
	p.add_argument("--num_workers", type=int, default=2)
	p.add_argument("--save_every", type=int, default=100_000, help="0 = save only at end")
	p.add_argument("--device", type=str, default=None)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--log_dir", type=str, default=str(Path("runs") / "vae"))
	p.add_argument("--log_every", type=int, default=20, help="0 = every step")
	p.add_argument("--validation_every", type=int, default=1000, help="0 = no mid-run val")
	p.add_argument("--val_batch_size", type=int, default=8, help="max frames per val eval")
	p.add_argument("--kl_weight", type=float, default=1e-6, help="KL term multiplier (small)")
	p.add_argument("--warmup_steps", type=int, default=500, help="linear LR warmup to --lr over this many optimizer steps")
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
	max_img: int = 8,
) -> tuple[float, float, float]:
	vae.eval()
	x = pixels.to(device=device, dtype=vae.dtype)
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
	def __init__(self, root: Path):
		super().__init__()
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
		return torch.from_numpy(f).permute(2, 0, 1).contiguous()


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

	train_ds = AllFramesDataset(Path(args.train_dir))
	val_ds = AllFramesDataset(Path(args.val_dir))
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
	print(f"train={len(train_ds):,}  val={len(val_ds):,}  device={device}\nTensorBoard: {log_dir.resolve()}")

	step = 0
	eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight)

	ep = 0
	while step < args.max_train_steps:
		ep += 1
		pbar = tqdm(loader, desc=f"epoch {ep}", total=min(len(loader), args.max_train_steps - step))
		for batch in pbar:
			vae.train()
			x = batch.to(device=device, dtype=vae.dtype)
			loss, parts = recon_loss(vae, x, lpips_fn, args.kl_weight)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			if args.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
			s = step + 1
			scale = min(1.0, s / args.warmup_steps) if args.warmup_steps > 0 else 1.0
			for pg in opt.param_groups:
				pg["lr"] = args.lr * scale
			opt.step()
			step += 1
			pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

			if args.log_every <= 0 or step % args.log_every == 0:
				writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
				writer.add_scalar("train/loss", loss.item(), step)
				writer.add_scalar("train/mse", parts["mse"].item(), step)
				writer.add_scalar("train/lpips", parts["lpips"].item(), step)
				writer.add_scalar("train/kl", parts["kl"].item(), step)

			if args.validation_every > 0 and step % args.validation_every == 0:
				eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight)
				vae.train()

			if args.save_every > 0 and step % args.save_every == 0:
				vae.eval()
				Path(args.output_dir).mkdir(parents=True, exist_ok=True)
				torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
				vae.train()

			if step >= args.max_train_steps:
				break

	eval_val(vae, val_x, device, writer, step, lpips_fn, args.kl_weight)
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	vae.eval()
	torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
	writer.close()
	print(f"Saved VAE -> {(Path(args.output_dir) / 'vae.pt').resolve()}")


if __name__ == "__main__":
	main()