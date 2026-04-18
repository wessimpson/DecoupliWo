"""
Train ASCIIVAE on GVGAI ASCII shards.
Train: data/transitions/train/**/shard_*/obs.npy  |  Val: data/transitions/test/**/shard_*/obs.npy
Loss: cross_entropy + kl_weight*KL.  TensorBoard: runs/ascii_vae/<timestamp>/

``obs.npy`` is ``uint8[N, H, W]`` printable ASCII bytes produced by GVGAI's
``RunDataCollectionAgent`` (see ``data/collect_transitions.py`` docstring). Each
frame is padded up to ``(CANVAS_H, CANVAS_W)`` with ``PAD_BYTE`` before encoding.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from world_model.ascii.constants import CANVAS_H, CANVAS_W, PAD_BYTE
from world_model.ascii.tokenizer import pad_to_canvas
from world_model.model.net.ascii_vae import ASCIIVAE


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--train_dir", type=str, default=str(Path("data") / "transitions" / "train"))
	p.add_argument("--val_dir", type=str, default=str(Path("data") / "transitions" / "test"))
	p.add_argument("--output_dir", type=str, default=str(Path("world_model") / "checkpoints" / "ascii_vae"))
	p.add_argument("--batch_size", type=int, default=64)
	p.add_argument("--epochs", type=int, default=4, help="stop after this many epochs (0 = unlimited, use --max_train_steps)")
	p.add_argument("--max_train_steps", type=int, default=50_000, help="stop after this many optimizer steps (0 = unlimited, use --epochs)")
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
	p.add_argument("--max_grad_norm", type=float, default=1.0, help="clip global grad norm (L2)")
	p.add_argument("--num_workers", type=int, default=0)
	p.add_argument("--save_every", type=int, default=100_000, help="0 = save only at end")
	p.add_argument("--device", type=str, default=None)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--log_dir", type=str, default=str(Path("runs") / "ascii_vae"))
	p.add_argument("--log_every", type=int, default=20, help="0 = every step")
	p.add_argument("--validation_every", type=int, default=5_000, help="0 = no mid-run val")
	p.add_argument("--val_batch_size", type=int, default=64, help="max frames per val eval")
	p.add_argument("--kl_weight", type=float, default=1e-3, help="KL term multiplier")
	p.add_argument("--warmup_steps", type=int, default=500, help="linear LR warmup to --lr over this many optimizer steps")
	p.add_argument("--canvas_h", type=int, default=CANVAS_H)
	p.add_argument("--canvas_w", type=int, default=CANVAS_W)
	return p.parse_args()


def discover_shards(root: Path) -> list[Path]:
	out = sorted(
		p
		for p in root.rglob("shard_*")
		if p.is_dir() and (p / "obs.npy").exists() and (p / "action.npy").exists()
	)
	assert out, f"no shards under {root}"
	return out


class AllAsciiFramesDataset(Dataset):
	def __init__(self, root: Path, canvas_h: int, canvas_w: int) -> None:
		super().__init__()
		self.canvas_h = int(canvas_h)
		self.canvas_w = int(canvas_w)
		self.paths: list[Path] = []
		self.ends: list[int] = []
		o = 0
		for p in discover_shards(root):
			arr = np.load(p / "obs.npy", mmap_mode="r")
			assert arr.dtype == np.uint8 and arr.ndim == 3, (
				f"expected uint8[N,H,W] ASCII shard, got dtype={arr.dtype} ndim={arr.ndim} at {p}"
			)
			n = int(arr.shape[0])
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
		frame = np.asarray(np.load(p / "obs.npy", mmap_mode="r")[r])  # [H, W] uint8
		padded = pad_to_canvas(frame, self.canvas_h, self.canvas_w, PAD_BYTE)
		return torch.from_numpy(padded).long()


def per_cell_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = logits.argmax(dim=-3)
	return float((preds == targets).float().mean().item())


@torch.no_grad()
def eval_val(
	vae: ASCIIVAE,
	ids: torch.Tensor,
	device: torch.device,
	writer: SummaryWriter | None,
	step: int,
	kl_w: float,
) -> tuple[float, float]:
	vae.train(False)
	ids = ids.to(device=device)
	logits, mu, logvar = vae(ids)
	total, stats = ASCIIVAE.elbo_loss(logits, mu, logvar, ids, kl_w)
	acc = per_cell_accuracy(logits, ids)
	if writer is not None:
		writer.add_scalar("val/loss", stats["total"], step)
		writer.add_scalar("val/ce", stats["ce"], step)
		writer.add_scalar("val/kl", stats["kl"], step)
		writer.add_scalar("val/accuracy", acc, step)
	return stats["total"], acc


@torch.no_grad()
def estimate_scaling_factor(vae: ASCIIVAE, ids: torch.Tensor, device: torch.device) -> float:
	vae.train(False)
	mu, _ = vae.encoder(ids.to(device))
	std = float(mu.detach().float().std().item())
	return 1.0 / max(std, 1e-8)


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

	train_ds = AllAsciiFramesDataset(Path(args.train_dir), args.canvas_h, args.canvas_w)
	val_ds = AllAsciiFramesDataset(Path(args.val_dir), args.canvas_h, args.canvas_w)
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
	val_ids = torch.stack([val_ds[int(i)] for i in idx])

	log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir.parent.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	vae = ASCIIVAE().to(device)
	opt = torch.optim.AdamW(
		vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
	)
	print(f"train={len(train_ds):,}  val={len(val_ds):,}  device={device}\nTensorBoard: {log_dir.resolve()}")

	step = 0
	eval_val(vae, val_ids, device, writer, step, args.kl_weight)

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
			vae.train(True)
			ids = batch.to(device=device)
			logits, mu, logvar = vae(ids)
			loss, parts = ASCIIVAE.elbo_loss(logits, mu, logvar, ids, args.kl_weight)
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
			pbar.update(1)
			pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

			if args.log_every <= 0 or step % args.log_every == 0:
				with torch.no_grad():
					acc = per_cell_accuracy(logits, ids)
				writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
				writer.add_scalar("train/loss", parts["total"], step)
				writer.add_scalar("train/ce", parts["ce"], step)
				writer.add_scalar("train/kl", parts["kl"], step)
				writer.add_scalar("train/accuracy", acc, step)

			if args.validation_every > 0 and step % args.validation_every == 0:
				eval_val(vae, val_ids, device, writer, step, args.kl_weight)
				vae.train(True)

			if args.save_every > 0 and step % args.save_every == 0:
				Path(args.output_dir).mkdir(parents=True, exist_ok=True)
				torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")

			if step >= total_steps:
				break
	pbar.close()

	eval_val(vae, val_ids, device, writer, step, args.kl_weight)
	scaling_factor = estimate_scaling_factor(vae, val_ids[: min(64, len(val_ids))], device)
	vae.set_scaling_factor(scaling_factor)
	writer.add_scalar("val/scaling_factor", scaling_factor, step)
	print(f"scaling_factor = {scaling_factor:.4f}")

	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
	writer.close()
	print(f"Saved ASCIIVAE -> {(Path(args.output_dir) / 'vae.pt').resolve()}")


if __name__ == "__main__":
	main()
