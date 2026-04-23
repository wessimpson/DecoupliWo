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
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from world_model.ascii.constants import CANVAS_H, CANVAS_W, PAD_BYTE
from world_model.ascii.renderer import GvgaiRenderer
from world_model.ascii.tokenizer import pad_to_canvas
from world_model.model.net.ascii_vae import ASCIIVAE

DEFAULT_RENDER_GAMES: str = "aliens,chopper,waves"
RENDER_FAILURE_TINT_RED: int = 80


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
	p.add_argument("--refresh_every", type=int, default=0,
		help="rescan --train_dir every N optimizer steps to pick up newly-written shards (0 = off)")
	p.add_argument("--sync_from", type=str, default=None,
		help="rsync this path into --train_dir before each refresh (e.g. Drive -> local SSD streaming)")
	p.add_argument("--render_rgb", action="store_true",
		help="log RGB original|reconstruction panels to TensorBoard via GvgaiRenderer "
		"(requires gvgai/out/production/gvgai/tracks/singlePlayer/rendering/AsciiRenderServer.class; "
		"no-ops with a warning if the class is missing)")
	p.add_argument("--render_games", type=str, default=DEFAULT_RENDER_GAMES,
		help="comma-separated games to render in the TensorBoard RGB panel (only used when --render_rgb)")
	p.add_argument("--gvgai_root", type=str, default="gvgai",
		help="path to the GVGAI build tree used by the JVM renderer (only used when --render_rgb)")
	return p.parse_args()


def _scan_shards(root: Path) -> list[Path]:
	return sorted(
		p
		for p in root.rglob("shard_*")
		if p.is_dir() and (p / "obs.npy").exists() and (p / "action.npy").exists()
	)


def discover_shards(root: Path) -> list[Path]:
	out = _scan_shards(root)
	assert out, f"no shards under {root}"
	return out


class AllAsciiFramesDataset(Dataset):
	def __init__(self, root: Path, canvas_h: int, canvas_w: int) -> None:
		super().__init__()
		self.root = Path(root)
		self.canvas_h = int(canvas_h)
		self.canvas_w = int(canvas_w)
		self.paths: list[Path] = []
		self.ends: list[int] = []
		self.n = 0
		self.refresh()
		assert self.ends, f"empty dataset at {self.root}"

	def __len__(self) -> int:
		return self.n

	def refresh(self) -> int:
		"""Rescan ``self.root`` and append any previously-unseen shards; returns the number of new frames added."""
		seen = set(self.paths)
		added = 0
		o = self.n
		for p in _scan_shards(self.root):
			if p in seen:
				continue
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
			added += n
		self.n = o
		return added

	def _loc(self, i: int) -> tuple[Path, int]:
		si = bisect.bisect_right(self.ends, i)
		return self.paths[si], i - (self.ends[si - 1] if si else 0)

	def __getitem__(self, i: int) -> torch.Tensor:
		p, r = self._loc(i)
		frame = np.asarray(np.load(p / "obs.npy", mmap_mode="r")[r])  # [H, W] uint8
		padded = pad_to_canvas(frame, self.canvas_h, self.canvas_w, PAD_BYTE)
		return torch.from_numpy(padded).long()


def _maybe_sync(args: argparse.Namespace) -> None:
	if not args.sync_from:
		return
	src = str(args.sync_from).rstrip("/") + "/"
	dst = str(args.train_dir).rstrip("/") + "/"
	subprocess.run(["rsync", "-a", src, dst], check=True)


def _build_train_loader(train_ds: AllAsciiFramesDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
	return DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
		persistent_workers=args.num_workers > 0,
	)


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


def _frame_game(val_ds: AllAsciiFramesDataset, i: int) -> str:
	shard_path, _ = val_ds._loc(i)
	return shard_path.relative_to(val_ds.root).parts[0]


def pick_val_frames_per_game(val_ds: AllAsciiFramesDataset, games: list[str]) -> dict[str, torch.Tensor]:
	"""Return ``{game: padded_canvas_ids_long [H, W]}`` for the first val frame found per game."""
	found: dict[str, torch.Tensor] = {}
	wanted = set(games)
	for i in range(len(val_ds)):
		game = _frame_game(val_ds, i)
		if game in wanted and game not in found:
			found[game] = val_ds[i]
			if len(found) == len(wanted):
				break
	return found


def start_renderers(gvgai_root: Path, games: list[str]) -> dict[str, GvgaiRenderer]:
	"""Best-effort: start one :class:`GvgaiRenderer` per game; skip games whose JVM fails to boot."""
	renderers: dict[str, GvgaiRenderer] = {}
	for game in games:
		renderer = GvgaiRenderer(gvgai_root=gvgai_root, game=game)
		try:
			renderer.start()
		except Exception as err:
			print(f"[render] skipping {game}: {err.__class__.__name__}: {err}")
			continue
		renderers[game] = renderer
	return renderers


def close_renderers(renderers: dict[str, GvgaiRenderer]) -> None:
	for renderer in renderers.values():
		try:
			renderer.close()
		except Exception:
			pass


def _render_or_placeholder(renderer: GvgaiRenderer, grid: np.ndarray) -> np.ndarray:
	"""Render ``grid``; on failure return a red-tinted placeholder sized to the game's screen."""
	try:
		return renderer.render(grid)
	except Exception as err:
		print(f"[render] {renderer.game} render failed: {err.__class__.__name__}: {err}")
		height = renderer.screen_h or 64
		width = renderer.screen_w or 128
		placeholder = np.zeros((height, width, 3), dtype=np.uint8)
		placeholder[..., 0] = RENDER_FAILURE_TINT_RED
		return placeholder


@torch.no_grad()
def log_rgb_reconstruction(
	vae: ASCIIVAE,
	renderers: dict[str, GvgaiRenderer],
	samples: dict[str, torch.Tensor],
	writer: SummaryWriter,
	step: int,
	device: torch.device,
) -> None:
	"""Render one ``[original | reconstruction]`` row per game and log the vertical stack."""
	if not renderers:
		return
	vae.train(False)
	panels: list[np.ndarray] = []
	for game, renderer in renderers.items():
		ids = samples[game].to(device).unsqueeze(0)
		logits, _, _ = vae(ids)
		pred_ids = ASCIIVAE.logits_to_ids(logits).squeeze(0).to("cpu").numpy().astype(np.uint8)
		orig_ids = ids.squeeze(0).to("cpu").numpy().astype(np.uint8)
		rgb_orig = _render_or_placeholder(renderer, orig_ids)
		rgb_pred = _render_or_placeholder(renderer, pred_ids)
		panels.append(np.concatenate([rgb_orig, rgb_pred], axis=1))
	if not panels:
		return
	max_w = max(panel.shape[1] for panel in panels)
	padded = [
		np.pad(panel, ((0, 0), (0, max_w - panel.shape[1]), (0, 0)), constant_values=0)
		for panel in panels
	]
	stacked = np.concatenate(padded, axis=0)
	writer.add_image("val/rgb_reconstruction", stacked, step, dataformats="HWC")


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

	_maybe_sync(args)
	train_ds = AllAsciiFramesDataset(Path(args.train_dir), args.canvas_h, args.canvas_w)
	val_ds = AllAsciiFramesDataset(Path(args.val_dir), args.canvas_h, args.canvas_w)
	loader = _build_train_loader(train_ds, args, device)
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

	render_samples: dict[str, torch.Tensor] = {}
	renderers: dict[str, GvgaiRenderer] = {}
	if args.render_rgb:
		requested_games = [g.strip() for g in args.render_games.split(",") if g.strip()]
		render_samples = pick_val_frames_per_game(val_ds, requested_games)
		missing = [g for g in requested_games if g not in render_samples]
		if missing:
			print(f"[render] no val frames found for: {missing}")
		renderers = start_renderers(Path(args.gvgai_root), list(render_samples.keys()))
		if renderers:
			print(f"[render] TensorBoard RGB panels enabled for: {sorted(renderers.keys())}")
		else:
			print("[render] RGB logging disabled (no renderers started)")

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
				log_rgb_reconstruction(vae, renderers, render_samples, writer, step, device)
				vae.train(True)

			refreshed = False
			if args.refresh_every > 0 and step % args.refresh_every == 0:
				_maybe_sync(args)
				added = train_ds.refresh()
				if added > 0:
					writer.add_scalar("train/dataset_frames", float(len(train_ds)), step)
					print(f"[step {step}] refreshed +{added:,} frames -> {len(train_ds):,}")
					refreshed = True

			if step >= total_steps:
				break
			if refreshed:
				loader = _build_train_loader(train_ds, args, device)
				break
	pbar.close()

	eval_val(vae, val_ids, device, writer, step, args.kl_weight)
	scaling_factor = estimate_scaling_factor(vae, val_ids[: min(64, len(val_ids))], device)
	vae.set_scaling_factor(scaling_factor)
	writer.add_scalar("val/scaling_factor", scaling_factor, step)
	print(f"scaling_factor = {scaling_factor:.4f}")

	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	torch.save(vae.state_dict(), Path(args.output_dir) / "vae.pt")
	log_rgb_reconstruction(vae, renderers, render_samples, writer, step, device)
	close_renderers(renderers)
	writer.close()
	print(f"Saved ASCIIVAE -> {(Path(args.output_dir) / 'vae.pt').resolve()}")


if __name__ == "__main__":
	main()
