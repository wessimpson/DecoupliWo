"""
Train a small MLP to predict next-step delta (dx, dy) from **current** position and action.

Expects ``data/transitions/{train,test}/<env>/shard_*/player_x.npy``, ``player_y.npy``, ``action.npy``
(see ``collect_transitions.py``). Row ``t`` uses ``(x_t, y_t)`` and ``a[t]`` to predict
``(x_{t+1} - x_t, y_{t+1} - y_t)``.
TensorBoard: train loss, val loss, rollout MSE at 10 and 30 steps ahead,
``val/mask_rollout_{1,10,30}step`` — single overlay (alpha-blended green=target, red=pred; amber if same cell) per horizon.
**sprite_size** (square cell in raw px/py) defaults to 32; run ``check_transition_player_xy.py`` on the same env to infer it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def _grid_bounds_and_shape(
	xy: np.ndarray, sprite_size: float
) -> tuple[float, float, int, int]:
	"""Axis-aligned bounds and cell counts (nx, ny) for integer grid indices."""
	xmin, ymin = float(xy[:, 0].min()), float(xy[:, 1].min())
	xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())
	ss = max(float(sprite_size), 1e-6)
	nx = int(np.floor((xmax - xmin) / ss)) + 1
	ny = int(np.floor((ymax - ymin) / ss)) + 1
	nx, ny = max(nx, 1), max(ny, 1)
	return xmin, ymin, nx, ny


def _sprite_target_pred_overlay_chw(
	px_t: float,
	py_t: float,
	px_p: float,
	py_p: float,
	xmin: float,
	ymin: float,
	nx: int,
	ny: int,
	sprite_size: float,
	alpha_t: float = 0.55,
	alpha_p: float = 0.55,
) -> np.ndarray:
	"""One grid image: white background, target/pred cells alpha-blended (RGB CHW [0, 1])."""
	ss = max(float(sprite_size), 1e-6)

	def cell(px: float, py: float) -> tuple[int, int]:
		ix = int(np.floor((px - xmin) / ss))
		iy = int(np.floor((py - ymin) / ss))
		return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))

	def blend_cell(
		img: np.ndarray, iy: int, ix: int, rgb: tuple[float, float, float], a: float
	) -> None:
		c = np.array(rgb, dtype=np.float32)
		img[iy, ix] = (1.0 - a) * img[iy, ix] + a * c

	img = np.ones((ny, nx, 3), dtype=np.float32)
	ix_t, iy_t = cell(px_t, py_t)
	ix_p, iy_p = cell(px_p, py_p)
	green = (0.15, 0.85, 0.15)
	red = (0.9, 0.15, 0.15)
	amber = (0.95, 0.75, 0.1)
	if (ix_t, iy_t) == (ix_p, iy_p):
		blend_cell(img, iy_t, ix_t, amber, max(alpha_t, alpha_p))
	else:
		blend_cell(img, iy_t, ix_t, green, alpha_t)
		blend_cell(img, iy_p, ix_p, red, alpha_p)
	return np.transpose(img, (2, 0, 1))


@torch.no_grad()
def _log_val_mask_images(
	writer: SummaryWriter,
	model: nn.Module,
	test_sh: list[tuple[np.ndarray, np.ndarray]],
	mu: torch.Tensor,
	std: torch.Tensor,
	sprite_size: float,
	bounds_xy: np.ndarray,
	device: torch.device,
	step: int,
) -> None:
	"""Log alpha overlay masks for 1-, 10-, and 30-step rollouts (GT actions)."""
	model.eval()
	mu_b = mu.to(device).float()
	std_b = std.to(device).float()
	xmin, ymin, nx, ny = _grid_bounds_and_shape(bounds_xy, sprite_size)

	xy_a: tuple[np.ndarray, np.ndarray] | None = None
	t0 = 0
	max_h = 30
	for xy, act in test_sh:
		if xy.shape[0] > t0 + max_h and act.shape[0] == xy.shape[0]:
			xy_a = (xy, act)
			break
	if xy_a is None:
		return
	xy, act = xy_a

	def denorm(p: torch.Tensor) -> np.ndarray:
		x = p * std_b + mu_b
		return x.detach().cpu().numpy()

	for h, tag in ((1, "1step"), (10, "10step"), (30, "30step")):
		if xy.shape[0] <= t0 + h:
			continue
		cur = torch.from_numpy(xy[t0].copy()).float().to(device)
		for s in range(h):
			a = torch.tensor([int(act[t0 + s])], device=device, dtype=torch.long)
			delta = model(cur.unsqueeze(0), a).squeeze(0)
			cur = cur + delta
		gt_t = torch.from_numpy(xy[t0 + h].copy()).float().to(device)
		p0 = denorm(cur)
		t0n = denorm(gt_t)
		overlay = _sprite_target_pred_overlay_chw(
			float(t0n[0]),
			float(t0n[1]),
			float(p0[0]),
			float(p0[1]),
			xmin,
			ymin,
			nx,
			ny,
			sprite_size,
		)
		writer.add_image(f"val/mask_rollout_{tag}", torch.as_tensor(overlay), step)


def _load_xy_action_shards(env_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
	"""Aligned (xy, action) per shard; rows where xy is finite and action is valid."""
	shards = sorted(p for p in env_dir.glob("shard_*") if p.is_dir())
	out: list[tuple[np.ndarray, np.ndarray]] = []
	for s in shards:
		px, py, pa = s / "player_x.npy", s / "player_y.npy", s / "action.npy"
		if not px.is_file() or not py.is_file() or not pa.is_file():
			continue
		x = np.load(px).astype(np.float64)
		y = np.load(py).astype(np.float64)
		a = np.load(pa).astype(np.int64).ravel()
		n = min(x.size, y.size, a.size)
		if n == 0:
			continue
		x, y, a = x.ravel()[:n], y.ravel()[:n], a[:n]
		xy = np.stack([x, y], axis=1)
		m = np.isfinite(xy).all(axis=1) & np.isfinite(a.astype(np.float64))
		xy = xy[m].astype(np.float32)
		a = a[m].astype(np.int64)
		if xy.shape[0] > 0:
			out.append((xy, a))
	return out


def _norm_stats(shards: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
	cat = np.concatenate([s[0] for s in shards], axis=0) if shards else np.zeros((0, 2), np.float32)
	mu = cat.mean(axis=0) if len(cat) else np.zeros(2, np.float32)
	std = cat.std(axis=0).clip(min=1e-6)
	return mu.astype(np.float32), std.astype(np.float32)


def _apply_norm(
	shards: list[tuple[np.ndarray, np.ndarray]], mu: np.ndarray, std: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
	return [(((xy - mu) / std).astype(np.float32), act.copy()) for xy, act in shards]


def _infer_num_actions(shards: list[tuple[np.ndarray, np.ndarray]]) -> int:
	if not shards:
		return 2
	a = np.concatenate([s[1] for s in shards], axis=0)
	mx = int(a.max()) if a.size else 0
	return max(mx + 1, 2)


class StepDataset(Dataset):
	"""One sample = (xy[t], action[t]) -> (xy[t+1] - xy[t])."""

	def __init__(self, shards: list[tuple[np.ndarray, np.ndarray]]) -> None:
		self.shards = shards
		self.idx: list[tuple[int, int]] = []
		for si, (xy, act) in enumerate(shards):
			tmax = xy.shape[0]
			if tmax < 2 or act.shape[0] != tmax:
				continue
			for t in range(tmax - 1):
				if (
					np.isfinite(xy[t]).all()
					and np.isfinite(xy[t + 1]).all()
					and np.isfinite(float(act[t]))
				):
					self.idx.append((si, t))

	def __len__(self) -> int:
		return len(self.idx)

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		si, t = self.idx[i]
		xy, act = self.shards[si]
		return (
			torch.from_numpy(xy[t].copy()),
			torch.tensor(int(act[t]), dtype=torch.long),
			torch.from_numpy((xy[t + 1] - xy[t]).copy()),
		)


class PositionMLP(nn.Module):
	def __init__(
		self,
		num_actions: int,
		action_embed_dim: int = 32,
		hidden_dim: int = 128,
		num_hidden_layers: int = 2,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		self.num_actions = num_actions
		self.act_emb = nn.Embedding(num_actions, action_embed_dim)
		d_in = 2 + action_embed_dim
		layers: list[nn.Module] = []
		d = d_in
		for _ in range(num_hidden_layers):
			layers += [
				nn.Linear(d, hidden_dim),
				nn.ReLU(inplace=True),
				nn.Dropout(dropout),
			]
			d = hidden_dim
		layers.append(nn.Linear(d, 2))
		self.net = nn.Sequential(*layers)

	def forward(self, xy: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		a = actions.clamp(min=0, max=self.num_actions - 1)
		e = self.act_emb(a)
		h = torch.cat([xy, e], dim=-1)
		return self.net(h)


@torch.no_grad()
def rollout_mse(
	model: nn.Module,
	shards: list[tuple[np.ndarray, np.ndarray]],
	device: torch.device,
	max_h: int,
	n_samples: int,
) -> tuple[float, float]:
	"""Mean MSE at 10th and 30th autoregressive steps (normalized space).

	Rolls forward with predicted deltas and ground-truth actions from the shard.
	"""
	model.eval()
	errs10: list[float] = []
	errs30: list[float] = []
	tries = 0
	rng = np.random.default_rng(0)
	while (len(errs10) < n_samples or len(errs30) < n_samples) and tries < n_samples * 40:
		tries += 1
		si = int(rng.integers(0, len(shards)))
		xy, act = shards[si]
		tmax = xy.shape[0]
		if tmax <= max_h or act.shape[0] != tmax:
			continue
		t0 = int(rng.integers(0, tmax - max_h))
		cur = torch.from_numpy(xy[t0].copy()).float().to(device)
		for step in range(max_h):
			a = torch.tensor([int(act[t0 + step])], device=device, dtype=torch.long)
			pred_delta = model(cur.unsqueeze(0), a).squeeze(0)
			pred = cur + pred_delta
			if step == 9 and len(errs10) < n_samples:
				gt = torch.from_numpy(xy[t0 + 10]).float().to(device)
				errs10.append((pred - gt).pow(2).mean().item())
			if step == 29 and len(errs30) < n_samples:
				gt = torch.from_numpy(xy[t0 + 30]).float().to(device)
				errs30.append((pred - gt).pow(2).mean().item())
			cur = pred
	return float(np.mean(errs10)) if errs10 else float("nan"), float(np.mean(errs30)) if errs30 else float("nan")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "transitions"))
	p.add_argument("--batch_size", type=int, default=512)
	p.add_argument("--epochs", type=int, default=50)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--hidden_dim", type=int, default=128)
	p.add_argument("--num_hidden_layers", type=int, default=2)
	p.add_argument("--dropout", type=float, default=0.1)
	p.add_argument("--log_dir", type=str, default="runs/position_transformer")
	p.add_argument("--val_rollout_samples", type=int, default=512)
	p.add_argument("--num_workers", type=int, default=0)
	p.add_argument(
		"--sprite_size",
		type=float,
		default=32.0,
		help="Square cell side in raw px/py space (TensorBoard masks). Infer with data/check_transition_player_xy.py.",
	)
	p.add_argument(
		"--num_actions",
		type=int,
		default=0,
		help="Embedding size upper bound (max_action+1); 0 = infer from train action.npy (max id + 1).",
	)
	p.add_argument("--action_embed_dim", type=int, default=32)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = Path(args.data_root)
	train_dir = root / "train" / args.env
	test_dir = root / "test" / args.env
	if not train_dir.is_dir():
		raise FileNotFoundError(train_dir)

	train_raw = _load_xy_action_shards(train_dir)
	test_raw = _load_xy_action_shards(test_dir) if test_dir.is_dir() else []
	if not train_raw:
		raise RuntimeError(f"No player_x/player_y/action shards under {train_dir}")
	if not test_raw:
		print(f"Warning: no test shards under {test_dir}; single-step val/rollout skipped.")

	train_cat = np.concatenate([s[0] for s in train_raw], axis=0)
	sprite_size = float(args.sprite_size)
	print(f"sprite_size={sprite_size}")

	mu_np, std_np = _norm_stats(train_raw)
	mu_t = torch.from_numpy(mu_np)
	std_t = torch.from_numpy(std_np)
	train_sh = _apply_norm(train_raw, mu_np, std_np)
	test_sh = _apply_norm(test_raw, mu_np, std_np) if test_raw else []

	ds_tr = StepDataset(train_sh)
	ds_va = StepDataset(test_sh) if test_sh else None
	if len(ds_tr) == 0:
		raise RuntimeError("No (pos, action) -> delta_pos pairs; check shards.")

	tr_loader = DataLoader(
		ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False,
	)
	va_loader = (
		DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
		if ds_va is not None and len(ds_va) > 0
		else None
	)

	num_actions = args.num_actions if args.num_actions > 0 else _infer_num_actions(train_raw)
	print(f"num_actions={num_actions}")
	model = PositionMLP(
		num_actions,
		action_embed_dim=args.action_embed_dim,
		hidden_dim=args.hidden_dim,
		num_hidden_layers=args.num_hidden_layers,
		dropout=args.dropout,
	).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
	loss_fn = nn.MSELoss()

	writer = SummaryWriter(log_dir=str(Path(args.log_dir)))
	global_step = 0

	for epoch in range(args.epochs):
		model.train()
		pbar = tqdm(tr_loader, desc=f"epoch {epoch+1}/{args.epochs}")
		for xy_in, act_in, tgt in pbar:
			xy_in = xy_in.to(device)
			act_in = act_in.to(device)
			tgt = tgt.to(device)
			opt.zero_grad(set_to_none=True)
			pred = model(xy_in, act_in)
			loss = loss_fn(pred, tgt)
			loss.backward()
			opt.step()
			writer.add_scalar("train/loss", loss.item(), global_step)
			global_step += 1
			pbar.set_postfix(loss=f"{loss.item():.6f}")

		model.eval()
		if va_loader is not None and test_sh:
			tot, n = 0.0, 0
			with torch.no_grad():
				for xy_in, act_in, tgt in va_loader:
					xy_in = xy_in.to(device)
					act_in = act_in.to(device)
					tgt = tgt.to(device)
					pred = model(xy_in, act_in)
					tot += (pred - tgt).pow(2).mean().item() * xy_in.size(0)
					n += xy_in.size(0)
			val_loss = tot / max(n, 1)
			writer.add_scalar("val/loss", val_loss, epoch)
			e10, e30 = rollout_mse(
				model, test_sh, device, 30, args.val_rollout_samples,
			)
			if not np.isnan(e10):
				writer.add_scalar("val/rollout_mse_10step", e10, epoch)
			if not np.isnan(e30):
				writer.add_scalar("val/rollout_mse_30step", e30, epoch)
			writer.add_scalar("val/sprite_size", sprite_size, epoch)
			_log_val_mask_images(
				writer,
				model,
				test_sh,
				mu_t,
				std_t,
				sprite_size,
				train_cat,
				device,
				epoch,
			)

	writer.close()
	print("Done.", f"train_steps={len(ds_tr)}", f"val_steps={len(ds_va) if ds_va else 0}")


if __name__ == "__main__":
	main()
