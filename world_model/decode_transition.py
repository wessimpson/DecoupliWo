"""
Decode encoded transition shards (latent.npy) back to RGB and display frames.

Reads from:
  {transitions_root}/{encoded_subdir}/{split}/{env}/shard_*/latent.npy

The viewer iterates over all selected shards and shows decoded frames in order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from world_model.model.net.vae import DEFAULT_VAE_PT, VAE


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Decode encoded transition latents and display frames.")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded")
	p.add_argument("--env", type=str, default=None, help="Environment folder name. If omitted, decode all envs.")
	p.add_argument("--split", type=str, choices=("train", "test", "both"), default="test")
	p.add_argument("--vae_checkpoint", type=str, default=str(DEFAULT_VAE_PT))
	p.add_argument("--batch_size", type=int, default=32, help="Decode batch size in latent frames.")
	p.add_argument("--max_frames_per_shard", type=int, default=120, help="0 = all frames, else cap per shard.")
	p.add_argument("--stride", type=int, default=1, help="Show every Nth frame (>=1).")
	p.add_argument("--fps", type=float, default=12.0, help="Autoplay speed (used only when --autoplay is set).")
	p.add_argument("--autoplay", action="store_true", help="Play frames automatically instead of stepping with space.")
	return p.parse_args()


def _to_imshow01(frame_chw: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] -> [H,W,3] in [0,1]."""
	return ((frame_chw.clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _iter_encoded_env_dirs(root: Path, split: str, env: str | None) -> list[tuple[str, Path]]:
	splits = ("train", "test") if split == "both" else (split,)
	out: list[tuple[str, Path]] = []
	for sp in splits:
		sp_root = root / sp
		if not sp_root.is_dir():
			continue
		if env is not None:
			p = sp_root / env
			if p.is_dir():
				out.append((sp, p))
			continue
		for p in sorted(sp_root.iterdir()):
			if p.is_dir() and any((p / s.name / "latent.npy").is_file() for s in p.glob("shard_*")):
				out.append((sp, p))
	return out


def _decode_latent_shard(
	vae: VAE,
	device: torch.device,
	latent_path: Path,
	batch_size: int,
	max_frames: int,
	stride: int,
	decode_one_by_one: bool,
) -> Iterator[np.ndarray]:
	lat = np.load(latent_path, mmap_mode="r")
	n = int(lat.shape[0])
	if max_frames > 0:
		n = min(n, max_frames)
	idx = np.arange(0, n, max(1, stride), dtype=np.int64)
	if idx.size == 0:
		return
	if decode_one_by_one:
		for j in idx:
			z = torch.from_numpy(np.asarray(lat[j : j + 1], dtype=np.float32)).to(device=device, dtype=vae._dtype())
			with torch.no_grad():
				px = vae.decode_latents(z).cpu()
			yield _to_imshow01(px[0])
	else:
		for i in range(0, idx.size, batch_size):
			bi = idx[i : i + batch_size]
			z = torch.from_numpy(np.asarray(lat[bi], dtype=np.float32)).to(device=device, dtype=vae._dtype())
			with torch.no_grad():
				px = vae.decode_latents(z).cpu()
			for f in px:
				yield _to_imshow01(f)


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoded_root = Path(args.transitions_root) / args.encoded_subdir
	if not encoded_root.is_dir():
		raise FileNotFoundError(f"Encoded root not found: {encoded_root}")

	vae = VAE(checkpoint=Path(args.vae_checkpoint))
	vae.freeze()
	vae.to(device)

	env_dirs = _iter_encoded_env_dirs(encoded_root, args.split, args.env)
	if not env_dirs:
		raise FileNotFoundError(
			f"No encoded env dirs under {encoded_root} for split={args.split!r}, env={args.env!r}"
		)

	all_shards: list[tuple[str, str, Path]] = []
	for sp, env_dir in env_dirs:
		shards = sorted(p for p in env_dir.glob("shard_*") if (p / "latent.npy").is_file())
		for s in shards:
			all_shards.append((sp, env_dir.name, s))
	if not all_shards:
		raise FileNotFoundError("No shard_*/latent.npy found in selected encoded dirs.")
	print(
		f"Decoding {len(all_shards)} shard(s) from split={args.split!r}, env={args.env!r}, "
		f"max_frames_per_shard={args.max_frames_per_shard}, stride={args.stride}, fps={args.fps}."
	)

	plt.ion()
	fig, ax = plt.subplots(figsize=(6.4, 6.4))
	ax.axis("off")
	img = None
	delay = max(1e-3, 1.0 / max(0.1, float(args.fps)))
	next_frame = False
	quit_viewer = False

	def on_key(event):
		nonlocal next_frame, quit_viewer
		k = (event.key or "").lower()
		if k == " ":
			next_frame = True
		elif k in {"escape", "q"}:
			quit_viewer = True

	fig.canvas.mpl_connect("key_press_event", on_key)

	for sp, env_name, shard_dir in tqdm(all_shards, desc="decode shards", dynamic_ncols=True):
		latent_path = shard_dir / "latent.npy"
		title_prefix = f"{sp}/{env_name}/{shard_dir.name}"
		any_frame = False
		for fi, frame in enumerate(
			_decode_latent_shard(
				vae=vae,
				device=device,
				latent_path=latent_path,
				batch_size=max(1, int(args.batch_size)),
				max_frames=max(0, int(args.max_frames_per_shard)),
				stride=max(1, int(args.stride)),
				decode_one_by_one=(not args.autoplay),
			)
		):
			any_frame = True
			if img is None:
				img = ax.imshow(frame, vmin=0.0, vmax=1.0, interpolation="nearest")
			else:
				img.set_data(frame)
			ax.set_title(f"{title_prefix}  frame={fi}", fontsize=10)
			fig.canvas.draw_idle()
			if args.autoplay:
				plt.pause(delay)
			else:
				# Manual stepping: press SPACE for next frame, ESC/Q to quit.
				next_frame = False
				while plt.fignum_exists(fig.number) and not next_frame and not quit_viewer:
					plt.pause(0.03)
			if not plt.fignum_exists(fig.number) or quit_viewer:
				return
		if not any_frame:
			continue

	print("Done.")
	if plt.fignum_exists(fig.number):
		plt.ioff()
		plt.show()


if __name__ == "__main__":
	main()

