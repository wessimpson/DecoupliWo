"""
Decode VAE latents from **encoded** transition shards (``latent.npy``), default env aliens.

Layout: ``{transitions_root}/{encoded_subdir}/{split}/{env}/shard_*/latent.npy`` (same as
``encode_transition.py`` output).

With **no** ``--shard``: walks every ``shard_*`` in sorted order; each row is one latent frame.

**Space** / **Right** = next row, decode latent to RGB. **Left** = previous. **q** quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from world_model.model.net.vae import VAE


def _pixels_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
	"""[1,3,H,W] in [-1,1] -> uint8 [H,W,3]."""
	t = x[0].detach().float().cpu().clamp(-1, 1).add(1).mul(0.5).mul(255).byte()
	return t.permute(1, 2, 0).numpy()


def _list_shards(encoded_env_dir: Path) -> list[Path]:
	if not encoded_env_dir.is_dir():
		raise FileNotFoundError(f"Missing {encoded_env_dir}")
	shards = sorted(p for p in encoded_env_dir.glob("shard_*") if (p / "latent.npy").is_file())
	if not shards:
		raise FileNotFoundError(f"No shard_*/latent.npy under {encoded_env_dir}")
	return shards


def _unpack_global(global_idx: int, lengths: list[int]) -> tuple[int, int]:
	"""Linear index -> (shard_index, row_within_shard)."""
	g = int(global_idx)
	for si, L in enumerate(lengths):
		if g < L:
			return si, g
		g -= L
	raise IndexError("global_idx out of range")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded", help="Under transitions_root, same as encode_transition.")
	p.add_argument("--split", type=str, choices=("train", "test"), default="test")
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument(
		"--shard",
		type=str,
		default=None,
		help="If set, only this shard (e.g. shard_00000). If omitted, all shards in order.",
	)
	p.add_argument(
		"--start",
		type=int,
		default=0,
		help="Starting linear row index across selected shard(s) (0 = first row of first shard).",
	)
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="",
		help="Optional local Wan VAE state dict. Empty uses pretrained Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae.",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	encoded_env = Path(args.transitions_root) / args.encoded_subdir / args.split / args.env
	if args.shard:
		shards = [encoded_env / args.shard]
	else:
		shards = _list_shards(encoded_env)

	for s in shards:
		if not (s / "latent.npy").is_file():
			raise FileNotFoundError(f"No latent.npy at {s}")

	lengths = [int(np.load(s / "latent.npy", mmap_mode="r").shape[0]) for s in shards]
	total = sum(lengths)
	if total == 0:
		raise RuntimeError("Empty latent.npy across shards")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vae = VAE(checkpoint=args.vae_checkpoint.strip() or None)
	vae.freeze()
	vae.to(device)

	global_idx = int(args.start) % total

	fig, ax = plt.subplots(1, 1, figsize=(7, 6))

	_lat_mm: np.ndarray | None = None
	_lat_shard: Path | None = None

	def _latent_for_shard(shard_p: Path) -> np.ndarray:
		nonlocal _lat_mm, _lat_shard
		if _lat_shard != shard_p:
			_lat_mm = np.load(shard_p / "latent.npy", mmap_mode="r")
			_lat_shard = shard_p
		assert _lat_mm is not None
		return _lat_mm

	def refresh() -> None:
		si, li = _unpack_global(global_idx, lengths)
		shard_p = shards[si]
		lat = _latent_for_shard(shard_p)
		z_np = lat[li : li + 1]
		z = torch.from_numpy(np.asarray(z_np, dtype=np.float32)).to(device=device, dtype=vae._dtype())
		with torch.no_grad():
			out = vae.decode_latents(z)
		rgb = _pixels_to_hwc_uint8(out)

		ax.clear()
		ax.imshow(rgb)
		ax.set_title("VAE decode (latent row)")
		ax.axis("off")
		shard_name = shard_p.name
		fig.suptitle(
			f"{global_idx + 1}/{total} | {shard_name} row {li} | Space/Right=next | Left=prev | q=quit",
		)
		fig.canvas.draw_idle()

	def on_key(ev) -> None:
		nonlocal global_idx
		if ev.key is None:
			return
		k = ev.key.lower()
		if k in ("q", "escape"):
			plt.close(fig)
			return
		if k == " " or ev.key == "right":
			global_idx = (global_idx + 1) % total
			refresh()
			return
		if ev.key == "left":
			global_idx = (global_idx - 1) % total
			refresh()

	fig.canvas.mpl_connect("key_press_event", on_key)
	refresh()
	plt.show()


if __name__ == "__main__":
	main()
