"""
Encode raw transition shards (obs.npy) with the trained frozen VAE into latent.npy.

Writes mirrored layout under ``{transitions_root}/encoded/{train|test}/{env}/shard_*/``:
  latent.npy  float16 [N, C, h, w]  (scaled latents, same as training)
  action.npy, n_actions.npy  copied from source when present.

Preprocessing matches ``train_vae.py``:
- crop native frames to multiples of 8
- optional ``--down_scale`` (integer factor) mapped to a resize target divisible by 8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from world_model.dataset import obs_array_to_pixels
from world_model.model.net.vae import VAE


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="VAE-encode transition shards to latent.npy.")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded", help="Folder under transitions_root for outputs.")
	p.add_argument("--env", type=str, default=None, help="Environment folder name. If omitted, encode all envs in each split.")
	p.add_argument("--split", type=str, choices=("train", "test", "both"), default="both")
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="",
		help="Optional local Wan VAE state dict. Empty uses pretrained Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae.",
	)
	p.add_argument(
		"--down_scale",
		type=int,
		default=1,
		help="After div-8 crop, divide H/W by this integer (same rule as train_vae). 1 = no extra resize.",
	)
	p.add_argument("--batch_size", type=int, default=32)
	return p.parse_args()


def crop_hw_div8(h: int, w: int) -> tuple[int, int]:
	H, W = (h // 8) * 8, (w // 8) * 8
	assert H > 0 and W > 0, (h, w)
	return H, W


def downscaled_hw_div8(h: int, w: int, down_scale: int) -> tuple[int, int]:
	assert down_scale >= 1
	if down_scale <= 1:
		return h, w
	H = max(8, (h // down_scale // 8) * 8)
	W = max(8, (w // down_scale // 8) * 8)
	return H, W


def _encode_one_split(
	device: torch.device,
	vae: VAE,
	src_env_dir: Path,
	dst_env_dir: Path,
	down_scale: int,
	batch_size: int,
) -> None:
	shards = sorted(p for p in src_env_dir.glob("shard_*") if (p / "obs.npy").is_file() and (p / "action.npy").is_file())
	if not shards:
		raise FileNotFoundError(f"No shard_* with obs.npy+action.npy under {src_env_dir}")

	dst_env_dir.mkdir(parents=True, exist_ok=True)
	shard_desc = f"{src_env_dir.parent.name}/{src_env_dir.name}/shards"
	for shard in tqdm(shards, desc=shard_desc, dynamic_ncols=True):
		out_dir = dst_env_dir / shard.name
		out_dir.mkdir(parents=True, exist_ok=True)
		obs = np.load(shard / "obs.npy", mmap_mode="r")
		resize_to: tuple[int, int] | None = None
		if down_scale > 1:
			h, w = obs.shape[1], obs.shape[2]
			h, w = crop_hw_div8(h, w)
			resize_to = downscaled_hw_div8(h, w, down_scale)
		pixels = obs_array_to_pixels(obs, resize_to)
		N = pixels.shape[0]
		n_batches = (N + batch_size - 1) // batch_size
		chunks: list[np.ndarray] = []
		for i in tqdm(
			range(0, N, batch_size),
			desc=f"batches {shard.name}",
			total=n_batches,
			leave=False,
			dynamic_ncols=True,
		):
			b = pixels[i : i + batch_size].to(device=device, dtype=vae._dtype())
			with torch.no_grad():
				z = vae.encode_pixels(b).float().cpu().numpy()
			chunks.append(z)
		latent = np.concatenate(chunks, axis=0).astype(np.float16)
		np.save(out_dir / "latent.npy", latent)
		shutil.copy2(shard / "action.npy", out_dir / "action.npy")
		na = shard / "n_actions.npy"
		if na.is_file():
			shutil.copy2(na, out_dir / "n_actions.npy")


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = Path(args.transitions_root)
	encoded_base = root / args.encoded_subdir

	vae = VAE(checkpoint=args.vae_checkpoint.strip() or None)
	vae.freeze()
	vae.to(device)

	splits = ("train", "test") if args.split == "both" else (args.split,)
	for sp in tqdm(splits, desc="splits", dynamic_ncols=True):
		split_root = root / sp
		if not split_root.is_dir():
			print(f"Skip missing {split_root}")
			continue

		if args.env:
			env_names = [args.env]
		else:
			env_names = sorted(
				p.name
				for p in split_root.iterdir()
				if p.is_dir() and any(p.glob("shard_*/obs.npy"))
			)
			if not env_names:
				print(f"Skip {split_root}: no envs with shard_*/obs.npy")
				continue

		for env_name in env_names:
			src = split_root / env_name
			dst = encoded_base / sp / env_name
			if not src.is_dir():
				print(f"Skip missing {src}")
				continue
			print(f"Encoding {sp}/{env_name}")
			_encode_one_split(device, vae, src, dst, args.down_scale, args.batch_size)
	print("Done.")


if __name__ == "__main__":
	main()
