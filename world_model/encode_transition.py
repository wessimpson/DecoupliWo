"""
Encode raw transition shards (obs.npy) with the trained frozen VAE into latent.npy.

Writes mirrored layout under:

- **Train/test** (default): ``{transitions_root}/encoded/{train|test}/{env}/shard_*/``
- **Eval** (``--split eval``): raw rollouts live directly under ``transitions_root/<env>/shard_*``
  (e.g. ``data/eval/transitions/2ship``). Encoded output is
  ``{transitions_root}/encoded/{env}/shard_*/`` — e.g. ``data/eval/transitions/encoded/2ship/...``.
  latent.npy  float16 [N, C, h, w]  (scaled latents, same as training)
  action.npy, n_actions.npy  copied from source when present.

Preprocessing: bilinear resize to ``vae_pixel_hw()`` → latents ``[4, 11, 30]``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from world_model.dataset import obs_array_to_pixels
from world_model.model.net.vae import DEFAULT_VAE_PT, VAE, vae_latent_hw, vae_pixel_hw


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="VAE-encode transition shards to latent.npy.")
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded", help="Folder under transitions_root for outputs.")
	p.add_argument("--env", type=str, default=None, help="Environment folder name. If omitted, encode all envs in each split.")
	p.add_argument(
		"--split",
		type=str,
		choices=("train", "test", "both", "eval"),
		default="both",
		help="'eval': transitions_root/<env>/shard_* → transitions_root/encoded/<env>/ (for data/eval/transitions).",
	)
	p.add_argument("--vae_checkpoint", type=str, default=str(DEFAULT_VAE_PT))
	p.add_argument("--batch_size", type=int, default=32)
	return p.parse_args()


def _encode_one_split(
	device: torch.device,
	vae: VAE,
	src_env_dir: Path,
	dst_env_dir: Path,
	batch_size: int,
) -> None:
	shards = sorted(p for p in src_env_dir.glob("shard_*") if (p / "obs.npy").is_file() and (p / "action.npy").is_file())
	if not shards:
		raise FileNotFoundError(f"No shard_* with obs.npy+action.npy under {src_env_dir}")

	dst_env_dir.mkdir(parents=True, exist_ok=True)
	resize_to = vae_pixel_hw()
	shard_desc = f"{src_env_dir.parent.name}/{src_env_dir.name}/shards"
	for shard in tqdm(shards, desc=shard_desc, dynamic_ncols=True):
		out_dir = dst_env_dir / shard.name
		out_dir.mkdir(parents=True, exist_ok=True)
		obs = np.load(shard / "obs.npy", mmap_mode="r")
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
	lh, lw = vae_latent_hw()
	ph, pw = vae_pixel_hw()
	print(f"VAE encode: pixel={ph}x{pw} → latent Cx{lh}x{lw}")

	vae = VAE(checkpoint=Path(args.vae_checkpoint))
	vae.freeze()
	vae.to(device)

	if args.split == "eval":
		encoded_base = root / args.encoded_subdir
		if args.env:
			env_names = [args.env]
		else:
			env_names = sorted(
				p.name
				for p in root.iterdir()
				if p.is_dir()
				and p.name != args.encoded_subdir
				and any(
					q.is_dir()
					and (q / "obs.npy").is_file()
					and (q / "action.npy").is_file()
					for q in p.glob("shard_*")
				)
			)
		if not env_names:
			print(f"Skip eval layout: no env dirs with shard_* under {root}")
		for env_name in tqdm(env_names, desc="eval envs", dynamic_ncols=True):
			src = root / env_name
			dst = encoded_base / env_name
			if not src.is_dir():
				print(f"Skip missing {src}")
				continue
			print(f"Encoding eval/{env_name} → {dst}")
			_encode_one_split(device, vae, src, dst, args.batch_size)
		print("Done.")
		return

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
			_encode_one_split(device, vae, src, dst, args.batch_size)
	print("Done.")


if __name__ == "__main__":
	main()
