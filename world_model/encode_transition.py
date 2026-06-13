"""
Encode raw transition shards (obs.npy) with the trained frozen VAE into latent.npy.

Writes mirrored layout under:

- **Train/test** (default): ``{transitions_root}/encoded/{train|test}/{env}/shard_*/``
- **Eval** (``--split eval``): raw rollouts live directly under ``transitions_root/<env>/shard_*``
  (e.g. ``data/eval/transitions/2ship``). Encoded output is
  ``{transitions_root}/encoded/{env}/shard_*/`` — e.g. ``data/eval/transitions/encoded/2ship/...``.
  latent.npy  float16 [N, C, h, w]  (scaled latents, same as training)
  action.npy, n_actions.npy  copied from source when present.

Native 120×120 frames → latents ``[4, 15, 15]``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from world_model.dataset import obs_array_to_pixels
from world_model.model.net.vae import DEFAULT_VAE_PT, VAE


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


def _list_shards(src_env_dir: Path) -> list[Path]:
	return sorted(
		p for p in src_env_dir.glob("shard_*")
		if (p / "obs.npy").is_file() and (p / "action.npy").is_file()
	)


def _plan_encode_jobs(
	root: Path,
	encoded_base: Path,
	encoded_subdir: str,
	split: str,
	env: str | None,
) -> list[tuple[Path, Path, str]]:
	"""Return ``(src_env_dir, dst_env_dir, label)`` for each env to encode."""
	jobs: list[tuple[Path, Path, str]] = []

	if split == "eval":
		if env:
			env_names = [env]
		else:
			env_names = sorted(
				p.name
				for p in root.iterdir()
				if p.is_dir()
				and p.name != encoded_subdir
				and _list_shards(p)
			)
		for env_name in env_names:
			src = root / env_name
			if not src.is_dir():
				continue
			jobs.append((src, encoded_base / env_name, f"eval/{env_name}"))
		return jobs

	splits = ("train", "test") if split == "both" else (split,)
	for sp in splits:
		split_root = root / sp
		if not split_root.is_dir():
			continue
		if env:
			env_names = [env]
		else:
			env_names = sorted(
				p.name for p in split_root.iterdir() if p.is_dir() and _list_shards(p)
			)
		for env_name in env_names:
			src = split_root / env_name
			if not src.is_dir():
				continue
			jobs.append((src, encoded_base / sp / env_name, f"{sp}/{env_name}"))
	return jobs


def _encode_one_split(
	device: torch.device,
	vae: VAE,
	src_env_dir: Path,
	dst_env_dir: Path,
	batch_size: int,
	whole_pbar: tqdm | None = None,
	job_label: str = "",
) -> None:
	shards = _list_shards(src_env_dir)
	if not shards:
		raise FileNotFoundError(f"No shard_* with obs.npy+action.npy under {src_env_dir}")

	dst_env_dir.mkdir(parents=True, exist_ok=True)
	resize_to = VAE.pixel_hw
	for shard in shards:
		if whole_pbar is not None:
			whole_pbar.set_postfix_str(f"{job_label}/{shard.name}", refresh=False)
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
		if whole_pbar is not None:
			whole_pbar.update(1)


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = Path(args.transitions_root)
	encoded_base = root / args.encoded_subdir
	lh, lw = VAE.latent_hw
	ph, pw = VAE.pixel_hw
	print(f"VAE encode: pixel={ph}x{pw} → latent Cx{lh}x{lw}")

	vae = VAE(Path(args.vae_checkpoint))
	vae.freeze()
	vae.to(device)

	jobs = _plan_encode_jobs(root, encoded_base, args.encoded_subdir, args.split, args.env)
	if not jobs:
		print("No encode jobs found.")
		return

	total_shards = sum(len(_list_shards(src)) for src, _, _ in jobs)
	print(f"Encoding {total_shards} shards across {len(jobs)} env(s) → {encoded_base}")

	with tqdm(total=total_shards, desc="encode (whole)", dynamic_ncols=True) as whole_pbar:
		for src, dst, label in jobs:
			print(f"  {label} → {dst}")
			_encode_one_split(device, vae, src, dst, args.batch_size, whole_pbar=whole_pbar, job_label=label)
	print("Done.")


if __name__ == "__main__":
	main()
