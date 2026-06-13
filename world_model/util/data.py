"""Transition shard indexing and frame loading for VAE training."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from world_model.dataset import obs_array_to_pixels


def discover_shards(root: Path, max_per_env: int | None = None) -> list[Path]:
	"""Collect ``shard_*`` under each env folder; optionally keep only the first ``max_per_env`` per folder."""
	by_env: dict[Path, list[Path]] = {}
	for p in sorted(root.rglob("shard_*")):
		if not p.is_dir() or not (p / "obs.npy").is_file() or not (p / "action.npy").is_file():
			continue
		by_env.setdefault(p.parent, []).append(p)
	out: list[Path] = []
	for env_dir in sorted(by_env):
		shards = sorted(by_env[env_dir])
		if max_per_env is not None and max_per_env > 0:
			shards = shards[:max_per_env]
		out.extend(shards)
	assert out, f"no shards under {root}"
	return out


class FrameDataset(Dataset):
	"""Flat index over frames in ``root/<env>/shard_*/obs.npy`` (120×120 RGB, [-1,1])."""

	def __init__(self, root: Path, max_shards_per_env: int | None = None) -> None:
		self.paths: list[Path] = []
		self.ends: list[int] = []
		o = 0
		for p in discover_shards(root, max_per_env=max_shards_per_env):
			n = int(np.load(p / "obs.npy", mmap_mode="r").shape[0])
			if n <= 0:
				continue
			self.paths.append(p)
			o += n
			self.ends.append(o)
		assert self.ends

	def __len__(self) -> int:
		return self.ends[-1]

	def _loc(self, i: int) -> tuple[Path, int]:
		si = bisect.bisect_right(self.ends, i)
		return self.paths[si], i - (self.ends[si - 1] if si else 0)

	def __getitem__(self, i: int) -> torch.Tensor:
		p, r = self._loc(i)
		f = np.copy(np.asarray(np.load(p / "obs.npy", mmap_mode="r")[r]))
		return obs_array_to_pixels(f[np.newaxis], resize_to=None)[0]
