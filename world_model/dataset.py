"""Metadata-indexed rollout dataset with optional sample transforms."""

from __future__ import annotations

from bisect import bisect_right
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class ShardMeta:
	path: Path
	num_rows: int
	num_windows: int
	cumulative_windows: int
	n_actions: int
	game_name: str


SampleTransform = Callable[[dict[str, np.ndarray]], dict[str, torch.Tensor] | dict[str, np.ndarray]]


IMG_TRANSFORMS = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0),
	]
)


def preprocess(
	examples: dict[str, np.ndarray],
	buffer_size: int,
	resize_to: tuple[int, int] | None = (256, 256),
) -> dict[str, torch.Tensor]:
	"""
	Convert a raw sample to model-ready tensors:
	- Normalize to [-1,1]
	- Resize to `resize_to` if provided
	Returns context and target tensors shaped for training.
	"""
	tx = transforms.Compose([IMG_TRANSFORMS, transforms.Resize(resize_to, antialias=True)])
	frames = torch.stack([tx(frame) for frame in examples["obs"]], dim=0)  # [T,3,H,W]
	actions = examples["action"].astype(np.int64)
	return {
		"context_frames": frames[:buffer_size],  # [BUF,3,256,256]
		"target_frame": frames[buffer_size],     # [3,256,256]
		"last_action": torch.tensor(int(actions[buffer_size - 1]), dtype=torch.long),
	}


class RolloutVideoDataset(Dataset):
	"""
	Metadata-only indexed dataset over rollout shards.

	At init time it records only shard metadata and cumulative row offsets.
	Actual shard contents are opened lazily per worker and only the requested
	window is decoded in ``__getitem__``.
	"""

	def __init__(
		self,
		data_dir: str | Path,
		seq_len: int,
		stride: int = 1,
		num_actions: int | None = None,
		transform: SampleTransform | None = None,
	) -> None:
		super().__init__()
		self.data_dir = Path(data_dir)
		self.seq_len = int(seq_len)
		self.stride = max(1, int(stride))
		self.num_actions = None if num_actions is None else int(num_actions)
		self.transform = transform

		# Expect per-shard directories: shard_XXXXX with obs.npy/action.npy
		paths = sorted(p for p in self.data_dir.glob("shard_*") if (p / "obs.npy").exists() and (p / "action.npy").exists())
		if not paths:
			raise FileNotFoundError(f"No shard_* with obs.npy/action.npy under {self.data_dir}")

		self._shards: list[ShardMeta] = []
		self._cumulative_windows: list[int] = []
		self._worker_cache: dict[str, np.lib.npyio.NpzFile] = {}
		self.n_actions = 0
		total_windows = 0

		for path in paths:
			obs_path = path / "obs.npy"
			act_path = path / "action.npy"
			n_actions_path = path / "n_actions.npy"
			obs = np.load(obs_path, mmap_mode="r")
			num_rows = int(obs.shape[0])
			if n_actions_path.exists():
				n_actions = int(np.load(n_actions_path))
			elif self.num_actions is not None:
				n_actions = self.num_actions
			else:
				raise KeyError(f"{path} must contain 'n_actions.npy' or dataset must be given fixed num_actions")

			if self.num_actions is not None and n_actions != self.num_actions:
				raise ValueError(f"{path} has n_actions={n_actions}, expected {self.num_actions}")

			num_windows = max(0, (num_rows - self.seq_len) // self.stride + 1)
			total_windows += num_windows
			meta = ShardMeta(
				path=path,
				num_rows=num_rows,
				num_windows=num_windows,
				cumulative_windows=total_windows,
				n_actions=n_actions,
				game_name=self.data_dir.name,
			)
			self._shards.append(meta)
			self._cumulative_windows.append(total_windows)
			self.n_actions = max(self.n_actions, n_actions)

		self._total_windows = total_windows

	def with_transform(self, transform: SampleTransform) -> "RolloutVideoDataset":
		cloned = copy(self)
		cloned.transform = transform
		cloned._worker_cache = {}
		return cloned

	def __len__(self) -> int:
		return self._total_windows

	def _resolve_index(self, idx: int) -> tuple[ShardMeta, int]:
		if idx < 0 or idx >= self._total_windows:
			raise IndexError(idx)
		shard_idx = bisect_right(self._cumulative_windows, idx)
		meta = self._shards[shard_idx]
		prev_cumulative = 0 if shard_idx == 0 else self._cumulative_windows[shard_idx - 1]
		local_window_idx = idx - prev_cumulative
		row_start = local_window_idx * self.stride
		return meta, row_start

	def _get_shard_handle(self, meta: ShardMeta) -> dict[str, np.memmap]:
		cache_key = str(meta.path)
		handle = self._worker_cache.get(cache_key)
		if handle is None:
			handle = {
				"obs": np.load(meta.path / "obs.npy", mmap_mode="r"),
				"action": np.load(meta.path / "action.npy", mmap_mode="r"),
			}
			self._worker_cache[cache_key] = handle
		return handle

	def __getitem__(self, idx: int):
		meta, row_start = self._resolve_index(idx)
		shard = self._get_shard_handle(meta)
		sample = {
			"obs": shard["obs"][row_start : row_start + self.seq_len],
			"action": shard["action"][row_start : row_start + self.seq_len],
		}
		if self.transform is not None:
			return self.transform(sample)
		return sample


RolloutClipDataset = RolloutVideoDataset


def main() -> None:
	data_root = Path("data") / "transitions" / "space_invaders"
	ds = RolloutVideoDataset(data_root, seq_len=9, stride=1, num_actions=18)
	ds = ds.with_transform(partial(preprocess, buffer_size=8, resize_to=(210, 160)))
	print(f"shards={len(ds._shards)} windows={len(ds)} n_actions={ds.n_actions}")
	sample = ds[0]
	print("context_frames:", tuple(sample["context_frames"].shape))
	print("target_frame:", tuple(sample["target_frame"].shape))
	print("last_action:", int(sample["last_action"]))


if __name__ == "__main__":
	main()
