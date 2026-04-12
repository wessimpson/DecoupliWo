"""Metadata-indexed rollout dataset for chunk-based temporal world model."""

from __future__ import annotations

from bisect import bisect_right
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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


def _resize_hw_divisible_by_8(resize_to: tuple[int, int]) -> tuple[int, int]:
	h, w = int(resize_to[0]), int(resize_to[1])
	return max(8, (h // 8) * 8), max(8, (w // 8) * 8)


IMG_TRANSFORMS = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 2.0 - 1.0),
])


def preprocess(
	examples: dict[str, np.ndarray],
	history_len: int,
	chunk_len: int,
	resize_to: tuple[int, int] | None = (208, 160),
) -> dict[str, torch.Tensor]:
	"""Convert raw sample → {history_frames, target_frames, history_actions, future_actions}.

	Seq is sliced as: [0:K] = history, [K:K+N] = target chunk.
	history_actions[j] = action at history_frames[j]; future_actions[i] aligns with target chunk.
	"""
	if resize_to is not None:
		resize_to = _resize_hw_divisible_by_8(resize_to)
	tx = transforms.Compose([IMG_TRANSFORMS, transforms.Resize(resize_to, antialias=True)])

	frames = torch.stack([tx(f) for f in examples["obs"]], dim=0)  # [K+N, 3, H, W]
	actions = examples["action"].astype(np.int64)

	K, N = history_len, chunk_len
	return {
		"history_frames": frames[:K],                                  # [K, 3, H, W]
		"target_frames": frames[K : K + N],                            # [N, 3, H, W]
		"history_actions": torch.from_numpy(actions[:K]).long(),       # [K]
		"future_actions": torch.from_numpy(actions[K : K + N]).long(),  # [N]
	}


class RolloutVideoDataset(Dataset):
	"""Lazy, mmap-backed windowed dataset over rollout shards."""

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

		paths = sorted(
			p for p in self.data_dir.glob("shard_*")
			if (p / "obs.npy").exists() and (p / "action.npy").exists()
		)
		assert paths, f"No shard_* with obs.npy/action.npy under {self.data_dir}"

		self._shards: list[ShardMeta] = []
		self._cumulative_windows: list[int] = []
		self._worker_cache: dict[str, dict[str, np.memmap]] = {}
		self.n_actions = 0
		total_windows = 0

		for path in paths:
			obs = np.load(path / "obs.npy", mmap_mode="r")
			num_rows = int(obs.shape[0])

			n_actions_path = path / "n_actions.npy"
			if n_actions_path.exists():
				n_actions = int(np.load(n_actions_path))
			elif self.num_actions is not None:
				n_actions = self.num_actions
			else:
				raise KeyError(f"{path}: need n_actions.npy or fixed num_actions")

			num_windows = max(0, (num_rows - self.seq_len) // self.stride + 1)
			total_windows += num_windows
			self._shards.append(ShardMeta(
				path=path, num_rows=num_rows, num_windows=num_windows,
				cumulative_windows=total_windows, n_actions=n_actions,
				game_name=self.data_dir.name,
			))
			self._cumulative_windows.append(total_windows)
			self.n_actions = max(self.n_actions, n_actions)

		self._total_windows = total_windows

	def with_transform(self, transform: SampleTransform) -> RolloutVideoDataset:
		cloned = copy(self)
		cloned.transform = transform
		cloned._worker_cache = {}
		return cloned

	def __len__(self) -> int:
		return self._total_windows

	def _resolve_index(self, idx: int) -> tuple[ShardMeta, int]:
		assert 0 <= idx < self._total_windows
		si = bisect_right(self._cumulative_windows, idx)
		meta = self._shards[si]
		prev = 0 if si == 0 else self._cumulative_windows[si - 1]
		return meta, (idx - prev) * self.stride

	def _get_shard(self, meta: ShardMeta) -> dict[str, np.memmap]:
		key = str(meta.path)
		if key not in self._worker_cache:
			self._worker_cache[key] = {
				"obs": np.load(meta.path / "obs.npy", mmap_mode="r"),
				"action": np.load(meta.path / "action.npy", mmap_mode="r"),
			}
		return self._worker_cache[key]

	def __getitem__(self, idx: int):
		meta, row = self._resolve_index(idx)
		shard = self._get_shard(meta)
		sample = {
			"obs": shard["obs"][row : row + self.seq_len],
			"action": shard["action"][row : row + self.seq_len],
		}
		return self.transform(sample) if self.transform is not None else sample



"""test"""
def _find_test_env_dir() -> Path:
	root = Path(__file__).resolve().parent.parent / "data" / "transitions" / "test"
	assert root.is_dir(), f"expected {root}"
	for env in sorted(root.iterdir()):
		if env.is_dir() and any(env.glob("shard_*/obs.npy")):
			return env
	raise AssertionError(f"no env with shards under {root}")


def main() -> None:
	"""Smoke test: dataset + DataLoader batch shapes."""
	K, N = 4, 4
	H, W = 208, 160
	env_dir = _find_test_env_dir()
	tx = partial(preprocess, history_len=K, chunk_len=N, resize_to=(H, W))
	ds = RolloutVideoDataset(env_dir, seq_len=K + N, stride=1, num_actions=18).with_transform(tx)
	assert len(ds) > 0, "dataset empty"

	row0 = ds[0]
	for k in ("history_frames", "target_frames", "history_actions", "future_actions"):
		assert k in row0, k

	dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
	batch = next(iter(dl))
	B = batch["history_frames"].shape[0]
	assert batch["history_frames"].shape == (B, K, 3, H, W)
	assert batch["target_frames"].shape == (B, N, 3, H, W)
	assert batch["history_actions"].shape == (B, K)
	assert batch["future_actions"].shape == (B, N)
	assert batch["history_actions"].dtype == torch.long
	print(f"OK  env={env_dir.name}  len={len(ds)}  batch={B}")


if __name__ == "__main__":
	main()
