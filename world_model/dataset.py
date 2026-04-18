"""Metadata-indexed rollout dataset for next-frame temporal world model."""

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

from world_model.ascii.constants import CANVAS_H, CANVAS_W, PAD_BYTE


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


def crop_hw_div8(h: int, w: int) -> tuple[int, int]:
	H, W = (h // 8) * 8, (w // 8) * 8
	assert H > 0 and W > 0, (h, w)
	return H, W


def _rgb_hwc(f: np.ndarray) -> np.ndarray:
	return np.asarray(f)[..., -3:]


IMG_TRANSFORMS = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 2.0 - 1.0),
])


def preprocess(
	examples: dict[str, np.ndarray],
	history_len: int,
	resize_to: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
	"""Convert raw sample → {history_frames, target_frame, history_actions}.

	Seq: ``obs[i]`` with ``action[i]`` then env step → ``obs[i+1]`` (see ``collect_transitions``).
	History is ``obs[0:K]``; target is ``obs[K]``, produced by ``action[K-1]`` (last history action).
	``history_actions[j]`` is ``action[j]`` paired with ``history_frames[j]``.
	If ``resize_to`` is None, keep native resolution (crop H×W to multiples of 8 for the VAE).
	"""
	if resize_to is not None:
		resize_to = _resize_hw_divisible_by_8(resize_to)
		tx = transforms.Compose([IMG_TRANSFORMS, transforms.Resize(resize_to, antialias=True)])
		frames = torch.stack([tx(_rgb_hwc(f)) for f in examples["obs"]], dim=0)  # [K+1, 3, H, W]
	else:
		def native_tx(f: np.ndarray) -> torch.Tensor:
			r = _rgb_hwc(f)
			H, W = crop_hw_div8(*r.shape[:2])
			return IMG_TRANSFORMS(r[:H, :W])

		frames = torch.stack([native_tx(f) for f in examples["obs"]], dim=0)
	actions = examples["action"].astype(np.int64)

	K = history_len
	return {
		"history_frames": frames[:K],                            # [K, 3, H, W]
		"target_frame": frames[K],                               # [3, H, W]
		"history_actions": torch.from_numpy(actions[:K]).long(),  # [K]
	}


def _pad_ascii_to_canvas(
	frames: np.ndarray,
	canvas_h: int,
	canvas_w: int,
	pad_byte: int,
) -> np.ndarray:
	"""Pad ``[N, h, w]`` uint8 ASCII frames up to ``[N, canvas_h, canvas_w]`` with ``pad_byte``."""
	assert frames.dtype == np.uint8, f"ASCII obs must be uint8, got {frames.dtype}"
	assert frames.ndim == 3, f"ASCII obs must be [N,H,W], got shape {frames.shape}"
	n, h, w = frames.shape
	assert h <= canvas_h and w <= canvas_w, (
		f"native grid {h}x{w} exceeds canvas {canvas_h}x{canvas_w}"
	)
	if h == canvas_h and w == canvas_w:
		return np.ascontiguousarray(frames)
	out = np.full((n, canvas_h, canvas_w), pad_byte, dtype=np.uint8)
	out[:, :h, :w] = frames
	return out


def preprocess_ascii(
	examples: dict[str, np.ndarray],
	history_len: int,
	canvas: tuple[int, int] = (CANVAS_H, CANVAS_W),
	pad_byte: int = PAD_BYTE,
) -> dict[str, torch.Tensor]:
	"""Convert a raw ASCII sample -> ``{history_ids, target_ids, history_actions}``.

	``examples["obs"]`` is ``[K+1, h, w]`` uint8 ASCII bytes (see
	``data/collect_transitions.py`` docstring for the shard schema). Frames are
	padded up to the unified ``canvas`` and cast to ``long`` for the VAE's
	embedding lookup. History/target/action alignment matches :func:`preprocess`.
	"""
	canvas_h, canvas_w = int(canvas[0]), int(canvas[1])
	padded = _pad_ascii_to_canvas(np.asarray(examples["obs"]), canvas_h, canvas_w, int(pad_byte))
	ids = torch.from_numpy(padded).long()  # [K+1, H, W]
	actions = np.asarray(examples["action"]).astype(np.int64)

	k = history_len
	return {
		"history_ids": ids[:k],                                  # [K, H, W]
		"target_ids": ids[k],                                    # [H, W]
		"history_actions": torch.from_numpy(actions[:k]).long(),  # [K]
	}


def preprocess_contiguous_ar_ascii(
	examples: dict[str, np.ndarray],
	history_len: int,
	num_future_frames: int,
	canvas: tuple[int, int] = (CANVAS_H, CANVAS_W),
	pad_byte: int = PAD_BYTE,
) -> dict[str, torch.Tensor]:
	"""ASCII counterpart of :func:`preprocess_contiguous_ar` for AR eval rollouts."""
	k, hn = history_len, int(num_future_frames)
	need = k + hn
	if examples["obs"].shape[0] < need or examples["action"].shape[0] < need:
		raise ValueError(f"need {need} rows, got obs={examples['obs'].shape[0]}")

	canvas_h, canvas_w = int(canvas[0]), int(canvas[1])
	padded = _pad_ascii_to_canvas(
		np.asarray(examples["obs"][:need]), canvas_h, canvas_w, int(pad_byte),
	)
	ids = torch.from_numpy(padded).long()  # [K+Hn, H, W]
	actions = np.asarray(examples["action"][:need]).astype(np.int64)

	return {
		"history_ids": ids[:k],
		"history_actions": torch.from_numpy(actions[:k]).long(),
		"gt_future_ids": ids[k:],                                           # [Hn, H, W]
		"future_action_frames": torch.from_numpy(actions[k:]).long(),        # [Hn]
	}


def preprocess_contiguous_ar(
	examples: dict[str, np.ndarray],
	history_len: int,
	num_future_frames: int,
	resize_to: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
	"""Same timeline as ``preprocess``, plus ``num_future_frames`` ground-truth steps for AR eval.

	Requires ``len(obs) >= history_len + num_future_frames``.
	Returns ``gt_future_frames`` [H, 3, H, W] and ``future_action_frames`` [H] where
	``future_action_frames[s] = action[K+s]`` (``obs[K+s] → obs[K+s+1]``). First AR step
	uses ``history_actions[:, -1]`` (= ``action[K-1]``) to predict ``obs[K]``.
	"""
	K, Hn = history_len, int(num_future_frames)
	need = K + Hn
	if examples["obs"].shape[0] < need or examples["action"].shape[0] < need:
		raise ValueError(f"need {need} rows, got obs={examples['obs'].shape[0]}")

	if resize_to is not None:
		resize_to = _resize_hw_divisible_by_8(resize_to)
		tx = transforms.Compose([IMG_TRANSFORMS, transforms.Resize(resize_to, antialias=True)])
		frames = torch.stack([tx(_rgb_hwc(f)) for f in examples["obs"][:need]], dim=0)
	else:
		def native_tx(f: np.ndarray) -> torch.Tensor:
			r = _rgb_hwc(f)
			H, W = crop_hw_div8(*r.shape[:2])
			return IMG_TRANSFORMS(r[:H, :W])

		frames = torch.stack([native_tx(f) for f in examples["obs"][:need]], dim=0)
	actions = examples["action"][:need].astype(np.int64)

	return {
		"history_frames": frames[:K],
		"history_actions": torch.from_numpy(actions[:K]).long(),
		"gt_future_frames": frames[K:],  # [Hn, 3, H, W]
		"future_action_frames": torch.from_numpy(actions[K:]).long(),  # [Hn]
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

	def try_contiguous_ar(
		self,
		idx: int,
		history_len: int,
		num_future_frames: int,
		resize_to: tuple[int, int] | None,
	) -> dict[str, torch.Tensor] | None:
		"""Load ``history_len + num_future_frames`` rows from the shard of window ``idx``, or None if OOB."""
		meta, row = self._resolve_index(idx)
		need = history_len + int(num_future_frames)
		if row + need > meta.num_rows:
			return None
		shard = self._get_shard(meta)
		sample = {
			"obs": shard["obs"][row : row + need],
			"action": shard["action"][row : row + need],
		}
		return preprocess_contiguous_ar(sample, history_len, num_future_frames, resize_to)

	def try_contiguous_ar_ascii(
		self,
		idx: int,
		history_len: int,
		num_future_frames: int,
		canvas: tuple[int, int] = (CANVAS_H, CANVAS_W),
	) -> dict[str, torch.Tensor] | None:
		"""ASCII counterpart of :meth:`try_contiguous_ar`; returns ``history_ids``/``gt_future_ids``/etc."""
		meta, row = self._resolve_index(idx)
		need = history_len + int(num_future_frames)
		if row + need > meta.num_rows:
			return None
		shard = self._get_shard(meta)
		sample = {
			"obs": shard["obs"][row : row + need],
			"action": shard["action"][row : row + need],
		}
		return preprocess_contiguous_ar_ascii(sample, history_len, num_future_frames, canvas)



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
	K = 2
	env_dir = _find_test_env_dir()
	tx = partial(preprocess, history_len=K, resize_to=None)
	ds = RolloutVideoDataset(env_dir, seq_len=K + 1, stride=1, num_actions=18).with_transform(tx)
	assert len(ds) > 0, "dataset empty"

	row0 = ds[0]
	for k in ("history_frames", "target_frame", "history_actions"):
		assert k in row0, k
	H, W = row0["target_frame"].shape[-2:]

	dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
	batch = next(iter(dl))
	B = batch["history_frames"].shape[0]
	assert batch["history_frames"].shape == (B, K, 3, H, W)
	assert batch["target_frame"].shape == (B, 3, H, W)
	assert batch["history_actions"].shape == (B, K)
	assert batch["history_actions"].dtype == torch.long
	print(f"OK  env={env_dir.name}  len={len(ds)}  batch={B}")


if __name__ == "__main__":
	main()
