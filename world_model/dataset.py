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


def obs_array_to_pixels(obs: np.ndarray, resize_to: tuple[int, int] | None = None) -> torch.Tensor:
	"""Full-shard ``obs`` [N, …] → stacked pixels [N, 3, H, W] in [-1, 1] (same rules as ``preprocess``)."""
	if resize_to is not None:
		resize_to = _resize_hw_divisible_by_8(resize_to)
		tx = transforms.Compose([IMG_TRANSFORMS, transforms.Resize(resize_to, antialias=True)])
		return torch.stack([tx(_rgb_hwc(f)) for f in obs], dim=0)

	def native_tx(f: np.ndarray) -> torch.Tensor:
		r = _rgb_hwc(f)
		H, W = crop_hw_div8(*r.shape[:2])
		return IMG_TRANSFORMS(r[:H, :W])

	return torch.stack([native_tx(f) for f in obs], dim=0)


def preprocess_latent(
	examples: dict[str, np.ndarray],
	history_len: int,
) -> dict[str, torch.Tensor]:
	"""Windowed latent shard row → tensors (same timeline as ``preprocess``)."""
	z = torch.from_numpy(np.asarray(examples["latent"], dtype=np.float32))
	actions = examples["action"].astype(np.int64)
	K = history_len
	return {
		"history_latents": z[:K],
		"target_latent": z[K],
		"history_actions": torch.from_numpy(actions[:K]).long(),
	}


def preprocess_contiguous_latent_ar(
	examples: dict[str, np.ndarray],
	history_len: int,
	num_future_frames: int,
) -> dict[str, torch.Tensor]:
	"""AR eval slice over pre-encoded latents (decode to RGB in the trainer for PSNR if needed)."""
	K, Hn = history_len, int(num_future_frames)
	need = K + Hn
	if examples["latent"].shape[0] < need or examples["action"].shape[0] < need:
		raise ValueError(f"need {need} rows, got latent={examples['latent'].shape[0]}")
	z = torch.from_numpy(np.asarray(examples["latent"][:need], dtype=np.float32))
	actions = examples["action"][:need].astype(np.int64)
	return {
		"history_latents": z[:K],
		"history_actions": torch.from_numpy(actions[:K]).long(),
		"gt_future_latents": z[K:],
		"future_action_frames": torch.from_numpy(actions[K:]).long(),
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


class EncodedRolloutVideoDataset(Dataset):
	"""Same windowing as ``RolloutVideoDataset``, but each shard has ``latent.npy`` (+ ``action.npy``)."""

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
			if (p / "latent.npy").exists() and (p / "action.npy").exists()
		)
		assert paths, f"No shard_* with latent.npy/action.npy under {self.data_dir}"

		self._shards: list[ShardMeta] = []
		self._cumulative_windows: list[int] = []
		self._worker_cache: dict[str, dict[str, np.memmap]] = {}
		self.n_actions = 0
		total_windows = 0

		for path in paths:
			lat = np.load(path / "latent.npy", mmap_mode="r")
			num_rows = int(lat.shape[0])

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

	def with_transform(self, transform: SampleTransform) -> EncodedRolloutVideoDataset:
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
				"latent": np.load(meta.path / "latent.npy", mmap_mode="r"),
				"action": np.load(meta.path / "action.npy", mmap_mode="r"),
			}
		return self._worker_cache[key]

	def __getitem__(self, idx: int):
		meta, row = self._resolve_index(idx)
		shard = self._get_shard(meta)
		sample = {
			"latent": shard["latent"][row : row + self.seq_len],
			"action": shard["action"][row : row + self.seq_len],
		}
		return self.transform(sample) if self.transform is not None else sample

	def try_contiguous_ar(
		self,
		idx: int,
		history_len: int,
		num_future_frames: int,
	) -> dict[str, torch.Tensor] | None:
		meta, row = self._resolve_index(idx)
		need = history_len + int(num_future_frames)
		if row + need > meta.num_rows:
			return None
		shard = self._get_shard(meta)
		sample = {
			"latent": shard["latent"][row : row + need],
			"action": shard["action"][row : row + need],
		}
		return preprocess_contiguous_latent_ar(sample, history_len, num_future_frames)


# Multi-hot rule vector: ``RULE_TAGS[i]`` is the folder suffix after ``_rules_`` (e.g. ``aliens_rules_multishot`` → ``multishot``).
# Folders **without** ``_rules_*`` map to ``NULL``: all-zero vector (baseline / classifier-free ``rule`` dropout target).
RULE_TAGS: tuple[str, ...] = (
	"enemy_explode",
	"enemy_multishot",
	"multishot",
	"ricochet",
	"shoot_walls",
	"split_orthogonal",
	"two_hit_color",
)
RULE_TAG_TO_INDEX: dict[str, int] = {t: i for i, t in enumerate(RULE_TAGS)}
NUM_RULE_TYPES = len(RULE_TAGS)
# Suffixes treated as NULL (no rule slot); e.g. ``*_rules_fast`` data still loads without a ``fast`` dim.
LEGACY_NULL_RULE_TAGS: frozenset[str] = frozenset({"fast"})


def _zeros_rule_tuple() -> tuple[float, ...]:
	return tuple(0.0 for _ in range(NUM_RULE_TYPES))


def rule_multihot_tuple_from_env_name(env: str) -> tuple[float, ...]:
	"""Map encoded folder name to a length-``NUM_RULE_TYPES`` multi-hot (single tag) or all zeros (base / NULL)."""
	if "_rules_" not in env:
		return _zeros_rule_tuple()
	tag = env.split("_rules_", 1)[1]
	if tag in LEGACY_NULL_RULE_TAGS:
		return _zeros_rule_tuple()
	if tag not in RULE_TAG_TO_INDEX:
		known = ", ".join(RULE_TAGS)
		raise ValueError(
			f"Unknown rule tag {tag!r} in folder {env!r}. Extend RULE_TAGS in dataset.py. Known: {known}"
		)
	v = [0.0] * NUM_RULE_TYPES
	v[RULE_TAG_TO_INDEX[tag]] = 1.0
	return tuple(v)


def canonical_rule_onehots() -> tuple[torch.Tensor, ...]:
	"""For val bucketing: NULL (zeros) then one unit vector per ``RULE_TAGS`` slot (CPU float32)."""
	out: list[torch.Tensor] = [torch.tensor(_zeros_rule_tuple(), dtype=torch.float32)]
	for i in range(NUM_RULE_TYPES):
		row = [0.0] * NUM_RULE_TYPES
		row[i] = 1.0
		out.append(torch.tensor(row, dtype=torch.float32))
	return tuple(out)


def encoded_folder_base_game(folder_name: str) -> str:
	"""Strip ``_rules_<tag>`` so variants map to one base game (e.g. ``aliens_rules_fast`` → ``aliens``)."""
	if "_rules_" in folder_name:
		return folder_name.split("_rules_", 1)[0]
	return folder_name


def _dir_has_encoded_shards(p: Path) -> bool:
	return p.is_dir() and any(
		sp.is_dir() and (sp / "latent.npy").is_file() and (sp / "action.npy").is_file()
		for sp in p.glob("shard_*")
	)


def collect_unknown_rule_tags_under_split(encoded_split_dir: str | Path) -> list[str]:
	"""Tags after ``_rules_`` in immediate child dirs that are not in ``RULE_TAGS``."""
	split_dir = Path(encoded_split_dir)
	if not split_dir.is_dir():
		return []
	unknown: set[str] = set()
	for p in split_dir.iterdir():
		if not p.is_dir() or "_rules_" not in p.name:
			continue
		if not _dir_has_encoded_shards(p):
			continue
		tag = p.name.split("_rules_", 1)[1]
		if tag in LEGACY_NULL_RULE_TAGS:
			continue
		if tag not in RULE_TAG_TO_INDEX:
			unknown.add(tag)
	return sorted(unknown)


def encoded_dirs_with_rules(encoded_split_dir: str | Path, env: str) -> list[tuple[Path, tuple[float, ...]]]:
	"""Resolve ``env`` to one or more shard directories under ``encoded_split_dir`` (e.g. encoded/train).

	If ``env`` is a bare base name (no ``_rules_*`` in the string), every child ``{env}`` and ``{env}_rules_*``
	with encoded shards is included. If ``env`` names a specific folder (optionally with ``_rules_<tag>``),
	only that directory is used.

	Base game folder ``{env}`` (no suffix) uses the **NULL** rule vector (all zeros).
	"""
	split_dir = Path(encoded_split_dir)
	name = str(env)
	explicit = "_rules_" in name
	if explicit:
		p = split_dir / name
		if not _dir_has_encoded_shards(p):
			raise FileNotFoundError(f"No encoded shards under {p}")
		return [(p, rule_multihot_tuple_from_env_name(name))]

	out: list[tuple[Path, tuple[float, ...]]] = []
	p0 = split_dir / name
	if _dir_has_encoded_shards(p0):
		out.append((p0, rule_multihot_tuple_from_env_name(name)))
	for p in sorted(split_dir.glob(f"{name}_rules_*")):
		if _dir_has_encoded_shards(p):
			out.append((p, rule_multihot_tuple_from_env_name(p.name)))
	if not out:
		raise FileNotFoundError(
			f"No encoded shards for env={name!r} (tried {name!r} and {name}_rules_*) under {split_dir}",
		)
	return out


def encoded_dirs_all_under_split(encoded_split_dir: str | Path) -> list[tuple[Path, tuple[float, ...]]]:
	"""Every immediate subdirectory of ``encoded_split_dir`` that contains encoded shards.

	Rule conditioning is :func:`rule_multihot_tuple_from_env_name`: base folders → NULL (zeros); ``*_rules_<tag>`` → one-hot over ``RULE_TAGS``.
	"""
	split_dir = Path(encoded_split_dir)
	if not split_dir.is_dir():
		raise FileNotFoundError(f"encoded split dir not found: {split_dir}")
	out: list[tuple[Path, tuple[float, ...]]] = []
	for p in sorted(split_dir.iterdir()):
		if not p.is_dir():
			continue
		if not _dir_has_encoded_shards(p):
			continue
		out.append((p, rule_multihot_tuple_from_env_name(p.name)))
	if not out:
		raise FileNotFoundError(f"No encoded env dirs with shard_*/latent.npy+action.npy under {split_dir}")
	return out


@dataclass(frozen=True)
class EncodedShardRuleMeta:
	path: Path
	num_rows: int
	num_windows: int
	cumulative_windows: int
	n_actions: int
	game_name: str
	rule_onehot: tuple[float, ...]


class MixedEncodedRolloutVideoDataset(Dataset):
	"""Encoded windows over multiple env dirs; each tagged with a fixed rule multi-hot (see ``RULE_TAGS``)."""

	def __init__(
		self,
		dir_rule_pairs: list[tuple[Path, tuple[float, ...]]],
		seq_len: int,
		stride: int = 1,
		num_actions: int | None = None,
		transform: SampleTransform | None = None,
	) -> None:
		super().__init__()
		if not dir_rule_pairs:
			raise ValueError("dir_rule_pairs must be non-empty")
		self.seq_len = int(seq_len)
		self.stride = max(1, int(stride))
		self.num_actions = None if num_actions is None else int(num_actions)
		self.transform = transform

		self._shards: list[EncodedShardRuleMeta] = []
		self._cumulative_windows: list[int] = []
		self._worker_cache: dict[str, dict[str, np.memmap]] = {}
		self.n_actions = 0
		total_windows = 0

		for data_dir, rule_oh in dir_rule_pairs:
			data_dir = Path(data_dir)
			paths = sorted(
				p for p in data_dir.glob("shard_*")
				if (p / "latent.npy").exists() and (p / "action.npy").exists()
			)
			if not paths:
				continue
			for path in paths:
				lat = np.load(path / "latent.npy", mmap_mode="r")
				num_rows = int(lat.shape[0])

				n_actions_path = path / "n_actions.npy"
				if n_actions_path.exists():
					n_actions = int(np.load(n_actions_path))
				elif self.num_actions is not None:
					n_actions = self.num_actions
				else:
					raise KeyError(f"{path}: need n_actions.npy or fixed num_actions")

				num_windows = max(0, (num_rows - self.seq_len) // self.stride + 1)
				total_windows += num_windows
				self._shards.append(EncodedShardRuleMeta(
					path=path, num_rows=num_rows, num_windows=num_windows,
					cumulative_windows=total_windows, n_actions=n_actions,
					game_name=data_dir.name, rule_onehot=rule_oh,
				))
				self._cumulative_windows.append(total_windows)
				self.n_actions = max(self.n_actions, n_actions)

		if not self._shards:
			raise ValueError(f"No shard_* with latent.npy/action.npy in given dirs: {[str(d) for d, _ in dir_rule_pairs]}")
		self._total_windows = total_windows

	def with_transform(self, transform: SampleTransform) -> MixedEncodedRolloutVideoDataset:
		cloned = copy(self)
		cloned.transform = transform
		cloned._worker_cache = {}
		return cloned

	def __len__(self) -> int:
		return self._total_windows

	def _resolve_index(self, idx: int) -> tuple[EncodedShardRuleMeta, int]:
		assert 0 <= idx < self._total_windows
		si = bisect_right(self._cumulative_windows, idx)
		meta = self._shards[si]
		prev = 0 if si == 0 else self._cumulative_windows[si - 1]
		return meta, (idx - prev) * self.stride

	def window_game_folder(self, idx: int) -> str:
		"""Encoded root child folder name for this window (e.g. ``aliens`` or ``aliens_rules_fast``)."""
		meta, _ = self._resolve_index(idx)
		return meta.game_name

	def _get_shard(self, meta: EncodedShardRuleMeta) -> dict[str, np.memmap]:
		key = str(meta.path)
		if key not in self._worker_cache:
			self._worker_cache[key] = {
				"latent": np.load(meta.path / "latent.npy", mmap_mode="r"),
				"action": np.load(meta.path / "action.npy", mmap_mode="r"),
			}
		return self._worker_cache[key]

	def __getitem__(self, idx: int):
		meta, row = self._resolve_index(idx)
		shard = self._get_shard(meta)
		sample = {
			"latent": shard["latent"][row : row + self.seq_len],
			"action": shard["action"][row : row + self.seq_len],
		}
		out = self.transform(sample) if self.transform is not None else sample
		if not isinstance(out, dict):
			raise TypeError("transform must return dict for MixedEncodedRolloutVideoDataset")
		out = dict(out)
		out["rule_onehot"] = torch.tensor(meta.rule_onehot, dtype=torch.float32)
		return out

	def try_contiguous_ar(
		self,
		idx: int,
		history_len: int,
		num_future_frames: int,
	) -> dict[str, torch.Tensor] | None:
		meta, row = self._resolve_index(idx)
		need = history_len + int(num_future_frames)
		if row + need > meta.num_rows:
			return None
		shard = self._get_shard(meta)
		sample = {
			"latent": shard["latent"][row : row + need],
			"action": shard["action"][row : row + need],
		}
		row_dict = preprocess_contiguous_latent_ar(sample, history_len, num_future_frames)
		row_dict = dict(row_dict)
		row_dict["rule_onehot"] = torch.tensor(meta.rule_onehot, dtype=torch.float32)
		row_dict["game_name"] = meta.game_name
		return row_dict


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
	K = 8
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
