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


# Legacy rule one-hot order: normal, rules_fast, multishot, ricochet.
NUM_RULE_TYPES = 4
_RULE_NORMAL = (1.0, 0.0, 0.0, 0.0)
_RULE_FAST = (0.0, 1.0, 0.0, 0.0)
_RULE_MULTISHOT = (0.0, 0.0, 1.0, 0.0)
_RULE_RICOCHET = (0.0, 0.0, 0.0, 1.0)

# Residual correction rule order: rules_fast, multishot, ricochet.
# The original mode is not a shared rule anymore; it is the all-zero correction.
NUM_CORRECTION_RULE_TYPES = 3
_CORRECTION_NORMAL = (0.0, 0.0, 0.0)
_CORRECTION_FAST = (1.0, 0.0, 0.0)
_CORRECTION_MULTISHOT = (0.0, 1.0, 0.0)
_CORRECTION_RICOCHET = (0.0, 0.0, 1.0)
RULE_VARIANT_SUFFIXES = ("_rules_fast", "_rules_multishot", "_rules_ricochet")


def canonical_rule_onehots() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Fixed order: normal, rules_fast, multishot, ricochet (CPU float32 one-hot rows)."""
	return tuple(torch.tensor(t, dtype=torch.float32) for t in (_RULE_NORMAL, _RULE_FAST, _RULE_MULTISHOT, _RULE_RICOCHET))


def rule_onehot_tuple_from_env_name(env: str) -> tuple[float, float, float, float]:
	"""Map encoded folder name (…/train/<name>) to a length-4 one-hot tuple."""
	if env.endswith("_rules_ricochet"):
		return _RULE_RICOCHET
	if env.endswith("_rules_multishot"):
		return _RULE_MULTISHOT
	if env.endswith("_rules_fast"):
		return _RULE_FAST
	return _RULE_NORMAL


def is_rule_variant_name(env: str) -> bool:
	"""Whether an encoded folder name denotes a non-original rule variant."""
	return str(env).endswith(RULE_VARIANT_SUFFIXES)


def base_game_from_encoded_folder(folder: str) -> str:
	"""Strip a rule suffix from an encoded folder name."""
	name = str(folder)
	for suffix in RULE_VARIANT_SUFFIXES:
		if name.endswith(suffix):
			return name[: -len(suffix)]
	return name


def correction_rule_tuple_from_env_name(env: str) -> tuple[float, float, float]:
	"""Map encoded folder name to a 3D residual correction vector.

	Original folders map to all zeros. Rule variant folders map to the variant
	axis only, so the vector encodes the correction rather than "normal".
	"""
	name = str(env)
	if name.endswith("_rules_ricochet"):
		return _CORRECTION_RICOCHET
	if name.endswith("_rules_multishot"):
		return _CORRECTION_MULTISHOT
	if name.endswith("_rules_fast"):
		return _CORRECTION_FAST
	return _CORRECTION_NORMAL


def correction_rule_tensor_from_name(name: str) -> torch.Tensor:
	"""User-facing rule name to 3D correction tensor."""
	n = str(name).lower().strip()
	v = torch.zeros(NUM_CORRECTION_RULE_TYPES, dtype=torch.float32)
	if n in {"fast", "rules_fast"}:
		v[0] = 1.0
	elif n == "multishot":
		v[1] = 1.0
	elif n == "ricochet":
		v[2] = 1.0
	elif n in {"multishot+ricochet", "rule3+rule4", "combo34", "5"}:
		v[1] = 1.0
		v[2] = 1.0
	elif n not in {"", "normal", "original", "base", "0", "1"}:
		raise ValueError(f"unknown correction rule: {name!r}")
	return v


def canonical_correction_rule_vectors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Fixed order: normal-zero, rules_fast, multishot, ricochet."""
	return tuple(
		torch.tensor(t, dtype=torch.float32)
		for t in (_CORRECTION_NORMAL, _CORRECTION_FAST, _CORRECTION_MULTISHOT, _CORRECTION_RICOCHET)
	)


def _has_encoded_shards(p: Path) -> bool:
	return any(
		sp.is_dir() and (sp / "latent.npy").is_file() and (sp / "action.npy").is_file()
		for sp in p.glob("shard_*")
	)


def encoded_original_dirs_under_split(encoded_split_dir: str | Path, env: str | None = None) -> list[Path]:
	"""Original encoded folders under a split.

	If ``env`` is omitted, all immediate non-variant folders are returned. If it
	is provided, only the base/original folder for that game is returned.
	"""
	split_dir = Path(encoded_split_dir)
	if not split_dir.is_dir():
		raise FileNotFoundError(f"encoded split dir not found: {split_dir}")
	out: list[Path] = []
	if env is not None:
		base = base_game_from_encoded_folder(str(env).strip())
		p = split_dir / base
		if p.is_dir() and _has_encoded_shards(p):
			out.append(p)
	else:
		for p in sorted(split_dir.iterdir()):
			if p.is_dir() and not is_rule_variant_name(p.name) and _has_encoded_shards(p):
				out.append(p)
	if not out:
		raise FileNotFoundError(f"No original encoded dirs under {split_dir} for env={env!r}")
	return out


def encoded_variant_dirs_under_split(
	encoded_split_dir: str | Path,
	env: str | None = None,
) -> list[tuple[Path, tuple[float, float, float]]]:
	"""Rule-variant encoded folders with 3D correction vectors."""
	split_dir = Path(encoded_split_dir)
	if not split_dir.is_dir():
		raise FileNotFoundError(f"encoded split dir not found: {split_dir}")
	candidates: list[Path] = []
	if env is not None:
		name = str(env).strip()
		if is_rule_variant_name(name):
			candidates = [split_dir / name]
		else:
			candidates = [
				split_dir / f"{name}_rules_fast",
				split_dir / f"{name}_rules_multishot",
				split_dir / f"{name}_rules_ricochet",
			]
	else:
		candidates = [p for p in sorted(split_dir.iterdir()) if p.is_dir() and is_rule_variant_name(p.name)]

	out: list[tuple[Path, tuple[float, float, float]]] = []
	for p in candidates:
		if p.is_dir() and _has_encoded_shards(p):
			out.append((p, correction_rule_tuple_from_env_name(p.name)))
	if not out:
		raise FileNotFoundError(f"No variant encoded dirs under {split_dir} for env={env!r}")
	return out


def encoded_dirs_with_rules(encoded_split_dir: str | Path, env: str) -> list[tuple[Path, tuple[float, float, float, float]]]:
	"""Resolve ``env`` to one or more shard directories under ``encoded_split_dir`` (e.g. encoded/train).

	If ``env`` is a bare base name (no ``_rules_*`` suffix), every sibling variant that contains at least
	one ``shard_*/latent.npy`` is included (normal ``{env}``, ``{env}_rules_fast``, …), so ``--env aliens``
	mixes all available rule variants. If ``env`` names a specific variant folder, only that directory is used.

	Rule one-hot for each path is ``rule_onehot_tuple_from_env_name(folder_name)`` (suffix only), so
	different games with the same ``_rules_*`` suffix share the same conditioning. To load every game
	under a split, use :func:`encoded_dirs_all_under_split` instead.
	"""
	split_dir = Path(encoded_split_dir)
	name = str(env)
	explicit = name.endswith("_rules_fast") or name.endswith("_rules_multishot") or name.endswith("_rules_ricochet")
	if explicit:
		p = split_dir / name
		if not p.is_dir() or not any(p.glob("shard_*/latent.npy")):
			raise FileNotFoundError(f"No encoded shards under {p}")
		return [(p, rule_onehot_tuple_from_env_name(name))]

	variant_names = [
		name,
		f"{name}_rules_fast",
		f"{name}_rules_multishot",
		f"{name}_rules_ricochet",
	]
	out: list[tuple[Path, tuple[float, float, float, float]]] = []
	for vn in variant_names:
		p = split_dir / vn
		if p.is_dir() and any(p.glob("shard_*/latent.npy")):
			out.append((p, rule_onehot_tuple_from_env_name(vn)))
	if not out:
		raise FileNotFoundError(
			f"No encoded shards for env={name!r} (tried {variant_names[0]!r} and rule variants) under {split_dir}",
		)
	return out


def encoded_dirs_all_under_split(encoded_split_dir: str | Path) -> list[tuple[Path, tuple[float, float, float, float]]]:
	"""Every immediate subdirectory of ``encoded_split_dir`` that contains encoded shards.

	Rule conditioning is ``rule_onehot_tuple_from_env_name(dir.name)``: the ``_rules_*`` suffix
	determines the one-hot, so e.g. ``aliens_rules_fast`` and ``chopper_rules_fast`` share the
	same rule vector even though the game name differs.
	"""
	split_dir = Path(encoded_split_dir)
	if not split_dir.is_dir():
		raise FileNotFoundError(f"encoded split dir not found: {split_dir}")
	out: list[tuple[Path, tuple[float, float, float, float]]] = []
	for p in sorted(split_dir.iterdir()):
		if not p.is_dir():
			continue
		if not any(
			sp.is_dir() and (sp / "latent.npy").is_file() and (sp / "action.npy").is_file()
			for sp in p.glob("shard_*")
		):
			continue
		out.append((p, rule_onehot_tuple_from_env_name(p.name)))
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
	"""Encoded windows over multiple env dirs, each tagged with a fixed rule vector."""

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
