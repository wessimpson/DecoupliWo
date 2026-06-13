"""Rule multi-hot helpers: atomic ``RULE_TAGS`` slots (inference toggles, val inject)."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model.dataset import (
	NUM_RULE_TYPES,
	RULE_TAG_TO_INDEX,
	RULE_TAGS,
	active_rule_tags,
	rule_multihot_tuple_from_env_name,
)


def rule_panel_tags(transitions_split_dir: str | Path | None = None) -> list[str]:
	"""Inference / UI toggles: only tags present under ``data/transitions/test`` (or given split)."""
	return list(active_rule_tags(transitions_split_dir))


def rule_onehot_from_tag_list(tags: tuple[str, ...], *, device: torch.device | None = None) -> torch.Tensor:
	"""``[1, R]`` multi-hot with one or more ``RULE_TAGS`` slots set."""
	return rule_onehot_from_tags(frozenset(tags), device=device)


def rule_onehot_from_tags(tags: frozenset[str] | set[str], *, device: torch.device | None = None) -> torch.Tensor:
	"""``[1, R]`` multi-hot; empty tags = base game (all inactive atomic slots)."""
	v = [0.0] * NUM_RULE_TYPES
	for tag in tags:
		if tag not in RULE_TAG_TO_INDEX:
			raise KeyError(f"{tag!r} not in RULE_TAGS")
		v[RULE_TAG_TO_INDEX[tag]] = 1.0
	out = torch.tensor([v], dtype=torch.float32)
	return out.to(device) if device is not None else out


def inject_rule_tag(vec: torch.Tensor, tag: str) -> torch.Tensor:
	if tag not in RULE_TAG_TO_INDEX:
		raise KeyError(f"{tag!r} not in RULE_TAGS")
	out = vec.clone()
	out[..., RULE_TAG_TO_INDEX[tag]] = 1.0
	return out


def rule_multihot_tuple_from_env_folder(env: str) -> tuple[float, ...]:
	"""Dataset-aligned multi-hot from an encoded env directory name."""
	return rule_multihot_tuple_from_env_name(str(env).lower().strip())
