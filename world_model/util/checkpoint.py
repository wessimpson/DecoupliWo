"""Dynamics checkpoint I/O shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from world_model.model.world_model import WorldModel


def read_trainer_args(ckpt_dir: Path) -> dict[str, Any]:
	p = ckpt_dir / "trainer_state.pt"
	if not p.is_file():
		return {}
	blob = torch.load(p, map_location="cpu", weights_only=False)
	return dict(blob.get("args") or {})


def history_len_from_meta(meta: dict[str, Any], override: int | None, default: int = 8) -> int:
	if override is not None:
		return int(override)
	if meta.get("context_len") is not None:
		return int(meta["context_len"])
	return int(default)


def cfg_scale_from_meta(
	meta: dict[str, Any],
	key_new: str,
	key_legacy: str,
	override: float | None,
	default: float,
) -> float:
	if override is not None:
		return float(override)
	if key_new in meta and meta[key_new] is not None:
		return float(meta[key_new])
	if key_legacy in meta and meta[key_legacy] is not None:
		return float(meta[key_legacy])
	return float(default)


def resolve_checkpoint_dir(raw: str, base: Path) -> Path:
	candidate = Path(raw).expanduser()
	if candidate.is_dir():
		return candidate
	under = base / raw
	if under.is_dir():
		return under
	raise FileNotFoundError(f"Checkpoint not found: {raw!r} (tried {candidate} and {under})")


def load_world_model(
	ckpt_dir: Path | str,
	*,
	num_actions: int,
	history_len: int | None = None,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
	cross_attention_dim: int = 768,
	prediction_type: str = "v_prediction",
) -> WorldModel:
	ckpt_dir = Path(ckpt_dir)
	meta = read_trainer_args(ckpt_dir)
	K = history_len_from_meta(meta, history_len)
	if history_len is None and meta.get("context_len") is not None:
		print(f"Using context_len={K} from checkpoint trainer_state.pt")
	vae_eff = vae_checkpoint if vae_checkpoint is not None else meta.get("vae_checkpoint")
	wm = WorldModel(
		num_actions=num_actions,
		cross_attention_dim=cross_attention_dim,
		vae_checkpoint=vae_eff,
		prediction_type=prediction_type,
		history_len=K,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		cfg_scale_action=cfg_scale_from_meta(meta, "cfg_scale_action", "cfg_scale", cfg_scale_action, 1.5),
		cfg_scale_rule=cfg_scale_from_meta(meta, "cfg_scale_rule", "cfg_scale", cfg_scale_rule, 1.5),
		cfg_both_drop_prob=float(meta.get("cfg_both_drop_prob", 0.10)),
		cfg_action_drop_prob=float(meta.get("cfg_action_drop_prob", 0.05)),
		cfg_rule_drop_prob=float(meta.get("cfg_rule_drop_prob", 0.05)),
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = wm.to(device)
	wm.load_diffuser_checkpoint(ckpt_dir, device)
	return wm
