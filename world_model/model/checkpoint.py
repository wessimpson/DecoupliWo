"""Checkpoint helpers for base and residual dynamics models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from world_model.model.world_model import ResidualWorldModel, WorldModel


def load_sd(path: Path, device: torch.device) -> dict:
	return torch.load(path, map_location=device, weights_only=True)


def read_trainer_args(ckpt_dir: str | Path) -> dict[str, Any]:
	p = Path(ckpt_dir) / "trainer_state.pt"
	if not p.is_file():
		return {}
	blob = torch.load(p, map_location="cpu", weights_only=False)
	return dict(blob.get("args") or {})


def _coalesce(meta: dict[str, Any], key: str, override: Any, fallback: Any) -> Any:
	if override is not None:
		return override
	if key in meta and meta[key] is not None:
		return meta[key]
	return fallback


def _float_from_meta(
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


def build_world_model_from_meta(
	ckpt_dir: str | Path,
	num_actions: int,
	history_len: int,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
	num_rules: int = 0,
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
	zero_init_output: bool = False,
) -> WorldModel:
	meta = read_trainer_args(ckpt_dir)
	vae_eff = _coalesce(meta, "vae_checkpoint", vae_checkpoint, None)
	pretrained_eff = _coalesce(
		meta,
		"pretrained_model_name_or_path",
		pretrained_model_name_or_path,
		pretrained_model_name_or_path,
	)
	num_rules_eff = int(_coalesce(meta, "num_rules", num_rules, num_rules))
	c_sa = _float_from_meta(meta, "cfg_scale_action", "cfg_scale", cfg_scale_action, 1.5)
	c_sr = _float_from_meta(meta, "cfg_scale_rule", "cfg_scale", cfg_scale_rule, 1.5)
	return WorldModel(
		num_actions=num_actions,
		cross_attention_dim=768,
		vae_checkpoint=vae_eff,
		prediction_type=str(_coalesce(meta, "prediction_type", None, "v_prediction")),
		history_len=history_len,
		pretrained_model_name_or_path=str(pretrained_eff),
		num_rules=num_rules_eff,
		cfg_scale_action=c_sa,
		cfg_scale_rule=c_sr,
		zero_init_output=zero_init_output,
	)


def load_world_model_checkpoint(
	ckpt_dir: str | Path,
	num_actions: int,
	history_len: int,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
	num_rules: int = 0,
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
	device: torch.device | None = None,
) -> WorldModel:
	"""Load a base or residual denoiser checkpoint."""
	ckpt_dir = Path(ckpt_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
	wm = build_world_model_from_meta(
		ckpt_dir,
		num_actions=num_actions,
		history_len=history_len,
		vae_checkpoint=vae_checkpoint,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		num_rules=num_rules,
		cfg_scale_action=cfg_scale_action,
		cfg_scale_rule=cfg_scale_rule,
	)
	if wm.diffuser.num_rules != int(num_rules):
		raise RuntimeError(
			f"{ckpt_dir} was built with num_rules={wm.diffuser.num_rules}, expected num_rules={int(num_rules)}"
		)
	wm = wm.to(device)

	unet_path = ckpt_dir / "unet.pt"
	if not unet_path.is_file():
		raise FileNotFoundError(f"Missing UNet weights: {unet_path}")
	wm.diffuser.unet.load_state_dict(load_sd(unet_path, device), strict=True)

	emb_path = ckpt_dir / "action_embedding.pt"
	if not emb_path.is_file():
		raise FileNotFoundError(f"Missing action embedding weights: {emb_path}")
	wm.diffuser.action_embedding.load_state_dict(load_sd(emb_path, device), strict=True)

	rule_path = ckpt_dir / "rule_projection.pt"
	if wm.diffuser.num_rules > 0:
		if not rule_path.is_file():
			raise FileNotFoundError(f"Missing residual rule projection weights: {rule_path}")
		if wm.diffuser.rule_projection is None:
			raise RuntimeError("checkpoint requested rule weights but model has no rule projection")
		wm.diffuser.rule_projection.load_state_dict(load_sd(rule_path, device), strict=True)
	elif rule_path.is_file():
		raise RuntimeError(f"{ckpt_dir} has rule_projection.pt but model was built with num_rules=0")

	return wm


def load_residual_world_model(
	base_ckpt_dir: str | Path,
	residual_ckpt_dir: str | Path,
	num_actions: int,
	history_len: int,
	vae_checkpoint: str | Path | None = None,
	device: torch.device | None = None,
	cfg_scale_action: float | None = None,
	cfg_scale_rule: float | None = None,
) -> ResidualWorldModel:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
	base = load_world_model_checkpoint(
		base_ckpt_dir,
		num_actions=num_actions,
		history_len=history_len,
		vae_checkpoint=vae_checkpoint,
		num_rules=0,
		device=device,
		cfg_scale_action=cfg_scale_action,
	)
	residual = load_world_model_checkpoint(
		residual_ckpt_dir,
		num_actions=num_actions,
		history_len=history_len,
		vae_checkpoint=vae_checkpoint,
		num_rules=3,
		device=device,
		cfg_scale_action=cfg_scale_action,
		cfg_scale_rule=cfg_scale_rule,
	)
	return ResidualWorldModel(base, residual).to(device)
