"""Selective finetuning for Diffuser: freeze masks, motion-only, last UNet up-blocks, spatial-attn LoRA."""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn as nn

_TRAINABLE_POLICIES = frozenset(
	{
		"full",
		"action_head",
		"motion_modules",
		"action_and_motion",
		"unet_top",
		"lora_attn",
		"action_motion_lora",
		"action_motion_lora_unet_top",
	}
)

# argparse / CLI
TRAINABLE_PARTS_CHOICES: tuple[str, ...] = tuple(sorted(_TRAINABLE_POLICIES))

# Spatial transformer self/cross-attn projections (AnimateDiff UNet2D path, not motion_module internals).
_ATTN_PROJ_RE = re.compile(
	r"\.attn[12]\.(to_q|to_k|to_v|to_out\.0)$"
)


class LoRALinear(nn.Module):
	"""Low-rank adapter around a frozen nn.Linear (standard LoRA on the right)."""

	def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
		super().__init__()
		self.linear = linear
		for p in self.linear.parameters():
			p.requires_grad_(False)
		self.rank = rank
		self.scaling = alpha / rank if rank > 0 else 0.0
		self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
		self.lora_B = nn.Parameter(torch.empty(linear.out_features, rank))
		nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
		nn.init.zeros_(self.lora_B)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.linear(x)
		if self.rank <= 0 or self.scaling == 0:
			return out
		return out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def _is_attn_proj_linear(full_name: str, include_motion: bool) -> bool:
	if not _ATTN_PROJ_RE.search(full_name):
		return False
	if not include_motion and "motion_modules" in full_name:
		return False
	return True


def inject_spatial_attention_lora(
	unet: nn.Module,
	*,
	rank: int,
	alpha: float,
	include_motion: bool = False,
) -> int:
	"""Replace matching Linear modules with LoRALinear. Returns number of layers adapted."""
	if rank <= 0:
		return 0
	names = [
		n
		for n, m in unet.named_modules()
		if isinstance(m, nn.Linear) and _is_attn_proj_linear(n, include_motion) and "lora_" not in n
	]
	# Deepest paths first so parent paths remain valid while replacing leaves.
	names.sort(key=len, reverse=True)
	for full in names:
		linear = unet.get_submodule(full)
		assert isinstance(linear, nn.Linear)
		parent_path, _, attr = full.rpartition(".")
		parent = unet.get_submodule(parent_path) if parent_path else unet
		setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha))
	return len(names)


def count_trainable_params(module: nn.Module) -> tuple[int, int]:
	n_train, n_tot = 0, 0
	for p in module.parameters():
		n = p.numel()
		n_tot += n
		if p.requires_grad:
			n_train += n
	return n_train, n_tot


def apply_diffuser_train_policy(
	diffuser: Any,
	policy: str,
	*,
	unet_top_n_blocks: int = 2,
	lora_rank: int = 8,
	lora_alpha: float = 8.0,
	lora_include_motion: bool = False,
) -> None:
	"""Set requires_grad on Diffuser (UNet + action head). May inject LoRA once."""
	if policy not in _TRAINABLE_POLICIES:
		raise ValueError(f"trainable policy must be one of {sorted(_TRAINABLE_POLICIES)}, got {policy!r}")

	if policy in ("lora_attn", "action_motion_lora", "action_motion_lora_unet_top") and lora_rank <= 0:
		raise ValueError(f"trainable_parts={policy} requires lora_rank > 0")

	needs_lora = policy in ("lora_attn", "action_motion_lora", "action_motion_lora_unet_top")
	if needs_lora and not getattr(diffuser, "_attn_lora_injected", False):
		n = inject_spatial_attention_lora(
			diffuser.unet,
			rank=lora_rank,
			alpha=lora_alpha,
			include_motion=lora_include_motion,
		)
		diffuser._attn_lora_injected = True  # type: ignore[attr-defined]
		diffuser._attn_lora_layers = n  # type: ignore[attr-defined]

	# Default: train everything in diffuser.
	for p in diffuser.parameters():
		p.requires_grad_(True)

	if policy == "full":
		return

	# Freeze all; then enable selected subsets.
	diffuser.requires_grad_(False)

	def unfreeze_unet_by_name(pred) -> None:
		for name, p in diffuser.unet.named_parameters():
			if pred(name):
				p.requires_grad_(True)

	if policy == "action_head":
		diffuser.action_embedding.requires_grad_(True)
		diffuser.mlp.requires_grad_(True)
	elif policy == "motion_modules":
		unfreeze_unet_by_name(lambda n: "motion_modules" in n)
	elif policy == "action_and_motion":
		diffuser.action_embedding.requires_grad_(True)
		diffuser.mlp.requires_grad_(True)
		unfreeze_unet_by_name(lambda n: "motion_modules" in n)
	elif policy == "unet_top":
		n = max(0, int(unet_top_n_blocks))
		nu = len(diffuser.unet.up_blocks)
		keep = set(range(nu - n, nu)) if n > 0 and nu > 0 else set()

		def pred_top(name: str) -> bool:
			if name.startswith("conv_out."):
				return True
			for i in keep:
				if name.startswith(f"up_blocks.{i}."):
					return True
			return False

		unfreeze_unet_by_name(pred_top)
	elif policy == "lora_attn":
		for name, p in diffuser.unet.named_parameters():
			if "lora_A" in name or "lora_B" in name:
				p.requires_grad_(True)
	elif policy == "action_motion_lora":
		diffuser.action_embedding.requires_grad_(True)
		diffuser.mlp.requires_grad_(True)
		unfreeze_unet_by_name(lambda n: "motion_modules" in n)
		for name, p in diffuser.unet.named_parameters():
			if "lora_A" in name or "lora_B" in name:
				p.requires_grad_(True)
	elif policy == "action_motion_lora_unet_top":
		# Union: action head + motion + spatial-attn LoRA + last N up_blocks (full weights) + conv_out.
		diffuser.action_embedding.requires_grad_(True)
		diffuser.mlp.requires_grad_(True)
		unfreeze_unet_by_name(lambda n: "motion_modules" in n)
		for name, p in diffuser.unet.named_parameters():
			if "lora_A" in name or "lora_B" in name:
				p.requires_grad_(True)
		n = max(0, int(unet_top_n_blocks))
		nu = len(diffuser.unet.up_blocks)
		keep = set(range(nu - n, nu)) if n > 0 and nu > 0 else set()

		def pred_top(name: str) -> bool:
			if name.startswith("conv_out."):
				return True
			for i in keep:
				if name.startswith(f"up_blocks.{i}."):
					return True
			return False

		unfreeze_unet_by_name(pred_top)
	else:
		raise AssertionError(policy)
