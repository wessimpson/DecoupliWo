"""Shared metrics, AMP, and VAE eval for train_vae / train_world_model / train_dynamics."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter

LPIPS_W = 0.1


def amp_autocast(device: torch.device, mixed_precision: str):
	if mixed_precision == "no" or device.type != "cuda":
		return nullcontext()
	dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
	return torch.autocast(device_type="cuda", dtype=dtype)


def psnr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
	"""Mean PSNR (dB) in [0,1] space; inputs in [-1, 1]."""
	p = ((pred.clamp(-1, 1) + 1) * 0.5).float()
	t = ((tgt.clamp(-1, 1) + 1) * 0.5).float()
	mse = (p - t).pow(2).mean().item()
	if mse <= 0:
		return float("inf")
	return 10.0 * math.log10(1.0 / mse)


def vae_train_loss(
	ae: AutoencoderKL,
	x: torch.Tensor,
	lpips_fn: torch.nn.Module,
	kl_w: float,
	lpips_w: float = LPIPS_W,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	post = ae.encode(x).latent_dist
	recon = ae.decode(post.sample()).sample
	xf, rf = x.float(), recon.float()
	mse = F.mse_loss(rf, xf)
	lp = lpips_fn(xf, rf).mean()
	kl = post.kl().mean()
	loss = mse + lpips_w * lp + kl_w * kl
	return loss, {"mse": mse, "lpips": lp, "kl": kl}


@torch.no_grad()
def eval_vae(
	ae: AutoencoderKL,
	pixels: torch.Tensor,
	device: torch.device,
	lpips_fn: torch.nn.Module,
	kl_w: float,
	mixed_precision: str = "no",
	writer: SummaryWriter | None = None,
	step: int = 0,
	prefix: str = "val",
	max_images: int = 8,
	lpips_w: float = LPIPS_W,
) -> dict[str, float]:
	ae.eval()
	x = pixels.to(device=device, dtype=torch.float32)
	with amp_autocast(device, mixed_precision):
		post = ae.encode(x).latent_dist
		recon = ae.decode(post.mode()).sample
	xf, rf = x.float(), recon.float()
	mse = F.mse_loss(rf, xf)
	lp = lpips_fn(xf, rf).mean()
	kl = post.kl().mean()
	out = {
		"loss": float((mse + lpips_w * lp + kl_w * kl).item()),
		"mse": float(mse.item()),
		"lpips": float(lp.item()),
		"kl": float(kl.item()),
		"psnr": psnr(x, recon),
	}
	if writer is not None:
		for k in ("loss", "mse", "lpips", "kl", "psnr"):
			writer.add_scalar(f"{prefix}/{k}", out[k], step)
		n = min(max_images, x.shape[0])
		to01 = lambda t: (t[:n].float().cpu().clamp(-1, 1) + 1.0) * 0.5
		writer.add_images(f"{prefix}/input", to01(x), step)
		writer.add_images(f"{prefix}/reconstruction", to01(recon), step)
	return out


def log_scalars(writer: SummaryWriter, prefix: str, metrics: dict[str, Any], step: int) -> None:
	for k, v in metrics.items():
		if isinstance(v, (int, float)):
			writer.add_scalar(f"{prefix}/{k}", v, step)
