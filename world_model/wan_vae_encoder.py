"""
Frozen Wan 2.1 VAE (pretrained) -> fixed-size latent vectors per frame.

Expects ``Wan2.1_VAE.pth`` in a directory (e.g. HuggingFace ``Skywork/Matrix-Game-2.0`` or Wan2.1 release).
Adds ``Matrix-Game-2`` to ``sys.path`` so ``wan.*`` imports resolve without ``pip install`` that package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MG2 = _REPO_ROOT / "Matrix-Game-2"
if _MG2.is_dir() and str(_MG2) not in sys.path:
    sys.path.insert(0, str(_MG2))

try:
    from wan.vae.wanx_vae_src.vae import WanVAE
except ImportError as e:
    WanVAE = None  # type: ignore[misc, assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class FrozenWanLatentEncoder(nn.Module):
    """
    Encode RGB frames in [-1, 1] using pretrained Wan VAE (deterministic mu),
    spatially pool latents to a fixed vector size for ``CausalLatentFlowModel``.
    """

    def __init__(
        self,
        pretrained_dir: str | Path,
        image_size: int = 256,
        pool_size: int = 4,
        z_dim: int = 16,
    ) -> None:
        super().__init__()
        if WanVAE is None:
            raise ImportError(
                "Could not import Wan VAE. Install Matrix-Game-2 deps (see Matrix-Game-2/requirements.txt) "
                f"and ensure Matrix-Game-2 lives at {_MG2}. Original error: {_IMPORT_ERR}"
            )
        pretrained_dir = Path(pretrained_dir)
        ckpt = pretrained_dir / "Wan2.1_VAE.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Missing {ckpt}. Place ``Wan2.1_VAE.pth`` in ``world_model/checkpoint/vae/`` "
                "(default) or set ``--wan_vae_dir``. Download e.g. from "
                "https://huggingface.co/Skywork/Matrix-Game-2.0 or Wan-AI/Wan2.1-T2V-1.3B."
            )
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 for Wan VAE.")
        self.image_size = image_size
        self.pool_size = pool_size
        self.z_dim = z_dim
        self.wan = WanVAE(pretrained_path=str(ckpt), z_dim=z_dim)
        self.wan.eval()
        self.wan.requires_grad_(False)

    @property
    def latent_dim(self) -> int:
        return self.z_dim * self.pool_size * self.pool_size

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.wan.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_frames(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        x: [B, T, 3, H, W], values in [-1, 1].
        Returns z0: [B, T, latent_dim] on ``device``, dtype model dtype.
        """
        b, t, c, _, _ = x.shape
        if c != 3:
            raise ValueError(f"expected 3 RGB channels, got {c}")
        x = x.flatten(0, 1)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # Wan batching: first dim iterates clips [C, T, H, W]
        vid = x.unsqueeze(2)  # [N, 3, 1, H, H]
        dtype = next(self.wan.model.parameters()).dtype
        vid = vid.to(device=device, dtype=dtype)
        z = self.wan.encode(vid, device=device, tiled=False)  # [N, 16, T', h, w]
        if z.dim() == 5 and z.size(2) == 1:
            z = z.squeeze(2)
        z = F.adaptive_avg_pool2d(z, (self.pool_size, self.pool_size))
        z = z.flatten(1).to(device=device)
        return z.view(b, t, -1)
