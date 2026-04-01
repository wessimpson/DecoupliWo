"""Small convolutional VAE for single RGB frames (Atari-scale)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariFrameVAE(nn.Module):
    """Encode/decode one frame: [B, 3, H, W] in [-1, 1] <-> latent [B, latent_dim]."""

    def __init__(self, image_size: int = 84, latent_dim: int = 128) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.SiLU(),
        )
        with torch.no_grad():
            t = self.enc(torch.zeros(1, 3, image_size, image_size))
            self.register_buffer("_shape_buf", torch.tensor(t.shape[1:], dtype=torch.long))
            d = t.numel()
        self.to_mu = nn.Linear(d, latent_dim)
        self.to_logvar = nn.Linear(d, latent_dim)
        self.from_z = nn.Linear(latent_dim, d)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        shape = self._shape_buf.tolist()
        h = self.from_z(z).view(z.size(0), *shape)
        out = self.dec(h)
        out = F.interpolate(
            out,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.tanh(out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_recon_kl(self, x: torch.Tensor, kl_weight: float = 1e-3) -> torch.Tensor:
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl
