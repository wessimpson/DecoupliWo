# TODO: add causal attention mask to the transformer layers
"""Causal temporal transformer + flow matching on per-frame latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] in [0, 1] or arbitrary; returns [B, dim]."""
    half = dim // 2
    t = t.float().view(-1, 1) * torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    return torch.cat([t.sin(), t.cos()], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], causal_mask: [T, T] with -inf above diagonal
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att + causal_mask.view(1, 1, t, t)
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class AdaLNBlock(nn.Module):
    """Scale/shift norm from diffusion time (DiT-style, simplified)."""

    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: [B, time_dim] -> gamma, beta [B, 1, D]
        g, b = self.mod(t_emb).chunk(2, dim=-1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        h = self.norm(x)
        return h * (1 + g) + b


class CausalTransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff: int, time_dim: int) -> None:
        super().__init__()
        self.ada1 = AdaLNBlock(dim, time_dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ada2 = AdaLNBlock(dim, time_dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff, dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ada1(x, t_emb)
        x = x + self.attn(h, mask)
        h = self.ada2(x, t_emb)
        x = x + self.ff(h)
        return x


class CausalLatentFlowModel(nn.Module):
    """
    Predicts flow target v = eps - z0 for linear path z_t = (1-t) z0 + t eps.
    Causal over time: position i only attends to j <= i.
    Conditioned on discrete actions a_t (same index as latent z_t).
    """

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.time_dim = time_dim
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        self.act_emb = nn.Embedding(n_actions, d_model)
        self.time_in = nn.Linear(time_dim, time_dim)
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayer(d_model, n_heads, ff_mult * d_model, time_dim)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        m = torch.zeros(t, t, device=device)
        m = m.masked_fill(torch.triu(torch.ones(t, t, device=device), diagonal=1).bool(), float("-inf"))
        return m

    def forward(
        self,
        z_noisy: torch.Tensor,
        t_diff: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_noisy: [B, T, latent_dim]
        t_diff: [B] flow time in [0, 1]
        actions: [B, T] long
        """
        b, seq, _ = z_noisy.shape
        device = z_noisy.device
        if seq > self.pos_emb.size(1):
            raise ValueError(f"sequence {seq} > max pos {self.pos_emb.size(1)}")

        x = self.in_proj(z_noisy) + self.pos_emb[:, :seq, :] + self.act_emb(actions.clamp(min=0))
        te = sinusoidal_time_embed(t_diff, self.time_dim)
        te = self.time_in(te)
        mask = self._causal_mask(seq, device)
        for layer in self.layers:
            x = layer(x, te, mask)
        x = self.out_norm(x)
        return self.out_proj(x)
