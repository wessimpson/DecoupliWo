"""Minimal causal latent flow model for Atari-style (frame, action) rollouts.

Inspired by Matrix-Game-style causal + flow matching, without Wan/VAE/CLIP scale.
"""

from world_model.model.atari_vae import AtariVAEEncoder
from world_model.causal_latent_dit import CausalLatentFlowModel

__all__ = ["AtariVAEEncoder", "CausalLatentFlowModel"]
