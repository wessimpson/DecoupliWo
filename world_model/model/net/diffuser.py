from __future__ import annotations

from typing import Literal
import math

import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DConditionModel


class Diffuser(nn.Module):
	"""Class bundle for diffusion components used by the world model."""

	def __init__(
		self,
		num_actions: int,
		latent_channels: int,
		buffer_size: int,
		cross_attention_dim: int,
		num_train_timesteps: int,
		prediction_type: Literal["epsilon", "sample", "v_prediction"],
		temporal_downsample: int = 4,
		model_size: Literal["small", "base", "large"] = "small",
	) -> None:
		super().__init__()

		self.num_actions = num_actions
		self.latent_channels = latent_channels
		self.buffer_size = buffer_size
		self.cross_attention_dim = cross_attention_dim

		effective_context = max(1, math.ceil(buffer_size / max(1, int(temporal_downsample))))
		in_channels = latent_channels * (effective_context + 1)

		# Model size presets
		if model_size == "small":
			block_out_channels = (128, 256, 512, 512)
			layers_per_block = 1
		elif model_size == "large":
			block_out_channels = (320, 640, 1280, 1280)
			layers_per_block = 2
		else:  # "base"
			block_out_channels = (192, 384, 768, 768)
			layers_per_block = 2

		self.unet = UNet2DConditionModel(
			sample_size=None,
			in_channels=in_channels,
			out_channels=latent_channels,
			down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
			up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
			block_out_channels=block_out_channels,
			layers_per_block=layers_per_block,
			cross_attention_dim=cross_attention_dim,
			num_class_embeds=None,
		)
		self.action_embedding = nn.Embedding(
			num_embeddings=num_actions + 1,
			embedding_dim=cross_attention_dim,
		)
		nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)

		self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
		self.noise_scheduler.register_to_config(prediction_type=prediction_type)
