from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from data.pong_common import OBJECT_TYPE_TO_ID


@dataclass(frozen=True)
class ObjectGraph:
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    relative_position: torch.Tensor
    relative_velocity: torch.Tensor
    distance: torch.Tensor


class ObjectGraphBuilder(nn.Module):
    """Build differentiable object-interaction graphs from structured slots.

    Slots use the project-wide schema:
    ``x, y, vx, vy, width, height, type_id``. The returned tensors are shaped
    ``[batch, dst, src, ...]`` so messages can be summed over the source axis.
    """

    edge_feature_dim = 14

    def __init__(
        self,
        slot_scales: tuple[float, float, float, float, float, float],
        num_object_types: int,
        edge_mode: str = "hybrid",
        distance_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        if edge_mode not in {"fully_connected", "distance", "hybrid"}:
            raise ValueError(f"Unknown edge_mode={edge_mode!r}")
        self.edge_mode = edge_mode
        self.distance_threshold = float(distance_threshold)
        self.num_object_types = int(num_object_types)
        self.register_buffer("slot_scales", torch.tensor(slot_scales, dtype=torch.float32), persistent=False)

    def forward(self, slots: torch.Tensor, object_mask: torch.Tensor) -> ObjectGraph:
        slots = slots.to(torch.float32)
        object_mask = object_mask.to(torch.float32)
        device = slots.device
        scales = self.slot_scales.to(device=device, dtype=slots.dtype).clamp_min(1.0)

        pos = slots[..., 0:2] / scales[0:2]
        vel = slots[..., 2:4] / scales[2:4]
        size = slots[..., 4:6] / scales[4:6]
        type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1)
        type_norm = type_ids / max(self.num_object_types - 1, 1)

        pos_src = pos[:, None, :, :]
        pos_dst = pos[:, :, None, :]
        vel_src = vel[:, None, :, :]
        vel_dst = vel[:, :, None, :]
        size_src = size[:, None, :, :]
        size_dst = size[:, :, None, :]
        type_src = type_norm[:, None, :, None]
        type_dst = type_norm[:, :, None, None]

        relative_position = pos_src - pos_dst
        relative_velocity = vel_src - vel_dst
        abs_relative_position = relative_position.abs()
        distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
        same_type = (type_src == type_dst).to(slots.dtype)

        edge_features = torch.cat(
            [
                relative_position,
                relative_velocity,
                abs_relative_position,
                distance,
                size_src.expand(-1, slots.shape[1], -1, -1),
                size_dst.expand(-1, -1, slots.shape[1], -1),
                type_src.expand(-1, slots.shape[1], -1, -1),
                type_dst.expand(-1, -1, slots.shape[1], -1),
                same_type,
            ],
            dim=-1,
        )

        active = object_mask[:, :, None] * object_mask[:, None, :]
        nodes = slots.shape[1]
        non_self = 1.0 - torch.eye(nodes, dtype=slots.dtype, device=device)[None, :, :]
        base_mask = active * non_self

        if self.edge_mode == "fully_connected":
            edge_mask = base_mask
        else:
            near = (distance.squeeze(-1) <= self.distance_threshold).to(slots.dtype)
            if self.edge_mode == "distance":
                edge_mask = base_mask * near
            else:
                moving_types = torch.tensor(
                    [OBJECT_TYPE_TO_ID["ball"], OBJECT_TYPE_TO_ID["paddle"]],
                    dtype=torch.long,
                    device=device,
                )
                type_ids_long = type_ids.to(torch.long)
                moving = (type_ids_long[..., None] == moving_types).any(dim=-1)
                moving_pair = (moving[:, :, None] | moving[:, None, :]).to(slots.dtype)
                edge_mask = base_mask * torch.maximum(near, moving_pair)

        return ObjectGraph(
            edge_features=edge_features * base_mask[..., None],
            edge_mask=edge_mask,
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            distance=distance.squeeze(-1),
        )
