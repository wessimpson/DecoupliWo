from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.gns_shared_dataset import GNSNormalizationStats
from data.pong_common import GAME_TO_ID, OBJECT_TYPE_TO_ID


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, layer_norm: bool = True) -> nn.Sequential:
    parts: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(max(int(layers), 1)):
        parts.append(nn.Linear(dim, int(hidden_dim)))
        if layer_norm:
            parts.append(nn.LayerNorm(int(hidden_dim)))
        parts.append(nn.ReLU())
        dim = int(hidden_dim)
    parts.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*parts)


@dataclass(frozen=True)
class SharedSimulatorConstants:
    width: float = 640.0
    height: float = 480.0
    dt: float = 1.0 / 60.0
    paddle_speed: float = 360.0
    max_ball_speed: float = 720.0


class GNSSharedSimulator(nn.Module):
    def __init__(
        self,
        stats: GNSNormalizationStats,
        history_length: int = 6,
        connectivity_radius: float = 0.35,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        message_passing_steps: int = 10,
        type_dim: int = 16,
        rule_dim: int = 16,
        game_dim: int = 16,
        action_dim: int = 3,
        num_rules: int = 3,
        num_games: int = 2,
        num_object_types: int = 4,
        constants: SharedSimulatorConstants | None = None,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.connectivity_radius = float(connectivity_radius)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.action_dim = int(action_dim)
        self.constants = constants or SharedSimulatorConstants()
        self.num_object_types = int(num_object_types)
        self.register_buffer("pos_mean", torch.as_tensor(stats.pos_mean, dtype=torch.float32))
        self.register_buffer("pos_std", torch.as_tensor(stats.pos_std, dtype=torch.float32))
        self.register_buffer("vel_mean", torch.as_tensor(stats.vel_mean, dtype=torch.float32))
        self.register_buffer("vel_std", torch.as_tensor(stats.vel_std, dtype=torch.float32))

        node_in_dim = self.history_length * 2 + (self.history_length - 1) * 2 + 4 + 2 + action_dim + 3 + type_dim + rule_dim + game_dim
        edge_in_dim = 3
        self.type_embedding = nn.Embedding(num_object_types, type_dim)
        self.rule_embedding = nn.Embedding(num_rules, rule_dim)
        self.game_embedding = nn.Embedding(num_games, game_dim)
        self.node_encoder = _mlp(node_in_dim, hidden_dim, latent_dim)
        self.edge_encoder = _mlp(edge_in_dim, hidden_dim, latent_dim)
        self.processor_edges = nn.ModuleList([_mlp(latent_dim * 3, hidden_dim, latent_dim) for _ in range(self.message_passing_steps)])
        self.processor_nodes = nn.ModuleList([_mlp(latent_dim * 2, hidden_dim, latent_dim) for _ in range(self.message_passing_steps)])
        self.delta_head = _mlp(latent_dim, hidden_dim, 2)
        self.mask_head = _mlp(latent_dim, hidden_dim, 1)

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions - self.pos_mean.to(positions.device)) / self.pos_std.to(positions.device)

    def denormalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions * self.pos_std.to(positions.device) + self.pos_mean.to(positions.device)

    def normalize_velocities(self, velocities: torch.Tensor) -> torch.Tensor:
        return (velocities - self.vel_mean.to(velocities.device)) / self.vel_std.to(velocities.device)

    def analytic_paddle_update(self, latest_slots: torch.Tensor, latest_mask: torch.Tensor, action: torch.Tensor, game_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_slots = latest_slots.clone()
        next_mask = latest_mask.clone()
        action = action.to(torch.long)
        game_id = game_id.to(torch.long)
        direction_values = torch.tensor([0.0, -1.0, 1.0], dtype=latest_slots.dtype, device=latest_slots.device)
        directions = direction_values[action]
        paddle_mask = (latest_slots[..., 6].round().to(torch.long) == OBJECT_TYPE_TO_ID["paddle"]) & (latest_mask > 0.5)
        if not paddle_mask.any():
            return next_slots, next_mask
        paddle_indices = paddle_mask.nonzero(as_tuple=False)
        for batch_idx, node_idx in paddle_indices:
            slot = latest_slots[batch_idx, node_idx]
            direction = directions[batch_idx]
            if int(game_id[batch_idx].item()) == GAME_TO_ID["pong"]:
                old_y = slot[1]
                new_y = torch.clamp(old_y + direction * self.constants.paddle_speed * self.constants.dt, 0.0, self.constants.height - slot[5])
                next_slots[batch_idx, node_idx, 1] = new_y
                next_slots[batch_idx, node_idx, 3] = (new_y - old_y) / self.constants.dt
                next_slots[batch_idx, node_idx, 2] = 0.0
            else:
                old_x = slot[0]
                new_x = torch.clamp(old_x + direction * self.constants.paddle_speed * self.constants.dt, 0.0, self.constants.width - slot[4])
                next_slots[batch_idx, node_idx, 0] = new_x
                next_slots[batch_idx, node_idx, 2] = (new_x - old_x) / self.constants.dt
                next_slots[batch_idx, node_idx, 3] = 0.0
        return next_slots, next_mask

    def add_training_noise(self, history_slots: torch.Tensor, dynamic_pos_mask: torch.Tensor, noise_std: float) -> torch.Tensor:
        if float(noise_std) <= 0.0:
            return history_slots
        noisy = history_slots.clone()
        batch, time_steps, nodes, _ = history_slots.shape
        device = history_slots.device
        dtype = history_slots.dtype
        vel_noise = torch.randn(batch, time_steps - 1, nodes, 2, device=device, dtype=dtype) * (float(noise_std) / max((time_steps - 1) ** 0.5, 1.0))
        vel_noise = torch.cumsum(vel_noise, dim=1)
        pos_noise = torch.cat([torch.zeros(batch, 1, nodes, 2, device=device, dtype=dtype), torch.cumsum(vel_noise, dim=1)], dim=1)
        pos_noise = pos_noise * dynamic_pos_mask[:, None, :, None].to(dtype)
        noisy[..., :2] = noisy[..., :2] + pos_noise
        noisy[..., 2:4] = noisy[..., 2:4] + torch.cat([torch.zeros(batch, 1, nodes, 2, device=device, dtype=dtype), vel_noise], dim=1) * dynamic_pos_mask[:, None, :, None].to(dtype)
        return noisy

    def build_node_features(
        self,
        history_slots: torch.Tensor,
        history_mask: torch.Tensor,
        action: torch.Tensor,
        rule_id: torch.Tensor,
        game_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latest_slots = history_slots[:, -1]
        latest_mask = history_mask[:, -1]
        positions = history_slots[..., :2]
        norm_positions = self.normalize_positions(positions)
        vel = (positions[:, 1:] - positions[:, :-1]) / self.constants.dt
        norm_vel = self.normalize_velocities(vel)
        flat_pos = norm_positions.reshape(history_slots.shape[0], history_slots.shape[2], -1)
        flat_vel = norm_vel.reshape(history_slots.shape[0], history_slots.shape[2], -1)
        latest_pos = latest_slots[..., :2]
        boundary = torch.stack(
            [
                latest_pos[..., 0] / max(self.constants.width, 1.0),
                latest_pos[..., 1] / max(self.constants.height, 1.0),
                (self.constants.width - latest_pos[..., 0]) / max(self.constants.width, 1.0),
                (self.constants.height - latest_pos[..., 1]) / max(self.constants.height, 1.0),
            ],
            dim=-1,
        ).clamp(-2.0, 2.0)
        sizes = latest_slots[..., 4:6].clone()
        sizes[..., 0] = sizes[..., 0] / max(self.constants.width, 1.0)
        sizes[..., 1] = sizes[..., 1] / max(self.constants.height, 1.0)
        type_ids = latest_slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        type_emb = self.type_embedding(type_ids)
        rule_emb = self.rule_embedding(rule_id.to(torch.long))[:, None, :].expand(-1, latest_slots.shape[1], -1)
        game_emb = self.game_embedding(game_id.to(torch.long))[:, None, :].expand(-1, latest_slots.shape[1], -1)
        action_one_hot = F.one_hot(action.to(torch.long), num_classes=self.action_dim).to(latest_slots.dtype)
        action_nodes = action_one_hot[:, None, :].expand(-1, latest_slots.shape[1], -1)
        ball_mask = (type_ids == OBJECT_TYPE_TO_ID["ball"]).to(latest_slots.dtype)
        paddle_mask = (type_ids == OBJECT_TYPE_TO_ID["paddle"]).to(latest_slots.dtype)
        block_mask = (type_ids == OBJECT_TYPE_TO_ID["block"]).to(latest_slots.dtype)
        flags = torch.stack([latest_mask, paddle_mask, block_mask], dim=-1)
        node_features = torch.cat([flat_pos, flat_vel, boundary, sizes, action_nodes, flags, type_emb, rule_emb, game_emb], dim=-1)
        return node_features, latest_slots, latest_mask

    def build_connectivity(self, latest_slots: torch.Tensor, latest_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.normalize_positions(latest_slots[..., :2])
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        active = (latest_mask > 0.5).to(pos.dtype)
        pair_mask = active[:, :, None] * active[:, None, :]
        eye = torch.eye(latest_slots.shape[1], device=latest_slots.device, dtype=pair_mask.dtype)[None]
        pair_mask = pair_mask * (1.0 - eye)
        radius_mask = (dist[..., 0] <= float(self.connectivity_radius)).to(pair_mask.dtype)
        edge_mask = pair_mask * radius_mask
        edge_features = torch.cat([rel, dist], dim=-1)
        return edge_features, edge_mask

    def compose_prediction(
        self,
        latest_slots: torch.Tensor,
        latest_mask: torch.Tensor,
        action: torch.Tensor,
        game_id: torch.Tensor,
        delta_pos_norm: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_slots, next_mask = self.analytic_paddle_update(latest_slots, latest_mask, action, game_id)
        type_ids = latest_slots[..., 6].round().to(torch.long)
        ball_mask = (type_ids == OBJECT_TYPE_TO_ID["ball"]) & (latest_mask > 0.5)
        block_mask = (type_ids == OBJECT_TYPE_TO_ID["block"]) & (latest_mask > 0.5)

        latest_pos_norm = self.normalize_positions(latest_slots[..., :2])
        pred_pos = self.denormalize_positions(latest_pos_norm + delta_pos_norm)
        next_slots[..., :2] = torch.where(ball_mask[..., None], pred_pos, next_slots[..., :2])
        next_slots[..., 2:4] = torch.where(
            ball_mask[..., None],
            (next_slots[..., :2] - latest_slots[..., :2]) / self.constants.dt,
            next_slots[..., 2:4],
        )
        next_slots[..., 2:4] = torch.where(block_mask[..., None], torch.zeros_like(next_slots[..., 2:4]), next_slots[..., 2:4])

        strong_keep = torch.full_like(mask_logits[..., 0], 15.0)
        strong_drop = torch.full_like(mask_logits[..., 0], -15.0)
        final_logits = torch.where(block_mask, mask_logits[..., 0], torch.where(latest_mask > 0.5, strong_keep, strong_drop))
        next_mask = torch.where(block_mask, torch.sigmoid(final_logits), latest_mask)
        next_mask = torch.where(type_ids == OBJECT_TYPE_TO_ID["ball"], torch.ones_like(next_mask), next_mask)
        next_mask = torch.where(type_ids == OBJECT_TYPE_TO_ID["paddle"], torch.ones_like(next_mask), next_mask)
        return next_slots, final_logits, next_mask

    def forward(
        self,
        history_slots: torch.Tensor,
        history_mask: torch.Tensor,
        action: torch.Tensor,
        rule_id: torch.Tensor,
        game_id: torch.Tensor,
        dynamic_pos_mask: torch.Tensor | None = None,
        noise_std: float = 0.0,
        training_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        if training_noise and dynamic_pos_mask is not None:
            history_slots = self.add_training_noise(history_slots, dynamic_pos_mask, noise_std)
        node_features, latest_slots, latest_mask = self.build_node_features(history_slots, history_mask, action, rule_id, game_id)
        node_latent = self.node_encoder(node_features)
        edge_features, edge_mask = self.build_connectivity(latest_slots, latest_mask)
        edge_latent = self.edge_encoder(edge_features)

        current_nodes = node_latent
        current_edges = edge_latent
        for edge_net, node_net in zip(self.processor_edges, self.processor_nodes):
            src = current_nodes[:, :, None, :].expand_as(current_edges)
            dst = current_nodes[:, None, :, :].expand_as(current_edges)
            edge_input = torch.cat([src, dst, current_edges], dim=-1)
            updated_edges = edge_net(edge_input) * edge_mask[..., None]
            messages = updated_edges.sum(dim=2)
            updated_nodes = node_net(torch.cat([current_nodes, messages], dim=-1)) * latest_mask[..., None]
            current_nodes = current_nodes + updated_nodes
            current_edges = current_edges + updated_edges

        delta_pos_norm = self.delta_head(current_nodes)
        mask_logits = self.mask_head(current_nodes)
        pred_next_slots, pred_next_mask_logits, pred_next_mask_prob = self.compose_prediction(
            latest_slots, latest_mask, action, game_id, delta_pos_norm, mask_logits
        )
        return {
            "pred_next_slots": pred_next_slots,
            "pred_next_mask_logits": pred_next_mask_logits,
            "pred_next_mask_prob": pred_next_mask_prob,
            "delta_pos_norm": delta_pos_norm,
            "latest_slots": latest_slots,
            "latest_mask": latest_mask,
            "edge_mask": edge_mask,
        }


def build_gns_model_from_checkpoint(checkpoint: dict, device: torch.device) -> GNSSharedSimulator:
    args = checkpoint.get("args", {})
    stats = GNSNormalizationStats.from_dict(checkpoint["normalization_stats"])
    constants = SharedSimulatorConstants(**checkpoint.get("constants", {}))
    model = GNSSharedSimulator(
        stats=stats,
        history_length=int(args.get("history_length", 6)),
        connectivity_radius=float(args.get("connectivity_radius", 0.35)),
        latent_dim=int(args.get("latent_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        message_passing_steps=int(args.get("message_passing_steps", 10)),
        type_dim=int(args.get("type_dim", 16)),
        rule_dim=int(args.get("rule_dim", 16)),
        game_dim=int(args.get("game_dim", 16)),
        constants=constants,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


__all__ = ["GNSSharedSimulator", "GNSSharedSimulator", "SharedSimulatorConstants", "build_gns_model_from_checkpoint"]
