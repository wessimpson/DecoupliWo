from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from data.pong_common import (
    EVENTS,
    GAME_TO_ID,
    MAX_OBJECTS,
    OBJECT_TYPES,
    OBJECT_TYPE_TO_ID,
    RULE_TO_ID,
    SLOT_DIM,
    PongSlotConfig,
)
from models.object_graph import ObjectGraphBuilder


@dataclass(frozen=True)
class PongObjectConstants:
    width: float = 640.0
    height: float = 480.0
    dt: float = 1.0 / 60.0
    paddle_width: float = 12.0
    paddle_height: float = 88.0
    paddle_margin: float = 24.0
    paddle_speed: float = 360.0
    ball_radius: float = 8.0
    max_ball_speed: float = 720.0
    max_objects: int = MAX_OBJECTS

    @property
    def scales(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.width,
            self.height,
            self.max_ball_speed,
            self.max_ball_speed,
            max(1.0, self.height - self.paddle_height),
            max(1.0, self.paddle_speed),
        )

    @property
    def slot_scales(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.width,
            self.height,
            self.max_ball_speed,
            self.max_ball_speed,
            self.width,
            self.height,
        )

    @property
    def paddle_x(self) -> float:
        return self.width - self.paddle_margin - self.paddle_width

    def as_slot_config(self) -> PongSlotConfig:
        return PongSlotConfig(
            width=self.width,
            height=self.height,
            paddle_width=self.paddle_width,
            paddle_height=self.paddle_height,
            paddle_margin=self.paddle_margin,
            paddle_speed=self.paddle_speed,
            ball_radius=self.ball_radius,
            max_ball_speed=self.max_ball_speed,
        )


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(max(int(layers), 1)):
        modules += [nn.Linear(dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.ReLU()]
        dim = int(hidden_dim)
    modules.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*modules)


class PongStateNormalizer(nn.Module):
    def __init__(self, constants: PongObjectConstants | None = None):
        super().__init__()
        constants = constants or PongObjectConstants()
        self.constants = constants
        self.register_buffer("scales", torch.tensor(constants.scales, dtype=torch.float32))
        self.register_buffer("slot_scales", torch.tensor(constants.slot_scales, dtype=torch.float32))
        dt = max(float(constants.dt), 1e-6)
        self.register_buffer(
            "acceleration_scales",
            torch.tensor((constants.max_ball_speed / dt, constants.max_ball_speed / dt), dtype=torch.float32),
        )

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return state.to(torch.float32) / self.scales.to(state.device)

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        return state.to(torch.float32) * self.scales.to(state.device)

    def normalize_slots(self, slots: torch.Tensor) -> torch.Tensor:
        out = slots.to(torch.float32).clone()
        out[..., :6] = out[..., :6] / self.slot_scales.to(slots.device)
        return out

    def denormalize_slots(self, slots: torch.Tensor) -> torch.Tensor:
        out = slots.to(torch.float32).clone()
        out[..., :6] = out[..., :6] * self.slot_scales.to(slots.device)
        return out

    def normalize_acceleration(self, acceleration: torch.Tensor) -> torch.Tensor:
        return acceleration.to(torch.float32) / self.acceleration_scales.to(acceleration.device)

    def denormalize_acceleration(self, acceleration: torch.Tensor) -> torch.Tensor:
        return acceleration.to(torch.float32) * self.acceleration_scales.to(acceleration.device)


class RuleConditionedPongGNN(nn.Module):
    """Rule-conditioned object-centric neural game engine.

    The public name is kept for compatibility with the earlier Pong scripts,
    but the model is no longer Pong-specific internally. It consumes structured
    object slots, builds an interaction graph, sends rule/action-conditioned
    messages over explicit relative position/velocity/type edge features, and
    predicts residual object dynamics plus object liveness and event logits.
    """

    def __init__(
        self,
        num_rules: int = 3,
        action_dim: int = 3,
        num_object_types: int = len(OBJECT_TYPES),
        node_numeric_dim: int = 6,
        latent_dim: int = 64,
        rule_dim: int = 16,
        type_dim: int = 8,
        hidden_dim: int = 128,
        message_passing_steps: int = 2,
        mlp_layers: int = 2,
        history_steps: int = 6,
        constants: PongObjectConstants | None = None,
        edge_mode: str = "hybrid",
        edge_distance_threshold: float = 0.35,
        num_events: int = len(EVENTS),
    ):
        super().__init__()
        self.num_rules = int(num_rules)
        self.action_dim = int(action_dim)
        self.num_object_types = int(num_object_types)
        self.node_numeric_dim = int(node_numeric_dim)
        self.latent_dim = int(latent_dim)
        self.rule_dim = int(rule_dim)
        self.type_dim = int(type_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.history_steps = max(int(history_steps), 2)
        self.edge_mode = edge_mode
        self.edge_distance_threshold = float(edge_distance_threshold)
        self.num_events = int(num_events)
        self.boundary_feature_dim = 4
        self.history_feature_dim = 2 * (self.history_steps - 1)
        self.constants = constants or PongObjectConstants()
        self.max_objects = int(self.constants.max_objects)
        self.normalizer = PongStateNormalizer(self.constants)
        self.graph_builder = ObjectGraphBuilder(
            self.constants.slot_scales,
            num_object_types=self.num_object_types,
            edge_mode=edge_mode,
            distance_threshold=edge_distance_threshold,
        )

        self.rule_embedding = nn.Embedding(self.num_rules, self.rule_dim)
        self.type_embedding = nn.Embedding(self.num_object_types, self.type_dim)
        self.node_encoder = mlp(
            self.history_feature_dim + type_dim + self.boundary_feature_dim + action_dim + 1,
            hidden_dim,
            latent_dim,
            layers=mlp_layers,
        )
        edge_encoder_in_dim = self.graph_builder.edge_feature_dim + 2 * type_dim + action_dim + rule_dim
        self.edge_encoder = mlp(edge_encoder_in_dim, hidden_dim, latent_dim, layers=mlp_layers)
        self.processor_edge_mlps = nn.ModuleList(
            [
                mlp(3 * latent_dim + action_dim + rule_dim, hidden_dim, latent_dim, layers=mlp_layers)
                for _ in range(self.message_passing_steps)
            ]
        )
        self.processor_node_mlps = nn.ModuleList(
            [
                mlp(2 * latent_dim + action_dim + rule_dim, hidden_dim, latent_dim, layers=mlp_layers)
                for _ in range(self.message_passing_steps)
            ]
        )
        self.acceleration_decoder = mlp(latent_dim + rule_dim + type_dim, hidden_dim, 2, layers=mlp_layers)
        self.position_correction_decoder = mlp(latent_dim + rule_dim + type_dim, hidden_dim, 2, layers=mlp_layers)
        self.mask_decoder = mlp(latent_dim + rule_dim + type_dim, hidden_dim, 1, layers=mlp_layers)
        self.event_head = mlp(latent_dim + rule_dim + action_dim, hidden_dim, self.num_events, layers=mlp_layers)
        self.register_buffer("max_acceleration_norm", torch.tensor((1.5, 1.5), dtype=torch.float32), persistent=False)
        self.register_buffer("max_position_correction_norm", torch.tensor((1.2, 1.2), dtype=torch.float32), persistent=False)

    def state_to_slots(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.to(torch.float32)
        batch = state.shape[0]
        device = state.device
        c = self.constants
        slots = torch.zeros(batch, self.max_objects, SLOT_DIM, dtype=torch.float32, device=device)
        mask = torch.zeros(batch, self.max_objects, dtype=torch.float32, device=device)
        diameter = 2.0 * c.ball_radius
        slots[:, 0, 0] = state[:, 0]
        slots[:, 0, 1] = state[:, 1]
        slots[:, 0, 2] = state[:, 2]
        slots[:, 0, 3] = state[:, 3]
        slots[:, 0, 4] = diameter
        slots[:, 0, 5] = diameter
        slots[:, 0, 6] = OBJECT_TYPE_TO_ID["ball"]
        slots[:, 1, 0] = c.paddle_x
        slots[:, 1, 1] = state[:, 4]
        slots[:, 1, 3] = state[:, 5]
        slots[:, 1, 4] = c.paddle_width
        slots[:, 1, 5] = c.paddle_height
        slots[:, 1, 6] = OBJECT_TYPE_TO_ID["paddle"]
        mask[:, :2] = 1.0
        return slots, mask

    def slots_to_state(self, slots: torch.Tensor, game_id: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.zeros(slots.shape[0], 6, dtype=slots.dtype, device=slots.device)
        out[:, 0] = slots[:, 0, 0]
        out[:, 1] = slots[:, 0, 1]
        out[:, 2] = slots[:, 0, 2]
        out[:, 3] = slots[:, 0, 3]
        if game_id is None:
            out[:, 4] = slots[:, 1, 1]
            out[:, 5] = slots[:, 1, 3]
        else:
            game_id = game_id.to(slots.device)
            breakout = game_id == GAME_TO_ID["breakout"]
            out[:, 4] = torch.where(breakout, slots[:, 1, 0], slots[:, 1, 1])
            out[:, 5] = torch.where(breakout, slots[:, 1, 2], slots[:, 1, 3])
        return out

    def update_mask(self, slots: torch.Tensor, game_id: torch.Tensor | None = None) -> torch.Tensor:
        type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        mask = torch.zeros(*slots.shape[:2], 6, dtype=slots.dtype, device=slots.device)
        ball = type_ids == OBJECT_TYPE_TO_ID["ball"]
        paddle = type_ids == OBJECT_TYPE_TO_ID["paddle"]
        mask[..., 0:4] = ball[..., None].to(slots.dtype)
        if game_id is None:
            pong = torch.ones(slots.shape[0], dtype=torch.bool, device=slots.device)
        else:
            pong = game_id.to(slots.device) != GAME_TO_ID["breakout"]
        pong_paddle = paddle & pong[:, None]
        breakout_paddle = paddle & ~pong[:, None]
        mask[..., 1] = torch.where(pong_paddle, torch.ones_like(mask[..., 1]), mask[..., 1])
        mask[..., 3] = torch.where(pong_paddle, torch.ones_like(mask[..., 3]), mask[..., 3])
        mask[..., 0] = torch.where(breakout_paddle, torch.ones_like(mask[..., 0]), mask[..., 0])
        mask[..., 2] = torch.where(breakout_paddle, torch.ones_like(mask[..., 2]), mask[..., 2])
        return mask

    def apply_structural_constraints(
        self,
        pred_slots: torch.Tensor,
        slots: torch.Tensor,
        action: torch.Tensor,
        object_mask: torch.Tensor,
        game_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        ball = type_ids == OBJECT_TYPE_TO_ID["ball"]
        paddle = type_ids == OBJECT_TYPE_TO_ID["paddle"]
        block = type_ids == OBJECT_TYPE_TO_ID["block"]

        motion = torch.where(block[..., None], slots[..., 0:4], pred_slots[..., 0:4])

        max_component_speed = float(self.constants.max_ball_speed) * 1.25
        velocity = torch.where(
            ball[..., None],
            motion[..., 2:4].clamp(-max_component_speed, max_component_speed),
            motion[..., 2:4],
        )
        motion = torch.cat([motion[..., 0:2], velocity], dim=-1)

        if game_id is None:
            pong = torch.ones(slots.shape[0], dtype=torch.bool, device=slots.device)
        else:
            pong = game_id.to(slots.device) != GAME_TO_ID["breakout"]
        action_id = action.argmax(dim=-1) if action.ndim == 2 else action.to(torch.long)
        direction = torch.zeros_like(action_id, dtype=slots.dtype)
        direction = torch.where(action_id == 1, torch.full_like(direction, -1.0), direction)
        direction = torch.where(action_id == 2, torch.full_like(direction, 1.0), direction)
        step = direction[:, None] * float(self.constants.paddle_speed) * float(self.constants.dt)
        dt = max(float(self.constants.dt), 1e-6)

        old_x = slots[..., 0]
        old_y = slots[..., 1]
        width = slots[..., 4].clamp_min(1.0)
        height = slots[..., 5].clamp_min(1.0)

        pong_paddle = paddle & pong[:, None]
        new_y = (old_y + step).clamp(0.0, float(self.constants.height))
        new_y = torch.minimum(new_y, (float(self.constants.height) - height).clamp_min(0.0))
        pong_values = torch.stack(
            [
                torch.full_like(old_x, float(self.constants.paddle_x)),
                new_y,
                torch.zeros_like(old_x),
                (new_y - old_y) / dt,
            ],
            dim=-1,
        )

        breakout_paddle = paddle & ~pong[:, None]
        new_x = (old_x + step).clamp(0.0, float(self.constants.width))
        new_x = torch.minimum(new_x, (float(self.constants.width) - width).clamp_min(0.0))
        breakout_values = torch.stack(
            [
                new_x,
                old_y,
                (new_x - old_x) / dt,
                torch.zeros_like(old_x),
            ],
            dim=-1,
        )
        motion = torch.where(pong_paddle[..., None], pong_values, motion)
        motion = torch.where(breakout_paddle[..., None], breakout_values, motion)
        constrained = torch.cat([motion, slots[..., 4:6], slots[..., 6:7]], dim=-1)
        return constrained * object_mask[..., None].to(constrained.dtype)

    def action_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 2:
            return action.to(torch.float32)
        return F.one_hot(action.to(torch.int64), num_classes=self.action_dim).to(torch.float32)

    def action_to_nodes(self, action: torch.Tensor, slots: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        one_hot = self.action_one_hot(action)
        type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        paddle_mask = (type_ids == OBJECT_TYPE_TO_ID["paddle"]).to(torch.float32) * object_mask.to(torch.float32)
        return one_hot[:, None, :].expand(-1, slots.shape[1], -1) * paddle_mask[..., None]

    def boundary_node_features(self, slots: torch.Tensor) -> torch.Tensor:
        pos = self.normalizer.normalize_slots(slots)[..., 0:2]
        dist_to_lower = pos
        dist_to_upper = 1.0 - pos
        radius = max(float(self.edge_distance_threshold), 1e-6)
        return torch.clamp(torch.cat([dist_to_lower, dist_to_upper], dim=-1) / radius, -1.0, 1.0)

    def expand_history(self, slots: torch.Tensor, object_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            slots[:, None, ...].expand(-1, self.history_steps, -1, -1).contiguous(),
            object_mask[:, None, ...].expand(-1, self.history_steps, -1).contiguous(),
        )

    def history_features(
        self,
        slot_history: torch.Tensor,
        mask_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slot_history = slot_history.to(torch.float32)
        mask_history = mask_history.to(torch.float32)
        normalized_history = self.normalizer.normalize_slots(slot_history)
        position_history = normalized_history[..., 0:2]
        velocity_sequence = position_history[:, 1:] - position_history[:, :-1]
        velocity_features = velocity_sequence.permute(0, 2, 1, 3).reshape(
            slot_history.shape[0],
            slot_history.shape[2],
            self.history_feature_dim,
        )
        current_slots = slot_history[:, -1]
        current_mask = mask_history[:, -1]
        velocity_features = velocity_features * current_mask[..., None]
        return velocity_features, current_slots, current_mask

    def encode_slots(
        self,
        slots: torch.Tensor,
        object_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        slot_history: torch.Tensor | None = None,
        object_mask_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slots = slots.to(torch.float32)
        object_mask = object_mask.to(torch.float32)
        if slot_history is None or object_mask_history is None:
            slot_history, object_mask_history = self.expand_history(slots, object_mask)
        else:
            slot_history = slot_history.to(torch.float32)
            object_mask_history = object_mask_history.to(torch.float32)
        history_features, current_slots, current_mask = self.history_features(slot_history, object_mask_history)
        type_ids = current_slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        type_emb = self.type_embedding(type_ids)
        boundary_features = self.boundary_node_features(current_slots)
        if action is None:
            action_nodes = torch.zeros(slots.shape[0], slots.shape[1], self.action_dim, dtype=slots.dtype, device=slots.device)
        else:
            action_nodes = self.action_to_nodes(action, current_slots, current_mask)
        node_in = torch.cat([history_features, boundary_features, type_emb, action_nodes, current_mask[..., None]], dim=-1)
        return self.node_encoder(node_in) * current_mask[..., None]

    def encode_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        raw_state = self.normalizer.denormalize(normalized_state)
        slots, mask = self.state_to_slots(raw_state)
        return self.encode_slots(slots, mask)

    def transition_latents(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rule_id: torch.Tensor,
        slots: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, nodes, _ = z.shape
        rule = self.rule_embedding(rule_id.to(torch.int64))
        global_action = self.action_one_hot(action).to(z.device)
        if slots is None or object_mask is None:
            node_actions = torch.zeros(batch, nodes, self.action_dim, dtype=z.dtype, device=z.device)
            object_mask = torch.ones(batch, nodes, dtype=z.dtype, device=z.device)
            graph_mask = torch.ones(batch, nodes, nodes, dtype=z.dtype, device=z.device)
            eye = torch.eye(nodes, dtype=z.dtype, device=z.device)[None, :, :]
            graph_mask = graph_mask * (1.0 - eye)
            graph_features = torch.zeros(
                batch,
                nodes,
                nodes,
                self.graph_builder.edge_feature_dim,
                dtype=z.dtype,
                device=z.device,
            )
            type_ids = torch.zeros(batch, nodes, dtype=torch.long, device=z.device)
        else:
            node_actions = self.action_to_nodes(action, slots, object_mask).to(z.device)
            object_mask = object_mask.to(z.device).to(z.dtype)
            graph = self.graph_builder(slots.to(z.device), object_mask)
            graph_mask = graph.edge_mask.to(z.dtype)
            graph_features = graph.edge_features.to(z.dtype)
            type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long).to(z.device)
        type_emb = self.type_embedding(type_ids)
        src_type = type_emb[:, None, :, :].expand(batch, nodes, nodes, self.type_dim)
        dst_type = type_emb[:, :, None, :].expand(batch, nodes, nodes, self.type_dim)
        action_edges = global_action[:, None, None, :].expand(batch, nodes, nodes, self.action_dim)
        rule_edges = rule[:, None, None, :].expand(batch, nodes, nodes, self.rule_dim)
        encoded_edges = self.edge_encoder(torch.cat([graph_features, src_type, dst_type, action_edges, rule_edges], dim=-1))
        encoded_edges = encoded_edges * graph_mask[..., None]

        current = z
        current_edges = encoded_edges
        for edge_mlp, node_mlp in zip(self.processor_edge_mlps, self.processor_node_mlps):
            src = current[:, None, :, :].expand(batch, nodes, nodes, self.latent_dim)
            dst = current[:, :, None, :].expand(batch, nodes, nodes, self.latent_dim)
            edge_in = torch.cat([src, dst, current_edges, action_edges, rule_edges], dim=-1)
            edge_update = edge_mlp(edge_in) * graph_mask[..., None]
            current_edges = current_edges + edge_update
            message_tensor = current_edges.sum(dim=2)
            rule_nodes = rule[:, None, :].expand(batch, nodes, self.rule_dim)
            node_in = torch.cat([current, message_tensor, node_actions, rule_nodes], dim=-1)
            current = (current + node_mlp(node_in)) * object_mask[..., None]
        return current

    def decode_next_slots(
        self,
        z_next: torch.Tensor,
        slots: torch.Tensor,
        object_mask: torch.Tensor,
        rule_id: torch.Tensor,
        action: torch.Tensor,
        game_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rule = self.rule_embedding(rule_id.to(torch.int64))
        type_ids = slots[..., 6].round().clamp(0, self.num_object_types - 1).to(torch.long)
        type_emb = self.type_embedding(type_ids)
        ball = (type_ids == OBJECT_TYPE_TO_ID["ball"]).to(slots.dtype)[..., None]
        rule_nodes = rule[:, None, :].expand(slots.shape[0], slots.shape[1], self.rule_dim)
        decoder_in = torch.cat([z_next, rule_nodes, type_emb], dim=-1)
        acceleration_norm = torch.tanh(self.acceleration_decoder(decoder_in))
        acceleration_norm = acceleration_norm * self.max_acceleration_norm.to(acceleration_norm.device, acceleration_norm.dtype)
        position_correction_norm = torch.tanh(self.position_correction_decoder(decoder_in))
        position_correction_norm = position_correction_norm * self.max_position_correction_norm.to(position_correction_norm.device, position_correction_norm.dtype)
        mask_logits = self.mask_decoder(decoder_in).squeeze(-1)
        mask_logits[:, 0:2] = 20.0
        inactive = object_mask <= 0.0
        mask_logits = mask_logits.masked_fill(inactive, -20.0)
        acceleration = self.normalizer.denormalize_acceleration(acceleration_norm)
        dt = float(self.constants.dt)
        current_pos = slots[..., 0:2]
        current_vel = slots[..., 2:4]
        next_vel = torch.where(ball > 0.0, current_vel + acceleration * dt, current_vel)
        next_pos = torch.where(ball > 0.0, current_pos + next_vel * dt, current_pos)

        teleport_rule = (rule_id == RULE_TO_ID["teleport"]).to(slots.dtype)[:, None, None]
        position_correction = position_correction_norm * self.normalizer.slot_scales.to(slots.device, slots.dtype)[None, None, 0:2]
        next_pos = next_pos + position_correction * ball * teleport_rule

        pred_slots = slots.clone()
        pred_slots[..., 0:2] = next_pos
        pred_slots[..., 2:4] = next_vel
        pred_slots = self.apply_structural_constraints(pred_slots, slots, action, object_mask, game_id)
        return pred_slots, mask_logits, acceleration_norm

    def predict_events(self, z_next: torch.Tensor, action: torch.Tensor, rule_id: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        mask = object_mask.to(z_next.device).to(z_next.dtype)
        pooled = (z_next * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        rule = self.rule_embedding(rule_id.to(torch.int64))
        action_emb = self.action_one_hot(action).to(z_next.device)
        return self.event_head(torch.cat([pooled, rule, action_emb], dim=-1))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        rule_id: torch.Tensor,
        normalized: bool = False,
        object_slots: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
        slot_history: torch.Tensor | None = None,
        object_mask_history: torch.Tensor | None = None,
        game_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if object_slots is None or object_mask is None:
            raw_state = self.normalizer.denormalize(state) if normalized else state.to(torch.float32)
            object_slots, object_mask = self.state_to_slots(raw_state)
        else:
            object_slots = object_slots.to(torch.float32)
            object_mask = object_mask.to(torch.float32)
        if slot_history is None or object_mask_history is None:
            slot_history, object_mask_history = self.expand_history(object_slots, object_mask)
        else:
            slot_history = slot_history.to(torch.float32)
            object_mask_history = object_mask_history.to(torch.float32)
            object_slots = slot_history[:, -1]
            object_mask = object_mask_history[:, -1]

        z = self.encode_slots(object_slots, object_mask, action, slot_history, object_mask_history)
        z_next = self.transition_latents(z, action, rule_id, object_slots, object_mask)
        pred_next_slots, pred_next_mask_logits, pred_normalized_acceleration = self.decode_next_slots(
            z_next,
            object_slots,
            object_mask,
            rule_id,
            action,
            game_id,
        )
        pred_event_logits = self.predict_events(z_next, action, rule_id, object_mask)
        pred_next_normalized_slots = self.normalizer.normalize_slots(pred_next_slots)
        pred_next = self.slots_to_state(pred_next_slots, game_id)
        return {
            "pred_next_normalized_slots": pred_next_normalized_slots,
            "pred_next_slots": pred_next_slots,
            "pred_normalized_acceleration": pred_normalized_acceleration,
            "pred_acceleration": self.normalizer.denormalize_acceleration(pred_normalized_acceleration),
            "pred_next_mask_logits": pred_next_mask_logits,
            "pred_next_mask_prob": torch.sigmoid(pred_next_mask_logits),
            "pred_event_logits": pred_event_logits,
            "pred_event_prob": torch.softmax(pred_event_logits, dim=-1),
            "pred_next_normalized": self.normalizer.normalize(pred_next),
            "pred_next": pred_next,
            "z": z,
            "z_next_pred": z_next,
            "object_slots": object_slots,
            "object_mask": object_mask,
            "slot_history": slot_history,
            "object_mask_history": object_mask_history,
        }

    def contrastive_loss(
        self,
        z_next_pred: torch.Tensor,
        next_slots: torch.Tensor,
        next_mask: torch.Tensor,
        negative_slots: torch.Tensor,
        negative_mask: torch.Tensor,
        margin: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_true = self.encode_slots(next_slots, next_mask).detach()
        z_neg = self.encode_slots(negative_slots, negative_mask)
        mask = next_mask.to(z_next_pred.device).to(z_next_pred.dtype)
        node_count = mask.sum(dim=1).clamp_min(1.0)
        positive = 0.5 * ((z_next_pred - z_true).pow(2).mean(dim=-1) * mask).sum(dim=1) / node_count
        negative = 0.5 * ((z_neg - z_true).pow(2).mean(dim=-1) * mask).sum(dim=1) / node_count
        loss = positive.mean() + torch.relu(torch.as_tensor(margin, device=positive.device) - negative).mean()
        return loss, {
            "contrastive_positive": positive.mean().detach(),
            "contrastive_negative": negative.mean().detach(),
        }
