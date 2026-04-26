#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pong_common import EVENTS, GAME_TO_ID, ID_TO_EVENT, ID_TO_GAME, ID_TO_RULE, ID_TO_SOURCE, OBJECT_TYPE_TO_ID, RULE_TO_ID, load_shards
from models.object_losses import (
    LossWeights,
    compute_object_centric_losses,
    event_prediction_loss,
    kinematic_consistency_loss,
    masked_acceleration_loss,
    masked_delta_loss,
    relative_slot_loss,
    rollout_slot_loss,
)
from models.rule_conditioned_gnn import PongObjectConstants, RuleConditionedPongGNN


class PongTransitionDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        split: str,
        combos: set[tuple[int, int]] | None = None,
        exclude_combos: set[tuple[int, int]] | None = None,
        rule_ablation: str = "none",
        seed: int = 0,
        rollout_horizon: int = 1,
        history_length: int = 6,
        include_counterfactuals: bool = False,
        rule_id_map: dict[int, int] | None = None,
    ):
        data = load_shards(root, split)
        game_id = data.get("game_id", np.zeros_like(data["rule_id"], dtype=np.int64))
        rule_id = data["rule_id"].copy()
        keep = np.ones(int(data["action"].shape[0]), dtype=bool)
        if combos:
            keep &= np.asarray([(int(g), int(r)) in combos for g, r in zip(game_id, rule_id)], dtype=bool)
        if exclude_combos:
            keep &= np.asarray([(int(g), int(r)) not in exclude_combos for g, r in zip(game_id, rule_id)], dtype=bool)
        if not np.any(keep):
            raise ValueError(f"No rows left after combo filtering for split={split}")
        if rule_ablation == "zero":
            effective_rule_id = np.zeros_like(rule_id, dtype=np.int64)
        elif rule_ablation == "shuffle":
            rng = np.random.default_rng(int(seed) + (0 if split == "train" else 1))
            effective_rule_id = rng.permutation(rule_id)
        elif rule_ablation != "none":
            raise ValueError(f"Unknown rule_ablation={rule_ablation!r}")
        else:
            effective_rule_id = rule_id

        if rule_id_map is None:
            unique_rules = sorted(int(rule) for rule in np.unique(effective_rule_id[keep]).tolist())
            self.rule_id_map = {rule: idx for idx, rule in enumerate(unique_rules)}
        else:
            self.rule_id_map = {int(key): int(value) for key, value in rule_id_map.items()}
        missing_rules = sorted(int(rule) for rule in np.unique(effective_rule_id[keep]).tolist() if int(rule) not in self.rule_id_map)
        if missing_rules:
            raise ValueError(f"Rule map missing ids {missing_rules} for split={split}")
        mapped_rule_id = np.asarray([self.rule_id_map[int(rule)] for rule in effective_rule_id[keep]], dtype=np.int64)
        self.num_effective_rules = len(self.rule_id_map)

        self.state = torch.as_tensor(data["state"][keep], dtype=torch.float32)
        self.action = torch.as_tensor(data["action"][keep], dtype=torch.long)
        self.next_state = torch.as_tensor(data["next_state"][keep], dtype=torch.float32)
        self.rule_id = torch.as_tensor(mapped_rule_id, dtype=torch.long)
        self.true_rule_id = torch.as_tensor(data["rule_id"][keep], dtype=torch.long)
        self.game_id = torch.as_tensor(game_id[keep], dtype=torch.long)
        self.event_id = torch.as_tensor(data["event_id"][keep], dtype=torch.long)
        self.source_id = torch.as_tensor(data["source_id"][keep], dtype=torch.long)
        self.episode_id = torch.as_tensor(data["episode_id"][keep], dtype=torch.long)
        self.step = torch.as_tensor(data["step"][keep], dtype=torch.long)
        self.terminated = torch.as_tensor(data.get("terminated", np.zeros_like(data["action"], dtype=np.bool_))[keep], dtype=torch.bool)
        self.truncated = torch.as_tensor(data.get("truncated", np.zeros_like(data["action"], dtype=np.bool_))[keep], dtype=torch.bool)
        self.object_slots = torch.as_tensor(data["object_slots"][keep], dtype=torch.float32)
        self.next_object_slots = torch.as_tensor(data["next_object_slots"][keep], dtype=torch.float32)
        self.object_mask = torch.as_tensor(data["object_mask"][keep], dtype=torch.float32)
        self.next_object_mask = torch.as_tensor(data["next_object_mask"][keep], dtype=torch.float32)
        self.history_length = max(2, int(history_length))
        self.history_indices, self.history_valid = self._build_history_index()
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.rollout_indices, self.rollout_valid = self._build_rollout_index()
        self.include_counterfactuals = bool(include_counterfactuals)
        if self.include_counterfactuals:
            self.counterfactual_indices, self.counterfactual_valid = self._build_counterfactual_index()
            self.counterfactual_rule_id = torch.as_tensor(
                [mapped for _, mapped in sorted(self.rule_id_map.items(), key=lambda item: item[1])],
                dtype=torch.long,
            )

    def _build_history_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(self.action.shape[0])
        length = int(self.history_length)
        indices = torch.zeros(rows, length, dtype=torch.long)
        valid = torch.zeros(rows, length, dtype=torch.float32)
        lookup: dict[tuple[int, int, int, int], int] = {}
        game_np = self.game_id.numpy()
        rule_np = self.true_rule_id.numpy()
        episode_np = self.episode_id.numpy()
        step_np = self.step.numpy()
        for idx in range(rows):
            key = (int(game_np[idx]), int(rule_np[idx]), int(episode_np[idx]), int(step_np[idx]))
            lookup.setdefault(key, idx)
        for idx in range(rows):
            anchor = int(step_np[idx])
            earliest_idx = idx
            for offset in range(length):
                history_step = anchor - (length - 1 - offset)
                key = (
                    int(game_np[idx]),
                    int(rule_np[idx]),
                    int(episode_np[idx]),
                    int(history_step),
                )
                history_idx = lookup.get(key)
                if history_idx is None:
                    indices[idx, offset] = earliest_idx
                    continue
                indices[idx, offset] = int(history_idx)
                valid[idx, offset] = 1.0
                earliest_idx = int(history_idx)
        return indices, valid

    def _build_rollout_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(self.action.shape[0])
        horizon = int(self.rollout_horizon)
        indices = torch.zeros(rows, horizon, dtype=torch.long)
        valid = torch.zeros(rows, horizon, dtype=torch.float32)
        if horizon <= 1:
            indices[:, 0] = torch.arange(rows, dtype=torch.long)
            valid[:, 0] = 1.0
            return indices, valid

        lookup: dict[tuple[int, int, int, int], int] = {}
        game_np = self.game_id.numpy()
        rule_np = self.true_rule_id.numpy()
        episode_np = self.episode_id.numpy()
        step_np = self.step.numpy()
        for idx in range(rows):
            key = (int(game_np[idx]), int(rule_np[idx]), int(episode_np[idx]), int(step_np[idx]))
            lookup.setdefault(key, idx)

        for idx in range(rows):
            indices[idx, 0] = idx
            valid[idx, 0] = 1.0
            terminal_seen = bool(self.terminated[idx] or self.truncated[idx])
            for offset in range(1, horizon):
                key = (
                    int(game_np[idx]),
                    int(rule_np[idx]),
                    int(episode_np[idx]),
                    int(step_np[idx]) + offset,
                )
                next_idx = lookup.get(key)
                if terminal_seen or next_idx is None:
                    indices[idx, offset] = idx
                    continue
                indices[idx, offset] = int(next_idx)
                valid[idx, offset] = 1.0
                terminal_seen = bool(self.terminated[next_idx] or self.truncated[next_idx])
        return indices, valid

    def _build_counterfactual_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(self.action.shape[0])
        ordered_rules = [rule for rule, _ in sorted(self.rule_id_map.items(), key=lambda item: item[1])]
        num_rules = len(ordered_rules)
        indices = torch.zeros(rows, num_rules, dtype=torch.long)
        valid = torch.zeros(rows, num_rules, dtype=torch.float32)
        lookup: dict[tuple[int, int, int, int], int] = {}
        game_np = self.game_id.numpy()
        rule_np = self.true_rule_id.numpy()
        episode_np = self.episode_id.numpy()
        step_np = self.step.numpy()
        for idx in range(rows):
            key = (int(game_np[idx]), int(rule_np[idx]), int(episode_np[idx]), int(step_np[idx]))
            lookup.setdefault(key, idx)
        for idx in range(rows):
            for mapped_rule, original_rule in enumerate(ordered_rules):
                next_idx = lookup.get((int(game_np[idx]), int(original_rule), int(episode_np[idx]), int(step_np[idx])))
                if next_idx is not None:
                    indices[idx, mapped_rule] = int(next_idx)
                    valid[idx, mapped_rule] = 1.0
        return indices, valid

    def sampling_weights(self, rare_weight: float = 4.0) -> torch.Tensor:
        weights = torch.ones(len(self), dtype=torch.float64)
        rare = (self.event_id != EVENTS.index("step")) & (self.event_id != EVENTS.index("none"))
        weights[rare] = float(rare_weight)
        for event_name in ("paddle_hit", "wrapped", "left_wall_bounce", "top_bounce", "bottom_bounce", "miss"):
            weights[self.event_id == EVENTS.index(event_name)] = max(float(rare_weight), 6.0)
        weights[self.source_id != 0] = torch.maximum(weights[self.source_id != 0], torch.full_like(weights[self.source_id != 0], float(rare_weight)))
        return weights

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "state": self.state[idx],
            "action": self.action[idx],
            "next_state": self.next_state[idx],
            "rule_id": self.rule_id[idx],
            "true_rule_id": self.true_rule_id[idx],
            "game_id": self.game_id[idx],
            "event_id": self.event_id[idx],
            "source_id": self.source_id[idx],
            "object_slots": self.object_slots[idx],
            "next_object_slots": self.next_object_slots[idx],
            "object_mask": self.object_mask[idx],
            "next_object_mask": self.next_object_mask[idx],
            "history_object_slots": self.object_slots[self.history_indices[idx]],
            "history_object_mask": self.object_mask[self.history_indices[idx]],
            "history_valid": self.history_valid[idx],
        }
        if self.rollout_horizon > 1:
            rollout_idx = self.rollout_indices[idx]
            item.update(
                {
                    "rollout_action": self.action[rollout_idx],
                    "rollout_rule_id": self.rule_id[rollout_idx],
                    "rollout_next_object_slots": self.next_object_slots[rollout_idx],
                    "rollout_next_object_mask": self.next_object_mask[rollout_idx],
                    "rollout_valid": self.rollout_valid[idx],
                }
            )
        if self.include_counterfactuals:
            counterfactual_idx = self.counterfactual_indices[idx]
            item.update(
                {
                    "counterfactual_rule_id": self.counterfactual_rule_id,
                    "counterfactual_next_object_slots": self.next_object_slots[counterfactual_idx],
                    "counterfactual_next_object_mask": self.next_object_mask[counterfactual_idx],
                    "counterfactual_valid": self.counterfactual_valid[idx],
                }
            )
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rule-conditioned GNN dynamics model for custom Pong.")
    parser.add_argument("--dataset", required=True, help="Dataset root containing train/val shards.")
    parser.add_argument("--output", default="runs/pong_world_model", help="Output directory.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to continue training from, usually runs/.../latest.pt.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-size",
        choices=("custom", "small", "medium", "large", "xl"),
        default="custom",
        help="Named capacity preset. Explicit dimension flags override this.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--rule-dim", type=int, default=None)
    parser.add_argument("--type-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--message-passing-steps", type=int, default=None)
    parser.add_argument("--edge-mode", choices=("fully_connected", "distance", "hybrid"), default="hybrid")
    parser.add_argument("--edge-distance-threshold", type=float, default=0.35)
    parser.add_argument("--alpha-rel", type=float, default=0.1, help="Weight for relative object-state loss after --rel-start-epoch.")
    parser.add_argument("--beta-roll", type=float, default=0.1, help="Weight for rollout loss after --rollout-start-epoch.")
    parser.add_argument("--gamma-event", type=float, default=0.05, help="Weight for event classification after --event-start-epoch.")
    parser.add_argument("--delta-weight", type=float, default=0.5, help="Weight for residual delta loss that anchors predicted motion increments.")
    parser.add_argument("--kinematic-weight", type=float, default=0.2, help="Weight for free-flight x/y consistency with predicted vx/vy.")
    parser.add_argument(
        "--counterfactual-rule-weight",
        type=float,
        default=0.0,
        help="Weight for same-state, different-rule transition supervision when counterfactual rows are available.",
    )
    parser.add_argument("--rel-start-epoch", type=int, default=2, help="Epoch at which relative-state loss turns on.")
    parser.add_argument("--rollout-start-epoch", type=int, default=10, help="Epoch at which multi-step rollout loss turns on.")
    parser.add_argument("--event-start-epoch", type=int, default=20, help="Epoch at which event prediction loss turns on.")
    parser.add_argument("--rollout-horizon", type=int, default=3, help="Number of contiguous transitions used for rollout loss/eval.")
    parser.add_argument("--history-length", type=int, default=6, help="Number of past frames used to build DeepMind-style velocity-history node features.")
    parser.add_argument("--noise-std", type=float, default=6.7e-4, help="Normalized random-walk noise scale on history positions during training.")
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=0.1, help="Weight for active-object mask prediction loss.")
    parser.add_argument("--event-weight", type=float, default=2.0, help="Extra weight for rare event transitions.")
    parser.add_argument("--event-balanced-sampling", action=argparse.BooleanOptionalAction, default=False, help="Oversample rare/contact transitions in training.")
    parser.add_argument("--rare-sample-weight", type=float, default=6.0, help="Sampling multiplier for rare/contact transitions when balanced sampling is enabled.")
    parser.add_argument("--train-combos", nargs="*", default=None, help="Optional combos to train on, e.g. pong:normal pong:gravity.")
    parser.add_argument("--holdout-combos", nargs="*", default=None, help="Optional combos to exclude from train and report separately, e.g. pong:teleport.")
    parser.add_argument("--single-rule-mode", choices=tuple(RULE_TO_ID), default=None, help="Shortcut for training and evaluation on only one Pong rule, e.g. normal.")
    parser.add_argument("--rule-ablation", choices=("none", "zero", "shuffle"), default="none", help="Ablation for testing whether rule IDs are actually used.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


MODEL_PRESETS = {
    "custom": {"latent_dim": 64, "rule_dim": 16, "type_dim": 8, "hidden_dim": 128, "message_passing_steps": 2},
    "small": {"latent_dim": 64, "rule_dim": 16, "type_dim": 8, "hidden_dim": 128, "message_passing_steps": 2},
    "medium": {"latent_dim": 128, "rule_dim": 32, "type_dim": 16, "hidden_dim": 256, "message_passing_steps": 3},
    "large": {"latent_dim": 256, "rule_dim": 64, "type_dim": 32, "hidden_dim": 512, "message_passing_steps": 4},
    "xl": {"latent_dim": 384, "rule_dim": 96, "type_dim": 48, "hidden_dim": 768, "message_passing_steps": 5},
}


def apply_model_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODEL_PRESETS[args.model_size]
    for arg_name, default_value in preset.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, default_value)
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def constants_from_metadata(dataset_root: pathlib.Path) -> PongObjectConstants:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        return PongObjectConstants()
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata.get("env_config", {})
    return PongObjectConstants(
        width=float(cfg.get("width", 640.0)),
        height=float(cfg.get("height", 480.0)),
        dt=float(cfg.get("dt", 1.0 / 60.0)),
        paddle_width=float(cfg.get("paddle_width", 12.0)),
        paddle_height=float(cfg.get("paddle_height", 88.0)),
        paddle_margin=float(cfg.get("paddle_margin", 24.0)),
        paddle_speed=float(cfg.get("paddle_speed", 360.0)),
        ball_radius=float(cfg.get("ball_radius", 8.0)),
        max_ball_speed=float(cfg.get("max_ball_speed", 720.0)),
    )


def parse_combos(values: list[str] | None) -> set[tuple[int, int]] | None:
    if not values:
        return None
    combos: set[tuple[int, int]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Combo must be game:rule, got {value!r}")
        game_name, rule_name = value.split(":", 1)
        if game_name not in GAME_TO_ID:
            raise ValueError(f"Unknown game {game_name!r}; expected one of {sorted(GAME_TO_ID)}")
        if rule_name not in RULE_TO_ID:
            raise ValueError(f"Unknown rule {rule_name!r}; expected one of {sorted(RULE_TO_ID)}")
        combos.add((GAME_TO_ID[game_name], RULE_TO_ID[rule_name]))
    return combos


def apply_single_rule_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not args.single_rule_mode:
        return args
    single_combo = f"pong:{args.single_rule_mode}"
    args.train_combos = [single_combo]
    args.holdout_combos = None
    args.edge_mode = "distance"
    args.alpha_rel = 0.0
    args.beta_roll = 0.0
    args.gamma_event = 0.0
    args.delta_weight = 1.0
    args.kinematic_weight = 0.0
    args.counterfactual_rule_weight = 0.0
    args.mask_loss_weight = 0.0
    args.contrastive_weight = 0.0
    return args


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def transition_weights(event_id: torch.Tensor, event_weight: float) -> torch.Tensor:
    weights = torch.ones_like(event_id, dtype=torch.float32)
    if event_weight <= 1.0:
        return weights
    rare = event_id != EVENTS.index("step")
    rare &= event_id != EVENTS.index("none")
    weights[rare] = float(event_weight)
    return weights


def loss_weights_for_epoch(args: argparse.Namespace, epoch: int) -> LossWeights:
    return LossWeights(
        alpha_rel=float(args.alpha_rel) if epoch >= int(args.rel_start_epoch) else 0.0,
        beta_roll=float(args.beta_roll) if epoch >= int(args.rollout_start_epoch) else 0.0,
        gamma_event=float(args.gamma_event) if epoch >= int(args.event_start_epoch) else 0.0,
        delta=float(args.delta_weight),
        kinematic=float(args.kinematic_weight),
        counterfactual_rule=float(args.counterfactual_rule_weight),
        mask=float(args.mask_loss_weight),
        contrastive=float(args.contrastive_weight),
        event_weight=float(args.event_weight),
        contrastive_margin=float(args.contrastive_margin),
    )


def add_history_noise(
    slot_history: torch.Tensor,
    mask_history: torch.Tensor,
    model: RuleConditionedPongGNN,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise_std <= 0.0:
        return slot_history, slot_history[:, -1]

    noisy = slot_history.clone()
    type_ids = slot_history[:, -1, :, 6].round().to(torch.long)
    dynamic_mask = (type_ids == OBJECT_TYPE_TO_ID["ball"]).to(slot_history.dtype)[:, None, :, None] * mask_history[..., None].to(slot_history.dtype)
    steps = int(slot_history.shape[1])
    if steps <= 1 or dynamic_mask.sum().item() <= 0.0:
        return noisy, noisy[:, -1]

    pos_scales = model.normalizer.slot_scales.to(slot_history.device, slot_history.dtype)[0:2]
    velocity_noise = torch.randn(
        slot_history.shape[0],
        steps - 1,
        slot_history.shape[2],
        2,
        device=slot_history.device,
        dtype=slot_history.dtype,
    ) * (pos_scales * float(noise_std) / max((steps - 1) ** 0.5, 1.0))
    velocity_noise = torch.cumsum(velocity_noise, dim=1)
    position_noise = torch.cat([torch.zeros_like(velocity_noise[:, :1]), torch.cumsum(velocity_noise, dim=1)], dim=1)
    noisy[..., 0:2] = noisy[..., 0:2] + position_noise * dynamic_mask
    dt = max(float(model.constants.dt), 1e-6)
    derived_velocity = (noisy[:, 1:, :, 0:2] - noisy[:, :-1, :, 0:2]) / dt
    noisy[:, 1:, :, 2:4] = torch.where(dynamic_mask[:, 1:] > 0.0, derived_velocity, noisy[:, 1:, :, 2:4])
    noisy[:, 0, :, 2:4] = torch.where(dynamic_mask[:, 0] > 0.0, noisy[:, 1, :, 2:4], noisy[:, 0, :, 2:4])
    return noisy, noisy[:, -1]


def masked_slot_loss(
    pred_next_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    target_mask: torch.Tensor,
    model: RuleConditionedPongGNN,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_norm = model.normalizer.normalize_slots(pred_next_slots)[..., :6]
    target_norm = model.normalizer.normalize_slots(target_next_slots)[..., :6]
    per_slot = F.smooth_l1_loss(pred_norm, target_norm, reduction="none").mean(dim=-1)
    mask = target_mask.to(per_slot.device).to(per_slot.dtype)
    per_row = (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def object_mask_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target_mask, dtype=torch.float32)
    weights[:, 2:] = torch.where(input_mask[:, 2:] > 0.0, 2.0, 0.25)
    return F.binary_cross_entropy_with_logits(pred_logits, target_mask.to(pred_logits.dtype), weight=weights, reduction="mean")


def train_epoch(
    model: RuleConditionedPongGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: LossWeights,
    noise_std: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "state_loss": 0.0,
        "delta_loss": 0.0,
        "rel_loss": 0.0,
        "rollout_loss": 0.0,
        "event_loss": 0.0,
        "kinematic_loss": 0.0,
        "counterfactual_rule_loss": 0.0,
        "contrastive_loss": 0.0,
        "mask_loss": 0.0,
    }
    rows = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        model_history, model_slots = add_history_noise(batch["history_object_slots"], batch["history_object_mask"], model, noise_std)
        out = model(
            batch["state"],
            batch["action"],
            batch["rule_id"],
            normalized=False,
            object_slots=model_slots,
            object_mask=batch["history_object_mask"][:, -1],
            slot_history=model_history,
            object_mask_history=batch["history_object_mask"],
            game_id=batch["game_id"],
        )
        if noise_std > 0.0:
            batch = {**batch, "object_slots": model_slots, "object_mask": batch["history_object_mask"][:, -1], "history_object_slots": model_history}
        losses = compute_object_centric_losses(model, batch, out, loss_weights)
        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        batch_rows = int(batch["action"].shape[0])
        rows += batch_rows
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * batch_rows
    return {key: value / max(rows, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: RuleConditionedPongGNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sqerr = []
    flat_sqerr = []
    var_sqerr = []
    mask_losses = []
    rel_losses = []
    delta_losses = []
    kinematic_losses = []
    rollout_losses = []
    event_losses = []
    event_accuracies = []
    mask_accuracies = []
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}

    def add_group(name: str, values: np.ndarray) -> None:
        group_sums[name] = group_sums.get(name, 0.0) + float(values.sum())
        group_counts[name] = group_counts.get(name, 0) + int(values.size)

    def add_id_groups(prefix: str, values: np.ndarray, ids: np.ndarray, names: dict[int, str]) -> None:
        for group_id in np.unique(ids):
            selected = values[ids == group_id]
            add_group(f"{prefix}_{names.get(int(group_id), str(int(group_id)))}", selected)

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(
            batch["state"],
            batch["action"],
            batch["rule_id"],
            normalized=False,
            object_slots=batch["object_slots"],
            object_mask=batch["object_mask"],
            slot_history=batch["history_object_slots"],
            object_mask_history=batch["history_object_mask"],
            game_id=batch["game_id"],
        )
        slot_err = out["pred_next_slots"][..., :6] - batch["next_object_slots"][..., :6]
        mask = batch["next_object_mask"].to(slot_err.dtype)
        slot_sq = (slot_err.pow(2).mean(dim=-1) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        flat_sq = (out["pred_next"] - batch["next_state"]).pow(2)
        mask_loss = object_mask_loss(out["pred_next_mask_logits"], batch["next_object_mask"], batch["object_mask"])
        delta_loss = (
            masked_acceleration_loss(
                out["pred_normalized_acceleration"],
                batch["object_slots"],
                batch["next_object_slots"],
                batch["next_object_mask"],
                model,
            )
            if "pred_normalized_acceleration" in out
            else masked_delta_loss(out["pred_next_slots"], batch["object_slots"], batch["next_object_slots"], batch["next_object_mask"], model)
        )
        kinematic_loss = kinematic_consistency_loss(
            out["pred_next_slots"],
            batch["object_slots"],
            batch["next_object_mask"],
            batch["event_id"],
            model,
        )
        rel_loss = relative_slot_loss(out["pred_next_slots"], batch["next_object_slots"], batch["next_object_mask"], model)
        roll_loss = rollout_slot_loss(model, batch)
        event_loss = event_prediction_loss(out["pred_event_logits"], batch["event_id"])
        event_acc = (out["pred_event_logits"].argmax(dim=-1) == batch["event_id"]).to(torch.float32).mean()
        mask_pred = out["pred_next_mask_prob"] >= 0.5
        mask_true = batch["next_object_mask"] >= 0.5
        mask_acc = (mask_pred == mask_true).to(torch.float32).mean()
        sqerr.append(slot_sq.detach().cpu())
        flat_sqerr.append(flat_sq.mean(dim=-1).detach().cpu())
        var_sqerr.append(flat_sq.detach().cpu())
        mask_losses.append(mask_loss.detach().cpu())
        delta_losses.append(delta_loss.detach().cpu())
        kinematic_losses.append(kinematic_loss.detach().cpu())
        rel_losses.append(rel_loss.detach().cpu())
        rollout_losses.append(roll_loss.detach().cpu())
        event_losses.append(event_loss.detach().cpu())
        event_accuracies.append(event_acc.detach().cpu())
        mask_accuracies.append(mask_acc.detach().cpu())
        slot_values = slot_sq.detach().cpu().numpy()
        games = batch["game_id"].detach().cpu().numpy()
        rules = batch["true_rule_id"].detach().cpu().numpy()
        events = batch["event_id"].detach().cpu().numpy()
        sources = batch["source_id"].detach().cpu().numpy()
        add_id_groups("val_slot_mse/game", slot_values, games, ID_TO_GAME)
        add_id_groups("val_slot_mse/rule", slot_values, rules, ID_TO_RULE)
        add_id_groups("val_slot_mse/event", slot_values, events, ID_TO_EVENT)
        add_id_groups("val_slot_mse/source", slot_values, sources, ID_TO_SOURCE)
        for game in np.unique(games):
            for rule in np.unique(rules[games == game]):
                selected = (games == game) & (rules == rule)
                add_group(f"val_slot_mse/combo_{ID_TO_GAME.get(int(game), str(int(game)))}:{ID_TO_RULE.get(int(rule), str(int(rule)))}", slot_values[selected])
    all_sq = torch.cat(sqerr)
    all_flat_sq = torch.cat(flat_sqerr)
    var_sq = torch.cat(var_sqerr, dim=0)
    metrics: dict[str, float] = {
        "val_slot_mse": float(all_sq.mean().item()),
        "val_slot_rmse": float(torch.sqrt(all_sq.mean()).item()),
        "val_mse": float(all_flat_sq.mean().item()),
        "val_rmse": float(torch.sqrt(all_flat_sq.mean()).item()),
        "val_mask_loss": float(torch.stack(mask_losses).mean().item()),
        "val_mask_accuracy": float(torch.stack(mask_accuracies).mean().item()),
        "val_delta_loss": float(torch.stack(delta_losses).mean().item()),
        "val_kinematic_loss": float(torch.stack(kinematic_losses).mean().item()),
        "val_rel_loss": float(torch.stack(rel_losses).mean().item()),
        "val_rollout_loss": float(torch.stack(rollout_losses).mean().item()),
        "val_event_loss": float(torch.stack(event_losses).mean().item()),
        "val_event_accuracy": float(torch.stack(event_accuracies).mean().item()),
    }
    names = ("ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_pos", "paddle_vel")
    for idx, name in enumerate(names):
        metrics[f"val_mse/{name}"] = float(var_sq[:, idx].mean().item())
    for name, total in group_sums.items():
        metrics[name] = total / max(group_counts[name], 1)
    return metrics


def save_checkpoint(path: pathlib.Path, model: RuleConditionedPongGNN, optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int, metrics: dict[str, float]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "constants": model.constants.__dict__,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_training_checkpoint(
    path: pathlib.Path,
    model: RuleConditionedPongGNN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})
    best = float(metrics.get("val_mse", float("inf")))
    return epoch, best


def main() -> int:
    args = apply_single_rule_mode(apply_model_preset(parse_args()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_root = pathlib.Path(args.dataset).expanduser().resolve()
    output = pathlib.Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    device = choose_device(args.device)

    train_combos = parse_combos(args.train_combos)
    holdout_combos = parse_combos(args.holdout_combos)
    train_data = PongTransitionDataset(
        dataset_root,
        "train",
        combos=train_combos,
        exclude_combos=holdout_combos,
        rule_ablation=args.rule_ablation,
        seed=args.seed,
        rollout_horizon=args.rollout_horizon,
        history_length=args.history_length,
        include_counterfactuals=float(args.counterfactual_rule_weight) > 0.0,
    )
    args.num_rules = int(train_data.num_effective_rules)
    args.rule_id_map = dict(train_data.rule_id_map)
    val_data = PongTransitionDataset(
        dataset_root,
        "val",
        combos=train_combos,
        rule_ablation=args.rule_ablation,
        seed=args.seed,
        rollout_horizon=args.rollout_horizon,
        history_length=args.history_length,
        rule_id_map=train_data.rule_id_map,
    )
    holdout_data = (
        PongTransitionDataset(
            dataset_root,
            "val",
            combos=holdout_combos,
            rule_ablation=args.rule_ablation,
            seed=args.seed,
            rollout_horizon=args.rollout_horizon,
            history_length=args.history_length,
            rule_id_map=train_data.rule_id_map,
        )
        if holdout_combos
        else None
    )
    train_sampler = None
    shuffle_train = True
    if args.event_balanced_sampling:
        train_sampler = WeightedRandomSampler(
            train_data.sampling_weights(args.rare_sample_weight),
            num_samples=len(train_data),
            replacement=True,
        )
        shuffle_train = False
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    holdout_loader = (
        DataLoader(holdout_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        if holdout_data is not None
        else None
    )
    constants = constants_from_metadata(dataset_root)
    model = RuleConditionedPongGNN(
        num_rules=args.num_rules,
        latent_dim=args.latent_dim,
        rule_dim=args.rule_dim,
        type_dim=args.type_dim,
        hidden_dim=args.hidden_dim,
        message_passing_steps=args.message_passing_steps,
        history_steps=args.history_length,
        constants=constants,
        edge_mode=args.edge_mode,
        edge_distance_threshold=args.edge_distance_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    start_epoch = 1
    if args.resume:
        resume_path = pathlib.Path(args.resume).expanduser().resolve()
        loaded_epoch, loaded_best = load_training_checkpoint(resume_path, model, optimizer, device)
        best = loaded_best
        start_epoch = loaded_epoch + 1

    print(f"dataset={dataset_root}")
    print(f"output={output}")
    print(f"device={device}")
    print(
        f"model_size={args.model_size} latent={args.latent_dim} hidden={args.hidden_dim} "
        f"rule={args.rule_dim} type={args.type_dim} mp_steps={args.message_passing_steps} "
        f"edge={args.edge_mode}@{args.edge_distance_threshold} num_rules={args.num_rules}"
    )
    loss_schedule = (
        f"loss schedule: rel@{args.rel_start_epoch} roll@{args.rollout_start_epoch} "
        f"event@{args.event_start_epoch} delta={args.delta_weight} kinematic={args.kinematic_weight} "
    )
    if float(args.counterfactual_rule_weight) > 0.0:
        loss_schedule += f"counterfactual_rule={args.counterfactual_rule_weight} "
    else:
        loss_schedule += "counterfactual_rule=disabled "
    loss_schedule += f"rollout_horizon={args.rollout_horizon} noise_std={args.noise_std}"
    print(loss_schedule)
    print(f"event_balanced_sampling={args.event_balanced_sampling} rare_sample_weight={args.rare_sample_weight}")
    print(f"train rows={len(train_data)} val rows={len(val_data)}")
    if args.single_rule_mode:
        print(f"single_rule_mode={args.single_rule_mode} train_combos={args.train_combos}")
    if train_data.include_counterfactuals:
        valid_cf = int(train_data.counterfactual_valid.sum().item())
        total_cf = int(train_data.counterfactual_valid.numel())
        print(f"counterfactual rule targets={valid_cf}/{total_cf}")
    if holdout_data is not None:
        print(f"holdout rows={len(holdout_data)} combos={args.holdout_combos}")
    if args.resume:
        print(f"resumed from {pathlib.Path(args.resume).expanduser().resolve()} at epoch={start_epoch - 1}")
    if start_epoch > int(args.epochs):
        print(f"checkpoint already reached epoch {start_epoch - 1}; target epochs={args.epochs}")
        return 0
    for epoch in range(start_epoch, int(args.epochs) + 1):
        start = time.time()
        epoch_weights = loss_weights_for_epoch(args, epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch_weights,
            args.noise_std,
        )
        val_metrics = evaluate(model, val_loader, device)
        if holdout_loader is not None:
            holdout_metrics = evaluate(model, holdout_loader, device)
            val_metrics.update({f"holdout/{key}": value for key, value in holdout_metrics.items()})
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "time_sec": time.time() - start,
            "weight_alpha_rel": epoch_weights.alpha_rel,
            "weight_beta_roll": epoch_weights.beta_roll,
            "weight_gamma_event": epoch_weights.gamma_event,
            "weight_delta": epoch_weights.delta,
            "weight_kinematic": epoch_weights.kinematic,
            "weight_counterfactual_rule": epoch_weights.counterfactual_rule,
            **train_metrics,
            **val_metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        progress = (
            f"[{epoch:04d}] loss={metrics['loss']:.6f} state={metrics['state_loss']:.6f} "
            f"delta={metrics['delta_loss']:.6f} rel={metrics['rel_loss']:.6f} roll={metrics['rollout_loss']:.6f} "
            f"event={metrics['event_loss']:.6f} kin={metrics['kinematic_loss']:.6f} "
        )
        if float(args.counterfactual_rule_weight) > 0.0:
            progress += f"cf={metrics['counterfactual_rule_loss']:.6f} "
        progress += (
            f"mask={metrics['mask_loss']:.6f} contrast={metrics['contrastive_loss']:.6f} "
            f"val_slot_rmse={metrics['val_slot_rmse']:.4f} val_roll={metrics['val_rollout_loss']:.6f} "
            f"val_kin={metrics['val_kinematic_loss']:.6f} val_event_acc={metrics['val_event_accuracy']:.3f}"
        )
        print(progress)
        save_checkpoint(output / "latest.pt", model, optimizer, args, epoch, metrics)
        if float(val_metrics["val_mse"]) < best:
            best = float(val_metrics["val_mse"])
            save_checkpoint(output / "best.pt", model, optimizer, args, epoch, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
