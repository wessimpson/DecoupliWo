from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from data.pong_common import EVENTS, OBJECT_TYPE_TO_ID


@dataclass(frozen=True)
class LossWeights:
    alpha_rel: float = 0.1
    beta_roll: float = 0.0
    gamma_event: float = 0.0
    delta: float = 0.5
    kinematic: float = 0.2
    counterfactual_rule: float = 0.0
    mask: float = 0.1
    contrastive: float = 0.05
    event_weight: float = 2.0
    contrastive_margin: float = 1.0


SLOT_FEATURE_WEIGHTS = (4.0, 4.0, 2.0, 2.0, 0.05, 0.05)


def slot_feature_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(SLOT_FEATURE_WEIGHTS, dtype=dtype, device=device)


def transition_weights(event_id: torch.Tensor, event_weight: float) -> torch.Tensor:
    weights = torch.ones_like(event_id, dtype=torch.float32)
    if event_weight <= 1.0:
        return weights
    rare = event_id != EVENTS.index("step")
    rare &= event_id != EVENTS.index("none")
    weights[rare] = float(event_weight)
    return weights


def masked_slot_loss(
    pred_next_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    target_mask: torch.Tensor,
    model: Any,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_norm = model.normalizer.normalize_slots(pred_next_slots)[..., :6]
    target_norm = model.normalizer.normalize_slots(target_next_slots)[..., :6]
    weights = slot_feature_weights(pred_norm.device, pred_norm.dtype)
    per_feature = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    per_slot = (per_feature * weights).sum(dim=-1) / weights.sum().clamp_min(1.0)
    mask = target_mask.to(per_slot.device).to(per_slot.dtype)
    per_row = (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def masked_delta_loss(
    pred_next_slots: torch.Tensor,
    current_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    target_mask: torch.Tensor,
    model: Any,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_delta = model.normalizer.normalize_slots(pred_next_slots)[..., :6] - model.normalizer.normalize_slots(current_slots)[..., :6]
    target_delta = model.normalizer.normalize_slots(target_next_slots)[..., :6] - model.normalizer.normalize_slots(current_slots)[..., :6]
    weights = slot_feature_weights(pred_delta.device, pred_delta.dtype)
    per_feature = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
    per_slot = (per_feature * weights).sum(dim=-1) / weights.sum().clamp_min(1.0)
    mask = target_mask.to(per_slot.device).to(per_slot.dtype)
    per_row = (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def masked_acceleration_loss(
    pred_normalized_acceleration: torch.Tensor,
    current_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    target_mask: torch.Tensor,
    model: Any,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    dt = max(float(getattr(model.constants, "dt", 1.0 / 60.0)), 1e-6)
    type_ids = current_slots[..., 6].round().to(torch.long)
    dynamic_mask = (type_ids == OBJECT_TYPE_TO_ID["ball"]).to(pred_normalized_acceleration.dtype) * target_mask.to(pred_normalized_acceleration.dtype)
    if dynamic_mask.sum().item() <= 0.0:
        return torch.zeros((), dtype=pred_normalized_acceleration.dtype, device=pred_normalized_acceleration.device)

    target_acceleration = (target_next_slots[..., 2:4] - current_slots[..., 2:4]) / dt
    target_norm = model.normalizer.normalize_acceleration(target_acceleration)
    per_slot = F.smooth_l1_loss(pred_normalized_acceleration, target_norm, reduction="none").mean(dim=-1)
    per_row = (per_slot * dynamic_mask).sum(dim=1) / dynamic_mask.sum(dim=1).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def relative_slot_loss(
    pred_next_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    target_mask: torch.Tensor,
    model: Any,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = model.normalizer.normalize_slots(pred_next_slots)[..., :4]
    target = model.normalizer.normalize_slots(target_next_slots)[..., :4]
    pred_rel = pred[:, :, None, :] - pred[:, None, :, :]
    target_rel = target[:, :, None, :] - target[:, None, :, :]
    per_pair = F.smooth_l1_loss(pred_rel, target_rel, reduction="none").mean(dim=-1)
    mask = target_mask.to(per_pair.device).to(per_pair.dtype)
    nodes = mask.shape[1]
    non_self = 1.0 - torch.eye(nodes, dtype=per_pair.dtype, device=per_pair.device)[None, :, :]
    pair_mask = mask[:, :, None] * mask[:, None, :] * non_self
    per_row = (per_pair * pair_mask).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def object_mask_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target_mask, dtype=torch.float32)
    weights[:, 2:] = torch.where(input_mask[:, 2:] > 0.0, 2.0, 0.25)
    return F.binary_cross_entropy_with_logits(pred_logits, target_mask.to(pred_logits.dtype), weight=weights, reduction="mean")


def event_prediction_loss(pred_logits: torch.Tensor, event_id: torch.Tensor, row_weights: torch.Tensor | None = None) -> torch.Tensor:
    class_weights = torch.ones(pred_logits.shape[-1], dtype=pred_logits.dtype, device=pred_logits.device)
    for name in ("step", "none"):
        class_weights[EVENTS.index(name)] = 0.5
    for idx, name in enumerate(EVENTS):
        if name not in {"step", "none"}:
            class_weights[idx] = 2.0
    per_row = F.cross_entropy(pred_logits, event_id.to(torch.long), weight=class_weights, reduction="none")
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def kinematic_consistency_loss(
    pred_next_slots: torch.Tensor,
    current_slots: torch.Tensor,
    target_mask: torch.Tensor,
    event_id: torch.Tensor | None,
    model: Any,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    type_ids = current_slots[..., 6].round().to(torch.long)
    ball_mask = (type_ids == OBJECT_TYPE_TO_ID["ball"]).to(pred_next_slots.dtype) * target_mask.to(pred_next_slots.dtype)
    if event_id is not None:
        free = (event_id == EVENTS.index("step")).to(pred_next_slots.dtype)
        ball_mask = ball_mask * free[:, None]
    if ball_mask.sum().item() <= 0.0:
        return torch.zeros((), dtype=pred_next_slots.dtype, device=pred_next_slots.device)

    dt = float(getattr(model.constants, "dt", 1.0 / 60.0))
    scales = model.normalizer.slot_scales.to(pred_next_slots.device, pred_next_slots.dtype)
    expected_pos = current_slots[..., 0:2] + pred_next_slots[..., 2:4] * dt
    pred_pos = pred_next_slots[..., 0:2]
    per_slot = F.smooth_l1_loss(pred_pos / scales[0:2], expected_pos / scales[0:2], reduction="none").mean(dim=-1)
    per_row = (per_slot * ball_mask).sum(dim=1) / ball_mask.sum(dim=1).clamp_min(1.0)
    if row_weights is not None:
        per_row = per_row * row_weights.to(per_row.device)
    return per_row.mean()


def rollout_slot_loss(model: Any, batch: dict[str, torch.Tensor], row_weights: torch.Tensor | None = None) -> torch.Tensor:
    if "rollout_action" not in batch:
        return torch.zeros((), dtype=torch.float32, device=batch["object_slots"].device)

    actions = batch["rollout_action"]
    rules = batch["rollout_rule_id"]
    targets = batch["rollout_next_object_slots"]
    target_masks = batch["rollout_next_object_mask"]
    valid = batch["rollout_valid"].to(torch.float32)
    horizon = int(actions.shape[1])
    if horizon <= 1 or valid[:, 1:].sum().item() <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=batch["object_slots"].device)

    current_slots = batch["object_slots"]
    current_mask = batch["object_mask"]
    device = current_slots.device
    dummy_state = batch["state"]
    total = torch.zeros((), dtype=torch.float32, device=device)
    count = torch.zeros((), dtype=torch.float32, device=device)

    for step in range(horizon):
        out = model(
            dummy_state,
            actions[:, step],
            rules[:, step],
            normalized=False,
            object_slots=current_slots,
            object_mask=current_mask,
            game_id=batch.get("game_id"),
        )
        step_valid = valid[:, step]
        if step > 0 and step_valid.sum().item() > 0.0:
            weights = step_valid if row_weights is None else step_valid * row_weights.to(device)
            total = total + masked_slot_loss(out["pred_next_slots"], targets[:, step], target_masks[:, step], model, weights)
            count = count + 1.0
        current_slots = out["pred_next_slots"]
        current_mask = target_masks[:, step]

    return total / count.clamp_min(1.0)


def counterfactual_rule_loss(model: Any, batch: dict[str, torch.Tensor], row_weights: torch.Tensor | None = None) -> torch.Tensor:
    if "counterfactual_next_object_slots" not in batch:
        return torch.zeros((), dtype=torch.float32, device=batch["object_slots"].device)

    targets = batch["counterfactual_next_object_slots"]
    target_masks = batch["counterfactual_next_object_mask"]
    valid = batch["counterfactual_valid"].to(torch.float32)
    batch_size, num_rules = valid.shape
    if valid.sum().item() <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=batch["object_slots"].device)

    slots = batch["object_slots"][:, None].expand(-1, num_rules, -1, -1).reshape(batch_size * num_rules, *batch["object_slots"].shape[1:])
    masks = batch["object_mask"][:, None].expand(-1, num_rules, -1).reshape(batch_size * num_rules, batch["object_mask"].shape[1])
    actions = batch["action"][:, None].expand(-1, num_rules).reshape(batch_size * num_rules)
    rules = batch["counterfactual_rule_id"].reshape(batch_size * num_rules)
    game_ids = batch["game_id"][:, None].expand(-1, num_rules).reshape(batch_size * num_rules)
    dummy_state = batch["state"][:, None].expand(-1, num_rules, -1).reshape(batch_size * num_rules, batch["state"].shape[-1])

    out = model(dummy_state, actions, rules, normalized=False, object_slots=slots, object_mask=masks, game_id=game_ids)
    flat_targets = targets.reshape(batch_size * num_rules, *targets.shape[2:])
    flat_target_masks = target_masks.reshape(batch_size * num_rules, target_masks.shape[-1])
    flat_valid = valid.reshape(batch_size * num_rules)
    if row_weights is not None:
        flat_valid = flat_valid * row_weights[:, None].expand(-1, num_rules).reshape(batch_size * num_rules).to(flat_valid.device)
    return masked_slot_loss(out["pred_next_slots"], flat_targets, flat_target_masks, model, flat_valid)


def compute_object_centric_losses(
    model: Any,
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    row_weights = transition_weights(batch["event_id"], weights.event_weight).to(out["pred_next_slots"].device)
    obj = masked_slot_loss(out["pred_next_slots"], batch["next_object_slots"], batch["next_object_mask"], model, row_weights)
    delta = (
        masked_acceleration_loss(
            out["pred_normalized_acceleration"],
            batch["object_slots"],
            batch["next_object_slots"],
            batch["next_object_mask"],
            model,
            row_weights,
        )
        if "pred_normalized_acceleration" in out
        else masked_delta_loss(out["pred_next_slots"], batch["object_slots"], batch["next_object_slots"], batch["next_object_mask"], model, row_weights)
        if weights.delta > 0.0
        else torch.zeros((), dtype=obj.dtype, device=obj.device)
    )
    zero = torch.zeros((), dtype=obj.dtype, device=obj.device)
    rel = (
        relative_slot_loss(out["pred_next_slots"], batch["next_object_slots"], batch["next_object_mask"], model, row_weights)
        if weights.alpha_rel > 0.0
        else zero
    )
    mask = object_mask_loss(out["pred_next_mask_logits"], batch["next_object_mask"], batch["object_mask"])
    event = event_prediction_loss(out["pred_event_logits"], batch["event_id"], row_weights) if weights.gamma_event > 0.0 else zero
    roll = rollout_slot_loss(model, batch, row_weights) if weights.beta_roll > 0.0 else zero
    kinematic = (
        kinematic_consistency_loss(out["pred_next_slots"], batch["object_slots"], batch["next_object_mask"], batch["event_id"], model, row_weights)
        if weights.kinematic > 0.0
        else zero
    )
    cf_rule = counterfactual_rule_loss(model, batch, row_weights) if weights.counterfactual_rule > 0.0 else zero

    if weights.contrastive > 0.0:
        neg_perm = torch.randperm(batch["next_object_slots"].shape[0], device=out["pred_next_slots"].device)
        contrastive, contrastive_info = model.contrastive_loss(
            out["z_next_pred"],
            batch["next_object_slots"],
            batch["next_object_mask"],
            batch["next_object_slots"][neg_perm],
            batch["next_object_mask"][neg_perm],
            margin=weights.contrastive_margin,
        )
    else:
        contrastive = zero
        contrastive_info = {
            "contrastive_positive": zero.detach(),
            "contrastive_negative": zero.detach(),
        }

    total = (
        obj
        + float(weights.delta) * delta
        + float(weights.alpha_rel) * rel
        + float(weights.beta_roll) * roll
        + float(weights.gamma_event) * event
        + float(weights.kinematic) * kinematic
        + float(weights.counterfactual_rule) * cf_rule
        + float(weights.mask) * mask
        + float(weights.contrastive) * contrastive
    )
    return {
        "loss": total,
        "state_loss": obj,
        "delta_loss": delta,
        "rel_loss": rel,
        "rollout_loss": roll,
        "event_loss": event,
        "kinematic_loss": kinematic,
        "counterfactual_rule_loss": cf_rule,
        "mask_loss": mask,
        "contrastive_loss": contrastive,
        **contrastive_info,
    }
