from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from custom_breakout import BallState as BreakoutBallState
from custom_breakout import BlockState, BreakoutEnv, BreakoutState, PaddleState as BreakoutPaddleState
from custom_pong import BallState as PongBallState
from custom_pong import GameState as PongGameState
from custom_pong import PaddleState as PongPaddleState
from data.pong_common import GAME_TO_ID, OBJECT_TYPE_TO_ID
from models.gns_shared_simulator import GNSSharedSimulator


@dataclass
class SlotHistory:
    slots: deque[np.ndarray]
    masks: deque[np.ndarray]

    def append(self, slots: np.ndarray, mask: np.ndarray) -> None:
        self.slots.append(np.asarray(slots, dtype=np.float32).copy())
        self.masks.append(np.asarray(mask, dtype=np.float32).copy())

    def batch(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(list(self.slots), axis=0).astype(np.float32), np.stack(list(self.masks), axis=0).astype(np.float32)


def init_history(initial_slots: np.ndarray, initial_mask: np.ndarray, history_length: int) -> SlotHistory:
    slots = deque(maxlen=int(history_length))
    masks = deque(maxlen=int(history_length))
    for _ in range(int(history_length)):
        slots.append(np.asarray(initial_slots, dtype=np.float32).copy())
        masks.append(np.asarray(initial_mask, dtype=np.float32).copy())
    return SlotHistory(slots=slots, masks=masks)


@torch.no_grad()
def predict_next_from_history(
    model: GNSSharedSimulator,
    history: SlotHistory,
    action: int,
    rule_id: int,
    game_id: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    slots_np, masks_np = history.batch()
    latest_slots = slots_np[-1]
    type_ids = latest_slots[:, 6].round().astype(np.int64)
    dynamic_pos_mask = ((type_ids == OBJECT_TYPE_TO_ID["ball"]) & (masks_np[-1] > 0.5)).astype(np.float32)
    out = model(
        torch.as_tensor(slots_np[None], dtype=torch.float32, device=device),
        torch.as_tensor(masks_np[None], dtype=torch.float32, device=device),
        torch.as_tensor([action], dtype=torch.long, device=device),
        torch.as_tensor([rule_id], dtype=torch.long, device=device),
        torch.as_tensor([game_id], dtype=torch.long, device=device),
        dynamic_pos_mask=torch.as_tensor(dynamic_pos_mask[None], dtype=torch.float32, device=device),
        training_noise=False,
    )
    pred_slots = out["pred_next_slots"][0].detach().cpu().numpy().astype(np.float32)
    pred_mask = out["pred_next_mask_prob"][0].detach().cpu().numpy().astype(np.float32)
    return pred_slots, pred_mask


def slots_to_flat_state(slots: np.ndarray, game_id: int) -> np.ndarray:
    slots = np.asarray(slots, dtype=np.float32)
    if int(game_id) == GAME_TO_ID["pong"]:
        return np.asarray([slots[0, 0], slots[0, 1], slots[0, 2], slots[0, 3], slots[1, 1], slots[1, 3]], dtype=np.float32)
    return np.asarray([slots[0, 0], slots[0, 1], slots[0, 2], slots[0, 3], slots[1, 0], slots[1, 2]], dtype=np.float32)


def slots_to_pong_state(pred_slots: np.ndarray, env, prev_state: PongGameState, out_margin: float = 80.0) -> PongGameState:
    cfg = env.config
    slots = np.nan_to_num(np.asarray(pred_slots, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    max_speed = max(float(cfg.max_ball_speed), float(cfg.ball_speed), 1.0)
    slots[:, 0] = np.clip(slots[:, 0], -out_margin, cfg.width + out_margin)
    slots[:, 1] = np.clip(slots[:, 1], -cfg.ball_radius - out_margin, cfg.height + cfg.ball_radius + out_margin)
    slots[:, 2] = np.clip(slots[:, 2], -2.0 * max_speed, 2.0 * max_speed)
    slots[:, 3] = np.clip(slots[:, 3], -2.0 * max_speed, 2.0 * max_speed)
    paddle_y = float(np.clip(slots[1, 1], 0.0, cfg.height - cfg.paddle_height))
    ball = PongBallState(x=float(slots[0, 0]), y=float(slots[0, 1]), vx=float(slots[0, 2]), vy=float(slots[0, 3]), radius=cfg.ball_radius)
    paddle = PongPaddleState(x=cfg.width - cfg.paddle_margin - cfg.paddle_width, y=paddle_y, width=cfg.paddle_width, height=cfg.paddle_height, vy=float(np.clip(slots[1, 3], -cfg.paddle_speed, cfg.paddle_speed)))
    terminated = bool(ball.x - ball.radius > cfg.width + out_margin)
    truncated = bool(cfg.max_steps is not None and prev_state.step_count + 1 >= cfg.max_steps)
    return PongGameState(
        ball=ball,
        paddle=paddle,
        score=prev_state.score,
        hits=prev_state.hits,
        misses=prev_state.misses + int(terminated),
        step_count=prev_state.step_count + 1,
        last_action=prev_state.last_action,
        terminated=terminated,
        truncated=truncated,
        last_event="miss" if terminated else ("truncated" if truncated else "world_model"),
    )


def slots_to_breakout_state(pred_slots: np.ndarray, pred_mask: np.ndarray, env: BreakoutEnv, prev_state: BreakoutState, mask_threshold: float = 0.5) -> BreakoutState:
    cfg = env.config
    slots = np.nan_to_num(np.asarray(pred_slots, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    mask = np.asarray(pred_mask, dtype=np.float32)
    max_speed = max(float(cfg.max_ball_speed), float(cfg.ball_speed), 1.0)
    slots[:, 0] = np.clip(slots[:, 0], -cfg.width, 2.0 * cfg.width)
    slots[:, 1] = np.clip(slots[:, 1], -cfg.height, 2.0 * cfg.height)
    slots[:, 2] = np.clip(slots[:, 2], -2.0 * max_speed, 2.0 * max_speed)
    slots[:, 3] = np.clip(slots[:, 3], -2.0 * max_speed, 2.0 * max_speed)
    paddle = BreakoutPaddleState(
        x=float(np.clip(slots[1, 0], 0.0, cfg.width - cfg.paddle_width)),
        y=cfg.height - cfg.paddle_margin - cfg.paddle_height,
        width=cfg.paddle_width,
        height=cfg.paddle_height,
        vx=float(np.clip(slots[1, 2], -cfg.paddle_speed, cfg.paddle_speed)),
    )
    ball = BreakoutBallState(x=float(slots[0, 0]), y=float(slots[0, 1]), vx=float(slots[0, 2]), vy=float(slots[0, 3]), radius=cfg.ball_radius)
    blocks = []
    removed = 0
    for idx, prev_block in enumerate(prev_state.blocks, start=2):
        active = bool(mask[idx] >= float(mask_threshold))
        if prev_block.active and not active:
            removed += 1
        blocks.append(
            BlockState(
                x=float(slots[idx, 0]) if slots[idx, 4] > 0 else prev_block.x,
                y=float(slots[idx, 1]) if slots[idx, 5] > 0 else prev_block.y,
                width=float(slots[idx, 4]) if slots[idx, 4] > 0 else prev_block.width,
                height=float(slots[idx, 5]) if slots[idx, 5] > 0 else prev_block.height,
                active=active,
            )
        )
    missed = bool(ball.y - ball.radius > cfg.height + 80.0)
    cleared = not any(block.active for block in blocks)
    truncated = bool(cfg.max_steps is not None and prev_state.step_count + 1 >= cfg.max_steps)
    return BreakoutState(
        ball=ball,
        paddle=paddle,
        blocks=blocks,
        score=prev_state.score + removed,
        hits=prev_state.hits + removed,
        misses=prev_state.misses + int(missed),
        step_count=prev_state.step_count + 1,
        last_action=prev_state.last_action,
        terminated=missed or cleared,
        truncated=truncated,
        last_event="miss" if missed else ("cleared" if cleared else ("truncated" if truncated else "world_model")),
    )

