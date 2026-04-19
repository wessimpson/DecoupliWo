#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_breakout import ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, BallState, BlockState, BreakoutEnv, BreakoutState, PaddleState
from data.pong_common import GAME_TO_ID, MODES, RULE_TO_ID
from eval_pong_world_model import build_model, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Breakout-lite through a trained shared-slot world model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=MODES, default="normal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--gravity", type=float, default=420.0)
    parser.add_argument("--paddle-speed", type=float, default=360.0)
    parser.add_argument("--ball-speed", type=float, default=280.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless-steps", type=int, default=0)
    return parser.parse_args()


def require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required. Install with `python -m pip install pygame-ce`.") from exc
    return pygame


def build_env(args: argparse.Namespace, mode: str, render_mode: str | None = "human") -> BreakoutEnv:
    return BreakoutEnv(
        width=args.width,
        height=args.height,
        mode=mode,
        dt=args.dt,
        gravity=args.gravity,
        paddle_speed=args.paddle_speed,
        ball_speed=args.ball_speed,
        max_steps=args.max_steps,
        render_mode=render_mode,
        seed=args.seed,
    )


def sanitize_slots(slots: np.ndarray, env: BreakoutEnv) -> np.ndarray:
    cfg = env.config
    slots = np.nan_to_num(np.asarray(slots, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    max_speed = max(float(cfg.max_ball_speed), float(cfg.ball_speed), 1.0)
    slots[:, 0] = np.clip(slots[:, 0], -cfg.width, 2.0 * cfg.width)
    slots[:, 1] = np.clip(slots[:, 1], -cfg.height, 2.0 * cfg.height)
    slots[:, 2] = np.clip(slots[:, 2], -2.0 * max_speed, 2.0 * max_speed)
    slots[:, 3] = np.clip(slots[:, 3], -2.0 * max_speed, 2.0 * max_speed)
    slots[:, 4] = np.clip(slots[:, 4], 1.0, cfg.width)
    slots[:, 5] = np.clip(slots[:, 5], 1.0, cfg.height)
    return slots


def slots_to_breakout_state(pred_slots: np.ndarray, pred_mask: np.ndarray, env: BreakoutEnv, action: int, threshold: float) -> BreakoutState:
    cfg = env.config
    prev = env.get_state()
    slots = sanitize_slots(pred_slots, env)
    mask = np.asarray(pred_mask, dtype=np.float32)
    ball = BallState(
        x=float(slots[0, 0]),
        y=float(slots[0, 1]),
        vx=float(slots[0, 2]),
        vy=float(slots[0, 3]),
        radius=cfg.ball_radius,
    )
    paddle = PaddleState(
        x=float(np.clip(slots[1, 0], 0.0, cfg.width - cfg.paddle_width)),
        y=cfg.height - cfg.paddle_margin - cfg.paddle_height,
        width=cfg.paddle_width,
        height=cfg.paddle_height,
        vx=float(np.clip(slots[1, 2], -cfg.paddle_speed, cfg.paddle_speed)),
    )
    blocks = []
    removed = 0
    for idx, prev_block in enumerate(prev.blocks, start=2):
        active = bool(mask[idx] >= float(threshold))
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
    step_count = prev.step_count + 1
    missed = bool(ball.y - ball.radius > cfg.height + 80.0 or not np.isfinite(slots).all())
    truncated = bool(cfg.max_steps is not None and step_count >= cfg.max_steps)
    cleared = not any(block.active for block in blocks)
    return BreakoutState(
        ball=ball,
        paddle=paddle,
        blocks=blocks,
        score=prev.score + removed,
        hits=prev.hits + removed,
        misses=prev.misses + int(missed),
        step_count=step_count,
        last_action=int(action),
        terminated=missed or cleared,
        truncated=truncated,
        last_event="miss" if missed else ("cleared" if cleared else ("truncated" if truncated else "world_model")),
    )


@torch.no_grad()
def model_step(model, env: BreakoutEnv, action: int, rule_id: int, device: torch.device, threshold: float) -> None:
    state = torch.as_tensor(env.state_to_observation(), dtype=torch.float32, device=device).unsqueeze(0)
    slots_np, mask_np = env.state_to_slots()
    slots = torch.as_tensor(slots_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)
    out = model(
        state,
        torch.as_tensor([action], dtype=torch.long, device=device),
        torch.as_tensor([rule_id], dtype=torch.long, device=device),
        object_slots=slots,
        object_mask=mask,
        game_id=torch.as_tensor([GAME_TO_ID["breakout"]], dtype=torch.long, device=device),
    )
    pred_slots = out["pred_next_slots"][0].detach().cpu().numpy()
    pred_mask = out["pred_next_mask_prob"][0].detach().cpu().numpy()
    pred_mask[0:2] = 1.0
    env.set_state(slots_to_breakout_state(pred_slots, pred_mask, env, action, threshold))


def action_from_keys(pygame) -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        return ACTION_LEFT
    if keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        return ACTION_RIGHT
    return ACTION_STAY


def load_model(checkpoint_path: pathlib.Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint, device)
    return model


def run_headless(args: argparse.Namespace, model, device: torch.device) -> int:
    env = build_env(args, args.mode, render_mode=None)
    env.reset(seed=args.seed)
    try:
        for _ in range(int(args.headless_steps)):
            model_step(model, env, ACTION_STAY, RULE_TO_ID[args.mode], device, args.mask_threshold)
            if env.is_done:
                env.reset()
        print(f"Ran {args.headless_steps} headless Breakout world-model steps in mode={args.mode}.")
    finally:
        env.close()
    return 0


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(pathlib.Path(args.checkpoint).expanduser().resolve(), device)
    if args.headless_steps:
        return run_headless(args, model, device)

    pygame = require_pygame()
    env = build_env(args, args.mode, render_mode="human")
    env.reset(seed=args.seed)
    env.render()
    clock = pygame.time.Clock()
    running = True
    paused = False
    mode = args.mode
    print("Controls: Left/Right move paddle, R reset, 1 normal, 2 gravity, 3 teleport, P pause, Esc quit.")
    print("This is model rollout only: env.step() is not used after reset.")
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=args.seed)
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_1:
                        mode = "normal"
                        env.config.mode = mode
                        env.reset(seed=args.seed)
                    elif event.key == pygame.K_2:
                        mode = "gravity"
                        env.config.mode = mode
                        env.reset(seed=args.seed)
                    elif event.key == pygame.K_3:
                        mode = "teleport"
                        env.config.mode = mode
                        env.reset(seed=args.seed)
            if not paused and not env.is_done:
                model_step(model, env, action_from_keys(pygame), RULE_TO_ID[mode], device, args.mask_threshold)
            if args.auto_reset and env.is_done:
                env.reset()
            env.render()
            pygame.display.set_caption(f"World Model Breakout | rule={mode} | step={env.get_state().step_count}")
            clock.tick(int(args.fps))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
