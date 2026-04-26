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

from custom_pong import ACTION_DOWN, ACTION_STAY, ACTION_UP, PongEnv
from data.pong_common import MODES, RULE_TO_ID, flat_pong_state_to_slots
from gns_shared_rollout import init_history, predict_next_from_history, slots_to_pong_state
from models.gns_shared_simulator import build_gns_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Pong through the shared GNS simulator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=MODES, default="normal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required. Install with `python -m pip install pygame-ce`.") from exc
    return pygame


def action_from_keys(pygame) -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
        return ACTION_UP
    if keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
        return ACTION_DOWN
    return ACTION_STAY


def reset_env(env: PongEnv, random_start: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    if random_start:
        state = env.get_state()
        speed = float(rng.uniform(max(env.config.min_ball_speed, 120.0), max(env.config.ball_speed * 1.6, env.config.min_ball_speed)))
        angle = float(rng.uniform(-0.65, 0.65))
        state.ball.x = float(rng.uniform(0.2 * env.config.width, 0.65 * env.config.width))
        state.ball.y = float(rng.uniform(0.15 * env.config.height, 0.85 * env.config.height))
        state.ball.vx = abs(speed * np.cos(angle))
        state.ball.vy = speed * np.sin(angle)
        state.paddle.y = float(rng.uniform(0.0, env.config.height - env.config.paddle_height))
        env.set_state(state)
        obs = env.state_to_observation()
    slots, mask = flat_pong_state_to_slots(obs)
    return obs, slots, mask


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(pathlib.Path(args.checkpoint).expanduser().resolve(), map_location=device)
    model = build_gns_model_from_checkpoint(checkpoint, device)
    history_length = int(checkpoint.get("args", {}).get("history_length", 6))
    env = PongEnv(mode=args.mode, render_mode=None if args.headless_steps else "human", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    obs, slots, mask = reset_env(env, args.random_start, rng)
    history = init_history(slots, mask, history_length)
    if args.headless_steps:
        for _ in range(int(args.headless_steps)):
            pred_slots, pred_mask = predict_next_from_history(model, history, ACTION_STAY, RULE_TO_ID[args.mode], 0, device)
            env.set_state(slots_to_pong_state(pred_slots, env, env.get_state()))
            history.append(pred_slots, pred_mask)
            if env.is_done and args.auto_reset:
                obs, slots, mask = reset_env(env, args.random_start, rng)
                history = init_history(slots, mask, history_length)
        print(f"Ran {args.headless_steps} headless GNS Pong steps.")
        return 0

    pygame = require_pygame()
    env.render()
    clock = pygame.time.Clock()
    mode = args.mode
    paused = False
    running = True
    print("Controls: Up/Down move paddle, R reset, 1 normal, 2 gravity, 3 teleport, P pause, Esc quit.")
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, slots, mask = reset_env(env, args.random_start, rng)
                        history = init_history(slots, mask, history_length)
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_1:
                        mode = "normal"
                        env.config.mode = mode
                        obs, slots, mask = reset_env(env, args.random_start, rng)
                        history = init_history(slots, mask, history_length)
                    elif event.key == pygame.K_2:
                        mode = "gravity"
                        env.config.mode = mode
                        obs, slots, mask = reset_env(env, args.random_start, rng)
                        history = init_history(slots, mask, history_length)
                    elif event.key == pygame.K_3:
                        mode = "teleport"
                        env.config.mode = mode
                        obs, slots, mask = reset_env(env, args.random_start, rng)
                        history = init_history(slots, mask, history_length)
            if not paused and not env.is_done:
                action = action_from_keys(pygame)
                pred_slots, pred_mask = predict_next_from_history(model, history, action, RULE_TO_ID[mode], 0, device)
                env.set_state(slots_to_pong_state(pred_slots, env, env.get_state()))
                history.append(pred_slots, pred_mask)
            elif env.is_done and args.auto_reset:
                obs, slots, mask = reset_env(env, args.random_start, rng)
                history = init_history(slots, mask, history_length)
            env.render()
            pygame.display.set_caption(f"GNS World Model Pong | rule={mode} | step={env.get_state().step_count}")
            clock.tick(args.fps)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
