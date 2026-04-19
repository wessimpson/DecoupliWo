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

from custom_pong import ACTION_DOWN, ACTION_STAY, ACTION_UP, BallState, GameState, PaddleState, PongEnv
from data.pong_common import MODES, RULE_TO_ID
from eval_pong_world_model import build_model, choose_device, predict_next


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play custom Pong through a trained rule-conditioned world model.")
    parser.add_argument("--checkpoint", required=True, help="Path to world-model checkpoint, usually runs/.../best.pt")
    parser.add_argument("--mode", choices=MODES, default="normal", help="Rule variant used for the model rollout.")
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
    parser.add_argument("--start-ball-x", type=float, default=None)
    parser.add_argument("--start-ball-y", type=float, default=None)
    parser.add_argument("--start-ball-vx", type=float, default=None)
    parser.add_argument("--start-ball-vy", type=float, default=None)
    parser.add_argument("--start-paddle-y", type=float, default=None)
    parser.add_argument("--toward-paddle", action="store_true", help="Initialize ball moving right toward the paddle.")
    parser.add_argument("--random-start", action="store_true", help="Sample a new reproducible random initial state on each reset.")
    parser.add_argument("--no-clamp", action="store_true", help="Do not clamp predicted state values before rendering.")
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True, help="Reset when model rollout terminates or leaves the visible screen.")
    parser.add_argument("--out-margin", type=float, default=80.0, help="Reset if predicted ball is this far outside the screen.")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run N model steps without opening pygame; useful for smoke tests.")
    return parser.parse_args()


def require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required. Install with `python -m pip install pygame-ce`.") from exc
    return pygame


def build_env(args: argparse.Namespace, mode: str, render_mode: str | None = "human") -> PongEnv:
    return PongEnv(
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


def apply_start_state(env: PongEnv, args: argparse.Namespace, rng: np.random.Generator | None = None) -> None:
    state = env.get_state()
    cfg = env.config
    ball_x = state.ball.x
    ball_y = state.ball.y
    ball_vx = state.ball.vx
    ball_vy = state.ball.vy
    paddle_y = state.paddle.y

    if args.random_start:
        rng = rng or np.random.default_rng(args.seed)
        speed = float(rng.uniform(max(cfg.min_ball_speed, 120.0), max(cfg.ball_speed * 1.6, cfg.min_ball_speed)))
        angle = float(rng.uniform(-0.65, 0.65))
        direction = 1.0 if (args.toward_paddle or rng.random() < 0.5) else -1.0
        ball_x = float(rng.uniform(0.2 * cfg.width, 0.65 * cfg.width))
        ball_y = float(rng.uniform(0.15 * cfg.height, 0.85 * cfg.height))
        ball_vx = direction * speed * np.cos(angle)
        ball_vy = speed * np.sin(angle)
        paddle_y = float(rng.uniform(0.0, cfg.height - cfg.paddle_height))

    if args.toward_paddle and args.start_ball_x is None and not args.random_start:
        ball_x = cfg.width / 3.0
    if args.toward_paddle and args.start_ball_y is None and not args.random_start:
        ball_y = cfg.height / 2.0
    if args.toward_paddle and args.start_ball_vx is None:
        ball_vx = abs(ball_vx)

    if args.start_ball_x is not None:
        ball_x = args.start_ball_x
    if args.start_ball_y is not None:
        ball_y = args.start_ball_y
    if args.start_ball_vx is not None:
        ball_vx = args.start_ball_vx
    if args.start_ball_vy is not None:
        ball_vy = args.start_ball_vy
    if args.start_paddle_y is not None:
        paddle_y = args.start_paddle_y

    env.set_state(
        GameState(
            ball=BallState(
                x=float(np.clip(ball_x, 0.0, cfg.width)),
                y=float(np.clip(ball_y, 0.0, cfg.height)),
                vx=float(ball_vx),
                vy=float(ball_vy),
                radius=cfg.ball_radius,
            ),
            paddle=PaddleState(
                x=cfg.width - cfg.paddle_margin - cfg.paddle_width,
                y=float(np.clip(paddle_y, 0.0, cfg.height - cfg.paddle_height)),
                width=cfg.paddle_width,
                height=cfg.paddle_height,
                vy=state.paddle.vy,
            ),
            score=0,
            hits=0,
            misses=0,
            step_count=0,
            last_action=ACTION_STAY,
            terminated=False,
            truncated=False,
            last_event="custom_start",
        )
    )


def sanitize_prediction(pred: np.ndarray, env: PongEnv, clamp: bool) -> np.ndarray:
    cfg = env.config
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    if not clamp:
        return pred
    max_speed = max(float(cfg.max_ball_speed), float(cfg.ball_speed), 1.0)
    pred[0] = np.clip(pred[0], 0.0, cfg.width)
    pred[1] = np.clip(pred[1], 0.0, cfg.height)
    pred[2] = np.clip(pred[2], -2.0 * max_speed, 2.0 * max_speed)
    pred[3] = np.clip(pred[3], -2.0 * max_speed, 2.0 * max_speed)
    pred[4] = np.clip(pred[4], 0.0, cfg.height - cfg.paddle_height)
    pred[5] = np.clip(pred[5], -cfg.paddle_speed, cfg.paddle_speed)
    return pred


def prediction_is_invalid(raw_pred: np.ndarray, env: PongEnv, out_margin: float) -> bool:
    cfg = env.config
    pred = np.asarray(raw_pred, dtype=np.float32)
    if not np.isfinite(pred).all():
        return True
    return bool(
        pred[0] < -float(out_margin)
        or pred[0] > cfg.width + float(out_margin)
        or pred[1] < -float(out_margin)
        or pred[1] > cfg.height + float(out_margin)
    )


def flat_state_to_game_state(pred: np.ndarray, env: PongEnv, action: int, clamp: bool, out_margin: float) -> GameState:
    cfg = env.config
    prev = env.get_state()
    raw_pred = np.asarray(pred, dtype=np.float32)
    invalid = prediction_is_invalid(raw_pred, env, out_margin)
    pred = sanitize_prediction(raw_pred, env, clamp=clamp)
    step_count = prev.step_count + 1
    missed = bool(invalid or raw_pred[0] - cfg.ball_radius > cfg.width)
    truncated = bool(cfg.max_steps is not None and step_count >= cfg.max_steps)
    return GameState(
        ball=BallState(
            x=float(pred[0]),
            y=float(pred[1]),
            vx=float(pred[2]),
            vy=float(pred[3]),
            radius=cfg.ball_radius,
        ),
        paddle=PaddleState(
            x=cfg.width - cfg.paddle_margin - cfg.paddle_width,
            y=float(pred[4]),
            width=cfg.paddle_width,
            height=cfg.paddle_height,
            vy=float(pred[5]),
        ),
        score=prev.score,
        hits=prev.hits,
        misses=prev.misses + int(missed),
        step_count=step_count,
        last_action=int(action),
        terminated=missed,
        truncated=truncated,
        last_event="invalid_prediction" if invalid else ("miss" if missed else ("truncated" if truncated else "world_model")),
    )


def model_step(model, env: PongEnv, action: int, rule_id: int, device: torch.device, clamp: bool, out_margin: float) -> np.ndarray:
    state = env.state_to_observation()
    pred = predict_next(model, state, int(action), int(rule_id), device)
    env.set_state(flat_state_to_game_state(pred, env, int(action), clamp=clamp, out_margin=out_margin))
    return pred


def load_world_model(checkpoint_path: pathlib.Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint, device)
    return model


def action_from_keys(pygame) -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
        return ACTION_UP
    if keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
        return ACTION_DOWN
    return ACTION_STAY


def reset_with_start(env: PongEnv, args: argparse.Namespace, seed: int | None = None, rng: np.random.Generator | None = None) -> None:
    env.reset(seed=seed)
    apply_start_state(env, args, rng=rng)


def switch_mode(env: PongEnv, mode: str, args: argparse.Namespace, seed: int | None, rng: np.random.Generator | None = None) -> None:
    # Do not call env.close() here. close() calls pygame.quit(), which can
    # freeze some SDL backends when reinitializing inside the active event loop.
    env.config.mode = mode
    reset_with_start(env, args, seed=seed, rng=rng)
    env.render()


def run_headless(args: argparse.Namespace, model, device: torch.device) -> int:
    env = build_env(args, args.mode, render_mode=None)
    rng = np.random.default_rng(args.seed)
    reset_with_start(env, args, seed=args.seed, rng=rng)
    rule_id = RULE_TO_ID[args.mode]
    try:
        for _ in range(int(args.headless_steps)):
            model_step(model, env, ACTION_STAY, rule_id, device, clamp=not args.no_clamp, out_margin=args.out_margin)
            if env.is_done:
                reset_with_start(env, args, rng=rng)
        print(f"Ran {args.headless_steps} headless world-model steps in mode={args.mode}.")
    finally:
        env.close()
    return 0


def main() -> int:
    args = parse_args()
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    device = choose_device(args.device)
    model = load_world_model(checkpoint_path, device)

    if args.headless_steps:
        return run_headless(args, model, device)

    pygame = require_pygame()
    env = build_env(args, args.mode, render_mode="human")
    rng = np.random.default_rng(args.seed)
    reset_with_start(env, args, seed=args.seed, rng=rng)
    env.render()
    clock = pygame.time.Clock()
    running = True
    paused = False
    mode = args.mode

    print("Controls: Up/Down move paddle, R reset, 1 normal, 2 gravity, 3 teleport, P pause, S single-step, Esc quit.")
    print("This is model rollout only: env.step() is not used after reset.")

    try:
        while running:
            single_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_with_start(env, args, rng=rng)
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_s:
                        single_step = True
                    elif event.key == pygame.K_1:
                        mode = "normal"
                        switch_mode(env, mode, args, args.seed, rng=rng)
                    elif event.key == pygame.K_2:
                        mode = "gravity"
                        switch_mode(env, mode, args, args.seed, rng=rng)
                    elif event.key == pygame.K_3:
                        mode = "teleport"
                        switch_mode(env, mode, args, args.seed, rng=rng)

            action = action_from_keys(pygame)
            if not env.is_done and (not paused or single_step):
                model_step(model, env, action, RULE_TO_ID[mode], device, clamp=not args.no_clamp, out_margin=args.out_margin)
            if args.auto_reset and env.is_done:
                reset_with_start(env, args, rng=rng)

            env.render()
            pygame.display.set_caption(
                f"World Model Pong | rule={mode} | step={env.get_state().step_count} | action={env.action_meanings[action]}"
            )
            clock.tick(int(args.fps))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
