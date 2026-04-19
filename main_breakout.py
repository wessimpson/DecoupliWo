from __future__ import annotations

import argparse

from custom_breakout import BreakoutEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playable Breakout-lite demo.")
    parser.add_argument("--mode", choices=("normal", "gravity", "teleport"), default="normal")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--gravity", type=float, default=420.0)
    parser.add_argument("--paddle-speed", type=float, default=360.0)
    parser.add_argument("--ball-speed", type=float, default=280.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required for the playable demo. Install it with `pip install pygame-ce`.") from exc
    return pygame


def build_env(args: argparse.Namespace, mode: str) -> BreakoutEnv:
    return BreakoutEnv(
        width=args.width,
        height=args.height,
        mode=mode,
        dt=args.dt,
        gravity=args.gravity,
        paddle_speed=args.paddle_speed,
        ball_speed=args.ball_speed,
        max_steps=args.max_steps,
        render_mode="human",
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()
    pygame = _require_pygame()

    env = build_env(args, args.mode)
    _, info = env.reset(seed=args.seed)
    env.render()
    clock = pygame.time.Clock()
    running = True

    print("Controls: Left/Right move paddle, R reset, 1 normal, 2 gravity, 3 teleport, Esc quit.")

    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    _, info = env.reset()
                elif event.key == pygame.K_1:
                    env.close()
                    env = build_env(args, "normal")
                    _, info = env.reset(seed=args.seed)
                    env.render()
                elif event.key == pygame.K_2:
                    env.close()
                    env = build_env(args, "gravity")
                    _, info = env.reset(seed=args.seed)
                    env.render()
                elif event.key == pygame.K_3:
                    env.close()
                    env = build_env(args, "teleport")
                    _, info = env.reset(seed=args.seed)
                    env.render()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            action = 2

        if not env.is_done:
            _, _, terminated, truncated, info = env.step(action)
        else:
            terminated = info.get("terminated", False)
            truncated = info.get("truncated", False)

        env.render()
        pygame.display.set_caption(
            f"Breakout-lite - {env.config.mode} | blocks={info.get('active_blocks', 0)} | event={info.get('event', 'none')}"
        )
        clock.tick(env.config.render_fps)

        if terminated or truncated:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
