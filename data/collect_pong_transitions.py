#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pong_common import (
    MODES,
    RULE_TO_ID,
    SOURCE_TO_ID,
    SOURCES,
    TransitionShardWriter,
    choose_policy_action,
    copy_state_for_mode,
    count_rows,
    event_id,
    make_env,
    slot_config_from_env,
    split_episode_ids,
    state_config_metadata,
    write_metadata,
)
from custom_pong import ACTION_STAY, BallState, GameState, PaddleState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect state-action-next-state transitions for custom Pong.")
    parser.add_argument("--output", default="data/transitions/custom_pong/debug", help="Output dataset directory.")
    parser.add_argument("--policy", choices=("random", "heuristic", "mixed"), default="mixed")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per mode for normal collection.")
    parser.add_argument("--steps-per-episode", type=int, default=3000)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--counterfactual", action="store_true", help="For each sampled state/action, step all rule modes.")
    parser.add_argument("--counterfactual-base-mode", choices=MODES, default="normal")
    parser.add_argument("--rare-events", action="store_true", help="Append targeted rare/diverse state transitions.")
    parser.add_argument("--rare-samples-per-source", type=int, default=2000)
    parser.add_argument("--rare-sources", nargs="+", choices=SOURCES[1:], default=list(SOURCES[1:]))
    parser.add_argument("--rare-counterfactual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--gravity", type=float, default=420.0)
    parser.add_argument("--paddle-speed", type=float, default=360.0)
    parser.add_argument("--ball-speed", type=float, default=280.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def env_kwargs(args: argparse.Namespace) -> dict:
    return {
        "width": args.width,
        "height": args.height,
        "dt": args.dt,
        "gravity": args.gravity,
        "paddle_speed": args.paddle_speed,
        "ball_speed": args.ball_speed,
        "max_steps": args.max_steps,
    }


def _uniform_signed_speed(rng: random.Random, low: float, high: float, sign: int | None = None) -> float:
    value = rng.uniform(float(low), float(high))
    if sign is None:
        sign = -1 if rng.random() < 0.5 else 1
    return float(sign) * value


def _make_state(env, ball_x: float, ball_y: float, ball_vx: float, ball_vy: float, paddle_y: float, paddle_vy: float = 0.0) -> GameState:
    cfg = env.config
    return GameState(
        ball=BallState(
            x=float(ball_x),
            y=float(ball_y),
            vx=float(ball_vx),
            vy=float(ball_vy),
            radius=cfg.ball_radius,
        ),
        paddle=PaddleState(
            x=cfg.width - cfg.paddle_margin - cfg.paddle_width,
            y=float(paddle_y),
            width=cfg.paddle_width,
            height=cfg.paddle_height,
            vy=float(paddle_vy),
        ),
        last_event="rare_start",
    )


def sample_rare_state(source: str, env, rng: random.Random) -> tuple[GameState, int]:
    cfg = env.config
    max_speed = min(float(cfg.max_ball_speed), max(float(cfg.ball_speed) * 2.0, 360.0))
    min_speed = max(float(cfg.min_ball_speed), 140.0)
    paddle_y = rng.uniform(0.0, cfg.height - cfg.paddle_height)
    action = rng.choice((0, 1, 2))

    if source == "diverse":
        return _make_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=rng.uniform(-0.75 * max_speed, 0.75 * max_speed),
            paddle_y=paddle_y,
            paddle_vy=rng.choice((-cfg.paddle_speed, 0.0, cfg.paddle_speed)),
        ), action

    if source == "left_wall":
        return _make_state(
            env,
            ball_x=cfg.ball_radius + rng.uniform(0.0, 2.0),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            ball_vy=rng.uniform(-0.5 * max_speed, 0.5 * max_speed),
            paddle_y=paddle_y,
        ), action

    if source == "top_bounce":
        return _make_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=cfg.ball_radius + rng.uniform(0.0, 2.0),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            paddle_y=paddle_y,
        ), action

    if source == "bottom_bounce":
        return _make_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=cfg.height - cfg.ball_radius - rng.uniform(0.0, 2.0),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed, sign=1),
            paddle_y=paddle_y,
        ), action

    if source == "wrapped_top":
        return _make_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=-cfg.ball_radius + rng.uniform(-3.0, 1.0),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            paddle_y=paddle_y,
        ), action

    if source == "wrapped_bottom":
        return _make_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=cfg.height + cfg.ball_radius + rng.uniform(-1.0, 3.0),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed, sign=1),
            paddle_y=paddle_y,
        ), action

    if source == "paddle_hit":
        speed = rng.uniform(min_speed, max_speed)
        paddle_y = rng.uniform(0.0, cfg.height - cfg.paddle_height)
        hit_y = paddle_y + rng.uniform(0.15, 0.85) * cfg.paddle_height
        return _make_state(
            env,
            ball_x=cfg.width - cfg.paddle_margin - cfg.paddle_width - cfg.ball_radius - speed * cfg.dt * rng.uniform(0.25, 0.75),
            ball_y=hit_y,
            ball_vx=speed,
            ball_vy=rng.uniform(-0.2 * max_speed, 0.2 * max_speed),
            paddle_y=paddle_y,
        ), action

    if source == "miss":
        speed = rng.uniform(min_speed, max_speed)
        return _make_state(
            env,
            ball_x=cfg.width + cfg.ball_radius - speed * cfg.dt * rng.uniform(0.1, 0.7),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=speed,
            ball_vy=rng.uniform(-0.3 * max_speed, 0.3 * max_speed),
            paddle_y=paddle_y,
        ), action

    raise ValueError(f"Unknown rare source: {source}")


def collect_standard(args: argparse.Namespace, output: pathlib.Path, train_writer: TransitionShardWriter, val_writer: TransitionShardWriter) -> None:
    rng = random.Random(args.seed)
    total_episodes = int(args.episodes) * len(args.modes)
    val_ids = split_episode_ids(total_episodes, args.val_fraction, rng)
    episode_id = 0
    kwargs = env_kwargs(args)
    metadata_env = make_env(args.modes[0], seed=args.seed, **kwargs)
    env_metadata = state_config_metadata(metadata_env)
    metadata_env.close()

    for mode in args.modes:
        env = make_env(mode, seed=args.seed + RULE_TO_ID[mode], **kwargs)
        for _ in range(int(args.episodes)):
            writer = val_writer if episode_id in val_ids else train_writer
            obs, _ = env.reset()
            for step in range(int(args.steps_per_episode)):
                action = choose_policy_action(args.policy, obs, env, rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                writer.append(
                    obs,
                    action,
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    RULE_TO_ID[mode],
                    event_id(info),
                    episode_id,
                    step,
                    source_id=SOURCE_TO_ID["rollout"],
                )
                obs = next_obs
                if terminated or truncated:
                    break
            episode_id += 1
        env.close()
    write_metadata(
        output,
        {
            "collector": "standard",
            "policy": args.policy,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "env_config": env_metadata,
        },
    )


def collect_counterfactual(args: argparse.Namespace, output: pathlib.Path, train_writer: TransitionShardWriter, val_writer: TransitionShardWriter) -> None:
    rng = random.Random(args.seed)
    val_ids = split_episode_ids(args.episodes, args.val_fraction, rng)
    kwargs = env_kwargs(args)
    base_env = make_env(args.counterfactual_base_mode, seed=args.seed, **kwargs)
    rule_envs = {mode: make_env(mode, seed=args.seed + 100 + RULE_TO_ID[mode], **kwargs) for mode in args.modes}
    env_metadata = state_config_metadata(base_env)

    try:
        for episode_id in range(int(args.episodes)):
            writer = val_writer if episode_id in val_ids else train_writer
            obs, _ = base_env.reset()
            for step in range(int(args.steps_per_episode)):
                action = choose_policy_action(args.policy, obs, base_env, rng)
                base_state = base_env.get_state()
                for mode, env in rule_envs.items():
                    env.set_state(copy_state_for_mode(base_state))
                    state = env.state_to_observation()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    writer.append(
                        state,
                        action,
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        RULE_TO_ID[mode],
                        event_id(info),
                        episode_id,
                        step,
                        source_id=SOURCE_TO_ID["rollout"],
                    )
                obs, _, terminated, truncated, _ = base_env.step(action)
                if terminated or truncated:
                    break
    finally:
        base_env.close()
        for env in rule_envs.values():
            env.close()

    write_metadata(
        output,
        {
            "collector": "counterfactual",
            "policy": args.policy,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "counterfactual_base_mode": args.counterfactual_base_mode,
            "env_config": env_metadata,
        },
    )


def collect_rare(args: argparse.Namespace, output: pathlib.Path, train_writer: TransitionShardWriter, val_writer: TransitionShardWriter) -> None:
    rng = random.Random(args.seed + 99991)
    kwargs = env_kwargs(args)
    base_env = make_env(args.counterfactual_base_mode, seed=args.seed, **kwargs)
    rule_envs = {mode: make_env(mode, seed=args.seed + 300 + RULE_TO_ID[mode], **kwargs) for mode in args.modes}
    source_counts = {source: 0 for source in args.rare_sources}
    event_counts: dict[str, int] = {}
    episode_id = 10_000_000
    try:
        for source in args.rare_sources:
            for sample_idx in range(int(args.rare_samples_per_source)):
                base_state, action = sample_rare_state(source, base_env, rng)
                writer = val_writer if rng.random() < float(args.val_fraction) else train_writer
                modes = args.modes if args.rare_counterfactual else [base_env.mode]
                for mode in modes:
                    env = rule_envs[mode]
                    env.set_state(copy_state_for_mode(base_state))
                    state = env.state_to_observation()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    event_name = str(info.get("event", "none"))
                    event_counts[event_name] = event_counts.get(event_name, 0) + 1
                    writer.append(
                        state,
                        action,
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        RULE_TO_ID[mode],
                        event_id(info),
                        episode_id,
                        sample_idx,
                        source_id=SOURCE_TO_ID[source],
                    )
                source_counts[source] += 1
                episode_id += 1
    finally:
        base_env.close()
        for env in rule_envs.values():
            env.close()
    print(f"rare sources sampled: {source_counts}")
    print(f"rare resulting events: {event_counts}")


def main() -> int:
    args = parse_args()
    output = pathlib.Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_env = make_env(args.modes[0], seed=args.seed, **env_kwargs(args))
    slot_config = slot_config_from_env(metadata_env)
    metadata_env.close()
    train_writer = TransitionShardWriter(output, "train", chunk_size=args.chunk_size, slot_config=slot_config)
    val_writer = TransitionShardWriter(output, "val", chunk_size=args.chunk_size, slot_config=slot_config)

    if args.counterfactual:
        collect_counterfactual(args, output, train_writer, val_writer)
    elif int(args.episodes) > 0:
        collect_standard(args, output, train_writer, val_writer)
    if args.rare_events:
        collect_rare(args, output, train_writer, val_writer)
        metadata_env = make_env(args.modes[0], seed=args.seed, **env_kwargs(args))
        previous_collector = None
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            try:
                import json

                previous_collector = json.loads(metadata_path.read_text()).get("collector")
            except Exception:
                previous_collector = None
        collector_name = f"{previous_collector}+rare_events" if previous_collector else "rare_events"
        write_metadata(
            output,
            {
                "collector": collector_name,
                "previous_collector": previous_collector,
                "policy": args.policy,
                "episodes": args.episodes,
                "steps_per_episode": args.steps_per_episode,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "counterfactual_base_mode": args.counterfactual_base_mode,
                "rare_events": True,
                "rare_sources": args.rare_sources,
                "rare_samples_per_source": args.rare_samples_per_source,
                "rare_counterfactual": args.rare_counterfactual,
                "env_config": state_config_metadata(metadata_env),
            },
        )
        metadata_env.close()
    train_writer.flush()
    val_writer.flush()
    print(f"Saved dataset to {output}")
    print(f"train rows: {count_rows(output, 'train')}")
    print(f"val rows: {count_rows(output, 'val')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
