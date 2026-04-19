#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_breakout import BallState as BreakoutBallState
from custom_breakout import BreakoutEnv, BreakoutState
from custom_breakout import PaddleState as BreakoutPaddleState
from data.collect_pong_transitions import PONG_RARE_SOURCES, sample_rare_state
from data.pong_common import (
    GAME_TO_ID,
    GAMES,
    MODES,
    RULE_TO_ID,
    SOURCE_TO_ID,
    SOURCES,
    TransitionShardWriter,
    choose_policy_action,
    copy_state_for_mode,
    count_rows,
    event_id,
    flat_pong_state_to_slots,
    make_env,
    slot_config_from_env,
    split_episode_ids,
    state_config_metadata,
    write_metadata,
)

BREAKOUT_RARE_SOURCES = (
    "diverse",
    "left_wall",
    "right_wall",
    "top_bounce",
    "wrapped_left",
    "wrapped_right",
    "paddle_hit",
    "block_hit",
    "miss",
)
ALL_POLICIES = ("random", "heuristic", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect shared-slot transitions for Pong and Breakout-lite.")
    parser.add_argument("--output", default="data/transitions/editable_world/pong_breakout_counterfactual_seed0")
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--policy", choices=("random", "heuristic", "mixed", "all"), default="mixed")
    parser.add_argument("--policies", nargs="+", choices=ALL_POLICIES, default=None, help="Cycle these policies by episode; overrides --policy.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per game.")
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--counterfactual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--counterfactual-base-mode", choices=MODES, default="normal")
    parser.add_argument("--rare-events", action="store_true", help="Append targeted rare/diverse states after rollout collection.")
    parser.add_argument("--rare-samples-per-source", type=int, default=2000)
    parser.add_argument("--rare-sources", nargs="+", choices=SOURCES[1:], default=None, help="Rare sources to sample. Defaults to all supported sources per game.")
    parser.add_argument("--rare-counterfactual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--gravity", type=float, default=420.0)
    parser.add_argument("--paddle-speed", type=float, default=360.0)
    parser.add_argument("--ball-speed", type=float, default=280.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "width": args.width,
        "height": args.height,
        "dt": args.dt,
        "gravity": args.gravity,
        "paddle_speed": args.paddle_speed,
        "ball_speed": args.ball_speed,
        "max_steps": args.max_steps,
    }


def make_game_env(game: str, mode: str, seed: int | None, kwargs: dict[str, Any]):
    if game == "pong":
        return make_env(mode, seed=seed, **kwargs)
    if game == "breakout":
        return BreakoutEnv(mode=mode, render_mode=None, seed=seed, **kwargs)
    raise ValueError(f"Unknown game: {game}")


def active_policies(args: argparse.Namespace) -> tuple[str, ...]:
    if args.policies:
        return tuple(args.policies)
    if args.policy == "all":
        return ALL_POLICIES
    return (args.policy,)


def policy_for_episode(args: argparse.Namespace, episode_index: int) -> str:
    policies = active_policies(args)
    return policies[int(episode_index) % len(policies)]


def clone_for_mode(game: str, state):
    if game == "pong":
        return copy_state_for_mode(state)
    copied = state.copy()
    copied.terminated = False
    copied.truncated = False
    return copied


def state_slots(game: str, env, state=None) -> tuple[np.ndarray, np.ndarray]:
    if game == "pong":
        obs = env.state_to_observation(state) if state is not None else env.state_to_observation()
        return flat_pong_state_to_slots(obs, slot_config_from_env(env))
    return env.state_to_slots(state)


def heuristic_breakout_action(obs: np.ndarray, env: BreakoutEnv, rng: random.Random, epsilon: float = 0.05) -> int:
    if rng.random() < epsilon:
        return rng.choice((0, 1, 2))
    ball_x = float(obs[0])
    paddle_x = float(obs[4])
    center = paddle_x + env.config.paddle_width / 2.0
    margin = max(3.0, env.config.paddle_width * 0.06)
    if ball_x < center - margin:
        return 1
    if ball_x > center + margin:
        return 2
    return 0


def choose_action(game: str, policy: str, obs: np.ndarray, env, rng: random.Random) -> int:
    if game == "pong":
        return choose_policy_action(policy, obs, env, rng)
    if policy == "random":
        return rng.choice((0, 1, 2))
    if policy == "heuristic":
        return heuristic_breakout_action(obs, env, rng, epsilon=0.02)
    if policy == "mixed":
        if rng.random() < 0.5:
            return rng.choice((0, 1, 2))
        return heuristic_breakout_action(obs, env, rng, epsilon=0.08)
    raise ValueError(f"Unknown policy: {policy}")


def supported_rare_sources(game: str) -> tuple[str, ...]:
    if game == "pong":
        return PONG_RARE_SOURCES
    if game == "breakout":
        return BREAKOUT_RARE_SOURCES
    raise ValueError(f"Unknown game: {game}")


def rare_sources_for_game(args: argparse.Namespace, game: str) -> tuple[str, ...]:
    supported = supported_rare_sources(game)
    if args.rare_sources is None:
        return supported
    return tuple(source for source in args.rare_sources if source in supported)


def _uniform_signed_speed(rng: random.Random, low: float, high: float, sign: int | None = None) -> float:
    value = rng.uniform(float(low), float(high))
    if sign is None:
        sign = -1 if rng.random() < 0.5 else 1
    return float(sign) * value


def _make_breakout_state(
    env: BreakoutEnv,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    paddle_x: float,
    paddle_vx: float = 0.0,
) -> BreakoutState:
    cfg = env.config
    template = env.get_state()
    return BreakoutState(
        ball=BreakoutBallState(
            x=float(ball_x),
            y=float(ball_y),
            vx=float(ball_vx),
            vy=float(ball_vy),
            radius=cfg.ball_radius,
        ),
        paddle=BreakoutPaddleState(
            x=float(paddle_x),
            y=cfg.height - cfg.paddle_margin - cfg.paddle_height,
            width=cfg.paddle_width,
            height=cfg.paddle_height,
            vx=float(paddle_vx),
        ),
        blocks=[block.copy() for block in template.blocks],
        last_event="rare_start",
    )


def sample_breakout_rare_state(source: str, env: BreakoutEnv, rng: random.Random) -> tuple[BreakoutState, int]:
    cfg = env.config
    if env._state is None:
        env.reset()
    max_speed = min(float(cfg.max_ball_speed), max(float(cfg.ball_speed) * 2.0, 360.0))
    min_speed = max(float(cfg.min_ball_speed), 140.0)
    paddle_x = rng.uniform(0.0, cfg.width - cfg.paddle_width)
    action = rng.choice((0, 1, 2))

    if source == "diverse":
        return _make_breakout_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed),
            paddle_x=paddle_x,
            paddle_vx=rng.choice((-cfg.paddle_speed, 0.0, cfg.paddle_speed)),
        ), action

    if source == "left_wall":
        return _make_breakout_state(
            env,
            ball_x=cfg.ball_radius + rng.uniform(0.0, 2.0),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            ball_vy=rng.uniform(-0.5 * max_speed, 0.5 * max_speed),
            paddle_x=paddle_x,
        ), action

    if source == "right_wall":
        return _make_breakout_state(
            env,
            ball_x=cfg.width - cfg.ball_radius - rng.uniform(0.0, 2.0),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed, sign=1),
            ball_vy=rng.uniform(-0.5 * max_speed, 0.5 * max_speed),
            paddle_x=paddle_x,
        ), action

    if source == "top_bounce":
        return _make_breakout_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=cfg.ball_radius + rng.uniform(0.0, 2.0),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed),
            ball_vy=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            paddle_x=paddle_x,
        ), action

    if source == "wrapped_left":
        return _make_breakout_state(
            env,
            ball_x=-cfg.ball_radius + rng.uniform(-3.0, 1.0),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed, sign=-1),
            ball_vy=rng.uniform(-0.5 * max_speed, 0.5 * max_speed),
            paddle_x=paddle_x,
        ), action

    if source == "wrapped_right":
        return _make_breakout_state(
            env,
            ball_x=cfg.width + cfg.ball_radius + rng.uniform(-1.0, 3.0),
            ball_y=rng.uniform(cfg.ball_radius, cfg.height - cfg.ball_radius),
            ball_vx=_uniform_signed_speed(rng, min_speed, max_speed, sign=1),
            ball_vy=rng.uniform(-0.5 * max_speed, 0.5 * max_speed),
            paddle_x=paddle_x,
        ), action

    if source == "paddle_hit":
        paddle_x = rng.uniform(0.0, cfg.width - cfg.paddle_width)
        hit_x = paddle_x + rng.uniform(0.15, 0.85) * cfg.paddle_width
        speed = rng.uniform(min_speed, max_speed)
        return _make_breakout_state(
            env,
            ball_x=hit_x,
            ball_y=cfg.height - cfg.paddle_margin - cfg.paddle_height - cfg.ball_radius - speed * cfg.dt * rng.uniform(0.25, 0.75),
            ball_vx=rng.uniform(-0.2 * max_speed, 0.2 * max_speed),
            ball_vy=speed,
            paddle_x=paddle_x,
        ), action

    if source == "block_hit":
        template = env.get_state()
        block = rng.choice(template.blocks)
        speed = rng.uniform(min_speed, max_speed)
        state = _make_breakout_state(
            env,
            ball_x=block.x + rng.uniform(0.2, 0.8) * block.width,
            ball_y=block.y + block.height + cfg.ball_radius + speed * cfg.dt * rng.uniform(0.1, 0.7),
            ball_vx=rng.uniform(-0.15 * max_speed, 0.15 * max_speed),
            ball_vy=-speed,
            paddle_x=paddle_x,
        )
        return state, action

    if source == "miss":
        speed = rng.uniform(min_speed, max_speed)
        return _make_breakout_state(
            env,
            ball_x=rng.uniform(cfg.ball_radius, cfg.width - cfg.ball_radius),
            ball_y=cfg.height + cfg.ball_radius - speed * cfg.dt * rng.uniform(0.1, 0.7),
            ball_vx=rng.uniform(-0.25 * max_speed, 0.25 * max_speed),
            ball_vy=speed,
            paddle_x=paddle_x,
        ), action

    raise ValueError(f"Unsupported Breakout rare source: {source}")


def collect_standard_for_game(
    args: argparse.Namespace,
    game: str,
    train_writer: TransitionShardWriter,
    val_writer: TransitionShardWriter,
    start_episode_id: int,
    rng: random.Random,
) -> int:
    kwargs = env_kwargs(args)
    total_episodes = int(args.episodes) * len(args.modes)
    local_val_ids = split_episode_ids(total_episodes, args.val_fraction, rng)
    local_episode = 0
    episode_id = start_episode_id
    for mode in args.modes:
        env = make_game_env(game, mode, args.seed + 10_000 * GAME_TO_ID[game] + RULE_TO_ID[mode], kwargs)
        try:
            for _ in range(int(args.episodes)):
                policy = policy_for_episode(args, local_episode)
                writer = val_writer if local_episode in local_val_ids else train_writer
                obs, _ = env.reset()
                for step in range(int(args.steps_per_episode)):
                    action = choose_action(game, policy, obs, env, rng)
                    slots, mask = state_slots(game, env)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_slots, next_mask = state_slots(game, env)
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
                        game_id=GAME_TO_ID[game],
                        object_slots=slots,
                        next_object_slots=next_slots,
                        object_mask=mask,
                        next_object_mask=next_mask,
                    )
                    obs = next_obs
                    if terminated or truncated:
                        break
                local_episode += 1
                episode_id += 1
        finally:
            env.close()
    return episode_id


def collect_counterfactual_for_game(
    args: argparse.Namespace,
    game: str,
    train_writer: TransitionShardWriter,
    val_writer: TransitionShardWriter,
    start_episode_id: int,
    rng: random.Random,
) -> int:
    kwargs = env_kwargs(args)
    val_ids = split_episode_ids(args.episodes, args.val_fraction, rng)
    base_env = make_game_env(game, args.counterfactual_base_mode, args.seed + 20_000 * GAME_TO_ID[game], kwargs)
    rule_envs = {
        mode: make_game_env(game, mode, args.seed + 20_000 * GAME_TO_ID[game] + 100 + RULE_TO_ID[mode], kwargs)
        for mode in args.modes
    }
    episode_id = start_episode_id
    try:
        for local_episode in range(int(args.episodes)):
            policy = policy_for_episode(args, local_episode)
            writer = val_writer if local_episode in val_ids else train_writer
            obs, _ = base_env.reset()
            for step in range(int(args.steps_per_episode)):
                action = choose_action(game, policy, obs, base_env, rng)
                base_state = base_env.get_state()
                for mode, env in rule_envs.items():
                    env.set_state(clone_for_mode(game, base_state))
                    state_obs = env.state_to_observation()
                    slots, mask = state_slots(game, env)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_slots, next_mask = state_slots(game, env)
                    writer.append(
                        state_obs,
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
                        game_id=GAME_TO_ID[game],
                        object_slots=slots,
                        next_object_slots=next_slots,
                        object_mask=mask,
                        next_object_mask=next_mask,
                    )
                obs, _, terminated, truncated, _ = base_env.step(action)
                if terminated or truncated:
                    break
            episode_id += 1
    finally:
        base_env.close()
        for env in rule_envs.values():
            env.close()
    return episode_id


def collect_rare_for_game(
    args: argparse.Namespace,
    game: str,
    train_writer: TransitionShardWriter,
    val_writer: TransitionShardWriter,
    start_episode_id: int,
    rng: random.Random,
) -> int:
    kwargs = env_kwargs(args)
    base_env = make_game_env(game, args.counterfactual_base_mode, args.seed + 30_000 * GAME_TO_ID[game], kwargs)
    base_env.reset()
    rule_envs = {
        mode: make_game_env(game, mode, args.seed + 30_000 * GAME_TO_ID[game] + 100 + RULE_TO_ID[mode], kwargs)
        for mode in args.modes
    }
    for env in rule_envs.values():
        env.reset()

    sources = rare_sources_for_game(args, game)
    source_counts = {source: 0 for source in sources}
    event_counts: dict[str, int] = {}
    episode_id = start_episode_id
    try:
        for source in sources:
            for sample_idx in range(int(args.rare_samples_per_source)):
                if game == "pong":
                    base_state, action = sample_rare_state(source, base_env, rng)
                else:
                    base_state, action = sample_breakout_rare_state(source, base_env, rng)
                writer = val_writer if rng.random() < float(args.val_fraction) else train_writer
                modes = args.modes if args.rare_counterfactual else [args.counterfactual_base_mode]
                for mode in modes:
                    env = rule_envs[mode]
                    env.set_state(clone_for_mode(game, base_state))
                    state_obs = env.state_to_observation()
                    slots, mask = state_slots(game, env)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_slots, next_mask = state_slots(game, env)
                    event_name = str(info.get("event", "none"))
                    event_counts[event_name] = event_counts.get(event_name, 0) + 1
                    writer.append(
                        state_obs,
                        action,
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        RULE_TO_ID[mode],
                        event_id(info),
                        episode_id,
                        sample_idx,
                        source_id=SOURCE_TO_ID[source],
                        game_id=GAME_TO_ID[game],
                        object_slots=slots,
                        next_object_slots=next_slots,
                        object_mask=mask,
                        next_object_mask=next_mask,
                    )
                source_counts[source] += 1
                episode_id += 1
    finally:
        base_env.close()
        for env in rule_envs.values():
            env.close()
    print(f"{game} rare sources sampled: {source_counts}")
    print(f"{game} rare resulting events: {event_counts}")
    return episode_id


def main() -> int:
    args = parse_args()
    output = pathlib.Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_writer = TransitionShardWriter(output, "train", chunk_size=args.chunk_size)
    val_writer = TransitionShardWriter(output, "val", chunk_size=args.chunk_size)
    episode_id = 0
    for game in args.games:
        if args.counterfactual:
            episode_id = collect_counterfactual_for_game(args, game, train_writer, val_writer, episode_id, rng)
        else:
            episode_id = collect_standard_for_game(args, game, train_writer, val_writer, episode_id, rng)
        if args.rare_events:
            episode_id = collect_rare_for_game(args, game, train_writer, val_writer, episode_id, rng)
    train_writer.flush()
    val_writer.flush()

    metadata_env = make_game_env(args.games[0], args.modes[0], args.seed, env_kwargs(args))
    try:
        metadata = state_config_metadata(metadata_env)
    finally:
        metadata_env.close()
    write_metadata(
        output,
        {
            "collector": "editable_world_counterfactual" if args.counterfactual else "editable_world_standard",
            "games_collected": args.games,
            "modes_collected": args.modes,
            "policy": args.policy,
            "policies": list(active_policies(args)),
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "counterfactual": args.counterfactual,
            "counterfactual_base_mode": args.counterfactual_base_mode,
            "rare_events": args.rare_events,
            "rare_sources": args.rare_sources,
            "rare_samples_per_source": args.rare_samples_per_source,
            "rare_counterfactual": args.rare_counterfactual,
            "env_config": metadata,
        },
    )
    print(f"Saved editable-world dataset to {output}")
    print(f"train rows: {count_rows(output, 'train')}")
    print(f"val rows: {count_rows(output, 'val')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
