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

from custom_breakout import BreakoutEnv
from data.pong_common import (
    GAME_TO_ID,
    GAMES,
    MODES,
    RULE_TO_ID,
    SOURCE_TO_ID,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect shared-slot transitions for Pong and Breakout-lite.")
    parser.add_argument("--output", default="data/transitions/editable_world/pong_breakout_counterfactual_seed0")
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--policy", choices=("random", "heuristic", "mixed"), default="mixed")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per game.")
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--counterfactual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--counterfactual-base-mode", choices=MODES, default="normal")
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
                writer = val_writer if local_episode in local_val_ids else train_writer
                obs, _ = env.reset()
                for step in range(int(args.steps_per_episode)):
                    action = choose_action(game, args.policy, obs, env, rng)
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
            writer = val_writer if local_episode in val_ids else train_writer
            obs, _ = base_env.reset()
            for step in range(int(args.steps_per_episode)):
                action = choose_action(game, args.policy, obs, base_env, rng)
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
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "counterfactual": args.counterfactual,
            "counterfactual_base_mode": args.counterfactual_base_mode,
            "env_config": metadata,
        },
    )
    print(f"Saved editable-world dataset to {output}")
    print(f"train rows: {count_rows(output, 'train')}")
    print(f"val rows: {count_rows(output, 'val')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
