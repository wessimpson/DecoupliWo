#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pong_common import (
    MODES,
    RULE_TO_ID,
    TransitionShardWriter,
    count_rows,
    event_id,
    make_env,
    slot_config_from_env,
    state_config_metadata,
    write_metadata,
)
from models.rule_conditioned_gnn import PongObjectConstants, PongStateNormalizer


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.actor(h), self.value(h).squeeze(-1)


class RND(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(x)
        pred = self.predictor(x)
        return 0.5 * (pred - target).pow(2).mean(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO+RND on custom Pong and save transition data.")
    parser.add_argument("--output", default="data/transitions/custom_pong/ppo_rnd", help="Transition dataset output root.")
    parser.add_argument("--logdir", default="runs/pong_ppo_rnd")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--rnd-scale", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rnd-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--gravity", type=float, default=420.0)
    parser.add_argument("--paddle-speed", type=float, default=360.0)
    parser.add_argument("--ball-speed", type=float, default=280.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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


def one_hot_rule(rule_ids: torch.Tensor) -> torch.Tensor:
    return F.one_hot(rule_ids.to(torch.int64), num_classes=len(MODES)).to(torch.float32)


def make_input(obs: np.ndarray | torch.Tensor, rule_ids: np.ndarray | torch.Tensor, normalizer: PongStateNormalizer, device: torch.device) -> torch.Tensor:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    rule_t = torch.as_tensor(rule_ids, dtype=torch.long, device=device)
    return torch.cat([normalizer.normalize(obs_t), one_hot_rule(rule_t)], dim=-1)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    kwargs = env_kwargs(args)

    output = pathlib.Path(args.output).expanduser().resolve()
    logdir = pathlib.Path(args.logdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    envs = [make_env(mode, seed=args.seed + RULE_TO_ID[mode], **kwargs) for mode in args.modes]
    slot_config = slot_config_from_env(envs[0])
    train_writer = TransitionShardWriter(output, "train", args.chunk_size, slot_config=slot_config)
    val_writer = TransitionShardWriter(output, "val", args.chunk_size, slot_config=slot_config)
    rule_ids_np = np.asarray([RULE_TO_ID[env.mode] for env in envs], dtype=np.int64)
    constants = PongObjectConstants(
        width=float(args.width),
        height=float(args.height),
        paddle_speed=float(args.paddle_speed),
        max_ball_speed=float(envs[0].config.max_ball_speed),
    )
    normalizer = PongStateNormalizer(constants).to(device)
    input_dim = 6 + len(MODES)
    agent = ActorCritic(input_dim).to(device)
    rnd = RND(input_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=args.rnd_lr)

    obs = []
    episode_ids = []
    episode_steps = []
    split_is_val = []
    next_episode_id = 0
    for env in envs:
        initial_obs, _ = env.reset()
        obs.append(initial_obs)
        episode_ids.append(next_episode_id)
        episode_steps.append(0)
        split_is_val.append(rng.random() < args.val_fraction)
        next_episode_id += 1
    obs_np = np.stack(obs).astype(np.float32)
    episode_ids_np = np.asarray(episode_ids, dtype=np.int64)
    episode_steps_np = np.asarray(episode_steps, dtype=np.int64)
    split_is_val_np = np.asarray(split_is_val, dtype=np.bool_)
    global_step = 0
    metrics_path = logdir / "metrics.jsonl"

    write_metadata(
        output,
        {
            "collector": "ppo_rnd",
            "total_steps": args.total_steps,
            "rollout_steps": args.rollout_steps,
            "rnd_scale": args.rnd_scale,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "env_config": state_config_metadata(envs[0]),
        },
    )

    while global_step < int(args.total_steps):
        rollout = {key: [] for key in ("obs", "rule", "action", "logprob", "value", "reward", "ext_reward", "intr_reward", "done", "next_obs", "event")}
        for _ in range(int(args.rollout_steps)):
            inp = make_input(obs_np, rule_ids_np, normalizer, device)
            with torch.no_grad():
                logits, value = agent(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                intr_reward = rnd.error(inp)

            next_obs = []
            ext_rewards = []
            dones = []
            events = []
            for env_idx, env in enumerate(envs):
                action_i = int(action[env_idx].item())
                state_i = obs_np[env_idx].copy()
                next_i, reward_i, terminated, truncated, info = env.step(action_i)
                writer = val_writer if split_is_val_np[env_idx] else train_writer
                writer.append(
                    state_i,
                    action_i,
                    next_i,
                    reward_i,
                    terminated,
                    truncated,
                    int(rule_ids_np[env_idx]),
                    event_id(info),
                    int(episode_ids_np[env_idx]),
                    int(episode_steps_np[env_idx]),
                )
                next_obs.append(next_i)
                ext_rewards.append(float(reward_i))
                done_i = bool(terminated or truncated)
                dones.append(done_i)
                events.append(event_id(info))
                episode_steps_np[env_idx] += 1
                if done_i:
                    reset_obs, _ = env.reset()
                    next_obs[-1] = reset_obs
                    episode_ids_np[env_idx] = next_episode_id
                    episode_steps_np[env_idx] = 0
                    split_is_val_np[env_idx] = rng.random() < args.val_fraction
                    next_episode_id += 1

            next_obs_np = np.stack(next_obs).astype(np.float32)
            reward_np = np.asarray(ext_rewards, dtype=np.float32) + float(args.rnd_scale) * intr_reward.detach().cpu().numpy().astype(np.float32)
            done_np = np.asarray(dones, dtype=np.float32)
            rollout["obs"].append(obs_np.copy())
            rollout["rule"].append(rule_ids_np.copy())
            rollout["action"].append(action.detach().cpu().numpy())
            rollout["logprob"].append(logprob.detach().cpu().numpy())
            rollout["value"].append(value.detach().cpu().numpy())
            rollout["reward"].append(reward_np)
            rollout["ext_reward"].append(np.asarray(ext_rewards, dtype=np.float32))
            rollout["intr_reward"].append(intr_reward.detach().cpu().numpy().astype(np.float32))
            rollout["done"].append(done_np)
            rollout["next_obs"].append(next_obs_np.copy())
            rollout["event"].append(np.asarray(events, dtype=np.int64))
            obs_np = next_obs_np
            global_step += len(envs)
            if global_step >= int(args.total_steps):
                break

        obs_arr = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=device)
        rule_arr = torch.as_tensor(np.asarray(rollout["rule"]), dtype=torch.long, device=device)
        action_arr = torch.as_tensor(np.asarray(rollout["action"]), dtype=torch.long, device=device)
        old_logprob = torch.as_tensor(np.asarray(rollout["logprob"]), dtype=torch.float32, device=device)
        value_arr = torch.as_tensor(np.asarray(rollout["value"]), dtype=torch.float32, device=device)
        reward_arr = torch.as_tensor(np.asarray(rollout["reward"]), dtype=torch.float32, device=device)
        done_arr = torch.as_tensor(np.asarray(rollout["done"]), dtype=torch.float32, device=device)
        T, N = reward_arr.shape

        with torch.no_grad():
            next_inp = make_input(obs_np, rule_ids_np, normalizer, device)
            _, next_value = agent(next_inp)
            advantages = torch.zeros_like(reward_arr)
            lastgaelam = torch.zeros(N, dtype=torch.float32, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_nonterminal = 1.0 - done_arr[t]
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - done_arr[t + 1]
                    next_values = value_arr[t + 1]
                delta = reward_arr[t] + args.gamma * next_values * next_nonterminal - value_arr[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_arr

        flat_obs = obs_arr.reshape(T * N, 6)
        flat_rule = rule_arr.reshape(T * N)
        flat_action = action_arr.reshape(T * N)
        flat_old_logprob = old_logprob.reshape(T * N)
        flat_adv = advantages.reshape(T * N)
        flat_returns = returns.reshape(T * N)
        flat_values = value_arr.reshape(T * N)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
        batch_size = T * N
        indices = torch.arange(batch_size, device=device)
        policy_losses = []
        value_losses = []
        entropies = []
        rnd_losses = []
        for _ in range(int(args.epochs)):
            perm = indices[torch.randperm(batch_size, device=device)]
            for start in range(0, batch_size, int(args.minibatch_size)):
                mb = perm[start : start + int(args.minibatch_size)]
                mb_inp = make_input(flat_obs[mb], flat_rule[mb], normalizer, device)
                logits, new_value = agent(mb_inp)
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(flat_action[mb])
                ratio = (new_logprob - flat_old_logprob[mb]).exp()
                pg_loss1 = -flat_adv[mb] * ratio
                pg_loss2 = -flat_adv[mb] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = F.mse_loss(new_value, flat_returns[mb])
                entropy = dist.entropy().mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                rnd_loss = rnd.error(mb_inp.detach()).mean()
                rnd_optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()
                rnd_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                rnd_losses.append(float(rnd_loss.item()))

        metrics = {
            "step": int(global_step),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "rnd_loss": float(np.mean(rnd_losses)),
            "reward_mean": float(reward_arr.mean().item()),
            "ext_reward_mean": float(np.asarray(rollout["ext_reward"]).mean()),
            "intr_reward_mean": float(np.asarray(rollout["intr_reward"]).mean()),
            "episodes": int(next_episode_id),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"[{global_step}] policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.3f} rnd={metrics['rnd_loss']:.4f}"
        )
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "rnd_state_dict": rnd.state_dict(),
                "args": vars(args),
                "step": int(global_step),
            },
            logdir / "latest.pt",
        )

    train_writer.flush()
    val_writer.flush()
    for env in envs:
        env.close()
    print(f"Saved PPO/RND transitions to {output}")
    print(f"train rows: {count_rows(output, 'train')}")
    print(f"val rows: {count_rows(output, 'val')}")
    print(f"logdir: {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
