#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pong_common import MODES, RULE_TO_ID, choose_policy_action, copy_state_for_mode, make_env
from models.rule_conditioned_gnn import PongObjectConstants, RuleConditionedPongGNN
from train_pong_world_model import PongTransitionDataset, batch_to_device, evaluate, parse_combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained custom Pong rule-conditioned world model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, help="Optional dataset root for one-step validation metrics.")
    parser.add_argument("--holdout-combos", nargs="*", default=None, help="Optional val combos to report separately, e.g. pong:teleport.")
    parser.add_argument("--rule-ablation", choices=("none", "zero", "shuffle"), default=None, help="Override checkpoint rule ablation when loading dataset.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rollout-episodes", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=100)
    parser.add_argument("--policy", choices=("random", "heuristic", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(checkpoint: dict, device: torch.device) -> RuleConditionedPongGNN:
    args = checkpoint.get("args", {})
    constants = PongObjectConstants(**checkpoint.get("constants", {}))
    model = RuleConditionedPongGNN(
        latent_dim=int(args.get("latent_dim", 64)),
        rule_dim=int(args.get("rule_dim", 16)),
        type_dim=int(args.get("type_dim", 8)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        message_passing_steps=int(args.get("message_passing_steps", 2)),
        constants=constants,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def effective_rule_id(rule_id: int, rule_ablation: str) -> int:
    if rule_ablation == "zero":
        return 0
    if rule_ablation == "shuffle":
        return (int(rule_id) + 1) % len(MODES)
    return int(rule_id)


@torch.no_grad()
def predict_next(model: RuleConditionedPongGNN, state: np.ndarray, action: int, rule_id: int, device: torch.device) -> np.ndarray:
    batch_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    batch_action = torch.as_tensor([action], dtype=torch.long, device=device)
    batch_rule = torch.as_tensor([rule_id], dtype=torch.long, device=device)
    out = model(batch_state, batch_action, batch_rule, normalized=False)
    return out["pred_next"][0].detach().cpu().numpy()


def rollout_eval(model: RuleConditionedPongGNN, device: torch.device, episodes: int, horizon: int, policy: str, seed: int, rule_ablation: str) -> dict[str, float]:
    rng = random.Random(seed)
    metrics: dict[str, float] = {}
    for mode in MODES:
        env = make_env(mode, seed=seed + RULE_TO_ID[mode])
        per_step_errors: list[list[float]] = [[] for _ in range(horizon)]
        for _ in range(int(episodes)):
            obs, _ = env.reset()
            pred_state = obs.copy()
            for step in range(int(horizon)):
                action = choose_policy_action(policy, obs, env, rng)
                true_next, _, terminated, truncated, _ = env.step(action)
                pred_state = predict_next(model, pred_state, action, effective_rule_id(RULE_TO_ID[mode], rule_ablation), device)
                per_step_errors[step].append(float(np.mean((pred_state - true_next) ** 2)))
                obs = true_next
                if terminated or truncated:
                    break
        env.close()
        all_errors = [value for values in per_step_errors for value in values]
        metrics[f"rollout_mse/{mode}"] = float(np.mean(all_errors)) if all_errors else float("nan")
        for step in (0, 4, 19, horizon - 1):
            if 0 <= step < horizon and per_step_errors[step]:
                metrics[f"rollout_mse/{mode}/t{step + 1}"] = float(np.mean(per_step_errors[step]))
    return metrics


def counterfactual_eval(model: RuleConditionedPongGNN, device: torch.device, samples: int, policy: str, seed: int, rule_ablation: str) -> dict[str, float]:
    rng = random.Random(seed)
    base_env = make_env("normal", seed=seed)
    rule_envs = {mode: make_env(mode, seed=seed + 100 + RULE_TO_ID[mode]) for mode in MODES}
    errors = {mode: [] for mode in MODES}
    true_variance = []
    pred_variance = []
    try:
        obs, _ = base_env.reset()
        for _ in range(int(samples)):
            if base_env.is_done:
                obs, _ = base_env.reset()
            action = choose_policy_action(policy, obs, base_env, rng)
            base_state = base_env.get_state()
            true_nexts = []
            pred_nexts = []
            state_obs = None
            for mode, env in rule_envs.items():
                env.set_state(copy_state_for_mode(base_state))
                state_obs = env.state_to_observation()
                true_next, _, _, _, _ = env.step(action)
                pred_next = predict_next(model, state_obs, action, effective_rule_id(RULE_TO_ID[mode], rule_ablation), device)
                true_nexts.append(true_next)
                pred_nexts.append(pred_next)
                errors[mode].append(float(np.mean((pred_next - true_next) ** 2)))
            true_variance.append(float(np.mean(np.var(np.stack(true_nexts), axis=0))))
            pred_variance.append(float(np.mean(np.var(np.stack(pred_nexts), axis=0))))
            obs, _, terminated, truncated, _ = base_env.step(action)
            if terminated or truncated:
                obs, _ = base_env.reset()
    finally:
        base_env.close()
        for env in rule_envs.values():
            env.close()
    metrics = {f"counterfactual_mse/{mode}": float(np.mean(values)) for mode, values in errors.items()}
    metrics["counterfactual_true_rule_variance"] = float(np.mean(true_variance))
    metrics["counterfactual_pred_rule_variance"] = float(np.mean(pred_variance))
    metrics["counterfactual_rule_variance_ratio"] = metrics["counterfactual_pred_rule_variance"] / max(metrics["counterfactual_true_rule_variance"], 1e-8)
    return metrics


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint, device)
    rule_ablation = args.rule_ablation or checkpoint.get("args", {}).get("rule_ablation", "none")

    metrics: dict[str, float | str] = {"checkpoint_epoch": float(checkpoint.get("epoch", -1)), "rule_ablation": rule_ablation}
    if args.dataset:
        dataset_root = pathlib.Path(args.dataset).expanduser().resolve()
        val_data = PongTransitionDataset(dataset_root, "val", rule_ablation=rule_ablation, seed=args.seed)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        metrics.update(evaluate(model, val_loader, device))
        holdout_combos = parse_combos(args.holdout_combos)
        if holdout_combos:
            holdout_data = PongTransitionDataset(dataset_root, "val", combos=holdout_combos, rule_ablation=rule_ablation, seed=args.seed)
            holdout_loader = DataLoader(holdout_data, batch_size=args.batch_size, shuffle=False)
            metrics.update({f"holdout/{key}": value for key, value in evaluate(model, holdout_loader, device).items()})
    metrics.update(rollout_eval(model, device, args.rollout_episodes, args.rollout_horizon, args.policy, args.seed, rule_ablation))
    metrics.update(counterfactual_eval(model, device, args.rollout_episodes * args.rollout_horizon, args.policy, args.seed, rule_ablation))

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        output = pathlib.Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Saved metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
