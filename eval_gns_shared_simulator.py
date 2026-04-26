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

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_breakout import BreakoutEnv
from data.collect_editable_world_transitions import choose_action, clone_for_mode, env_kwargs, make_game_env
from data.gns_shared_dataset import GNSTrajectoryWindowDataset, batch_to_device
from data.pong_common import GAME_TO_ID, GAMES, ID_TO_EVENT, ID_TO_GAME, ID_TO_RULE, MODES, RULE_TO_ID
from gns_shared_rollout import init_history, predict_next_from_history, slots_to_flat_state
from models.gns_shared_simulator import build_gns_model_from_checkpoint
from train_gns_shared_simulator import parse_combos, transition_weights


def progress(iterable, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the shared GNS simulator for Pong + Breakout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--holdout-combos", nargs="*", default=None)
    parser.add_argument("--rule-ablation", choices=("none", "zero", "shuffle"), default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rollout-episodes", type=int, default=16)
    parser.add_argument("--rollout-horizon", type=int, default=100)
    parser.add_argument("--policy", choices=("random", "heuristic", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--eval-modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate_dataset(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals = {"val_ball_mse": 0.0, "val_mask_bce": 0.0}
    rows = 0
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}

    def add_group(name: str, values: np.ndarray) -> None:
        group_sums[name] = group_sums.get(name, 0.0) + float(values.sum())
        group_counts[name] = group_counts.get(name, 0) + int(values.size)

    model.eval()
    batch_iter = progress(loader, desc="eval-dataset", total=len(loader))
    for batch in batch_iter:
        batch = batch_to_device(batch, device)
        out = model(
            batch["history_slots"],
            batch["history_mask"],
            batch["action"],
            batch["rule_id"],
            batch["game_id"],
            dynamic_pos_mask=batch["dynamic_pos_mask"],
            training_noise=False,
        )
        ball_mask = batch["dynamic_pos_mask"].to(torch.float32)
        pred_pos = out["pred_next_slots"][..., :2]
        target_pos = batch["target_next_slots"][..., :2]
        ball_mse = (((pred_pos - target_pos) ** 2).mean(dim=-1) * ball_mask).sum(dim=1) / ball_mask.sum(dim=1).clamp_min(1.0)
        block_mask = batch["block_mask"].to(torch.float32)
        mask_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            out["pred_next_mask_logits"],
            batch["target_next_mask"].to(torch.float32),
            reduction="none",
        )
        mask_bce = (mask_bce * block_mask).sum(dim=1) / block_mask.sum(dim=1).clamp_min(1.0)
        batch_size = int(batch["action"].shape[0])
        rows += batch_size
        totals["val_ball_mse"] += float(ball_mse.mean().item()) * batch_size
        totals["val_mask_bce"] += float(mask_bce.mean().item()) * batch_size
        values = ball_mse.detach().cpu().numpy()
        games = batch["game_id"].detach().cpu().numpy()
        rules = batch["true_rule_id"].detach().cpu().numpy()
        events = batch["event_id"].detach().cpu().numpy()
        for gid in np.unique(games):
            add_group(f"val_ball_mse/game_{ID_TO_GAME[int(gid)]}", values[games == gid])
        for rid in np.unique(rules):
            add_group(f"val_ball_mse/rule_{ID_TO_RULE[int(rid)]}", values[rules == rid])
        for eid in np.unique(events):
            add_group(f"val_ball_mse/event_{ID_TO_EVENT[int(eid)]}", values[events == eid])
        if tqdm is not None:
            batch_iter.set_postfix(ball=f"{float(ball_mse.mean().item()):.4f}", mask=f"{float(mask_bce.mean().item()):.4f}")
    result = {key: value / max(rows, 1) for key, value in totals.items()}
    result["val_ball_rmse"] = float(np.sqrt(max(result["val_ball_mse"], 0.0)))
    for name, total in group_sums.items():
        result[name] = total / max(group_counts[name], 1)
    return result


def rollout_eval(model, device: torch.device, episodes: int, horizon: int, policy: str, seed: int, games: list[str], modes: list[str]) -> dict[str, float]:
    rng = random.Random(seed)
    metrics: dict[str, float] = {}
    kwargs = env_kwargs(argparse.Namespace(width=640, height=480, dt=1.0 / 60.0, gravity=420.0, paddle_speed=360.0, ball_speed=280.0, max_steps=3000))
    for game in games:
        for mode in modes:
            env = make_game_env(game, mode, seed + 1000 * GAME_TO_ID[game] + RULE_TO_ID[mode], kwargs)
            per_step_errors = [[] for _ in range(horizon)]
            try:
                ep_iter = progress(range(int(episodes)), desc=f"rollout {game}:{mode}", total=int(episodes))
                for _ in ep_iter:
                    obs, _ = env.reset()
                    init_slots, init_mask = env.state_to_slots() if game == "breakout" else __import__("data.pong_common", fromlist=["flat_pong_state_to_slots"]).flat_pong_state_to_slots(obs)
                    history = init_history(init_slots, init_mask, int(model.history_length))
                    for step in range(int(horizon)):
                        action = choose_action(game, policy, obs, env, rng)
                        true_next, _, terminated, truncated, _ = env.step(action)
                        pred_slots, pred_mask = predict_next_from_history(model, history, action, RULE_TO_ID[mode], GAME_TO_ID[game], device)
                        pred_state = slots_to_flat_state(pred_slots, GAME_TO_ID[game])
                        per_step_errors[step].append(float(np.mean((pred_state - true_next) ** 2)))
                        history.append(pred_slots, pred_mask)
                        obs = true_next
                        if terminated or truncated:
                            break
                    if tqdm is not None:
                        flattened = [v for row in per_step_errors for v in row]
                        if flattened:
                            ep_iter.set_postfix(mse=f"{float(np.mean(flattened)):.4f}")
            finally:
                env.close()
            all_errors = [v for row in per_step_errors for v in row]
            metrics[f"rollout_mse/{game}:{mode}"] = float(np.mean(all_errors)) if all_errors else float("nan")
            for idx in (0, 4, 19, horizon - 1):
                if 0 <= idx < horizon and per_step_errors[idx]:
                    metrics[f"rollout_mse/{game}:{mode}/t{idx + 1}"] = float(np.mean(per_step_errors[idx]))
    return metrics


def counterfactual_eval(model, device: torch.device, samples: int, policy: str, seed: int, games: list[str], modes: list[str]) -> dict[str, float]:
    rng = random.Random(seed)
    kwargs = env_kwargs(argparse.Namespace(width=640, height=480, dt=1.0 / 60.0, gravity=420.0, paddle_speed=360.0, ball_speed=280.0, max_steps=3000))
    metrics: dict[str, float] = {}
    for game in games:
        base_env = make_game_env(game, "normal", seed + 3000 * GAME_TO_ID[game], kwargs)
        rule_envs = {mode: make_game_env(game, mode, seed + 3000 * GAME_TO_ID[game] + 100 + RULE_TO_ID[mode], kwargs) for mode in modes}
        errors = {mode: [] for mode in modes}
        true_variance = []
        pred_variance = []
        try:
            obs, _ = base_env.reset()
            history_cache = {mode: None for mode in modes}
            sample_iter = progress(range(int(samples)), desc=f"counterfactual {game}", total=int(samples))
            for _ in sample_iter:
                if base_env.is_done:
                    obs, _ = base_env.reset()
                    history_cache = {mode: None for mode in modes}
                action = choose_action(game, policy, obs, base_env, rng)
                base_state = base_env.get_state()
                true_nexts = []
                pred_nexts = []
                for mode, env in rule_envs.items():
                    env.set_state(clone_for_mode(game, base_state))
                    state_obs = env.state_to_observation()
                    if history_cache[mode] is None:
                        slots, mask = env.state_to_slots() if game == "breakout" else __import__("data.pong_common", fromlist=["flat_pong_state_to_slots"]).flat_pong_state_to_slots(state_obs)
                        history_cache[mode] = init_history(slots, mask, int(model.history_length))
                    true_next, _, _, _, _ = env.step(action)
                    pred_slots, pred_mask = predict_next_from_history(model, history_cache[mode], action, RULE_TO_ID[mode], GAME_TO_ID[game], device)
                    pred_next = slots_to_flat_state(pred_slots, GAME_TO_ID[game])
                    errors[mode].append(float(np.mean((pred_next - true_next) ** 2)))
                    history_cache[mode].append(pred_slots, pred_mask)
                    true_nexts.append(true_next)
                    pred_nexts.append(pred_next)
                true_variance.append(float(np.mean(np.var(np.stack(true_nexts), axis=0))))
                pred_variance.append(float(np.mean(np.var(np.stack(pred_nexts), axis=0))))
                obs, _, terminated, truncated, _ = base_env.step(action)
                if terminated or truncated:
                    obs, _ = base_env.reset()
                    history_cache = {mode: None for mode in modes}
                if tqdm is not None:
                    current = {mode: float(np.mean(values)) for mode, values in errors.items() if values}
                    if current:
                        sample_iter.set_postfix(**{k.split(":")[-1]: f"{v:.2f}" for k, v in current.items()})
        finally:
            base_env.close()
            for env in rule_envs.values():
                env.close()
        for mode, values in errors.items():
            metrics[f"counterfactual_mse/{game}:{mode}"] = float(np.mean(values))
        metrics[f"counterfactual_true_rule_variance/{game}"] = float(np.mean(true_variance))
        metrics[f"counterfactual_pred_rule_variance/{game}"] = float(np.mean(pred_variance))
        metrics[f"counterfactual_rule_variance_ratio/{game}"] = metrics[f"counterfactual_pred_rule_variance/{game}"] / max(metrics[f"counterfactual_true_rule_variance/{game}"], 1e-8)
    return metrics


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(pathlib.Path(args.checkpoint).expanduser().resolve(), map_location=device)
    model = build_gns_model_from_checkpoint(checkpoint, device)
    rule_ablation = args.rule_ablation or checkpoint.get("args", {}).get("rule_ablation", "none")
    metrics: dict[str, float | str] = {"checkpoint_epoch": float(checkpoint.get("epoch", -1)), "rule_ablation": rule_ablation}
    if args.dataset:
        dataset_root = pathlib.Path(args.dataset).expanduser().resolve()
        val_data = GNSTrajectoryWindowDataset(dataset_root, "val", history_length=int(checkpoint["args"].get("history_length", 6)), rule_ablation=rule_ablation, seed=args.seed)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        metrics.update(evaluate_dataset(model, val_loader, device))
        holdout_combos = parse_combos(args.holdout_combos)
        if holdout_combos:
            holdout_data = GNSTrajectoryWindowDataset(
                dataset_root,
                "val",
                history_length=int(checkpoint["args"].get("history_length", 6)),
                combos=holdout_combos,
                rule_ablation=rule_ablation,
                seed=args.seed,
            )
            holdout_loader = DataLoader(holdout_data, batch_size=args.batch_size, shuffle=False)
            metrics.update({f"holdout/{key}": value for key, value in evaluate_dataset(model, holdout_loader, device).items()})
    metrics.update(rollout_eval(model, device, args.rollout_episodes, args.rollout_horizon, args.policy, args.seed, args.games, args.eval_modes))
    metrics.update(counterfactual_eval(model, device, args.rollout_episodes * args.rollout_horizon, args.policy, args.seed, args.games, args.eval_modes))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        output = pathlib.Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Saved metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
