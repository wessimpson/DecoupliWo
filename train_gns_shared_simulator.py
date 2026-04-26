#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gns_shared_dataset import GNSTrajectoryWindowDataset, batch_to_device, compute_gns_stats_from_rows
from data.pong_common import EVENTS, GAME_TO_ID, ID_TO_EVENT, ID_TO_GAME, ID_TO_RULE, RULE_TO_ID, load_shards
from models.gns_shared_simulator import GNSSharedSimulator, SharedSimulatorConstants


def progress(iterable, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepMind-style shared simulator for Pong + Breakout.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="runs/gns_shared_simulator")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--connectivity-radius", type=float, default=0.35)
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="large")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise-std", type=float, default=3e-2)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-weight", type=float, default=4.0)
    parser.add_argument("--train-combos", nargs="*", default=None)
    parser.add_argument("--holdout-combos", nargs="*", default=None)
    parser.add_argument("--rule-ablation", choices=("none", "zero", "shuffle"), default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


MODEL_PRESETS = {
    "small": {"latent_dim": 96, "hidden_dim": 192, "message_passing_steps": 6, "type_dim": 12, "rule_dim": 12, "game_dim": 12},
    "medium": {"latent_dim": 128, "hidden_dim": 256, "message_passing_steps": 8, "type_dim": 16, "rule_dim": 16, "game_dim": 16},
    "large": {"latent_dim": 192, "hidden_dim": 384, "message_passing_steps": 10, "type_dim": 24, "rule_dim": 24, "game_dim": 24},
}


def parse_combos(values: list[str] | None) -> set[tuple[int, int]] | None:
    if not values:
        return None
    combos: set[tuple[int, int]] = set()
    for value in values:
        game_name, rule_name = value.split(":", 1)
        combos.add((GAME_TO_ID[game_name], RULE_TO_ID[rule_name]))
    return combos


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def constants_from_metadata(dataset_root: pathlib.Path) -> SharedSimulatorConstants:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        return SharedSimulatorConstants()
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata.get("env_config", {})
    return SharedSimulatorConstants(
        width=float(cfg.get("width", 640.0)),
        height=float(cfg.get("height", 480.0)),
        dt=float(cfg.get("dt", 1.0 / 60.0)),
        paddle_speed=float(cfg.get("paddle_speed", 360.0)),
        max_ball_speed=float(cfg.get("max_ball_speed", 720.0)),
    )


def transition_weights(event_id: torch.Tensor, event_weight: float) -> torch.Tensor:
    weights = torch.ones_like(event_id, dtype=torch.float32)
    if event_weight <= 1.0:
        return weights
    rare = event_id != EVENTS.index("step")
    rare &= event_id != EVENTS.index("none")
    weights[rare] = float(event_weight)
    return weights


def compute_losses(model: GNSSharedSimulator, batch: dict[str, torch.Tensor], noise_std: float, mask_loss_weight: float, event_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(
        batch["history_slots"],
        batch["history_mask"],
        batch["action"],
        batch["rule_id"],
        batch["game_id"],
        dynamic_pos_mask=batch["dynamic_pos_mask"],
        noise_std=noise_std,
        training_noise=model.training,
    )
    weights = transition_weights(batch["event_id"], event_weight).to(batch["history_slots"].device)
    pred_pos = out["pred_next_slots"][..., :2]
    target_pos = batch["target_next_slots"][..., :2]
    dynamic_mask = batch["dynamic_pos_mask"].to(pred_pos.dtype)
    pos_err = ((pred_pos - target_pos) ** 2).mean(dim=-1)
    pos_loss = ((pos_err * dynamic_mask).sum(dim=1) / dynamic_mask.sum(dim=1).clamp_min(1.0)) * weights
    pos_loss = pos_loss.mean()

    block_mask = batch["block_mask"].to(pred_pos.dtype)
    target_mask = batch["target_next_mask"].to(pred_pos.dtype)
    mask_logits = out["pred_next_mask_logits"]
    mask_loss_per = F.binary_cross_entropy_with_logits(mask_logits, target_mask, reduction="none")
    mask_loss = ((mask_loss_per * block_mask).sum(dim=1) / block_mask.sum(dim=1).clamp_min(1.0)) * weights
    mask_loss = mask_loss.mean()
    loss = pos_loss + float(mask_loss_weight) * mask_loss
    return loss, {
        "loss": loss.detach(),
        "pos_loss": pos_loss.detach(),
        "mask_loss": mask_loss.detach(),
        "pred_next_slots": out["pred_next_slots"].detach(),
        "pred_next_mask_prob": out["pred_next_mask_prob"].detach(),
    }


def train_epoch(model: GNSSharedSimulator, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, noise_std: float, mask_loss_weight: float, event_weight: float) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "pos_loss": 0.0, "mask_loss": 0.0}
    rows = 0
    batch_iter = progress(loader, desc="train", total=len(loader))
    for batch in batch_iter:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_losses(model, batch, noise_std, mask_loss_weight, event_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        batch_size = int(batch["action"].shape[0])
        rows += batch_size
        for key in totals:
            totals[key] += float(metrics[key].item()) * batch_size
        if tqdm is not None:
            batch_iter.set_postfix(
                loss=f"{float(metrics['loss'].item()):.4f}",
                pos=f"{float(metrics['pos_loss'].item()):.4f}",
                mask=f"{float(metrics['mask_loss'].item()):.4f}",
            )
    return {key: value / max(rows, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: GNSSharedSimulator, loader: DataLoader, device: torch.device, mask_loss_weight: float, event_weight: float) -> dict[str, float]:
    model.eval()
    totals = {"val_loss": 0.0, "val_pos_loss": 0.0, "val_mask_loss": 0.0, "val_ball_mse": 0.0}
    rows = 0
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}

    def add_group(name: str, values: np.ndarray) -> None:
        group_sums[name] = group_sums.get(name, 0.0) + float(values.sum())
        group_counts[name] = group_counts.get(name, 0) + int(values.size)

    batch_iter = progress(loader, desc="eval", total=len(loader))
    for batch in batch_iter:
        batch = batch_to_device(batch, device)
        loss, metrics = compute_losses(model, batch, 0.0, mask_loss_weight, event_weight)
        ball_mask = batch["dynamic_pos_mask"].to(torch.float32)
        pred_pos = metrics["pred_next_slots"][..., :2]
        target_pos = batch["target_next_slots"][..., :2]
        ball_mse = (((pred_pos - target_pos) ** 2).mean(dim=-1) * ball_mask).sum(dim=1) / ball_mask.sum(dim=1).clamp_min(1.0)
        batch_size = int(batch["action"].shape[0])
        rows += batch_size
        totals["val_loss"] += float(loss.item()) * batch_size
        totals["val_pos_loss"] += float(metrics["pos_loss"].item()) * batch_size
        totals["val_mask_loss"] += float(metrics["mask_loss"].item()) * batch_size
        totals["val_ball_mse"] += float(ball_mse.mean().item()) * batch_size
        values = ball_mse.detach().cpu().numpy()
        games = batch["game_id"].detach().cpu().numpy()
        rules = batch["true_rule_id"].detach().cpu().numpy()
        events = batch["event_id"].detach().cpu().numpy()
        for group_id in np.unique(games):
            add_group(f"val_ball_mse/game_{ID_TO_GAME[int(group_id)]}", values[games == group_id])
        for group_id in np.unique(rules):
            add_group(f"val_ball_mse/rule_{ID_TO_RULE[int(group_id)]}", values[rules == group_id])
        for group_id in np.unique(events):
            add_group(f"val_ball_mse/event_{ID_TO_EVENT[int(group_id)]}", values[events == group_id])
        for game in np.unique(games):
            for rule in np.unique(rules[games == game]):
                selected = (games == game) & (rules == rule)
                add_group(f"val_ball_mse/combo_{ID_TO_GAME[int(game)]}:{ID_TO_RULE[int(rule)]}", values[selected])
        if tqdm is not None:
            batch_iter.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                ball=f"{float(ball_mse.mean().item()):.4f}",
                mask=f"{float(metrics['mask_loss'].item()):.4f}",
            )
    result = {key: value / max(rows, 1) for key, value in totals.items()}
    result["val_ball_rmse"] = float(np.sqrt(max(result["val_ball_mse"], 0.0)))
    for name, total in group_sums.items():
        result[name] = total / max(group_counts[name], 1)
    return result


def save_checkpoint(path: pathlib.Path, model: GNSSharedSimulator, optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int, metrics: dict[str, float], stats, constants: SharedSimulatorConstants) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "normalization_stats": stats.as_dict(),
        "constants": constants.__dict__,
        "model_family": "gns_shared_simulator",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_root = pathlib.Path(args.dataset).expanduser().resolve()
    output = pathlib.Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    train_combos = parse_combos(args.train_combos)
    holdout_combos = parse_combos(args.holdout_combos)

    train_data = GNSTrajectoryWindowDataset(
        dataset_root,
        "train",
        history_length=args.history_length,
        combos=train_combos,
        exclude_combos=holdout_combos,
        rule_ablation=args.rule_ablation,
        seed=args.seed,
    )
    val_data = GNSTrajectoryWindowDataset(
        dataset_root,
        "val",
        history_length=args.history_length,
        combos=train_combos,
        exclude_combos=holdout_combos,
        rule_ablation=args.rule_ablation,
        seed=args.seed,
    )
    holdout_data = (
        GNSTrajectoryWindowDataset(
            dataset_root,
            "val",
            history_length=args.history_length,
            combos=holdout_combos,
            rule_ablation=args.rule_ablation,
            seed=args.seed,
        )
        if holdout_combos
        else None
    )

    train_rows = load_shards(dataset_root, "train")
    if train_combos:
        game_ids = np.asarray(train_rows.get("game_id", np.zeros_like(train_rows["rule_id"], dtype=np.int64)), dtype=np.int64)
        rule_ids = np.asarray(train_rows["rule_id"], dtype=np.int64)
        keep = np.asarray([(int(g), int(r)) in train_combos for g, r in zip(game_ids, rule_ids)], dtype=bool)
        train_rows = {key: value[keep] for key, value in train_rows.items()}
    if holdout_combos:
        game_ids = np.asarray(train_rows.get("game_id", np.zeros_like(train_rows["rule_id"], dtype=np.int64)), dtype=np.int64)
        rule_ids = np.asarray(train_rows["rule_id"], dtype=np.int64)
        keep = np.asarray([(int(g), int(r)) not in holdout_combos for g, r in zip(game_ids, rule_ids)], dtype=bool)
        train_rows = {key: value[keep] for key, value in train_rows.items()}

    stats = compute_gns_stats_from_rows(train_rows, history_length=args.history_length)
    constants = constants_from_metadata(dataset_root)
    preset = MODEL_PRESETS[args.model_size]
    for key, value in preset.items():
        setattr(args, key, value)
    model = GNSSharedSimulator(
        stats=stats,
        history_length=args.history_length,
        connectivity_radius=args.connectivity_radius,
        constants=constants,
        **preset,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=max(args.lr * 0.05, 1e-6))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    holdout_loader = (
        DataLoader(holdout_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        if holdout_data is not None
        else None
    )

    best = float("inf")
    metrics_path = output / "metrics.jsonl"
    print(f"train windows={len(train_data)} val windows={len(val_data)} device={device}")
    if holdout_data is not None:
        print(f"holdout windows={len(holdout_data)} combos={args.holdout_combos}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.noise_std, args.mask_loss_weight, args.event_weight)
        val_metrics = evaluate(model, val_loader, device, args.mask_loss_weight, args.event_weight)
        if holdout_loader is not None:
            holdout_metrics = evaluate(model, holdout_loader, device, args.mask_loss_weight, args.event_weight)
            val_metrics.update({f"holdout/{key}": value for key, value in holdout_metrics.items()})
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.time() - start,
            **train_metrics,
            **val_metrics,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        save_checkpoint(output / "latest.pt", model, optimizer, args, epoch, metrics, stats, constants)
        if float(val_metrics["val_ball_mse"]) < best:
            best = float(val_metrics["val_ball_mse"])
            save_checkpoint(output / "best.pt", model, optimizer, args, epoch, metrics, stats, constants)
        print(
            f"[{epoch:04d}] loss={metrics['loss']:.6f} pos={metrics['pos_loss']:.6f} mask={metrics['mask_loss']:.6f} "
            f"val_ball_rmse={metrics['val_ball_rmse']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
