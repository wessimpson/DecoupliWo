from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.pong_common import GAME_TO_ID, OBJECT_TYPE_TO_ID, load_shards


@dataclass(frozen=True)
class GNSNormalizationStats:
    pos_mean: np.ndarray
    pos_std: np.ndarray
    vel_mean: np.ndarray
    vel_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "pos_mean": self.pos_mean.astype(float).tolist(),
            "pos_std": self.pos_std.astype(float).tolist(),
            "vel_mean": self.vel_mean.astype(float).tolist(),
            "vel_std": self.vel_std.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "GNSNormalizationStats":
        return cls(
            pos_mean=np.asarray(data["pos_mean"], dtype=np.float32),
            pos_std=np.asarray(data["pos_std"], dtype=np.float32),
            vel_mean=np.asarray(data["vel_mean"], dtype=np.float32),
            vel_std=np.asarray(data["vel_std"], dtype=np.float32),
        )


def _rule_ablation(rule_id: np.ndarray, split: str, rule_ablation: str, seed: int) -> np.ndarray:
    out = np.asarray(rule_id, dtype=np.int64).copy()
    if rule_ablation == "zero":
        out[:] = 0
    elif rule_ablation == "shuffle":
        rng = np.random.default_rng(int(seed) + (0 if split == "train" else 1))
        out = rng.permutation(out)
    elif rule_ablation != "none":
        raise ValueError(f"Unknown rule_ablation={rule_ablation!r}")
    return out


def compute_gns_stats_from_rows(data: dict[str, np.ndarray], history_length: int = 6, dt: float = 1.0 / 60.0) -> GNSNormalizationStats:
    slots = np.asarray(data["object_slots"], dtype=np.float32)
    next_slots = np.asarray(data["next_object_slots"], dtype=np.float32)
    mask = np.asarray(data["object_mask"], dtype=np.float32)
    next_mask = np.asarray(data["next_object_mask"], dtype=np.float32)
    all_slots = np.concatenate([slots, next_slots], axis=0)
    all_mask = np.concatenate([mask, next_mask], axis=0)
    active = all_mask > 0.5
    positions = all_slots[..., :2][active]
    velocities = all_slots[..., 2:4][active]
    if positions.size == 0:
        positions = np.zeros((1, 2), dtype=np.float32)
    if velocities.size == 0:
        velocities = np.zeros((1, 2), dtype=np.float32)
    pos_mean = positions.mean(axis=0).astype(np.float32)
    pos_std = positions.std(axis=0).astype(np.float32)
    vel_mean = velocities.mean(axis=0).astype(np.float32)
    vel_std = velocities.std(axis=0).astype(np.float32)
    pos_std = np.maximum(pos_std, 1e-3)
    vel_std = np.maximum(vel_std, 1e-3)
    return GNSNormalizationStats(pos_mean=pos_mean, pos_std=pos_std, vel_mean=vel_mean, vel_std=vel_std)


class GNSTrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        split: str,
        history_length: int = 6,
        combos: set[tuple[int, int]] | None = None,
        exclude_combos: set[tuple[int, int]] | None = None,
        rule_ablation: str = "none",
        seed: int = 0,
        min_episode_length: int | None = None,
    ):
        self.root = pathlib.Path(root).expanduser().resolve()
        self.split = split
        self.history_length = int(history_length)
        data = load_shards(self.root, split)
        game_id = np.asarray(data.get("game_id", np.zeros_like(data["rule_id"], dtype=np.int64)), dtype=np.int64)
        true_rule_id = np.asarray(data["rule_id"], dtype=np.int64)
        rule_id = _rule_ablation(true_rule_id, split, rule_ablation, seed)

        keep = np.ones(int(data["action"].shape[0]), dtype=bool)
        if combos:
            keep &= np.asarray([(int(g), int(r)) in combos for g, r in zip(game_id, true_rule_id)], dtype=bool)
        if exclude_combos:
            keep &= np.asarray([(int(g), int(r)) not in exclude_combos for g, r in zip(game_id, true_rule_id)], dtype=bool)
        if not np.any(keep):
            raise ValueError(f"No rows left after combo filtering for split={split}")

        order = np.lexsort((np.asarray(data["step"], dtype=np.int64)[keep], np.asarray(data["episode_id"], dtype=np.int64)[keep]))
        self.action = torch.as_tensor(np.asarray(data["action"], dtype=np.int64)[keep][order], dtype=torch.long)
        self.rule_id = torch.as_tensor(rule_id[keep][order], dtype=torch.long)
        self.true_rule_id = torch.as_tensor(true_rule_id[keep][order], dtype=torch.long)
        self.game_id = torch.as_tensor(game_id[keep][order], dtype=torch.long)
        self.event_id = torch.as_tensor(np.asarray(data["event_id"], dtype=np.int64)[keep][order], dtype=torch.long)
        self.source_id = torch.as_tensor(np.asarray(data["source_id"], dtype=np.int64)[keep][order], dtype=torch.long)
        self.episode_id = torch.as_tensor(np.asarray(data["episode_id"], dtype=np.int64)[keep][order], dtype=torch.long)
        self.step = torch.as_tensor(np.asarray(data["step"], dtype=np.int64)[keep][order], dtype=torch.long)
        self.state = torch.as_tensor(np.asarray(data["state"], dtype=np.float32)[keep][order], dtype=torch.float32)
        self.next_state = torch.as_tensor(np.asarray(data["next_state"], dtype=np.float32)[keep][order], dtype=torch.float32)
        self.object_slots = torch.as_tensor(np.asarray(data["object_slots"], dtype=np.float32)[keep][order], dtype=torch.float32)
        self.next_object_slots = torch.as_tensor(np.asarray(data["next_object_slots"], dtype=np.float32)[keep][order], dtype=torch.float32)
        self.object_mask = torch.as_tensor(np.asarray(data["object_mask"], dtype=np.float32)[keep][order], dtype=torch.float32)
        self.next_object_mask = torch.as_tensor(np.asarray(data["next_object_mask"], dtype=np.float32)[keep][order], dtype=torch.float32)

        self.sample_indices: list[int] = []
        min_len = int(min_episode_length or max(self.history_length + 1, 2))
        episodes = self.episode_id.cpu().numpy()
        steps = self.step.cpu().numpy()
        games = self.game_id.cpu().numpy()
        rules = self.true_rule_id.cpu().numpy()
        groups: dict[tuple[int, int, int], list[int]] = {}
        for idx, key in enumerate(zip(episodes, games, rules)):
            groups.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(idx)
        for indices in groups.values():
            if len(indices) < min_len:
                continue
            ordered = sorted(indices, key=lambda i: int(steps[i]))
            ordered_steps = steps[ordered]
            for pos in range(self.history_length - 1, len(ordered)):
                window_steps = ordered_steps[pos - self.history_length + 1 : pos + 1]
                if np.all(np.diff(window_steps) == 1):
                    self.sample_indices.append(ordered[pos])
        if not self.sample_indices:
            raise ValueError(
                f"No valid history windows of length {self.history_length} found for split={split}. "
                f"Regenerate data with longer rare rollouts."
            )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.sample_indices[idx]
        history = slice(row - self.history_length + 1, row + 1)
        latest_slots = self.object_slots[row]
        latest_mask = self.object_mask[row]
        type_ids = latest_slots[:, 6].round().clamp_min(0).to(torch.long)
        ball_mask = ((type_ids == OBJECT_TYPE_TO_ID["ball"]).to(torch.float32) * latest_mask)
        paddle_mask = ((type_ids == OBJECT_TYPE_TO_ID["paddle"]).to(torch.float32) * latest_mask)
        block_mask = ((type_ids == OBJECT_TYPE_TO_ID["block"]).to(torch.float32) * latest_mask)
        dynamic_pos_mask = ball_mask
        return {
            "history_slots": self.object_slots[history],
            "history_mask": self.object_mask[history],
            "action": self.action[row],
            "rule_id": self.rule_id[row],
            "true_rule_id": self.true_rule_id[row],
            "game_id": self.game_id[row],
            "event_id": self.event_id[row],
            "source_id": self.source_id[row],
            "target_next_slots": self.next_object_slots[row],
            "target_next_mask": self.next_object_mask[row],
            "target_next_state": self.next_state[row],
            "current_slots": latest_slots,
            "current_mask": latest_mask,
            "ball_mask": ball_mask,
            "kinematic_mask": paddle_mask,
            "block_mask": block_mask,
            "dynamic_pos_mask": dynamic_pos_mask,
            "episode_id": self.episode_id[row],
            "step": self.step[row],
        }


def load_gns_metadata(root: pathlib.Path) -> dict:
    path = pathlib.Path(root).expanduser() / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def pairwise_combo_filter(game_ids: np.ndarray, rule_ids: np.ndarray, combos: set[tuple[int, int]]) -> np.ndarray:
    return np.asarray([(int(g), int(r)) in combos for g, r in zip(game_ids, rule_ids)], dtype=bool)


__all__ = [
    "GNSNormalizationStats",
    "GNSTrajectoryWindowDataset",
    "batch_to_device",
    "compute_gns_stats_from_rows",
    "load_gns_metadata",
]
