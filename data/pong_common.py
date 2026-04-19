from __future__ import annotations

import json
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_pong import GameState, PongEnv, clone_state


MODES = ("normal", "gravity", "teleport")
RULE_TO_ID = {mode: idx for idx, mode in enumerate(MODES)}
ID_TO_RULE = {idx: mode for mode, idx in RULE_TO_ID.items()}
GAMES = ("pong", "breakout")
GAME_TO_ID = {game: idx for idx, game in enumerate(GAMES)}
ID_TO_GAME = {idx: game for game, idx in GAME_TO_ID.items()}
ACTIONS = (0, 1, 2)
ACTION_NAMES = {0: "stay", 1: "up_or_left", 2: "down_or_right"}
OBJECT_TYPES = ("empty", "ball", "paddle", "block")
OBJECT_TYPE_TO_ID = {name: idx for idx, name in enumerate(OBJECT_TYPES)}
ID_TO_OBJECT_TYPE = {idx: name for name, idx in OBJECT_TYPE_TO_ID.items()}
SLOT_FEATURE_NAMES = ("x", "y", "vx", "vy", "width", "height", "type_id")
SLOT_DIM = len(SLOT_FEATURE_NAMES)
MAX_OBJECTS = 10
SOURCES = (
    "rollout",
    "diverse",
    "left_wall",
    "right_wall",
    "top_bounce",
    "bottom_bounce",
    "wrapped_top",
    "wrapped_bottom",
    "wrapped_left",
    "wrapped_right",
    "paddle_hit",
    "block_hit",
    "miss",
)
SOURCE_TO_ID = {source: idx for idx, source in enumerate(SOURCES)}
ID_TO_SOURCE = {idx: source for source, idx in SOURCE_TO_ID.items()}
EVENTS = (
    "reset",
    "step",
    "paddle_hit",
    "miss",
    "left_wall_bounce",
    "top_bounce",
    "bottom_bounce",
    "wrapped",
    "truncated",
    "episode_done",
    "none",
    "block_hit",
    "cleared",
)
EVENT_TO_ID = {event: idx for idx, event in enumerate(EVENTS)}
ID_TO_EVENT = {idx: event for event, idx in EVENT_TO_ID.items()}


@dataclass(frozen=True)
class PongSlotConfig:
    width: float = 640.0
    height: float = 480.0
    paddle_width: float = 12.0
    paddle_height: float = 88.0
    paddle_margin: float = 24.0
    paddle_speed: float = 360.0
    ball_radius: float = 8.0
    max_ball_speed: float = 720.0

    @property
    def paddle_x(self) -> float:
        return self.width - self.paddle_margin - self.paddle_width

    @property
    def slot_scales(self) -> np.ndarray:
        return np.asarray(
            [
                self.width,
                self.height,
                max(self.max_ball_speed, 1.0),
                max(self.max_ball_speed, 1.0),
                self.width,
                self.height,
            ],
            dtype=np.float32,
        )


def make_env(mode: str, seed: int | None = None, **kwargs: Any) -> PongEnv:
    return PongEnv(mode=mode, render_mode=None, seed=seed, **kwargs)


def slot_config_from_env(env: PongEnv) -> PongSlotConfig:
    cfg = env.config
    return PongSlotConfig(
        width=float(cfg.width),
        height=float(cfg.height),
        paddle_width=float(cfg.paddle_width),
        paddle_height=float(cfg.paddle_height),
        paddle_margin=float(cfg.paddle_margin),
        paddle_speed=float(cfg.paddle_speed),
        ball_radius=float(cfg.ball_radius),
        max_ball_speed=float(cfg.max_ball_speed),
    )


def slot_config_from_metadata(metadata: dict[str, Any] | None) -> PongSlotConfig:
    cfg = (metadata or {}).get("env_config", {})
    return PongSlotConfig(
        width=float(cfg.get("width", 640.0)),
        height=float(cfg.get("height", 480.0)),
        paddle_width=float(cfg.get("paddle_width", 12.0)),
        paddle_height=float(cfg.get("paddle_height", 88.0)),
        paddle_margin=float(cfg.get("paddle_margin", 24.0)),
        paddle_speed=float(cfg.get("paddle_speed", 360.0)),
        ball_radius=float(cfg.get("ball_radius", 8.0)),
        max_ball_speed=float(cfg.get("max_ball_speed", 720.0)),
    )


def flat_pong_state_to_slots(
    state: np.ndarray,
    config: PongSlotConfig | None = None,
    max_objects: int = MAX_OBJECTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat Pong state to fixed object slots.

    Slot 0 is the ball and slot 1 is the player paddle. Remaining slots are
    inactive so this schema can later host block slots for Breakout-like games.
    """

    config = config or PongSlotConfig()
    flat = np.asarray(state, dtype=np.float32)
    if flat.ndim == 2:
        if flat.shape[-1] != 6:
            raise ValueError(f"Expected flat Pong states with 6 values, got shape {flat.shape}")
        rows = int(flat.shape[0])
        slots = np.zeros((rows, int(max_objects), SLOT_DIM), dtype=np.float32)
        mask = np.zeros((rows, int(max_objects)), dtype=np.float32)
        diameter = float(config.ball_radius) * 2.0
        slots[:, 0, 0] = flat[:, 0]
        slots[:, 0, 1] = flat[:, 1]
        slots[:, 0, 2] = flat[:, 2]
        slots[:, 0, 3] = flat[:, 3]
        slots[:, 0, 4] = diameter
        slots[:, 0, 5] = diameter
        slots[:, 0, 6] = OBJECT_TYPE_TO_ID["ball"]
        slots[:, 1, 0] = config.paddle_x
        slots[:, 1, 1] = flat[:, 4]
        slots[:, 1, 3] = flat[:, 5]
        slots[:, 1, 4] = config.paddle_width
        slots[:, 1, 5] = config.paddle_height
        slots[:, 1, 6] = OBJECT_TYPE_TO_ID["paddle"]
        mask[:, :2] = 1.0
        return slots, mask
    if flat.shape[-1] != 6:
        raise ValueError(f"Expected flat Pong state with 6 values, got shape {flat.shape}")

    slots = np.zeros((int(max_objects), SLOT_DIM), dtype=np.float32)
    mask = np.zeros((int(max_objects),), dtype=np.float32)
    diameter = float(config.ball_radius) * 2.0
    slots[0] = np.asarray(
        [
            flat[0],
            flat[1],
            flat[2],
            flat[3],
            diameter,
            diameter,
            OBJECT_TYPE_TO_ID["ball"],
        ],
        dtype=np.float32,
    )
    slots[1] = np.asarray(
        [
            config.paddle_x,
            flat[4],
            0.0,
            flat[5],
            config.paddle_width,
            config.paddle_height,
            OBJECT_TYPE_TO_ID["paddle"],
        ],
        dtype=np.float32,
    )
    mask[:2] = 1.0
    return slots, mask


def slots_to_flat_pong_state(slots: np.ndarray) -> np.ndarray:
    slots = np.asarray(slots, dtype=np.float32)
    if slots.ndim == 3:
        return np.stack([slots_to_flat_pong_state(row) for row in slots]).astype(np.float32)
    if slots.shape[-2:] != (MAX_OBJECTS, SLOT_DIM):
        raise ValueError(f"Expected slots shape ({MAX_OBJECTS}, {SLOT_DIM}), got {slots.shape}")
    return np.asarray(
        [slots[0, 0], slots[0, 1], slots[0, 2], slots[0, 3], slots[1, 1], slots[1, 3]],
        dtype=np.float32,
    )


def env_scales(env: PongEnv) -> np.ndarray:
    cfg = env.config
    return np.asarray(
        [
            cfg.width,
            cfg.height,
            max(cfg.max_ball_speed, cfg.ball_speed, 1.0),
            max(cfg.max_ball_speed, cfg.ball_speed, 1.0),
            max(1.0, cfg.height - cfg.paddle_height),
            max(cfg.paddle_speed, 1.0),
        ],
        dtype=np.float32,
    )


def normalize_state(state: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32) / np.asarray(scales, dtype=np.float32)


def denormalize_state(state: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32) * np.asarray(scales, dtype=np.float32)


def heuristic_action(obs: np.ndarray, env: PongEnv, rng: random.Random, epsilon: float = 0.05) -> int:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    ball_y = float(obs[1])
    paddle_y = float(obs[4])
    center = paddle_y + env.config.paddle_height / 2.0
    margin = max(3.0, env.config.paddle_height * 0.08)
    if ball_y < center - margin:
        return 1
    if ball_y > center + margin:
        return 2
    return 0


def choose_policy_action(policy: str, obs: np.ndarray, env: PongEnv, rng: random.Random) -> int:
    if policy == "random":
        return rng.choice(ACTIONS)
    if policy == "heuristic":
        return heuristic_action(obs, env, rng, epsilon=0.02)
    if policy == "mixed":
        if rng.random() < 0.5:
            return rng.choice(ACTIONS)
        return heuristic_action(obs, env, rng, epsilon=0.08)
    raise ValueError(f"Unknown policy: {policy}")


def event_id(info: dict[str, Any]) -> int:
    return EVENT_TO_ID.get(str(info.get("event", "none")), EVENT_TO_ID["none"])


def state_config_metadata(env: PongEnv) -> dict[str, Any]:
    cfg = env.config
    return {
        "width": cfg.width,
        "height": cfg.height,
        "dt": cfg.dt,
        "paddle_width": cfg.paddle_width,
        "paddle_height": cfg.paddle_height,
        "paddle_margin": cfg.paddle_margin,
        "paddle_speed": cfg.paddle_speed,
        "ball_radius": cfg.ball_radius,
        "ball_speed": cfg.ball_speed,
        "gravity": cfg.gravity,
        "max_steps": cfg.max_steps,
        "min_ball_speed": cfg.min_ball_speed,
        "max_ball_speed": cfg.max_ball_speed,
        "speedup_on_paddle_hit": getattr(cfg, "speedup_on_paddle_hit", 1.0),
        "state_scales": env_scales(env).tolist(),
    }


class TransitionShardWriter:
    def __init__(
        self,
        root: pathlib.Path,
        split: str,
        chunk_size: int = 10000,
        slot_config: PongSlotConfig | None = None,
        game_id: int = 0,
    ):
        self.root = pathlib.Path(root).expanduser() / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self.slot_config = slot_config or PongSlotConfig()
        self.game_id = int(game_id)
        self.shard_idx = self._next_shard_idx()
        self.buffer = self._empty()

    def append(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        rule_id: int,
        event: int,
        episode_id: int,
        step: int,
        source_id: int = 0,
        game_id: int | None = None,
        object_slots: np.ndarray | None = None,
        next_object_slots: np.ndarray | None = None,
        object_mask: np.ndarray | None = None,
        next_object_mask: np.ndarray | None = None,
    ) -> None:
        if object_slots is None or object_mask is None:
            object_slots, object_mask = flat_pong_state_to_slots(state, self.slot_config)
        if next_object_slots is None or next_object_mask is None:
            next_object_slots, next_object_mask = flat_pong_state_to_slots(next_state, self.slot_config)
        self.buffer["state"].append(np.asarray(state, dtype=np.float32))
        self.buffer["action"].append(int(action))
        self.buffer["next_state"].append(np.asarray(next_state, dtype=np.float32))
        self.buffer["reward"].append(float(reward))
        self.buffer["terminated"].append(bool(terminated))
        self.buffer["truncated"].append(bool(truncated))
        self.buffer["rule_id"].append(int(rule_id))
        self.buffer["event_id"].append(int(event))
        self.buffer["episode_id"].append(int(episode_id))
        self.buffer["step"].append(int(step))
        self.buffer["source_id"].append(int(source_id))
        self.buffer["game_id"].append(self.game_id if game_id is None else int(game_id))
        self.buffer["object_slots"].append(np.asarray(object_slots, dtype=np.float32))
        self.buffer["next_object_slots"].append(np.asarray(next_object_slots, dtype=np.float32))
        self.buffer["object_mask"].append(np.asarray(object_mask, dtype=np.float32))
        self.buffer["next_object_mask"].append(np.asarray(next_object_mask, dtype=np.float32))
        if len(self.buffer["action"]) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer["action"]:
            return
        path = self.root / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            state=np.stack(self.buffer["state"]).astype(np.float32),
            action=np.asarray(self.buffer["action"], dtype=np.int64),
            next_state=np.stack(self.buffer["next_state"]).astype(np.float32),
            reward=np.asarray(self.buffer["reward"], dtype=np.float32),
            terminated=np.asarray(self.buffer["terminated"], dtype=np.bool_),
            truncated=np.asarray(self.buffer["truncated"], dtype=np.bool_),
            rule_id=np.asarray(self.buffer["rule_id"], dtype=np.int64),
            event_id=np.asarray(self.buffer["event_id"], dtype=np.int64),
            episode_id=np.asarray(self.buffer["episode_id"], dtype=np.int64),
            step=np.asarray(self.buffer["step"], dtype=np.int64),
            source_id=np.asarray(self.buffer["source_id"], dtype=np.int64),
            game_id=np.asarray(self.buffer["game_id"], dtype=np.int64),
            object_slots=np.stack(self.buffer["object_slots"]).astype(np.float32),
            next_object_slots=np.stack(self.buffer["next_object_slots"]).astype(np.float32),
            object_mask=np.stack(self.buffer["object_mask"]).astype(np.float32),
            next_object_mask=np.stack(self.buffer["next_object_mask"]).astype(np.float32),
        )
        self.shard_idx += 1
        self.buffer = self._empty()

    def _next_shard_idx(self) -> int:
        indices = []
        for path in self.root.glob("shard_*.npz"):
            try:
                indices.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    @staticmethod
    def _empty() -> dict[str, list[Any]]:
        return {
            "state": [],
            "action": [],
            "next_state": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "rule_id": [],
            "event_id": [],
            "episode_id": [],
            "step": [],
            "source_id": [],
            "game_id": [],
            "object_slots": [],
            "next_object_slots": [],
            "object_mask": [],
            "next_object_mask": [],
        }


def write_metadata(root: pathlib.Path, metadata: dict[str, Any]) -> None:
    root = pathlib.Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "modes": list(MODES),
        "rule_to_id": RULE_TO_ID,
        "games": GAME_TO_ID,
        "actions": ACTION_NAMES,
        "events": EVENT_TO_ID,
        "sources": SOURCE_TO_ID,
        "object_types": OBJECT_TYPE_TO_ID,
        "slot_feature_names": SLOT_FEATURE_NAMES,
        "max_objects": MAX_OBJECTS,
        **metadata,
    }
    (root / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_shards(root: pathlib.Path, split: str) -> dict[str, np.ndarray]:
    root = pathlib.Path(root).expanduser()
    split_root = pathlib.Path(root).expanduser() / split
    paths = sorted(split_root.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards found under {split_root}")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    slot_config = slot_config_from_metadata(metadata)
    chunks: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as data:
            files = set(data.files)
            row_count = int(data["action"].shape[0])
            state = data["state"]
            next_state = data["next_state"]
            object_slots, object_mask = (
                (data["object_slots"], data["object_mask"])
                if {"object_slots", "object_mask"}.issubset(files)
                else flat_pong_state_to_slots(state, slot_config)
            )
            next_object_slots, next_object_mask = (
                (data["next_object_slots"], data["next_object_mask"])
                if {"next_object_slots", "next_object_mask"}.issubset(files)
                else flat_pong_state_to_slots(next_state, slot_config)
            )
            normalized = {
                key: data[key]
                for key in files
                if key not in {"object_slots", "object_mask", "next_object_slots", "next_object_mask"}
            }
            normalized.setdefault("source_id", np.zeros(row_count, dtype=np.int64))
            normalized.setdefault("game_id", np.full(row_count, GAME_TO_ID["pong"], dtype=np.int64))
            normalized["object_slots"] = object_slots.astype(np.float32)
            normalized["next_object_slots"] = next_object_slots.astype(np.float32)
            normalized["object_mask"] = object_mask.astype(np.float32)
            normalized["next_object_mask"] = next_object_mask.astype(np.float32)
            for key, value in normalized.items():
                chunks.setdefault(key, []).append(value)
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def split_episode_ids(total_episodes: int, val_fraction: float, rng: random.Random) -> set[int]:
    ids = list(range(int(total_episodes)))
    rng.shuffle(ids)
    val_count = int(round(len(ids) * float(val_fraction)))
    return set(ids[:val_count])


def copy_state_for_mode(state: GameState) -> GameState:
    copied = clone_state(state)
    copied.terminated = False
    copied.truncated = False
    return copied


def count_rows(root: pathlib.Path, split: str) -> int:
    total = 0
    for path in sorted((pathlib.Path(root) / split).glob("shard_*.npz")):
        with np.load(path) as data:
            total += int(data["action"].shape[0])
    return total


def iter_shard_paths(root: pathlib.Path, split: str) -> Iterable[pathlib.Path]:
    return sorted((pathlib.Path(root).expanduser() / split).glob("shard_*.npz"))
