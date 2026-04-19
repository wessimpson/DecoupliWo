from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from data.pong_common import MAX_OBJECTS, OBJECT_TYPE_TO_ID, SLOT_DIM

Mode = Literal["normal", "gravity", "teleport"]
RenderMode = Literal["human", "rgb_array"] | None

ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_TO_DIRECTION = {ACTION_STAY: 0.0, ACTION_LEFT: -1.0, ACTION_RIGHT: 1.0}
OBSERVATION_NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "paddle_vx")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _event_if_empty(event_info: dict[str, Any], event_name: str) -> None:
    if event_info["event"] == "none":
        event_info["event"] = event_name


@dataclass(slots=True)
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def copy(self) -> "BallState":
        return replace(self)


@dataclass(slots=True)
class PaddleState:
    x: float
    y: float
    width: float
    height: float
    vx: float = 0.0

    def copy(self) -> "PaddleState":
        return replace(self)


@dataclass(slots=True)
class BlockState:
    x: float
    y: float
    width: float
    height: float
    active: bool = True

    def copy(self) -> "BlockState":
        return replace(self)


@dataclass(slots=True)
class BreakoutState:
    ball: BallState
    paddle: PaddleState
    blocks: list[BlockState]
    score: int = 0
    hits: int = 0
    misses: int = 0
    step_count: int = 0
    last_action: int = ACTION_STAY
    terminated: bool = False
    truncated: bool = False
    last_event: str = "reset"

    def copy(self) -> "BreakoutState":
        return BreakoutState(
            ball=self.ball.copy(),
            paddle=self.paddle.copy(),
            blocks=[block.copy() for block in self.blocks],
            score=self.score,
            hits=self.hits,
            misses=self.misses,
            step_count=self.step_count,
            last_action=self.last_action,
            terminated=self.terminated,
            truncated=self.truncated,
            last_event=self.last_event,
        )


@dataclass(slots=True)
class BreakoutConfig:
    width: int = 640
    height: int = 480
    mode: Mode = "normal"
    dt: float = 1.0 / 60.0
    paddle_width: float = 88.0
    paddle_height: float = 12.0
    paddle_margin: float = 28.0
    paddle_speed: float = 360.0
    ball_radius: float = 8.0
    ball_speed: float = 280.0
    gravity: float = 420.0
    max_steps: int | None = 3000
    block_rows: int = 2
    block_cols: int = 4
    block_width: float = 92.0
    block_height: float = 24.0
    block_gap: float = 14.0
    block_top: float = 70.0
    reward_block: float = 1.0
    reward_miss: float = -1.0
    reward_step: float = 0.0
    max_reset_angle_deg: float = 35.0
    min_horizontal_speed: float = 80.0
    min_vertical_speed: float = 120.0
    min_ball_speed: float = 220.0
    max_ball_speed: float = 720.0
    render_mode: RenderMode = None
    render_fps: int = 60

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "gravity", "teleport"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.block_rows * self.block_cols > MAX_OBJECTS - 2:
            raise ValueError(f"Breakout supports at most {MAX_OBJECTS - 2} blocks with the shared slot schema")


def clone_state(state: BreakoutState) -> BreakoutState:
    return state.copy()


def _initial_blocks(config: BreakoutConfig) -> list[BlockState]:
    total_width = config.block_cols * config.block_width + (config.block_cols - 1) * config.block_gap
    left = (config.width - total_width) / 2.0
    blocks = []
    for row in range(config.block_rows):
        for col in range(config.block_cols):
            blocks.append(
                BlockState(
                    x=left + col * (config.block_width + config.block_gap),
                    y=config.block_top + row * (config.block_height + config.block_gap),
                    width=config.block_width,
                    height=config.block_height,
                    active=True,
                )
            )
    return blocks


def _initial_ball_velocity(config: BreakoutConfig, rng: random.Random) -> tuple[float, float]:
    angle = math.radians(rng.uniform(-config.max_reset_angle_deg, config.max_reset_angle_deg))
    vx = config.ball_speed * math.sin(angle)
    vy = -config.ball_speed * math.cos(angle)
    if abs(vx) < config.min_horizontal_speed:
        vx = math.copysign(config.min_horizontal_speed, vx if vx != 0.0 else rng.choice([-1.0, 1.0]))
    return vx, vy


def make_initial_state(config: BreakoutConfig, rng: random.Random) -> BreakoutState:
    paddle_x = (config.width - config.paddle_width) / 2.0
    paddle_y = config.height - config.paddle_margin - config.paddle_height
    vx, vy = _initial_ball_velocity(config, rng)
    return BreakoutState(
        ball=BallState(
            x=config.width / 2.0,
            y=paddle_y - config.ball_radius - 4.0,
            vx=vx,
            vy=vy,
            radius=config.ball_radius,
        ),
        paddle=PaddleState(
            x=paddle_x,
            y=paddle_y,
            width=config.paddle_width,
            height=config.paddle_height,
        ),
        blocks=_initial_blocks(config),
    )


def state_to_observation(state: BreakoutState) -> np.ndarray:
    return np.asarray(
        [
            state.ball.x,
            state.ball.y,
            state.ball.vx,
            state.ball.vy,
            state.paddle.x,
            state.paddle.vx,
        ],
        dtype=np.float32,
    )


def state_to_slots(state: BreakoutState, max_objects: int = MAX_OBJECTS) -> tuple[np.ndarray, np.ndarray]:
    slots = np.zeros((int(max_objects), SLOT_DIM), dtype=np.float32)
    mask = np.zeros((int(max_objects),), dtype=np.float32)
    diameter = state.ball.radius * 2.0
    slots[0] = np.asarray(
        [state.ball.x, state.ball.y, state.ball.vx, state.ball.vy, diameter, diameter, OBJECT_TYPE_TO_ID["ball"]],
        dtype=np.float32,
    )
    slots[1] = np.asarray(
        [
            state.paddle.x,
            state.paddle.y,
            state.paddle.vx,
            0.0,
            state.paddle.width,
            state.paddle.height,
            OBJECT_TYPE_TO_ID["paddle"],
        ],
        dtype=np.float32,
    )
    mask[:2] = 1.0
    for idx, block in enumerate(state.blocks[: max_objects - 2], start=2):
        slots[idx] = np.asarray(
            [block.x, block.y, 0.0, 0.0, block.width, block.height, OBJECT_TYPE_TO_ID["block"]],
            dtype=np.float32,
        )
        mask[idx] = 1.0 if block.active else 0.0
    return slots, mask


def _circle_rect_overlap(ball: BallState, rect: PaddleState | BlockState) -> bool:
    closest_x = _clamp(ball.x, rect.x, rect.x + rect.width)
    closest_y = _clamp(ball.y, rect.y, rect.y + rect.height)
    return (ball.x - closest_x) ** 2 + (ball.y - closest_y) ** 2 <= ball.radius**2


def _apply_paddle_bounce(ball: BallState, paddle: PaddleState, config: BreakoutConfig) -> None:
    speed = _clamp(math.hypot(ball.vx, ball.vy), config.min_ball_speed, config.max_ball_speed)
    rel = (ball.x - (paddle.x + paddle.width / 2.0)) / (paddle.width / 2.0)
    rel = _clamp(rel, -1.0, 1.0)
    angle = rel * math.radians(65.0)
    ball.x = _clamp(ball.x, paddle.x, paddle.x + paddle.width)
    ball.y = paddle.y - ball.radius
    ball.vx = speed * math.sin(angle)
    ball.vy = -max(abs(speed * math.cos(angle)), config.min_vertical_speed)


def _apply_block_bounce(ball: BallState, block: BlockState, previous_x: float, previous_y: float) -> None:
    from_left = previous_x + ball.radius <= block.x
    from_right = previous_x - ball.radius >= block.x + block.width
    from_top = previous_y + ball.radius <= block.y
    from_bottom = previous_y - ball.radius >= block.y + block.height
    if from_left or from_right:
        ball.vx = -ball.vx
    elif from_top or from_bottom:
        ball.vy = -ball.vy
    else:
        ball.vy = -ball.vy


def _compute_substeps(state: BreakoutState, config: BreakoutConfig) -> int:
    speed = max(abs(state.ball.vx), abs(state.ball.vy) + abs(config.gravity) * config.dt)
    travel = speed * config.dt
    return max(1, min(12, math.ceil(travel / max(2.0, config.ball_radius))))


def simulate_step(state: BreakoutState, action: int, config: BreakoutConfig) -> tuple[BreakoutState, dict[str, Any]]:
    if action not in ACTION_TO_DIRECTION:
        raise ValueError(f"Invalid action {action}. Expected one of {tuple(ACTION_TO_DIRECTION)}")

    next_state = clone_state(state)
    event_info: dict[str, Any] = {
        "action": action,
        "event": "none",
        "paddle_hit": False,
        "block_hit": False,
        "miss": False,
        "left_wall_bounce": False,
        "top_bounce": False,
        "bottom_bounce": False,
        "wrapped": False,
        "reward": config.reward_step,
    }
    if next_state.terminated or next_state.truncated:
        event_info["event"] = "episode_done"
        event_info["reward"] = 0.0
        return next_state, event_info

    next_state.step_count += 1
    next_state.last_action = action
    direction = ACTION_TO_DIRECTION[action]
    previous_paddle_x = next_state.paddle.x
    next_state.paddle.x = _clamp(
        previous_paddle_x + direction * config.paddle_speed * config.dt,
        0.0,
        config.width - next_state.paddle.width,
    )
    next_state.paddle.vx = (next_state.paddle.x - previous_paddle_x) / config.dt

    ball = next_state.ball
    substeps = _compute_substeps(next_state, config)
    sub_dt = config.dt / substeps

    for _ in range(substeps):
        previous_x = ball.x
        previous_y = ball.y
        if config.mode == "gravity":
            ball.vy += config.gravity * sub_dt
        ball.x += ball.vx * sub_dt
        ball.y += ball.vy * sub_dt

        if config.mode == "teleport":
            span = config.width + 2.0 * ball.radius
            wrapped = False
            while ball.x < -ball.radius:
                ball.x += span
                wrapped = True
            while ball.x > config.width + ball.radius:
                ball.x -= span
                wrapped = True
            if wrapped:
                event_info["wrapped"] = True
                _event_if_empty(event_info, "wrapped")
        else:
            if ball.x < ball.radius:
                ball.x = 2.0 * ball.radius - ball.x
                ball.vx = abs(ball.vx)
                event_info["left_wall_bounce"] = True
                _event_if_empty(event_info, "left_wall_bounce")
            elif ball.x > config.width - ball.radius:
                ball.x = 2.0 * (config.width - ball.radius) - ball.x
                ball.vx = -abs(ball.vx)
                event_info["left_wall_bounce"] = True
                _event_if_empty(event_info, "left_wall_bounce")

        if ball.y < ball.radius:
            ball.y = 2.0 * ball.radius - ball.y
            ball.vy = max(abs(ball.vy), config.min_vertical_speed)
            event_info["top_bounce"] = True
            _event_if_empty(event_info, "top_bounce")

        crossed_paddle = previous_y + ball.radius <= next_state.paddle.y and ball.y + ball.radius >= next_state.paddle.y
        if ball.vy > 0.0 and crossed_paddle and _circle_rect_overlap(ball, next_state.paddle):
            _apply_paddle_bounce(ball, next_state.paddle, config)
            event_info["paddle_hit"] = True
            event_info["event"] = "paddle_hit"
            next_state.last_event = "paddle_hit"

        for block in next_state.blocks:
            if block.active and _circle_rect_overlap(ball, block):
                block.active = False
                next_state.hits += 1
                next_state.score += 1
                event_info["block_hit"] = True
                event_info["reward"] += config.reward_block
                event_info["event"] = "block_hit"
                next_state.last_event = "block_hit"
                _apply_block_bounce(ball, block, previous_x, previous_y)
                break

        if ball.y - ball.radius > config.height:
            next_state.misses += 1
            next_state.terminated = True
            event_info["miss"] = True
            event_info["reward"] += config.reward_miss
            event_info["event"] = "miss"
            next_state.last_event = "miss"
            break

    if not any(block.active for block in next_state.blocks):
        next_state.terminated = True
        if event_info["event"] == "none":
            event_info["event"] = "cleared"
            next_state.last_event = "cleared"
    if not next_state.terminated and config.max_steps is not None and next_state.step_count >= config.max_steps:
        next_state.truncated = True
        if event_info["event"] == "none":
            event_info["event"] = "truncated"
        next_state.last_event = "truncated"
    elif event_info["event"] == "none":
        event_info["event"] = "step"
        next_state.last_event = "step"
    return next_state, event_info


class BreakoutEnv:
    metadata = {"name": "CustomBreakout-v0", "render_modes": [None, "human", "rgb_array"]}
    action_meanings = {ACTION_STAY: "stay", ACTION_LEFT: "left", ACTION_RIGHT: "right"}
    observation_names = OBSERVATION_NAMES

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        mode: Mode = "normal",
        paddle_speed: float = 360.0,
        ball_speed: float = 280.0,
        gravity: float = 420.0,
        dt: float = 1.0 / 60.0,
        max_steps: int | None = 3000,
        render_mode: RenderMode = None,
        seed: int | None = None,
        render_fps: int = 60,
    ) -> None:
        self.config = BreakoutConfig(
            width=width,
            height=height,
            mode=mode,
            dt=dt,
            paddle_speed=paddle_speed,
            ball_speed=ball_speed,
            gravity=gravity,
            max_steps=max_steps,
            render_mode=render_mode,
            render_fps=render_fps,
        )
        self.n_actions = 3
        self.observation_shape = (6,)
        self._rng = random.Random()
        self._state: BreakoutState | None = None
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._font: Any | None = None
        if seed is not None:
            self.seed(seed)

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def is_done(self) -> bool:
        return bool(self._state and (self._state.terminated or self._state.truncated))

    def seed(self, seed: int | None = None) -> int | None:
        self._rng.seed(seed)
        return seed

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        if options and "state" in options:
            self._state = clone_state(options["state"])
            self._state.terminated = False
            self._state.truncated = False
            self._state.last_event = "reset"
        else:
            self._state = make_initial_state(self.config, self._rng)
        return self.state_to_observation(), self._build_info({"event": "reset"})

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.terminated or self._state.truncated:
            raise RuntimeError("Episode is finished. Call reset() before step().")
        self._state, event_info = simulate_step(self._state, action, self.config)
        return (
            self.state_to_observation(),
            float(event_info["reward"]),
            self._state.terminated,
            self._state.truncated,
            self._build_info(event_info),
        )

    def get_state(self) -> BreakoutState:
        if self._state is None:
            raise RuntimeError("Call reset() before get_state().")
        return clone_state(self._state)

    def set_state(self, state: BreakoutState) -> None:
        self._state = clone_state(state)

    def state_to_observation(self, state: BreakoutState | None = None) -> np.ndarray:
        return state_to_observation(self.get_state() if state is None else state)

    def state_to_slots(self, state: BreakoutState | None = None) -> tuple[np.ndarray, np.ndarray]:
        return state_to_slots(self.get_state() if state is None else state)

    def predict_next_state(self, state: BreakoutState, action: int) -> tuple[BreakoutState, dict[str, Any]]:
        return simulate_step(state, action, self.config)

    def sample_random_action(self) -> int:
        return self._rng.randrange(self.n_actions)

    def _build_info(self, event_info: dict[str, Any]) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Call reset() before accessing info.")
        return {
            "mode": self.config.mode,
            "event": event_info.get("event", "none"),
            "action": event_info.get("action", self._state.last_action),
            "paddle_hit": bool(event_info.get("paddle_hit", False)),
            "block_hit": bool(event_info.get("block_hit", False)),
            "miss": bool(event_info.get("miss", False)),
            "left_wall_bounce": bool(event_info.get("left_wall_bounce", False)),
            "top_bounce": bool(event_info.get("top_bounce", False)),
            "bottom_bounce": bool(event_info.get("bottom_bounce", False)),
            "wrapped": bool(event_info.get("wrapped", False)),
            "score": self._state.score,
            "hits": self._state.hits,
            "misses": self._state.misses,
            "active_blocks": sum(1 for block in self._state.blocks if block.active),
            "step_count": self._state.step_count,
            "terminated": self._state.terminated,
            "truncated": self._state.truncated,
            "observation_names": self.observation_names,
        }

    def render(self) -> np.ndarray | None:
        if self._state is None:
            raise RuntimeError("Call reset() before render().")
        if self.config.render_mode is None:
            return None
        if self.config.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.config.render_mode == "human":
            self._render_human()
            return None
        raise ValueError(f"Unsupported render_mode: {self.config.render_mode}")

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._font = None

    def _render_rgb_array(self) -> np.ndarray:
        state = self.get_state()
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        for block in state.blocks:
            if block.active:
                x0 = max(0, int(round(block.x)))
                y0 = max(0, int(round(block.y)))
                x1 = min(self.config.width, int(round(block.x + block.width)))
                y1 = min(self.config.height, int(round(block.y + block.height)))
                frame[y0:y1, x0:x1] = (80, 180, 255)
        paddle = state.paddle
        px0 = max(0, int(round(paddle.x)))
        py0 = max(0, int(round(paddle.y)))
        px1 = min(self.config.width, int(round(paddle.x + paddle.width)))
        py1 = min(self.config.height, int(round(paddle.y + paddle.height)))
        frame[py0:py1, px0:px1] = 255
        cx = int(round(state.ball.x))
        cy = int(round(state.ball.y))
        radius = int(math.ceil(state.ball.radius))
        y0 = max(0, cy - radius)
        y1 = min(self.config.height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(self.config.width, cx + radius + 1)
        if y0 < y1 and x0 < x1:
            ys, xs = np.ogrid[y0:y1, x0:x1]
            mask = (xs - state.ball.x) ** 2 + (ys - state.ball.y) ** 2 <= state.ball.radius**2
            frame[y0:y1, x0:x1][mask] = 255
        return frame

    def _render_human(self) -> None:
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise ImportError("pygame is required for render_mode='human'. Install with `pip install pygame`.") from exc
            pygame.init()
            pygame.font.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode((self.config.width, self.config.height))
            self._font = pygame.font.SysFont("consolas", 18)
        pygame = self._pygame
        assert self._screen is not None
        state = self.get_state()
        self._screen.fill((0, 0, 0))
        for block in state.blocks:
            if block.active:
                pygame.draw.rect(
                    self._screen,
                    (80, 180, 255),
                    pygame.Rect(int(block.x), int(block.y), int(block.width), int(block.height)),
                )
        pygame.draw.rect(
            self._screen,
            (255, 255, 255),
            pygame.Rect(int(state.paddle.x), int(state.paddle.y), int(state.paddle.width), int(state.paddle.height)),
        )
        pygame.draw.circle(self._screen, (255, 255, 255), (int(state.ball.x), int(state.ball.y)), int(state.ball.radius))
        if self._font is not None:
            overlay = f"mode={self.config.mode} blocks={sum(b.active for b in state.blocks)} steps={state.step_count}"
            self._screen.blit(self._font.render(overlay, True, (255, 255, 255)), (10, 10))
        pygame.display.flip()


__all__ = [
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STAY",
    "BallState",
    "BlockState",
    "BreakoutConfig",
    "BreakoutEnv",
    "BreakoutState",
    "PaddleState",
    "clone_state",
    "make_initial_state",
    "simulate_step",
    "state_to_observation",
    "state_to_slots",
]
