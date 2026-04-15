from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

Mode = Literal["normal", "gravity", "teleport"]
RenderMode = Literal["human", "rgb_array"] | None

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_TO_DIRECTION = {ACTION_STAY: 0.0, ACTION_UP: -1.0, ACTION_DOWN: 1.0}
OBSERVATION_NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_y", "paddle_vy")


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

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy, "radius": self.radius}


@dataclass(slots=True)
class PaddleState:
    x: float
    y: float
    width: float
    height: float
    vy: float = 0.0

    def copy(self) -> "PaddleState":
        return replace(self)

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "vy": self.vy,
        }


@dataclass(slots=True)
class GameState:
    ball: BallState
    paddle: PaddleState
    score: int = 0
    hits: int = 0
    misses: int = 0
    step_count: int = 0
    last_action: int = ACTION_STAY
    terminated: bool = False
    truncated: bool = False
    last_event: str = "reset"

    def copy(self) -> "GameState":
        return GameState(
            ball=self.ball.copy(),
            paddle=self.paddle.copy(),
            score=self.score,
            hits=self.hits,
            misses=self.misses,
            step_count=self.step_count,
            last_action=self.last_action,
            terminated=self.terminated,
            truncated=self.truncated,
            last_event=self.last_event,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ball": self.ball.as_dict(),
            "paddle": self.paddle.as_dict(),
            "score": self.score,
            "hits": self.hits,
            "misses": self.misses,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "last_event": self.last_event,
        }


@dataclass(slots=True)
class PongConfig:
    width: int = 640
    height: int = 480
    mode: Mode = "normal"
    dt: float = 1.0 / 60.0
    paddle_width: float = 12.0
    paddle_height: float = 88.0
    paddle_margin: float = 24.0
    paddle_speed: float = 360.0
    ball_radius: float = 8.0
    ball_speed: float = 280.0
    gravity: float = 420.0
    max_steps: int | None = 3000
    reward_hit: float = 1.0
    reward_miss: float = -1.0
    reward_step: float = 0.0
    normalize_observation: bool = False
    random_reset_angle: bool = True
    max_reset_angle_deg: float = 25.0
    fixed_reset_angle_deg: float = 15.0
    max_bounce_angle_deg: float = 60.0
    min_horizontal_speed: float = 120.0
    min_vertical_bounce_speed: float = 90.0
    min_ball_speed: float = 220.0
    max_ball_speed: float = 720.0
    speedup_on_paddle_hit: float = 1.0
    render_mode: RenderMode = None
    render_fps: int = 60

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "gravity", "teleport"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.paddle_width <= 0.0 or self.paddle_height <= 0.0:
            raise ValueError("Paddle dimensions must be positive")
        if self.ball_radius <= 0.0 or self.ball_speed <= 0.0:
            raise ValueError("Ball radius and speed must be positive")
        if self.min_ball_speed <= 0.0 or self.max_ball_speed <= 0.0:
            raise ValueError("Ball speed limits must be positive")
        if self.min_ball_speed > self.max_ball_speed:
            raise ValueError("min_ball_speed must not exceed max_ball_speed")
        if self.speedup_on_paddle_hit <= 0.0:
            raise ValueError("speedup_on_paddle_hit must be positive")
        if self.render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive")


def clone_state(state: GameState) -> GameState:
    return state.copy()


def _initial_ball_velocity(config: PongConfig, rng: random.Random) -> tuple[float, float]:
    angle_deg = (
        rng.uniform(-config.max_reset_angle_deg, config.max_reset_angle_deg)
        if config.random_reset_angle
        else config.fixed_reset_angle_deg
    )
    angle = math.radians(angle_deg)
    return -config.ball_speed * math.cos(angle), config.ball_speed * math.sin(angle)


def make_initial_state(config: PongConfig, rng: random.Random) -> GameState:
    paddle_x = config.width - config.paddle_margin - config.paddle_width
    paddle_y = (config.height - config.paddle_height) / 2.0
    ball_vx, ball_vy = _initial_ball_velocity(config, rng)
    return GameState(
        ball=BallState(
            x=config.width / 2.0,
            y=config.height / 2.0,
            vx=ball_vx,
            vy=ball_vy,
            radius=config.ball_radius,
        ),
        paddle=PaddleState(
            x=paddle_x,
            y=paddle_y,
            width=config.paddle_width,
            height=config.paddle_height,
        ),
    )


def state_to_observation(
    state: GameState,
    config: PongConfig,
    normalize: bool | None = None,
) -> np.ndarray:
    use_normalized = config.normalize_observation if normalize is None else normalize
    raw = np.array(
        [
            state.ball.x,
            state.ball.y,
            state.ball.vx,
            state.ball.vy,
            state.paddle.y,
            state.paddle.vy,
        ],
        dtype=np.float32,
    )
    if not use_normalized:
        return raw

    paddle_y_scale = max(1.0, config.height - state.paddle.height)
    velocity_scale = max(config.max_ball_speed, config.ball_speed, 1.0)
    return np.array(
        [
            raw[0] / config.width,
            raw[1] / config.height,
            np.clip(raw[2] / velocity_scale, -1.0, 1.0),
            np.clip(raw[3] / velocity_scale, -1.0, 1.0),
            raw[4] / paddle_y_scale,
            np.clip(raw[5] / max(config.paddle_speed, 1.0), -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def state_to_dict_observation(
    state: GameState,
    config: PongConfig,
    normalize: bool | None = None,
) -> dict[str, float]:
    values = state_to_observation(state, config, normalize=normalize)
    return {name: float(value) for name, value in zip(OBSERVATION_NAMES, values)}


def _compute_substeps(state: GameState, config: PongConfig) -> int:
    ball = state.ball
    max_vertical_speed = abs(ball.vy) + abs(config.gravity) * config.dt
    max_component_speed = max(abs(ball.vx), max_vertical_speed)
    max_travel = max_component_speed * config.dt
    travel_limit = max(2.0, min(ball.radius, config.paddle_width * 0.5))
    return max(1, min(12, math.ceil(max_travel / travel_limit)))


def _ball_overlaps_paddle(ball: BallState, paddle: PaddleState) -> bool:
    return (
        paddle.x - ball.radius <= ball.x <= paddle.x + paddle.width + ball.radius
        and paddle.y - ball.radius <= ball.y <= paddle.y + paddle.height + ball.radius
    )


def _handle_left_wall(ball: BallState, config: PongConfig, event_info: dict[str, Any]) -> None:
    if ball.x - ball.radius < 0.0:
        ball.x = 2.0 * ball.radius - ball.x
        ball.vx = max(abs(ball.vx), config.min_horizontal_speed)
        event_info["left_wall_bounce"] = True
        _event_if_empty(event_info, "left_wall_bounce")


def _handle_vertical_bounds(ball: BallState, config: PongConfig, event_info: dict[str, Any]) -> None:
    if config.mode == "teleport":
        span = config.height + 2.0 * ball.radius
        wrapped = False
        while ball.y < -ball.radius:
            ball.y += span
            wrapped = True
        while ball.y > config.height + ball.radius:
            ball.y -= span
            wrapped = True
        if wrapped:
            event_info["wrapped"] = True
            _event_if_empty(event_info, "wrapped")
        return

    ceiling = ball.radius
    floor = config.height - ball.radius
    if ball.y < ceiling:
        ball.y = 2.0 * ceiling - ball.y
        ball.vy = max(abs(ball.vy), config.min_vertical_bounce_speed)
        event_info["top_bounce"] = True
        _event_if_empty(event_info, "top_bounce")
    elif ball.y > floor:
        ball.y = 2.0 * floor - ball.y
        ball.vy = -max(abs(ball.vy), config.min_vertical_bounce_speed)
        event_info["bottom_bounce"] = True
        _event_if_empty(event_info, "bottom_bounce")


def _apply_paddle_bounce(ball: BallState, paddle: PaddleState, config: PongConfig) -> None:
    current_speed = math.hypot(ball.vx, ball.vy) * config.speedup_on_paddle_hit
    max_angle = math.radians(config.max_bounce_angle_deg)
    min_speed_for_angle = config.min_horizontal_speed / max(math.cos(max_angle), 1e-6)
    speed = _clamp(current_speed, max(config.min_ball_speed, min_speed_for_angle), config.max_ball_speed)

    relative_hit = (ball.y - (paddle.y + paddle.height / 2.0)) / (paddle.height / 2.0)
    relative_hit = _clamp(relative_hit, -1.0, 1.0)
    angle = relative_hit * max_angle

    ball.x = paddle.x - ball.radius
    ball.vx = -speed * math.cos(angle)
    ball.vy = speed * math.sin(angle)

    if config.mode == "gravity" and abs(ball.vy) < config.min_vertical_bounce_speed * 0.5 and abs(relative_hit) > 0.2:
        ball.vy = math.copysign(config.min_vertical_bounce_speed * 0.5, relative_hit)


def simulate_step(state: GameState, action: int, config: PongConfig) -> tuple[GameState, dict[str, Any]]:
    if action not in ACTION_TO_DIRECTION:
        raise ValueError(f"Invalid action {action}. Expected one of {tuple(ACTION_TO_DIRECTION)}")

    next_state = clone_state(state)
    event_info: dict[str, Any] = {
        "action": action,
        "event": "none",
        "paddle_hit": False,
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
    previous_paddle_y = next_state.paddle.y
    requested_paddle_y = previous_paddle_y + direction * config.paddle_speed * config.dt
    next_state.paddle.y = _clamp(requested_paddle_y, 0.0, config.height - next_state.paddle.height)
    next_state.paddle.vy = (next_state.paddle.y - previous_paddle_y) / config.dt

    ball = next_state.ball
    substeps = _compute_substeps(next_state, config)
    sub_dt = config.dt / substeps

    for _ in range(substeps):
        previous_x = ball.x

        if config.mode == "gravity":
            ball.vy += config.gravity * sub_dt

        ball.x += ball.vx * sub_dt
        ball.y += ball.vy * sub_dt

        _handle_left_wall(ball, config, event_info)
        _handle_vertical_bounds(ball, config, event_info)

        crossed_paddle_face = previous_x + ball.radius <= next_state.paddle.x and ball.x + ball.radius >= next_state.paddle.x
        if ball.vx > 0.0 and crossed_paddle_face and _ball_overlaps_paddle(ball, next_state.paddle):
            _apply_paddle_bounce(ball, next_state.paddle, config)
            next_state.hits += 1
            next_state.score += 1
            event_info["paddle_hit"] = True
            event_info["reward"] += config.reward_hit
            event_info["event"] = "paddle_hit"
            next_state.last_event = "paddle_hit"

        if ball.x - ball.radius > config.width:
            next_state.misses += 1
            next_state.terminated = True
            event_info["miss"] = True
            event_info["reward"] += config.reward_miss
            event_info["event"] = "miss"
            next_state.last_event = "miss"
            break

    if not next_state.terminated and config.max_steps is not None and next_state.step_count >= config.max_steps:
        next_state.truncated = True
        if event_info["event"] == "none":
            event_info["event"] = "truncated"
        next_state.last_event = "truncated"
    elif event_info["event"] == "none":
        event_info["event"] = "step"
        next_state.last_event = "step"
    elif next_state.last_event == "reset":
        next_state.last_event = event_info["event"]

    return next_state, event_info


class PongEnv:
    """Lightweight single-player Pong environment with Gym-like semantics."""

    metadata = {"name": "CustomPong-v0", "render_modes": [None, "human", "rgb_array"]}
    action_meanings = {ACTION_STAY: "stay", ACTION_UP: "up", ACTION_DOWN: "down"}
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
        normalize_observation: bool = False,
        random_reset_angle: bool = True,
        reward_hit: float = 1.0,
        reward_miss: float = -1.0,
        reward_step: float = 0.0,
        speedup_on_paddle_hit: float = 1.0,
        render_fps: int = 60,
    ) -> None:
        self.config = PongConfig(
            width=width,
            height=height,
            mode=mode,
            dt=dt,
            paddle_speed=paddle_speed,
            ball_speed=ball_speed,
            gravity=gravity,
            max_steps=max_steps,
            reward_hit=reward_hit,
            reward_miss=reward_miss,
            reward_step=reward_step,
            normalize_observation=normalize_observation,
            random_reset_angle=random_reset_angle,
            speedup_on_paddle_hit=speedup_on_paddle_hit,
            render_mode=render_mode,
            render_fps=render_fps,
        )
        self.n_actions = 3
        self.observation_shape = (6,)
        self._rng = random.Random()
        self._seed: int | None = None
        self._state: GameState | None = None
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._font: Any | None = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int | None = None) -> int | None:
        self._seed = seed
        self._rng.seed(seed)
        return seed

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def is_done(self) -> bool:
        return bool(self._state and (self._state.terminated or self._state.truncated))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
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

    def get_state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("Call reset() before get_state().")
        return clone_state(self._state)

    def clone_state(self, state: GameState | None = None) -> GameState:
        return clone_state(self.get_state() if state is None else state)

    def set_state(self, state: GameState) -> None:
        self._state = clone_state(state)

    def state_to_observation(
        self,
        state: GameState | None = None,
        normalize: bool | None = None,
    ) -> np.ndarray:
        return state_to_observation(self.get_state() if state is None else state, self.config, normalize=normalize)

    def state_to_dict_observation(
        self,
        state: GameState | None = None,
        normalize: bool | None = None,
    ) -> dict[str, float]:
        return state_to_dict_observation(self.get_state() if state is None else state, self.config, normalize=normalize)

    def predict_next_state(self, state: GameState, action: int) -> tuple[GameState, dict[str, Any]]:
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
            "miss": bool(event_info.get("miss", False)),
            "left_wall_bounce": bool(event_info.get("left_wall_bounce", False)),
            "top_bounce": bool(event_info.get("top_bounce", False)),
            "bottom_bounce": bool(event_info.get("bottom_bounce", False)),
            "wrapped": bool(event_info.get("wrapped", False)),
            "score": self._state.score,
            "hits": self._state.hits,
            "misses": self._state.misses,
            "step_count": self._state.step_count,
            "terminated": self._state.terminated,
            "truncated": self._state.truncated,
            "observation_names": self.observation_names,
        }

    def _render_rgb_array(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Call reset() before render().")
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        frame[:, :2] = 255
        paddle = self._state.paddle
        ball = self._state.ball

        px0 = max(0, int(round(paddle.x)))
        px1 = min(self.config.width, int(round(paddle.x + paddle.width)))
        py0 = max(0, int(round(paddle.y)))
        py1 = min(self.config.height, int(round(paddle.y + paddle.height)))
        frame[py0:py1, px0:px1] = 255

        cx = int(round(ball.x))
        cy = int(round(ball.y))
        radius = int(math.ceil(ball.radius))
        y0 = max(0, cy - radius)
        y1 = min(self.config.height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(self.config.width, cx + radius + 1)
        if y0 < y1 and x0 < x1:
            ys, xs = np.ogrid[y0:y1, x0:x1]
            mask = (xs - ball.x) ** 2 + (ys - ball.y) ** 2 <= ball.radius ** 2
            frame[y0:y1, x0:x1][mask] = 255
        return frame

    def _render_human(self) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before render().")
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise ImportError(
                    "pygame is required for render_mode='human'. Install it with `pip install pygame`."
                ) from exc
            pygame.init()
            pygame.font.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode((self.config.width, self.config.height))
            self._font = pygame.font.SysFont("consolas", 18)

        pygame = self._pygame
        pygame.display.set_caption(f"Custom Pong - {self.config.mode}")
        assert self._screen is not None

        self._screen.fill((0, 0, 0))
        pygame.draw.line(self._screen, (255, 255, 255), (1, 0), (1, self.config.height), 2)
        paddle = self._state.paddle
        ball = self._state.ball
        pygame.draw.rect(
            self._screen,
            (255, 255, 255),
            pygame.Rect(int(round(paddle.x)), int(round(paddle.y)), int(round(paddle.width)), int(round(paddle.height))),
        )
        pygame.draw.circle(
            self._screen,
            (255, 255, 255),
            (int(round(ball.x)), int(round(ball.y))),
            int(round(ball.radius)),
        )

        if self._font is not None:
            overlay = f"mode={self.config.mode} hits={self._state.hits} steps={self._state.step_count}"
            self._screen.blit(self._font.render(overlay, True, (255, 255, 255)), (10, 10))
            if self._state.terminated or self._state.truncated:
                self._screen.blit(
                    self._font.render("Episode ended - press R to reset", True, (255, 255, 255)),
                    (10, 34),
                )
        pygame.display.flip()


__all__ = [
    "ACTION_DOWN",
    "ACTION_STAY",
    "ACTION_UP",
    "BallState",
    "GameState",
    "Mode",
    "PaddleState",
    "PongConfig",
    "PongEnv",
    "clone_state",
    "make_initial_state",
    "simulate_step",
    "state_to_dict_observation",
    "state_to_observation",
]
