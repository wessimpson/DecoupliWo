from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


_JVM_STARTED = False


def _default_gvgai_root() -> Path:
	vendored_root = Path(__file__).resolve().parents[1] / "gvgai" / "gym_gvgai" / "envs" / "gvgai"
	if vendored_root.is_dir():
		return vendored_root
	try:
		import gym_gvgai
	except ImportError as exc:
		raise ImportError(
			"gym_gvgai is not importable. Install the JPype fork with "
			"`pip install -e training_data_gvgai/gvgai` and run "
			"`python training_data_gvgai/gvgai/build.py` first."
		) from exc
	return Path(gym_gvgai.__file__).resolve().parent / "envs" / "gvgai"


def _ensure_jvm(gvgai_root: Path) -> None:
	global _JVM_STARTED
	if _JVM_STARTED:
		return

	import jpype
	import jpype.imports  # noqa: F401

	if jpype.isJVMStarted():
		_JVM_STARTED = True
		return

	build_dir = gvgai_root / "GVGAI_Build"
	if not build_dir.is_dir():
		raise FileNotFoundError(f"GVGAI build directory not found: {build_dir}")

	jpype.startJVM(
		jpype.getDefaultJVMPath(),
		f"-Djava.class.path={build_dir}",
		"-Djava.awt.headless=true",
		"-Xmx1024m",
		convertStrings=False,
	)
	_JVM_STARTED = True


class GVGAIFileEnv(gym.Env):
	"""Gymnasium env over the GVGAI_jpype Java bridge using explicit VGDL files."""

	metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

	def __init__(
		self,
		game_file: str | Path,
		level_files: Sequence[str | Path],
		level: int = 0,
		gvgai_root: str | Path | None = None,
		max_episode_steps: int = 2_000,
	) -> None:
		super().__init__()
		self.game_file = str(Path(game_file).expanduser().resolve())
		self.level_files = [str(Path(level_file).expanduser().resolve()) for level_file in level_files]
		self.level = int(level)
		self.max_episode_steps = int(max_episode_steps)
		self._elapsed_steps = 0
		self._gvgai_root = Path(gvgai_root).expanduser().resolve() if gvgai_root else _default_gvgai_root()

		if not Path(self.game_file).is_file():
			raise FileNotFoundError(f"GVGAI game file not found: {self.game_file}")
		if not self.level_files:
			raise ValueError("level_files must not be empty")
		for level_file in self.level_files:
			if level_file and not Path(level_file).is_file():
				raise FileNotFoundError(f"GVGAI level file not found: {level_file}")

		_ensure_jvm(self._gvgai_root)

		import jpype
		from jpype import JClass

		competition_parameters = JClass("core.competition.CompetitionParameters")
		sprites_dir = self._gvgai_root / "sprites"
		competition_parameters.IMG_PATH = str(sprites_dir) + "/"

		bridge_cls = JClass("core.game.GVGAIBridge")
		self._bridge = bridge_cls(self.game_file, jpype.JArray(jpype.JString)(self.level_files))

		self._bridge.reset(self.level, 0)
		self._actions = self._get_actions()
		self._current_img = self._get_image()

		self.action_space = spaces.Discrete(len(self._actions))
		self.observation_space = spaces.Box(low=0, high=255, shape=self._current_img.shape, dtype=np.uint8)

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		self._elapsed_steps = 0
		if options and "level" in options:
			self.level = int(options["level"])
		rng_seed = int(seed if seed is not None else 0)
		self._bridge.reset(self.level, rng_seed)
		self._actions = self._get_actions()
		self._current_img = self._get_image()
		return self._current_img, self._info()

	def step(self, action: int):
		action_id = int(action)
		if action_id < 0 or action_id >= len(self._actions):
			raise ValueError(f"action {action_id} outside [0, {len(self._actions)})")

		self._bridge.step(self._actions[action_id])
		self._elapsed_steps += 1
		reward = float(self._bridge.getScoreDelta())
		terminated = bool(self._bridge.isGameOver())
		truncated = self._elapsed_steps >= self.max_episode_steps and not terminated
		self._current_img = self._get_image()
		return self._current_img, reward, terminated, truncated, self._info()

	def render(self):
		return self._current_img[:, :, :3]

	def close(self) -> None:
		return None

	def get_action_meanings(self) -> list[str]:
		return list(self._actions)

	def _get_actions(self) -> list[str]:
		java_actions = self._bridge.getAvailableActions()
		return ["ACTION_NIL"] + [str(action) for action in java_actions]

	def _get_image(self) -> np.ndarray:
		raw = bytes(self._bridge.renderToBytes())
		if not raw:
			if hasattr(self, "_current_img"):
				return np.zeros_like(self._current_img)
			return np.zeros((64, 64, 4), dtype=np.uint8)
		return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)

	def _info(self) -> dict:
		avatar = np.asarray(self._bridge.getAvatarPosition(), dtype=np.float32)
		ascii_state = str(self._bridge.getObservationString())
		return {
			"actions": list(self._actions),
			"ascii": ascii_state,
			"avatar_xy": avatar,
			"block_size": int(self._bridge.getBlockSize()),
			"game_tick": int(self._bridge.getGameTick()),
			"score": float(self._bridge.getGameScore()),
			"winner": str(self._bridge.getWinner()),
		}
