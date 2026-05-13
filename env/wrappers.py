from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AttachPlayerInfo(gym.Wrapper):
	"""Adds 'player_xy' to info dict each step, extracted from OCAtari objects."""

	def reset(self, seed=None, options=None):
		return self.env.reset(seed=seed, options=options)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		player_xy = None
		objects = getattr(self.env.unwrapped, "objects", None)
		if objects:
			for obj in objects:
				if getattr(obj, "category", None) == "Player":
					player_xy = getattr(obj, "xy", None)
					break
		info = dict(info) if info is not None else {}
		info["player_xy"] = player_xy
		return obs, reward, terminated, truncated, info


class EnsureUint8Obs(gym.ObservationWrapper):
	"""Cast observations to uint8 and fix observation_space dtype accordingly."""

	def __init__(self, env: gym.Env):
		super().__init__(env)
		obs_space = self.env.observation_space
		if not isinstance(obs_space, spaces.Box):
			raise TypeError("EnsureUint8Obs only supports Box observation spaces")
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=obs_space.shape,
			dtype=np.uint8,
		)

	def observation(self, observation):
		# Assumes observation within [0, 255] or [0,1]; scale if needed
		obs = np.asarray(observation)
		if obs.dtype != np.uint8:
			# If inputs look like [0,1], rescale to [0,255]
			if obs.max() <= 1.0:
				obs = (obs * 255.0).round()
			obs = obs.clip(0, 255).astype(np.uint8)
		return obs


class HoldActionBetweenDecisions(gym.Wrapper):
	"""Repeat the last executed action until the next decision step."""

	def __init__(self, env: gym.Env, interval: int) -> None:
		if interval < 1:
			raise ValueError("interval must be >= 1")
		super().__init__(env)
		self.interval = interval
		self._step_idx = 0
		self._held = 0

	def reset(self, seed=None, options=None):
		self._step_idx = 0
		self._held = 0
		return self.env.reset(seed=seed, options=options)

	def step(self, action):
		if self._step_idx % self.interval == 0:
			self._held = int(action)
		obs, reward, terminated, truncated, info = self.env.step(self._held)
		self._step_idx += 1
		return obs, reward, terminated, truncated, info

def main() -> None:
	# Minimal smoke test: build Space Invaders via shared builder and print player_xy once
	from env.env import make_atari_env
	env = make_atari_env(
		game="space_invaders",
		render_mode=None,
		mode="ram",
		obs_mode="ori",
		hud=True,
		frameskip=1,
		decision_interval=1,
	)
	env = AttachPlayerInfo(env)
	obs, info = env.reset()
	print("reset player_xy:", info.get("player_xy"))
	obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
	print("step player_xy:", info.get("player_xy"))
	env.close()

if __name__ == "__main__":
	main()
