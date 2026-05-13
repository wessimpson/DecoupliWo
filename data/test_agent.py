import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv

from env.wrappers import AttachPlayerInfo, EnsureUint8Obs
from env.env import make_env_factory


DEFAULT_CHECKPOINT_DIR = Path("agent") / "checkpoints"
ACTION_DECISION_INTERVAL = 8


def build_vec_env(env_name: str, render_mode: Optional[str], obs_mode: str, frameskip: int, decision_interval: int, hud: bool) -> VecEnv:
	make_env_fn = make_env_factory(
		game=env_name,
		render_mode=render_mode,
		mode="ram",
		obs_mode=obs_mode,
		hud=hud,
		frameskip=frameskip,
		decision_interval=decision_interval,
	)
	vec = make_vec_env(
		make_env_fn,
		n_envs=1,
		vec_env_cls=DummyVecEnv,
		wrapper_class=lambda e: EnsureUint8Obs(AttachPlayerInfo(e)),
		env_kwargs={},
	)
	# Stack 4 frames only if using raw RGB ('ori'); 'dqn' is already stacked.
	if obs_mode != "dqn":
		vec = VecFrameStack(vec, n_stack=4, channels_order="last")
	return vec


def resolve_checkpoint_path(env_name: str, checkpoint_path: Optional[str]) -> Path:
	if checkpoint_path:
		path = Path(checkpoint_path)
		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {path}")
		return path
	# Default layout: agent/checkpoints/<env>/<env>_ppo.zip
	default_path = DEFAULT_CHECKPOINT_DIR / env_name / f"{env_name}_ppo.zip"
	if not default_path.exists():
		raise FileNotFoundError(
			f"Default checkpoint not found: {default_path}\n"
			f"Provide --checkpoint to specify a custom path."
		)
	return default_path


def play_interactive(
	env_name: str,
	checkpoint_path: Optional[str],
	max_steps: int,
	fps: int,
	decision_interval: int,
	deterministic: bool,
	obs_mode: str,
	frameskip: int,
	hud: bool,
) -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	vec = build_vec_env(env_name, render_mode="human", obs_mode=obs_mode, frameskip=frameskip, decision_interval=decision_interval, hud=hud)
	model_path = resolve_checkpoint_path(env_name, checkpoint_path)
	model: PPO = PPO.load(model_path, device=device)

	obs = vec.reset()
	# SB3 VecEnv reset returns np.ndarray with shape (n_envs, ...) even for n_envs=1
	assert isinstance(obs, np.ndarray)
	last_action = None
	step_counter = 0
	frame_duration = 1.0 / max(1, fps)

	try:
		while step_counter < max_steps:
			# Choose action only every `decision_interval` env steps, repeat otherwise.
			if (step_counter % decision_interval == 0) or last_action is None:
				action, _ = model.predict(obs, deterministic=deterministic)
				last_action = action
			else:
				action = last_action

			obs, rewards, dones, infos = vec.step(action)

			# Explicitly render the first (and only) sub-env.
			try:
				vec.envs[0].render()
			except Exception:
				# Fallback to VecEnv render (no-op for some envs).
				_ = vec.render(mode="human")

			step_counter += 1

			# Handle episode termination
			if dones[0]:
				obs = vec.reset()
				last_action = None

			# Basic real-time pacing
			time.sleep(frame_duration)
	finally:
		vec.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a trained PPO agent and render interactively.")
	parser.add_argument(
		"--env",
		type=str,
		default="space_invaders",
		help="Environment module name under env/ (e.g., space_invaders)",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Path to PPO checkpoint (.zip). Defaults to agent/checkpoints/<env>/<env>_ppo.zip",
	)
	parser.add_argument(
		"--max_steps",
		type=int,
		default=50_000,
		help="Maximum number of environment steps to run.",
	)
	parser.add_argument(
		"--fps",
		type=int,
		default=30,
		help="Approximate render FPS for pacing the loop.",
	)
	parser.add_argument(
		"--decision_interval",
		type=int,
		default=ACTION_DECISION_INTERVAL,
		help="Repeat the chosen action for this many steps (matches training).",
	)
	parser.add_argument(
		"--deterministic",
		action="store_true",
		help="Use deterministic policy for action selection.",
	)
	parser.add_argument(
		"--obs_mode",
		type=str,
		default="ori",
		choices=["ori", "dqn"],
		help="Observation mode from OCAtari ('ori' RGB or 'dqn' stacked 84x84).",
	)
	parser.add_argument(
		"--frameskip",
		type=int,
		default=1,
		help="OCAtari frameskip value (1 = no skip).",
	)
	parser.add_argument(
		"--hud",
		type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
		default=True,
		help="Whether to show HUD elements in the observation.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	play_interactive(
		env_name=args.env,
		checkpoint_path=args.checkpoint,
		max_steps=args.max_steps,
		fps=args.fps,
		decision_interval=args.decision_interval,
		deterministic=args.deterministic,
		obs_mode=args.obs_mode,
		frameskip=args.frameskip,
		hud=args.hud,
	)


if __name__ == "__main__":
	main()

