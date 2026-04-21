"""
Collect transitions from the SB3 PPO agent on Atari-style envs.

Layout matches GVGAI headless runs from ``RunDataCollectionAgent`` (Java): each shard directory
contains ``obs.npy``, ``action.npy``, ``n_actions.npy``, ``player_x.npy``, ``player_y.npy``.
This script writes under ``data/transitions/train/<env>/``. GVGAI ``RunDataCollectionAgent`` writes the
same shard layout to ``data/transitions/train/<game_stem>/``.
``obs.npy`` here is stacked RGB uint8; GVGAI uses a fixed grid encoding uint8 ``[N,H,W,3]``.

GVGAI MCTS collection (separate repo, not this file): ``examples/data_collection/run_mcts_data_collection.ps1``.
"""
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecEnv
from tqdm import tqdm

from env.env import make_env_factory
from env.wrappers import AttachPlayerInfo, EnsureUint8Obs


DEFAULT_CHECKPOINT_DIR = Path("agent") / "checkpoints"
TRANSITIONS_DIR = Path("data") / "transitions"
# Match training configuration
DECISION_INTERVAL = 4
NUM_ACTIONS = 18


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


def build_vec_env(
	env_name: str,
	render_mode: Optional[str],
	n_envs: int,
) -> VecEnv:
	make_env_fn = make_env_factory(
		game=env_name,
		render_mode=render_mode,
		mode="ram",
		obs_mode="ori",
		hud=True,
		frameskip=1,
		decision_interval=DECISION_INTERVAL,
	)
	vec = make_vec_env(
		make_env_fn,
		n_envs=n_envs,
		vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
		wrapper_class=lambda e: EnsureUint8Obs(AttachPlayerInfo(e)),
		env_kwargs={},
	)
	# Training uses 'ori' RGB + VecFrameStack(4)
	vec = VecFrameStack(vec, n_stack=4, channels_order="last")
	return vec


def save_shard(
	output_dir: Path,
	env_name: str,
	global_shard_id: int,
	obs_buf: List[np.ndarray],
	action_buf: List[np.ndarray],
	player_xy_buf: List[Tuple[Optional[float], Optional[float]]],
) -> None:
	"""
	Save one shard as uncompressed .npy files in:
	data/transitions/train/<env>/shard_XXXXX/{obs.npy, action.npy, n_actions.npy, player_x.npy, player_y.npy}
	"""
	root = output_dir / env_name / f"shard_{global_shard_id:05d}"
	root.mkdir(parents=True, exist_ok=True)
	# main arrays
	np.save(root / "obs.npy", np.asarray(obs_buf, dtype=np.uint8))
	np.save(root / "action.npy", np.asarray(action_buf, dtype=np.int64).reshape(-1))
	np.save(root / "n_actions.npy", np.array(NUM_ACTIONS, dtype=np.int64))
	# optional metadata
	player_x = np.array([p[0] if p is not None else np.nan for p in player_xy_buf], dtype=np.float32)
	player_y = np.array([p[1] if p is not None else np.nan for p in player_xy_buf], dtype=np.float32)
	np.save(root / "player_x.npy", player_x)
	np.save(root / "player_y.npy", player_y)


def collect_transitions(
	env_name: str,
	total_frames: int,
	chunk_size: int,
	n_envs: int,
	frame_save_freq: int = 1,
) -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_path = resolve_checkpoint_path(env_name, checkpoint_path=None)
	model: PPO = PPO.load(model_path, device=device)

	frame_save_freq = max(1, int(frame_save_freq))

	vec = build_vec_env(env_name, render_mode=None, n_envs=n_envs)

	obs = vec.reset()
	assert isinstance(obs, np.ndarray)

	# per-env buffers
	obs_bufs: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
	action_bufs: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
	player_xy_bufs: List[List[Tuple[Optional[float], Optional[float]]]] = [[] for _ in range(n_envs)]
	global_shard_id = 0
	# Policy timing: same as before (one tick per vec step batch, +n_envs per iteration).
	frame_counter = 0
	saved_total = 0
	step_loop_idx = 0
	# keep last_action per env
	last_action: Optional[np.ndarray] = None

	pbar = tqdm(total=total_frames, desc="Collecting frames", unit="frame")
	try:
		while saved_total < total_frames:
			# Decide action every decision_interval steps
			if (frame_counter % DECISION_INTERVAL == 0) or last_action is None:
				actions, _ = model.predict(obs, deterministic=True)
				# ensure shape [n_envs, ...]
				actions = np.array(actions)
				if actions.ndim == 0:
					actions = np.repeat(actions[None], n_envs, axis=0)
				last_action = actions
			else:
				actions = last_action

			next_obs, _rewards, dones, infos = vec.step(actions)

			# Optionally subsample what we persist (env still steps every iteration).
			if step_loop_idx % frame_save_freq == 0:
				# Extract and store most recent RGB frame (last 3 channels) for ALL envs
				for e in range(n_envs):
					frame = obs[e]  # (H, W, 3 * n_stack)
					if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] >= 3:
						frame = frame[..., -3:]
					frame = np.asarray(frame)
					if frame.dtype != np.uint8:
						if frame.max() <= 1.0:
							frame = (frame * 255.0).round()
						frame = frame.clip(0, 255).astype(np.uint8)
					obs_bufs[e].append(frame)
					# action for env e (store scalar int)
					action_e = int(actions[e]) if getattr(actions, "ndim", 0) > 0 else int(actions)
					action_bufs[e].append(action_e)
					info_e = infos[e] if len(infos) > e else {}
					player_xy = info_e.get("player_xy", (None, None))
					player_xy_bufs[e].append(player_xy)

				saved_total += n_envs
				pbar.update(n_envs)

			frame_counter += n_envs
			step_loop_idx += 1
			obs = next_obs

			# reset last_action for done envs to force new decision
			if np.any(dones):
				if last_action is not None:
					for e, d in enumerate(dones):
						if d:
							last_action[e] = -1  # sentinel; will be replaced on next decision interval

			# Save per-env when that env's buffer reaches chunk_size
			for e in range(n_envs):
				if len(obs_bufs[e]) >= chunk_size:
					save_shard(TRANSITIONS_DIR / "train", env_name, global_shard_id, obs_bufs[e], action_bufs[e], player_xy_bufs[e])
					global_shard_id += 1
					obs_bufs[e].clear()
					action_bufs[e].clear()
					player_xy_bufs[e].clear()
	finally:
		# Flush remaining
		for e in range(n_envs):
			if action_bufs[e]:
				save_shard(TRANSITIONS_DIR / "train", env_name, global_shard_id, obs_bufs[e], action_bufs[e], player_xy_bufs[e])
				global_shard_id += 1
		vec.close()
		pbar.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Collect transitions from a trained PPO agent (headless).")
	parser.add_argument(
		"--env",
		type=str,
		required=True,
		help="Game name (e.g., space_invaders, assault).",
	)
	parser.add_argument(
		"--frames",
		type=int,
		default=10_000_000,
		help="Number of frames to collect before stopping (across all envs).",
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=10_000,
		help="Number of steps per saved shard.",
	)
	parser.add_argument(
		"--n_envs",
		type=int,
		default=8,
		help="Number of parallel environments.",
	)
	parser.add_argument(
		"--frame_save_freq",
		type=int,
		default=4,
		help="Append to shards only every N vec-env steps (1 = every step).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	collect_transitions(
		env_name=args.env,
		total_frames=args.frames,
		chunk_size=args.chunk_size,
		n_envs=args.n_envs,
		frame_save_freq=args.frame_save_freq,
	)


if __name__ == "__main__":
	main()

