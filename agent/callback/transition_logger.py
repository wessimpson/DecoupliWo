from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TransitionLoggerCallback(BaseCallback):
	"""Logs transitions (obs, action, reward, done, player_xy) to chunked .npz files.

	Note: Observations can be large; consider reducing CHUNK_SIZE or disabling obs logging if storage is tight.
	When using stacked RGB observations (VecFrameStack with channels_order='last'), only the most recent
	RGB frame (last 3 channels) is saved to reduce storage.
	"""

	def __init__(
		self,
		output_dir: str | Path,
		env_name: str,
		chunk_size: int = 100_000,
		save_observations: bool = True,
		verbose: int = 0,
	):
		super().__init__(verbose=verbose)
		self.output_dir = Path(output_dir)
		self.env_name = env_name
		self.chunk_size = int(chunk_size)
		self.save_observations = save_observations

		self._buf_obs: List[np.ndarray] = []
		self._buf_action: List[np.ndarray] = []
		self._buf_reward: List[float] = []
		self._buf_done: List[bool] = []
		self._buf_player_xy: List[Tuple[float | None, float | None]] = []
		self._sample_count: int = 0
		self._shard_idx: int = 0

	def _on_training_start(self) -> None:
		(self.output_dir / self.env_name).mkdir(parents=True, exist_ok=True)

	def _save_shard(self) -> None:
		if not self._buf_action:
			return
		shard_path = self.output_dir / self.env_name / f"shard_{self._shard_idx:05d}.npz"
		player_x = np.array([p[0] if p is not None else np.nan for p in self._buf_player_xy], dtype=np.float32)
		player_y = np.array([p[1] if p is not None else np.nan for p in self._buf_player_xy], dtype=np.float32)
		pack: Dict[str, Any] = {
			"action": np.array(self._buf_action, dtype=np.int64).squeeze(-1),
			"reward": np.array(self._buf_reward, dtype=np.float32),
			"done": np.array(self._buf_done, dtype=np.bool_),
			"player_x": player_x,
			"player_y": player_y,
		}
		if self.save_observations:
			# Store observations as uint8 to save space
			pack["obs"] = np.array(self._buf_obs, dtype=np.uint8)
		np.savez_compressed(shard_path, **pack)
		self._shard_idx += 1
		self._buf_obs.clear()
		self._buf_action.clear()
		self._buf_reward.clear()
		self._buf_done.clear()
		self._buf_player_xy.clear()

	def _maybe_flush(self) -> None:
		if len(self._buf_action) >= self.chunk_size:
			self._save_shard()

	def _on_step(self) -> bool:
		# In SB3 callbacks, these are vectorized across environments
		obs = self.locals.get("obs")
		actions = self.locals.get("actions")
		rewards = self.locals.get("rewards")
		dones = self.locals.get("dones")
		infos = self.locals.get("infos")
		if obs is None or actions is None or rewards is None or dones is None or infos is None:
			return True

		# Normalize shapes
		obs_arr: np.ndarray = np.asarray(obs)
		act_arr: np.ndarray = np.asarray(actions).reshape(-1, 1)
		rew_arr: np.ndarray = np.asarray(rewards).reshape(-1)
		done_arr: np.ndarray = np.asarray(dones).reshape(-1)
		info_list: List[Dict[str, Any]] = list(infos)

		n_envs = act_arr.shape[0]
		for i in range(n_envs):
			if self.save_observations:
				frame = obs_arr[i]
				# If stacked RGB (H, W, 3 * n_stack), keep only most recent RGB (last 3 channels)
				if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] % 3 == 0 and frame.shape[-1] > 3:
					frame = frame[..., -3:]
				# Ensure uint8
				if frame.dtype != np.uint8:
					if frame.max() <= 1.0:
						frame = (frame * 255.0).round()
					frame = frame.clip(0, 255).astype(np.uint8)
				self._buf_obs.append(frame)
			self._buf_action.append(act_arr[i])
			self._buf_reward.append(float(rew_arr[i]))
			self._buf_done.append(bool(done_arr[i]))
			player_xy = info_list[i].get("player_xy") if i < len(info_list) else None
			self._buf_player_xy.append(player_xy if player_xy is not None else (None, None))
			self._sample_count += 1
		self._maybe_flush()
		return True

	def _on_training_end(self) -> None:
		self._save_shard()

