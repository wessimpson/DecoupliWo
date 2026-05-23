from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from training_data_gvgai.data.gvgai_jpype_env import GVGAIFileEnv


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PACKAGE_ROOT / "data" / "gvgai_variant_catalog.world_model.json"
DEFAULT_OUT_ROOT = PACKAGE_ROOT / "data" / "gvgai_rollouts"
SCHEMA_VERSION = "gvgai-rollout-v1"


@dataclass(frozen=True)
class VariantJob:
	game: str
	variant: str
	env_id: str | None
	game_file: Path | None
	level_files: tuple[Path, ...] | None
	levels: tuple[int, ...]
	rule_flags: dict[str, float | int | bool]
	description: str

	@property
	def env_key(self) -> str:
		return f"{self.game}__{self.variant}"


def _read_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _as_tuple_ints(raw: Iterable[Any]) -> tuple[int, ...]:
	values = tuple(int(value) for value in raw)
	if not values:
		raise ValueError("levels must not be empty")
	return values


def _resolve_path(base: Path, value: str | None) -> Path | None:
	if value is None:
		return None
	path = Path(value).expanduser()
	return path if path.is_absolute() else (base / path)


def load_jobs(catalog_path: Path, selected_games: set[str] | None, selected_variants: set[str] | None) -> list[VariantJob]:
	catalog = _read_json(catalog_path)
	base = catalog_path.resolve().parent
	default_levels = _as_tuple_ints(catalog.get("defaults", {}).get("levels", [0, 1, 2, 3, 4]))
	jobs: list[VariantJob] = []

	for game, game_spec in sorted(catalog.get("games", {}).items()):
		if selected_games is not None and game not in selected_games:
			continue
		game_levels = _as_tuple_ints(game_spec.get("levels", default_levels))
		for variant, variant_spec in sorted(game_spec.get("variants", {}).items()):
			if selected_variants is not None and variant not in selected_variants:
				continue

			env_id = variant_spec.get("env_id")
			game_file = _resolve_path(base, variant_spec.get("game_file"))
			level_files_raw = variant_spec.get("level_files")
			level_files = None
			if level_files_raw is not None:
				level_files = tuple(
					_resolve_path(base, str(level_file)) for level_file in level_files_raw
				)
				if any(level_file is None for level_file in level_files):
					raise ValueError(f"{game}/{variant}: invalid level_files")
			if env_id is None and game_file is None:
				env_game = variant_spec.get("env_game", game)
				version = int(variant_spec.get("version", 0))
				env_id = f"gvgai-{env_game}-lvl{{level}}-v{version}"

			jobs.append(VariantJob(
				game=game,
				variant=variant,
				env_id=env_id,
				game_file=game_file,
				level_files=level_files,
				levels=_as_tuple_ints(variant_spec.get("levels", game_levels)),
				rule_flags=dict(variant_spec.get("rule_flags", {})),
				description=str(variant_spec.get("description", "")),
			))

	if not jobs:
		raise ValueError("catalog selection produced no collection jobs")
	return jobs


def parse_filter(raw: str) -> set[str] | None:
	if raw.strip().lower() in {"all", "*"}:
		return None
	return {item.strip() for item in raw.split(",") if item.strip()}


def make_env(job: VariantJob, level: int, seed: int, gvgai_root: Path | None, max_episode_steps: int):
	if job.game_file is not None:
		if job.level_files is None:
			raise ValueError(f"{job.game}/{job.variant}: explicit game_file requires level_files")
		env = GVGAIFileEnv(
			game_file=job.game_file,
			level_files=job.level_files,
			level=level,
			gvgai_root=gvgai_root,
			max_episode_steps=max_episode_steps,
		)
		env.reset(seed=seed, options={"level": level})
		return env

	import gym_gvgai as gvgai

	env_id = str(job.env_id).format(level=level)
	env = gvgai.make(env_id)
	env.reset(seed=seed)
	return env


class ShardWriter:
	def __init__(
		self,
		out_dir: Path,
		job: VariantJob,
		split: str,
		chunk_size: int,
		flag_names: tuple[str, ...],
		source: dict[str, Any],
	) -> None:
		self.out_dir = out_dir
		self.job = job
		self.split = split
		self.chunk_size = max(1, int(chunk_size))
		self.flag_names = flag_names
		self.source = source
		self.shard_idx = self._next_shard_index()
		self.rows: list[dict[str, Any]] = []
		self.manifest_path = out_dir / "manifest.jsonl"

	def append(self, row: dict[str, Any]) -> None:
		self.rows.append(row)
		if len(self.rows) >= self.chunk_size:
			self.flush()

	def flush(self) -> None:
		if not self.rows:
			return

		shard_dir = self.out_dir / f"shard_{self.shard_idx:05d}"
		shard_dir.mkdir(parents=True, exist_ok=False)
		self.shard_idx += 1

		obs = np.stack([row["obs"] for row in self.rows]).astype(np.uint8, copy=False)
		next_obs = np.stack([row["next_obs"] for row in self.rows]).astype(np.uint8, copy=False)
		actions = np.asarray([row["action"] for row in self.rows], dtype=np.int64)
		rewards = np.asarray([row["reward"] for row in self.rows], dtype=np.float32)
		terminated = np.asarray([row["terminated"] for row in self.rows], dtype=np.bool_)
		truncated = np.asarray([row["truncated"] for row in self.rows], dtype=np.bool_)
		episode_id = np.asarray([row["episode_id"] for row in self.rows], dtype=np.int64)
		step_in_episode = np.asarray([row["step_in_episode"] for row in self.rows], dtype=np.int64)
		level = np.asarray([row["level"] for row in self.rows], dtype=np.int64)
		seed = np.asarray([row["seed"] for row in self.rows], dtype=np.int64)
		avatar_xy = np.stack([row["avatar_xy"] for row in self.rows]).astype(np.float32, copy=False)
		flag_vector = np.asarray(
			[float(self.job.rule_flags.get(name, 0.0)) for name in self.flag_names],
			dtype=np.float32,
		)
		rule_flags = np.repeat(flag_vector[None, :], len(self.rows), axis=0)

		np.save(shard_dir / "obs.npy", obs)
		np.save(shard_dir / "next_obs.npy", next_obs)
		np.save(shard_dir / "action.npy", actions)
		np.save(shard_dir / "reward.npy", rewards)
		np.save(shard_dir / "terminated.npy", terminated)
		np.save(shard_dir / "truncated.npy", truncated)
		np.save(shard_dir / "episode_id.npy", episode_id)
		np.save(shard_dir / "step_in_episode.npy", step_in_episode)
		np.save(shard_dir / "level.npy", level)
		np.save(shard_dir / "seed.npy", seed)
		np.save(shard_dir / "avatar_xy.npy", avatar_xy)
		np.save(shard_dir / "rule_flags.npy", rule_flags)
		np.save(shard_dir / "n_actions.npy", np.asarray(int(max(row["n_actions"] for row in self.rows)), dtype=np.int64))

		metadata = {
			"schema_version": SCHEMA_VERSION,
			"split": self.split,
			"game": self.job.game,
			"variant": self.job.variant,
			"env_key": self.job.env_key,
			"description": self.job.description,
			"rule_flags": self.job.rule_flags,
			"flag_names": list(self.flag_names),
			"source": self.source,
			"num_rows": len(self.rows),
			"obs_shape": list(obs.shape[1:]),
			"has_next_obs": True,
			"action_meanings": self.rows[0]["action_meanings"],
			"created_at_unix": time.time(),
		}
		(shard_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

		record = {
			"schema_version": SCHEMA_VERSION,
			"split": self.split,
			"game": self.job.game,
			"variant": self.job.variant,
			"env_key": self.job.env_key,
			"shard": shard_dir.name,
			"num_rows": len(self.rows),
			"path": str(shard_dir),
			"obs_shape": list(obs.shape[1:]),
			"rule_flags": self.job.rule_flags,
		}
		with self.manifest_path.open("a", encoding="utf-8") as handle:
			handle.write(json.dumps(record, sort_keys=True) + "\n")

		self.rows.clear()

	def _next_shard_index(self) -> int:
		if not self.out_dir.is_dir():
			return 0
		max_idx = -1
		for path in self.out_dir.glob("shard_*"):
			try:
				max_idx = max(max_idx, int(path.name.removeprefix("shard_")))
			except ValueError:
				continue
		return max_idx + 1


def as_rgb(obs: Any) -> np.ndarray:
	arr = np.asarray(obs)
	if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
		raise ValueError(f"expected HWC RGB/RGBA observation, got shape={arr.shape}")
	if arr.shape[-1] == 4:
		arr = arr[:, :, :3]
	if arr.dtype != np.uint8:
		arr = arr.clip(0, 255).astype(np.uint8)
	return arr


def choose_action(policy: str, action_space: gym.Space, rng: random.Random, last_action: int | None) -> int:
	if policy == "noop":
		return 0
	if policy == "repeat_random" and last_action is not None and rng.random() < 0.85:
		return int(last_action)
	if not isinstance(action_space, gym.spaces.Discrete):
		raise TypeError("only Discrete action spaces are supported")
	return rng.randrange(int(action_space.n))


def collect_job(
	job: VariantJob,
	split: str,
	out_root: Path,
	total_frames: int,
	chunk_size: int,
	policy: str,
	seed: int,
	gvgai_root: Path | None,
	max_episode_steps: int,
	flag_names: tuple[str, ...],
	source: dict[str, Any],
) -> None:
	out_dir = out_root / split / job.game / job.variant
	out_dir.mkdir(parents=True, exist_ok=True)
	writer = ShardWriter(out_dir, job, split, chunk_size, flag_names, source)
	rng = random.Random(seed)
	frames = 0
	episode_count = 0
	episode_base = writer.shard_idx * 1_000_000

	with tqdm(total=total_frames, desc=f"{split}/{job.game}/{job.variant}", unit="frame") as progress:
		while frames < total_frames:
			episode_id = episode_base + episode_count
			level = job.levels[episode_count % len(job.levels)]
			episode_seed = rng.randrange(2**31 - 1)
			env = make_env(job, level=level, seed=episode_seed, gvgai_root=gvgai_root, max_episode_steps=max_episode_steps)
			try:
				reset_out = env.reset(seed=episode_seed, options={"level": level})
				obs, _info = reset_out if isinstance(reset_out, tuple) else (reset_out, {})
				last_action: int | None = None
				step_in_episode = 0
				done = False

				while not done and frames < total_frames:
					action = choose_action(policy, env.action_space, rng, last_action)
					next_obs, reward, terminated, truncated, info = env.step(action)
					action_meanings = (
						env.get_action_meanings()
						if hasattr(env, "get_action_meanings")
						else list(info.get("actions", []))
					)
					avatar_xy = np.asarray(info.get("avatar_xy", info.get("player_xy", [-1.0, -1.0])), dtype=np.float32)
					if avatar_xy.shape != (2,):
						avatar_xy = np.asarray([-1.0, -1.0], dtype=np.float32)

					writer.append({
						"obs": as_rgb(obs),
						"next_obs": as_rgb(next_obs),
						"action": int(action),
						"reward": float(reward),
						"terminated": bool(terminated),
						"truncated": bool(truncated),
						"episode_id": int(episode_id),
						"step_in_episode": int(step_in_episode),
						"level": int(level),
						"seed": int(episode_seed),
						"avatar_xy": avatar_xy,
						"n_actions": int(getattr(env.action_space, "n", len(action_meanings))),
						"action_meanings": action_meanings,
					})
					frames += 1
					progress.update(1)
					obs = next_obs
					last_action = action
					step_in_episode += 1
					done = bool(terminated or truncated)
			finally:
				env.close()
			episode_count += 1
	writer.flush()


def all_flag_names(jobs: Iterable[VariantJob]) -> tuple[str, ...]:
	names: set[str] = set()
	for job in jobs:
		names.update(job.rule_flags)
	return tuple(sorted(names))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Collect variant-aware GVGAI rollouts with the JPype fork.")
	parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
	parser.add_argument("--games", default="all", help="Comma-separated game names or all.")
	parser.add_argument("--variants", default="all", help="Comma-separated variant names or all.")
	parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--frames-per-variant", type=int, default=10_000)
	parser.add_argument("--chunk-size", type=int, default=2_000)
	parser.add_argument("--policy", choices=["random", "repeat_random", "noop"], default="repeat_random")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--gvgai-root", type=Path, default=None)
	parser.add_argument("--max-episode-steps", type=int, default=2_000)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	jobs = load_jobs(args.catalog, parse_filter(args.games), parse_filter(args.variants))
	flag_names = all_flag_names(jobs)
	source = {
		"catalog": str(args.catalog),
		"policy": args.policy,
		"collector": Path(__file__).name,
	}
	for index, job in enumerate(jobs):
		collect_job(
			job=job,
			split=args.split,
			out_root=args.out_root,
			total_frames=args.frames_per_variant,
			chunk_size=args.chunk_size,
			policy=args.policy,
			seed=args.seed + index,
			gvgai_root=args.gvgai_root,
			max_episode_steps=args.max_episode_steps,
			flag_names=flag_names,
			source=source,
		)


if __name__ == "__main__":
	main()
