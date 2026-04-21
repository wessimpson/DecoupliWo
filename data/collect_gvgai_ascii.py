"""
Collect ASCII transitions from GVGAI's MCTS agent for VAE training.

Wraps the Java class ``tracks.singlePlayer.ascii.RunAsciiCollectionMCTS`` (see
``gvgai_java_stubs/src/tracks/singlePlayer/ascii/``) with a small CLI that

1. splits the target frame budget into a 90/10 train/test split (deterministic
   via ``--seed``) and
2. runs the Java collector once per (game, split), writing shards to
   ``<out>/<split>/<game>/shard_XXXXX/`` in the exact layout
   ``world_model/train_ascii_vae.py::AllAsciiFramesDataset`` reads.

Typical usage (after the ``gvgai/`` submodule is cloned and built):

	python -m data.collect_gvgai_ascii \\
		--games aliens,chopper,waves \\
		--frames-per-game 500000 \\
		--mcts-ms 40

Prereqs:
  * A ``gvgai/`` submodule at the repo root, built to ``out/production/gvgai``
    (matches the assumption in ``world_model/ascii/renderer.py``).
  * The collector Java sources copied from
    ``gvgai_java_stubs/src/tracks/singlePlayer/ascii/`` into
    ``gvgai/src/tracks/singlePlayer/ascii/`` and compiled against the GVGAI
    build tree. See ``gvgai_java_stubs/README.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GVGAI_ROOT = REPO_ROOT / "gvgai"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "transitions"
DEFAULT_MAPPINGS_ROOT = REPO_ROOT / "world_model" / "ascii" / "mappings"
MAIN_CLASS = "tracks.singlePlayer.ascii.RunAsciiCollectionMCTS"
DEFAULT_GAMES = ("aliens", "chopper", "waves")
TRAIN_TEST_SPLIT = 0.9


@dataclass(frozen=True)
class Run:
	game: str
	split: str
	frames: int
	seed: int


def default_classpath(gvgai_root: Path) -> str:
	override = os.environ.get("GVGAI_CLASSPATH")
	if override:
		return override
	return str(gvgai_root / "out" / "production" / "gvgai")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--games", type=str, default=",".join(DEFAULT_GAMES),
		help=f"comma-separated game names (default: {','.join(DEFAULT_GAMES)})")
	p.add_argument("--frames-per-game", type=int, default=500_000,
		help="total frames (train + test) to collect per game")
	p.add_argument("--mcts-ms", type=int, default=40, help="per-tick MCTS budget (ms)")
	p.add_argument("--levels", type=str, default="0,1,2,3,4", help="comma-separated level indices to rotate")
	p.add_argument("--chunk-size", type=int, default=5_000, help="frames per shard")
	p.add_argument("--seed", type=int, default=42, help="base RNG seed; splits get deterministic offsets")
	p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
		help="root under which <split>/<game>/shard_* directories are written")
	p.add_argument("--gvgai-root", type=Path, default=DEFAULT_GVGAI_ROOT,
		help="path to the gvgai submodule (must be cloned + built)")
	p.add_argument("--classpath", type=str, default=None,
		help="override Java classpath (default: <gvgai-root>/out/production/gvgai, or $GVGAI_CLASSPATH)")
	p.add_argument("--java-bin", type=str, default="java", help="path to java executable")
	p.add_argument("--mappings-root", type=Path, default=DEFAULT_MAPPINGS_ROOT,
		help="directory containing <game>.json sprite-to-char mappings")
	p.add_argument("--dry-run", action="store_true",
		help="print the java commands that would run and exit")
	return p.parse_args()


def build_runs(games: list[str], frames_per_game: int, base_seed: int) -> list[Run]:
	train_frames = int(frames_per_game * TRAIN_TEST_SPLIT)
	test_frames = frames_per_game - train_frames
	runs: list[Run] = []
	for g_idx, game in enumerate(games):
		game_seed = base_seed + 10_000 * g_idx
		runs.append(Run(game=game, split="train", frames=train_frames, seed=game_seed))
		runs.append(Run(game=game, split="test", frames=test_frames, seed=game_seed + 1))
	return runs


def run_java_collector(args: argparse.Namespace, run: Run) -> int:
	classpath = args.classpath or default_classpath(args.gvgai_root)
	mapping_path = args.mappings_root / f"{run.game}.json"
	if not mapping_path.is_file():
		raise FileNotFoundError(f"mapping not found: {mapping_path}")

	out_dir = args.out_root / run.split / run.game
	out_dir.mkdir(parents=True, exist_ok=True)

	cmd = [
		args.java_bin, "-cp", classpath, MAIN_CLASS,
		"--gvgai-root", str(args.gvgai_root),
		"--repo-root", str(REPO_ROOT),
		"--game", run.game,
		"--out", str(out_dir),
		"--mapping", str(mapping_path),
		"--frames", str(run.frames),
		"--levels", args.levels,
		"--mcts-ms", str(args.mcts_ms),
		"--chunk-size", str(args.chunk_size),
		"--seed", str(run.seed),
	]

	print(f"\n=== [{run.split}] {run.game}  frames={run.frames:,}  seed={run.seed} ===")
	print("  " + " ".join(cmd))
	if args.dry_run:
		return 0
	proc = subprocess.run(cmd, cwd=str(args.gvgai_root))
	return proc.returncode


def main() -> int:
	args = parse_args()
	games = [g.strip() for g in args.games.split(",") if g.strip()]
	if not games:
		print("no games specified", file=sys.stderr)
		return 2
	if not args.gvgai_root.is_dir():
		print(f"gvgai root missing: {args.gvgai_root}", file=sys.stderr)
		return 2

	runs = build_runs(games, args.frames_per_game, args.seed)
	for run in runs:
		rc = run_java_collector(args, run)
		if rc != 0:
			print(f"collector failed for {run.split}/{run.game} (exit {rc})", file=sys.stderr)
			return rc
	print(f"\nall done -> {args.out_root}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
