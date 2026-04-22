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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GVGAI_ROOT = REPO_ROOT / "gvgai"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "transitions"
DEFAULT_MAPPINGS_ROOT = REPO_ROOT / "world_model" / "ascii" / "mappings"
MAIN_CLASS = "tracks.singlePlayer.ascii.RunAsciiCollectionMCTS"
DEFAULT_GAMES = ("aliens", "chopper", "waves")
DEFAULT_VARIANTS = ("stock", "physics_a", "physics_b", "physics_c")
TRAIN_TEST_SPLIT = 0.9
DEFAULT_JOBS = max(1, (os.cpu_count() or 1) - 1)


@dataclass(frozen=True)
class Run:
	game: str
	variant: str
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
	p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS),
		help=("comma-separated VGDL variants per game; the per-game frame budget is "
			f"divided evenly across variants (default: {','.join(DEFAULT_VARIANTS)}). "
			"Valid values: stock, physics_a, physics_b, physics_c."))
	p.add_argument("--frames-per-game", type=int, default=500_000,
		help="total frames (train + test) to collect per game, split evenly across variants")
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
	p.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
		help=(f"number of Java collectors to run in parallel (default: {DEFAULT_JOBS}, "
			f"i.e. os.cpu_count()-1). Each run is single-threaded CPU-bound MCTS, so "
			f"wall-clock scales ~linearly with --jobs up to the vCPU count."))
	p.add_argument("--dry-run", action="store_true",
		help="print the java commands that would run and exit")
	return p.parse_args()


def build_runs(
	games: list[str],
	variants: list[str],
	frames_per_game: int,
	base_seed: int,
) -> list[Run]:
	frames_per_variant = frames_per_game // len(variants)
	train_frames = int(frames_per_variant * TRAIN_TEST_SPLIT)
	test_frames = frames_per_variant - train_frames
	runs: list[Run] = []
	for g_idx, game in enumerate(games):
		for v_idx, variant in enumerate(variants):
			seed = base_seed + 10_000 * g_idx + 100 * v_idx
			runs.append(Run(game=game, variant=variant, split="train", frames=train_frames, seed=seed))
			runs.append(Run(game=game, variant=variant, split="test", frames=test_frames, seed=seed + 1))
	return runs


def round_robin_by_game(runs: list[Run]) -> list[Run]:
	"""Re-order ``runs`` so the first slice contains one run per game, then the second, etc.

	``build_runs`` emits runs grouped game-by-game. ``ProcessPoolExecutor`` submits in that
	order, so with ``--jobs < len(games)*len(variants)*2`` the first workers all collect the
	same game before the next game starts. Interleaving by game here means the first
	``len(games)`` workers each cover a different game, so shards from every game begin
	appearing on disk (and on Drive, once the uploader flushes them) roughly in parallel —
	which is what the downstream VAE trainer wants so each mini-batch mixes games even when
	collection is still in-progress.
	"""
	by_game: dict[str, list[Run]] = {}
	for r in runs:
		by_game.setdefault(r.game, []).append(r)
	out: list[Run] = []
	while by_game:
		for game in list(by_game):
			bucket = by_game[game]
			out.append(bucket.pop(0))
			if not bucket:
				del by_game[game]
	return out


def build_java_command(args: argparse.Namespace, run: Run) -> tuple[list[str], Path]:
	"""Return (cmd, cwd) for the Java collector for this run.

	Each (game, variant, split) writes to its own subdirectory
	``<out-root>/<split>/<game>/<variant>/shard_*`` so concurrent runs never
	race on ``ShardAccumulator.nextFreeShardIndex``. The training dataset
	loader recurses with ``rglob("shard_*")``, so this nested layout is
	transparent to downstream code.
	"""
	classpath = args.classpath or default_classpath(args.gvgai_root)
	mapping_path = args.mappings_root / f"{run.game}.json"
	if not mapping_path.is_file():
		raise FileNotFoundError(f"mapping not found: {mapping_path}")

	out_dir = args.out_root / run.split / run.game / run.variant
	out_dir.mkdir(parents=True, exist_ok=True)

	cmd = [
		args.java_bin, "-cp", classpath, MAIN_CLASS,
		"--gvgai-root", str(args.gvgai_root),
		"--repo-root", str(REPO_ROOT),
		"--game", run.game,
		"--variant", run.variant,
		"--out", str(out_dir),
		"--mapping", str(mapping_path),
		"--frames", str(run.frames),
		"--levels", args.levels,
		"--mcts-ms", str(args.mcts_ms),
		"--chunk-size", str(args.chunk_size),
		"--seed", str(run.seed),
	]
	return cmd, Path(args.gvgai_root)


def _invoke(cmd: list[str], cwd: str) -> int:
	"""ProcessPool worker: fork/exec the Java collector and wait.

	Kept at module scope so it is picklable by ``ProcessPoolExecutor``.
	"""
	return subprocess.run(cmd, cwd=cwd).returncode


def run_all_collectors(args: argparse.Namespace, runs: list[Run]) -> int:
	"""Launch all Java collectors, up to ``args.jobs`` at a time.

	Returns the first non-zero exit code encountered, or 0 if all succeeded.
	A failed run cancels not-yet-started runs but lets in-flight ones finish
	(killing them mid-episode would leave half-written shards).
	"""
	jobs = max(1, min(int(args.jobs), len(runs)))
	prepared: list[tuple[Run, list[str], Path]] = []
	for run in runs:
		cmd, cwd = build_java_command(args, run)
		prepared.append((run, cmd, cwd))
		print(f"=== [{run.split}] {run.game}/{run.variant}  frames={run.frames:,}  seed={run.seed} ===")
		print("  " + " ".join(cmd))

	if args.dry_run:
		return 0

	print(f"\nlaunching {len(prepared)} run(s) with --jobs={jobs}\n")

	if jobs == 1:
		for run, cmd, cwd in prepared:
			rc = _invoke(cmd, str(cwd))
			if rc != 0:
				print(f"collector failed for {run.split}/{run.game}/{run.variant} (exit {rc})", file=sys.stderr)
				return rc
		return 0

	first_failure = 0
	with ProcessPoolExecutor(max_workers=jobs) as pool:
		future_to_run = {
			pool.submit(_invoke, cmd, str(cwd)): run
			for run, cmd, cwd in prepared
		}
		for fut in as_completed(future_to_run):
			run = future_to_run[fut]
			try:
				rc = fut.result()
			except Exception as exc:
				print(f"collector crashed for {run.split}/{run.game}/{run.variant}: {exc}", file=sys.stderr)
				first_failure = first_failure or 1
				continue
			if rc != 0:
				print(f"collector failed for {run.split}/{run.game}/{run.variant} (exit {rc})", file=sys.stderr)
				first_failure = first_failure or rc
	return first_failure


def main() -> int:
	args = parse_args()
	games = [g.strip() for g in args.games.split(",") if g.strip()]
	variants = [v.strip() for v in args.variants.split(",") if v.strip()]
	if not games:
		print("no games specified", file=sys.stderr)
		return 2
	if not variants:
		print("no variants specified", file=sys.stderr)
		return 2
	allowed = set(DEFAULT_VARIANTS)
	bad = [v for v in variants if v not in allowed]
	if bad:
		print(f"unknown variant(s): {bad} (valid: {sorted(allowed)})", file=sys.stderr)
		return 2
	if not args.gvgai_root.is_dir():
		print(f"gvgai root missing: {args.gvgai_root}", file=sys.stderr)
		return 2

	runs = round_robin_by_game(build_runs(games, variants, args.frames_per_game, args.seed))
	rc = run_all_collectors(args, runs)
	if rc != 0:
		return rc
	print(f"\nall done -> {args.out_root}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
