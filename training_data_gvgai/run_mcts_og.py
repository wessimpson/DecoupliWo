#!/usr/bin/env python
"""
Play GVGAI gridphysics examples with sampleMCTS and live display.

Games live in gvgai/examples/gridphysics/ ({game}.txt + {game}_lvl*.txt).

Run from repo root or this folder:
  python training_data_gvgai/run_mcts_og.py --env zelda --level 0 --show
  python training_data_gvgai/run_mcts_og.py --env chopper --level 0,1 --mcts-ms 40
  python training_data_gvgai/run_mcts_og.py --list-games
  python training_data_gvgai/run_mcts_og.py --env all --level 0 --show
    (S or Ctrl+C skips the current game and continues to the next)
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
GVGAI_JAVA_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "gvgai"
GRIDPHYSICS_ROOT = GVGAI_JAVA_ROOT / "examples" / "gridphysics"
_LEVEL_FILE_RE = re.compile(r"^.+_lvl\d+\.txt$")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_data_gvgai.data.gvgai_jpype_env import GVGAIFileEnv
from training_data_gvgai.run_mcts import _refresh_frame, _title, _video_out_path
from training_data_gvgai.run_random_action import (
    _ensure_build,
    parse_levels_arg,
)


class GameSkipped(Exception):
    """User pressed S (or equivalent) to skip the current episode."""


class SkipControl:
    """Press S in the terminal or game window to request a skip."""

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mpl_cid: int | None = None

    def request(self) -> None:
        self._flag.set()

    def clear(self) -> None:
        self._flag.clear()

    def check(self) -> None:
        if self._flag.is_set():
            raise GameSkipped()

    def bind_figure(self, fig) -> None:
        def on_key(event) -> None:
            if event.key and str(event.key).lower() == "s":
                self.request()

        self._mpl_cid = fig.canvas.mpl_connect("key_press_event", on_key)
        try:
            manager = fig.canvas.manager
            title = manager.get_window_title()
            if "[S=skip]" not in title:
                manager.set_window_title(f"{title}  [S=skip]")
        except Exception:
            pass

    def start_terminal_listener(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._terminal_loop, daemon=True)
        self._thread.start()

    def stop(self, fig=None) -> None:
        self._stop.set()
        if fig is not None and self._mpl_cid is not None:
            try:
                fig.canvas.mpl_disconnect(self._mpl_cid)
            except Exception:
                pass
            self._mpl_cid = None

    def _terminal_loop(self) -> None:
        if sys.platform == "win32":
            import msvcrt

            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("s", "S"):
                        self.request()
                time.sleep(0.05)
            return

        import select

        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ("s", "S"):
                        self.request()
            except Exception:
                time.sleep(0.1)


def _is_level_file(name: str) -> bool:
    return bool(_LEVEL_FILE_RE.match(name))


def list_og_games() -> list[tuple[str, int, list[str], int]]:
    """Return (base, version, vgdl_stems, n_levels) for each gridphysics game."""
    out: list[tuple[str, int, list[str], int]] = []
    if not GRIDPHYSICS_ROOT.is_dir():
        return out
    for path in sorted(GRIDPHYSICS_ROOT.glob("*.txt")):
        if _is_level_file(path.name):
            continue
        base = path.stem
        n_levels = len(list(GRIDPHYSICS_ROOT.glob(f"{base}_lvl*.txt")))
        if n_levels == 0:
            continue
        out.append((base, 0, [base], n_levels))
    return out


def resolve_og_paths(
    base: str,
    version: int,
    vgdl_stem: str | None = None,
) -> tuple[Path, list[str]]:
    """Resolve VGDL + level paths under gvgai/examples/gridphysics/."""
    del version  # flat layout; kept for CLI compatibility
    if not GRIDPHYSICS_ROOT.is_dir():
        raise FileNotFoundError(f"Gridphysics folder not found: {GRIDPHYSICS_ROOT}")

    stem = (vgdl_stem or base).strip()
    game_file = GRIDPHYSICS_ROOT / f"{stem}.txt"
    if not game_file.is_file():
        available = ", ".join(p.stem for p in sorted(GRIDPHYSICS_ROOT.glob("*.txt")) if not _is_level_file(p.name))
        raise FileNotFoundError(
            f"VGDL not found: {game_file}\nAvailable games in gridphysics: {available}"
        )

    level_files: list[str] = []
    for path in sorted(
        GRIDPHYSICS_ROOT.glob(f"{base}_lvl*.txt"),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    ):
        level_files.append(str(path.resolve()))
    if not level_files:
        raise FileNotFoundError(f"No level files matching {base}_lvl*.txt in {GRIDPHYSICS_ROOT}")
    level_files.append("")
    return game_file.resolve(), level_files


def build_og_env_id(
    base: str,
    level: int,
    version: int = 0,
    vgdl_stem: str | None = None,
) -> str:
    stem = (vgdl_stem or base).strip()
    return f"gvgai-{stem}-lvl{level}-v{version}"


def iter_og_run_targets(
    env_arg: str,
    *,
    version: int,
    levels: list[int],
    vgdl_stem: str | None,
) -> list[tuple[str, int, int, str | None]]:
    """(base, package_version, level, vgdl_stem) for each episode to run."""
    env_arg = env_arg.strip()
    if env_arg.lower() != "all":
        base = env_arg
        return [(base, version, lvl, vgdl_stem) for lvl in levels]

    targets: list[tuple[str, int, int, str | None]] = []
    for base, _pkg_version, _stems, n_levels in list_og_games():
        for lvl in levels:
            if lvl < n_levels:
                targets.append((base, 0, lvl, None))
    return targets


def add_og_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        default="zelda",
        metavar="GAME",
        help="Base game name (e.g. zelda, jaws, chopper), or 'all' to run every "
        "game in gvgai/examples/gridphysics sequentially",
    )
    parser.add_argument(
        "--vgdl",
        default="",
        metavar="STEM",
        help="VGDL file stem (default: same as --env). Example: aliens_ggame",
    )
    parser.add_argument(
        "--level",
        nargs="+",
        default=["0"],
        metavar="N",
        help="Level index(es): --level 0  or  --level 0,1,2",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Ignored (gridphysics uses a flat folder). Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List games under gvgai/examples/gridphysics/ and exit",
    )


def _run_og_episode(
    env_id: str,
    *,
    base: str,
    version: int,
    level: int,
    vgdl_stem: str | None,
    steps: int,
    mcts_ms: int,
    scale: int,
    delay: float,
    show: bool,
    video: Path | None,
    fps: float,
    block_show: bool = True,
    skip: SkipControl | None = None,
) -> None:
    game_file, level_files = resolve_og_paths(base, version, vgdl_stem)
    env = GVGAIFileEnv(
        game_file,
        level_files,
        level=level,
        gvgai_root=GVGAI_JAVA_ROOT,
        max_episode_steps=steps,
    )

    frames: list[np.ndarray] = []
    plt = None
    im = None
    fig = None
    if skip is not None:
        skip.clear()
        skip.start_terminal_listener()

    from training_data_gvgai.run_random_action import _obs_rgb, _save_video, _upscale

    try:
        obs, info = env.reset()
        if skip is not None:
            skip.check()
        frame = _upscale(_obs_rgb(obs), scale)
        frames.append(frame)
        game_tick = int(info.get("game_tick", -1))
        print(f"[{env_id}] reset game_tick={game_tick} winner={info.get('winner')}", flush=True)

        if show:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(frame)
            ax.set_title(_title(env_id, max(game_tick, 0), 0.0, str(info.get("winner", ""))))
            ax.axis("off")
            fig.tight_layout()
            if skip is not None:
                skip.bind_figure(fig)
            _refresh_frame(fig, im, frame, ax.get_title(), delay)

        for step_idx in range(steps):
            if skip is not None:
                skip.check()
            if show and fig is not None:
                import matplotlib.pyplot as plt

                plt.pause(0.001)
                fig.canvas.flush_events()

            obs, reward, terminated, truncated, info = env.step_mcts(mcts_ms)
            if skip is not None:
                skip.check()
            done = terminated or truncated
            game_tick = int(info.get("game_tick", step_idx))
            frame = _upscale(_obs_rgb(obs), scale)
            frames.append(frame)

            print(
                f"[{env_id}] step {step_idx + 1} game_tick={game_tick} r={reward:.1f} "
                f"winner={info.get('winner')} done={done}",
                flush=True,
            )

            if show and im is not None and fig is not None:
                _refresh_frame(
                    fig,
                    im,
                    frame,
                    _title(
                        env_id,
                        max(game_tick, 0),
                        reward,
                        str(info.get("winner", "")),
                        done,
                    ),
                    delay,
                )

            if done:
                break

        if show and plt is not None:
            if block_show:
                print("Close the window to exit.", flush=True)
                plt.ioff()
                try:
                    plt.show()
                except KeyboardInterrupt:
                    plt.close("all")
                    raise
            elif fig is not None:
                import matplotlib.pyplot as plt

                plt.close(fig)

        if video is not None:
            _save_video(frames, video, fps)
    finally:
        if skip is not None:
            skip.stop(fig)
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sampleMCTS on gvgai/examples/gridphysics games"
    )
    add_og_cli(parser)
    parser.add_argument("--steps", type=int, default=500, help="Max env steps")
    parser.add_argument(
        "--mcts-ms",
        type=int,
        default=40,
        help="Per-action MCTS CPU budget in milliseconds",
    )
    parser.add_argument("--scale", type=int, default=2, help="Nearest-neighbor upscale")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback / video FPS")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between frames when showing")
    parser.add_argument("--show", action="store_true", help="Open live matplotlib window (default)")
    parser.add_argument("--no-show", action="store_true", help="No live window (use with --video)")
    parser.add_argument("--video", default="", help="Write replay (.mp4 or .gif)")
    args = parser.parse_args()
    delay = args.delay if args.delay is not None else (1.0 / args.fps)
    show = args.show if args.show else not args.no_show
    vgdl_stem = args.vgdl.strip() or None

    if args.list_games:
        rows = list_og_games()
        if not rows:
            print(f"No games found under {GRIDPHYSICS_ROOT}")
            return
        for base, _version, stems, n_levels in rows:
            extra = [s for s in stems if s != base]
            extra_txt = f"  alt VGDL: {', '.join(extra)}" if extra else ""
            print(f"{base}  levels={n_levels}{extra_txt}")
        return

    _ensure_build()
    levels = parse_levels_arg([str(x) for x in args.level])
    run_all = args.env.strip().lower() == "all"
    if run_all and vgdl_stem:
        print("Note: --vgdl is ignored with --env all (uses each game's base VGDL).", flush=True)

    targets = iter_og_run_targets(
        args.env,
        version=args.version,
        levels=levels,
        vgdl_stem=None if run_all else vgdl_stem,
    )
    if not targets:
        print("No games/levels matched.", flush=True)
        return

    multi = len(targets) > 1
    video_base = Path(args.video) if args.video else None

    if not show and not video_base:
        print("No --video and --no-show: nothing to display.", flush=True)
        return

    if run_all:
        print(f"Running {len(targets)} game(s) sequentially…", flush=True)
        print("Press S (terminal or game window) or Ctrl+C to skip the current game.", flush=True)

    skip = SkipControl()
    for idx, (base, pkg_version, level, stem) in enumerate(targets):
        env_id = build_og_env_id(base, level, pkg_version, stem)
        out = _video_out_path(video_base, env_id, multi) if video_base else None
        block_show = show and (not run_all or idx == len(targets) - 1)
        if run_all:
            print(f"\n=== [{idx + 1}/{len(targets)}] {env_id} ===", flush=True)
        try:
            _run_og_episode(
                env_id,
                base=base,
                version=pkg_version,
                level=level,
                vgdl_stem=stem,
                steps=args.steps,
                mcts_ms=args.mcts_ms,
                scale=args.scale,
                delay=delay,
                show=show,
                video=out,
                fps=args.fps,
                block_show=block_show,
                skip=skip,
            )
        except GameSkipped:
            if not run_all:
                print(f"\n[{env_id}] skipped.", flush=True)
                raise
            print(f"\n[{env_id}] skipped (S) — next game…", flush=True)
        except KeyboardInterrupt:
            if not run_all:
                print("\nStopped.", flush=True)
                raise
            print(f"\n[{env_id}] skipped — next game…", flush=True)
        except Exception as exc:
            print(f"[{env_id}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
