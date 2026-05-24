#!/usr/bin/env python
"""
Play a GVGAI env with random actions; display frames live or save a video/GIF.

Run from repo root or this folder:
  python training_data_gvgai/run_random_action.py
  python run_random_action.py --env aliens --rules multishot --level 0 --show
  python run_random_action.py --env waves --rules split_orthogonal --level 0 --show
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
GVGAI_JAVA_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "gvgai"
GAMES_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "games_world_model"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_data_gvgai.data.gvgai_jpype_env import GVGAIFileEnv


def _ensure_build() -> None:
    if (GVGAI_JAVA_ROOT / "GVGAI_Build").is_dir():
        return
    subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "gvgai" / "build.py")],
        check=True,
    )


def resolve_game_stem(base: str, rules: str | None) -> str:
    """Map base game + optional rule tag to games_world_model VGDL stem."""
    base = base.strip()
    if not rules or rules.strip().lower() in ("", "base", "none"):
        return base
    tag = rules.strip()
    if tag.startswith(f"{base}_rules_"):
        return tag
    if tag.startswith("rules_"):
        tag = tag[len("rules_") :]
    return f"{base}_rules_{tag}"


def build_env_id(
    base: str,
    rules: str | None,
    level: int,
    version: int = 0,
) -> str:
    stem = resolve_game_stem(base, rules)
    return f"gvgai-{stem}-lvl{level}-v{version}"


def list_rule_tags(base: str) -> list[str]:
    """Return sorted rule tags for games_world_model/{base}/."""
    game_dir = GAMES_ROOT / base.strip()
    if not game_dir.is_dir():
        return []
    prefix = f"{base.strip()}_rules_"
    return [
        p.stem[len(prefix) :]
        for p in sorted(game_dir.glob(f"{base.strip()}_rules_*.txt"))
    ]


def parse_rules_arg(rules: str | None, base: str | None = None) -> list[str | None]:
    """Comma/slash-separated rule tags; empty string => base game only."""
    if rules is not None and str(rules).strip().lower() == "all":
        if not base:
            raise ValueError("--rules all requires --env <game>")
        tags = list_rule_tags(base)
        return tags or [None]
    if rules is None or not str(rules).strip():
        return [None]
    parts = re.split(r"[,/]+", str(rules).strip())
    out: list[str | None] = []
    for part in parts:
        part = part.strip()
        if not part or part.lower() in ("base", "none"):
            out.append(None)
        else:
            out.append(part)
    return out or [None]


def parse_levels_arg(level_tokens: list[str]) -> list[int]:
    """Accept --level 0 1 2 or --level 0,1,2."""
    levels: list[int] = []
    for token in level_tokens:
        for part in re.split(r"[,/]+", str(token).strip()):
            if part:
                levels.append(int(part))
    return levels or [0]


def add_world_model_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        default="aliens",
        metavar="GAME",
        help="Base game name (e.g. aliens, chopper, waves)",
    )
    parser.add_argument(
        "--rules",
        default="",
        help="Rule variant tag(s), comma- or slash-separated, or 'all' for every "
        "variant in games_world_model/<env>/ (multishot, ricochet, ...). Omit for base.",
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
        help="GVGAI env version suffix (default: 0 -> ...-v0)",
    )


def _paths_from_env_id(env_id: str) -> tuple[Path, list[str], int]:
    match = re.fullmatch(r"gvgai-(.+)-lvl(\d+)-v(\d+)", env_id)
    if not match:
        raise ValueError(f"Expected gvgai-<game>-lvl<N>-v<V>, got {env_id!r}")
    name, level, version = match.group(1), int(match.group(2)), int(match.group(3))
    envs_dir = GAMES_ROOT.parent
    mod_path = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "world_model_paths.py"
    spec = importlib.util.spec_from_file_location("world_model_paths", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    game_file, level_files = mod.resolve_gvgai_paths(str(envs_dir), name, version)
    return Path(game_file), level_files, level


def _obs_rgb(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 3 and obs.shape[-1] >= 3:
        return np.asarray(obs[..., :3], dtype=np.uint8)
    return np.asarray(obs, dtype=np.uint8)


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return rgb
    h, w = rgb.shape[:2]
    return np.array(
        Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST),
        dtype=np.uint8,
    )


def _save_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]

    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(path, frames, fps=fps, loop=0)
    else:
        if suffix != ".mp4":
            path = path.with_suffix(".mp4")
        imageio.mimsave(path, frames, fps=fps)
    print(f"Saved {len(frames)} frames → {path}")


def _video_out_path(base: Path, env_id: str, multi: bool) -> Path:
    if not multi:
        return base
    tag = env_id.removeprefix("gvgai-")
    suffix = base.suffix or ".mp4"
    return base.with_name(f"{base.stem}_{tag}{suffix}")


def _run_episode(
    env_id: str,
    *,
    steps: int,
    scale: int,
    delay: float,
    show: bool,
    video: Path | None,
    fps: float,
    block_on_close: bool = True,
) -> None:
    game_file, level_files, level = _paths_from_env_id(env_id)
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

    try:
        obs, info = env.reset()
        frame = _upscale(_obs_rgb(obs), scale)
        frames.append(frame)

        if show:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(frame)
            ax.set_title(_title(env_id, 0, 0.0, info.get("winner", "")))
            ax.axis("off")
            fig.tight_layout()
            plt.pause(max(delay, 0.001))

        for t in range(steps):
            action_id = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action_id)
            done = terminated or truncated
            frame = _upscale(_obs_rgb(obs), scale)
            frames.append(frame)

            if show and im is not None and plt is not None:
                im.set_data(frame)
                im.axes.set_title(
                    _title(env_id, t + 1, reward, info.get("winner", ""), done)
                )
                fig.canvas.draw_idle()
                plt.pause(delay)

            if done:
                break

        if show and plt is not None:
            if block_on_close:
                print("Close the window to exit.")
                plt.ioff()
                plt.show()
            else:
                plt.pause(0.05)

        if video is not None:
            _save_video(frames, video, fps)
    finally:
        if show:
            import matplotlib.pyplot as plt

            if fig is not None:
                plt.close(fig)
            elif plt is not None:
                plt.close("all")
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random agent with visual playback")
    add_world_model_cli(parser)
    parser.add_argument("--steps", type=int, default=500, help="Max env steps")
    parser.add_argument("--scale", type=int, default=4, help="Nearest-neighbor upscale")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback / video FPS")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between frames when showing")
    parser.add_argument("--show", action="store_true", help="Open live matplotlib window (default)")
    parser.add_argument("--no-show", action="store_true", help="No live window (use with --video)")
    parser.add_argument("--video", default="", help="Write replay (.mp4 or .gif)")
    args = parser.parse_args()
    delay = args.delay if args.delay is not None else (1.0 / args.fps)
    show = args.show if args.show else not args.no_show

    _ensure_build()
    rule_tags = parse_rules_arg(args.rules, args.env)
    levels = parse_levels_arg([str(x) for x in args.level])
    configs = [
        build_env_id(args.env, tag, lvl, args.version)
        for tag in rule_tags
        for lvl in levels
    ]
    multi = len(configs) > 1
    video_base = Path(args.video) if args.video else None

    if not show and not video_base and configs:
        print("Running headless (no window, no video).", flush=True)

    for env_id in configs:
        out = _video_out_path(video_base, env_id, multi) if video_base else None
        try:
            _run_episode(
                env_id,
                steps=args.steps,
                scale=args.scale,
                delay=delay,
                show=show,
                video=out,
                fps=args.fps,
                block_on_close=not multi,
            )
        except Exception as exc:
            print(f"[{env_id}] ERROR: {exc}", flush=True)
            if not multi:
                raise
            print(f"[{env_id}] skipping to next variant.", flush=True)


def _title(env: str, tick: int, reward: float, winner: str, done: bool = False) -> str:
    status = "DONE" if done else "playing"
    w = f" | {winner}" if winner else ""
    return f"{env} | tick {tick} | r={reward:.1f}{w} | {status}"


if __name__ == "__main__":
    main()
