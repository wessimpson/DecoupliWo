#!/usr/bin/env python
"""
Play a GVGAI env with a Java search agent; display frames live or save a video/GIF.

Run from repo root or this folder:
  python training_data_gvgai/run_mcts.py
  python run_mcts.py --env aliens --rules multishot --level 0 --mcts-ms 40
  python run_mcts.py --env aliens --rules multishot/ricochet --level 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
GVGAI_JAVA_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "gvgai"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_data_gvgai.data.gvgai_jpype_env import GVGAIFileEnv
from training_data_gvgai.run_random_action import (
    _ensure_build,
    _obs_rgb,
    _paths_from_env_id,
    _save_video,
    _upscale,
    add_world_model_cli,
    build_env_id,
    parse_levels_arg,
    parse_rules_arg,
)


AGENT_CLASSES = {
    "sample_mcts": "tracks.singlePlayer.advanced.sampleMCTS.Agent",
    "olets": "tracks.singlePlayer.advanced.olets.Agent",
}

MCTS_PROFILE_PROPS = {
    "mcts_default": {},
    "mcts_exploit": {
        "mcts.k": "0.50",
        "mcts.rolloutDepth": "12",
        "mcts.maxIterations": "150",
        "mcts.finalSelection": "best_value",
        "mcts.temperature": "0.25",
        "mcts.actionEpsilon": "0.0",
    },
    "mcts_balanced": {
        "mcts.k": "1.41421356237",
        "mcts.rolloutDepth": "10",
        "mcts.maxIterations": "100",
        "mcts.finalSelection": "visit_softmax",
        "mcts.temperature": "0.35",
        "mcts.actionEpsilon": "0.02",
    },
    "mcts_explore": {
        "mcts.k": "2.50",
        "mcts.rolloutDepth": "12",
        "mcts.maxIterations": "80",
        "mcts.finalSelection": "visit_softmax",
        "mcts.temperature": "1.00",
        "mcts.actionEpsilon": "0.05",
    },
    "mcts_scout": {
        "mcts.k": "4.00",
        "mcts.rolloutDepth": "6",
        "mcts.maxIterations": "40",
        "mcts.finalSelection": "visit_softmax",
        "mcts.temperature": "1.50",
        "mcts.actionEpsilon": "0.15",
    },
}

MCTS_PROPERTY_KEYS = tuple(
    sorted({key for props in MCTS_PROFILE_PROPS.values() for key in props})
)


def _apply_mcts_profile(profile: str) -> None:
    from jpype import JClass

    props = MCTS_PROFILE_PROPS[profile]
    system = JClass("java.lang.System")
    for key in MCTS_PROPERTY_KEYS:
        if key in props:
            system.setProperty(key, props[key])
        else:
            system.clearProperty(key)


def _agent_class(agent: str, agent_class: str | None) -> str:
    if agent_class:
        return agent_class
    return AGENT_CLASSES[agent]


def _title(env: str, tick: int, reward: float, winner: str, done: bool = False) -> str:
    status = "DONE" if done else "playing"
    w = f" | {winner}" if winner else ""
    return f"{env} | tick {tick} | r={reward:.1f}{w} | {status}"


def _refresh_frame(fig, im, frame: np.ndarray, title: str, delay: float) -> None:
    import matplotlib.pyplot as plt

    im.set_data(frame)
    im.axes.set_title(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(max(delay, 0.001))


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
    mcts_ms: int,
    profile: str,
    agent_class: str,
    scale: int,
    delay: float,
    show: bool,
    video: Path | None,
    fps: float,
) -> None:
    game_file, level_files, level = _paths_from_env_id(env_id)
    env = GVGAIFileEnv(
        game_file,
        level_files,
        level=level,
        gvgai_root=GVGAI_JAVA_ROOT,
        max_episode_steps=steps,
    )
    _apply_mcts_profile(profile)

    frames: list[np.ndarray] = []
    plt = None
    im = None
    fig = None

    try:
        obs, info = env.reset()
        frame = _upscale(_obs_rgb(obs), scale)
        frames.append(frame)
        game_tick = int(info.get("game_tick", -1))
        print(
            f"[{env_id}] agent={agent_class} profile={profile} reset game_tick={game_tick} "
            f"winner={info.get('winner')}",
            flush=True,
        )

        if show:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(frame)
            ax.set_title(_title(env_id, max(game_tick, 0), 0.0, str(info.get("winner", ""))))
            ax.axis("off")
            fig.tight_layout()
            _refresh_frame(fig, im, frame, ax.get_title(), delay)

        for step_idx in range(steps):
            if show and fig is not None:
                import matplotlib.pyplot as plt

                plt.pause(0.001)
                fig.canvas.flush_events()

            obs, reward, terminated, truncated, info = env.step_agent(agent_class, mcts_ms)
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
                if step_idx == 0:
                    print(
                        f"[{env_id}] WARNING: game ended on the first MCTS step. "
                        "Rebuild levels (build_world_model_games.py) or try another --level.",
                        flush=True,
                    )
                break

        if show and plt is not None:
            print("Close the window to exit.", flush=True)
            plt.ioff()
            plt.show()

        if video is not None:
            _save_video(frames, video, fps)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Java GVGAI search agent with visual playback")
    add_world_model_cli(parser)
    parser.add_argument("--steps", type=int, default=500, help="Max env steps")
    parser.add_argument(
        "--mcts-ms",
        type=int,
        default=40,
        help="Per-action MCTS CPU budget in milliseconds",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(MCTS_PROFILE_PROPS),
        default="mcts_default",
        help="sampleMCTS profile to visualize; OLETS ignores these properties.",
    )
    parser.add_argument(
        "--agent",
        choices=tuple(AGENT_CLASSES),
        default="sample_mcts",
        help="Java agent shortcut. Use --agent olets for the stronger open-loop MCTS agent.",
    )
    parser.add_argument(
        "--agent-class",
        default="",
        help="Fully-qualified Java agent class; overrides --agent.",
    )
    parser.add_argument("--scale", type=int, default=1, help="Nearest-neighbor upscale")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback / video FPS")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between frames when showing")
    parser.add_argument("--show", action="store_true", help="Open live matplotlib window (default)")
    parser.add_argument("--no-show", action="store_true", help="No live window (use with --video)")
    parser.add_argument("--video", default="", help="Write replay (.mp4 or .gif)")
    args = parser.parse_args()
    delay = args.delay if args.delay is not None else (1.0 / args.fps)
    show = args.show if args.show else not args.no_show

    _ensure_build()
    rule_tags = parse_rules_arg(args.rules)
    levels = parse_levels_arg([str(x) for x in args.level])
    configs = [
        build_env_id(args.env, tag, lvl, args.version)
        for tag in rule_tags
        for lvl in levels
    ]
    multi = len(configs) > 1
    video_base = Path(args.video) if args.video else None
    agent_class = _agent_class(args.agent, args.agent_class or None)

    if not show and not video_base and configs:
        print("No --video and --no-show: nothing to display.", flush=True)
        return

    for env_id in configs:
        out = _video_out_path(video_base, env_id, multi) if video_base else None
        _run_episode(
            env_id,
            steps=args.steps,
            mcts_ms=args.mcts_ms,
            profile=args.profile,
            agent_class=agent_class,
            scale=args.scale,
            delay=delay,
            show=show,
            video=out,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
