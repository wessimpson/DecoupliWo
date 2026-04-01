import argparse
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from agent.ppo import build_ppo
from agent.vec_video import VecSub0ClipRecorder
from agent.callback.transition_logger import TransitionLoggerCallback
from env.wrappers import AttachPlayerInfo, EnsureUint8Obs
from env.env import make_env_factory

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 8
CHECKPOINT_DIR = Path("agent") / "checkpoints"
VIDEO_DIR = Path("videos")
TENSORBOARD_DIR = Path("agent") / "tb_logs"

# Matches SB3 log ``total_timesteps`` (not raw vec-step count; see ``VecSub0ClipRecorder``).
VIDEO_EVERY_STEPS = 100_000
VIDEO_LENGTH = 1000
VIDEO_UPSCALE = 1
VIDEO_FPS = 15
ACTION_DECISION_INTERVAL = 4  # unified across envs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on selected Atari game (OCAtari)")
    # Available environments:
    # - air_raid (ALE/AirRaid-v5)
    # - assault (ALE/Assault-v5)
    # - beam_rider (ALE/BeamRider-v5)
    # - breakout (ALE/Breakout-v5)
    # - carnival (ALE/Carnival-v5)
    # - centipede (ALE/Centipede-v5)
    # - galaxian (ALE/Galaxian-v5)
    # - space_invaders (ALE/SpaceInvaders-v5)
    parser.add_argument(
        "--env",
        type=str,
        default="space_invaders",
        help="Environment module name under env/ (e.g., space_invaders, galaxian)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record periodic MP4 clips under videos/ (slower; needs moviepy)",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Build env factory via registry (dependency injection of configuration)
    render_mode = "rgb_array" if args.video else None
    make_env_fn = make_env_factory(
        game=args.env,
        render_mode=render_mode,
        mode="ram",
        obs_mode="ori",
        hud=True,
        frameskip=1,
        decision_interval=ACTION_DECISION_INTERVAL,
    )
    vec = make_vec_env(
        make_env_fn,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        wrapper_class=lambda e: EnsureUint8Obs(AttachPlayerInfo(e)),
        env_kwargs={},
    )
    # Always frame-stack for 'ori' RGB observations.
    vec = VecFrameStack(vec, n_stack=4, channels_order="last")
    if args.video:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        vec = VecSub0ClipRecorder(
            vec,
            str(VIDEO_DIR),
            fps=VIDEO_FPS,
            record_video_trigger=lambda step_id: step_id % VIDEO_EVERY_STEPS == 0,
            video_length=VIDEO_LENGTH,
            name_prefix=args.env,
            upscale=VIDEO_UPSCALE,
        )

    model = build_ppo(vec, tensorboard_log=str(TENSORBOARD_DIR))
    transitions_dir = Path("agent") / "transitions"
    callback = TransitionLoggerCallback(
        output_dir=str(transitions_dir),
        env_name=args.env,
        chunk_size=100_000,
        save_observations=True,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=args.env, callback=callback)
    # Save under per-environment subfolder
    env_name = args.env
    (CHECKPOINT_DIR / env_name).mkdir(parents=True, exist_ok=True)
    model.save(str(CHECKPOINT_DIR / env_name / f"{env_name}_ppo"))
    vec.close()


if __name__ == "__main__":
    main()
