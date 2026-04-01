"""Collect Space Invaders rollouts as ``.npz`` (frames + actions) for ``world_model`` training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env.space_invaders import make_space_invaders_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/space_invaders_rollouts")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = make_space_invaders_env(render_mode="rgb_array")
    n_act = env.action_space.n
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        frames = []
        actions = []
        for _ in range(args.max_steps):
            frames.append(np.asarray(obs, dtype=np.uint8).copy())
            a = env.action_space.sample()
            actions.append(int(a))
            obs, _r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if len(frames) < 16:
            continue
        fp = out / f"rollout_{ep:05d}.npz"
        np.savez_compressed(
            fp,
            frames=np.stack(frames, axis=0),
            actions=np.asarray(actions, dtype=np.int64),
            n_actions=np.array([n_act], dtype=np.int64),
        )
        print(f"wrote {fp} ({len(frames)} steps)")
    env.close()


if __name__ == "__main__":
    main()
