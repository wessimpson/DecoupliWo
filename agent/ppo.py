import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def build_ppo(env: VecEnv, tensorboard_log: str | None = None) -> PPO:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available (PyTorch was likely installed CPU-only from PyPI). "
        )
    # Linear schedule helpers per SB3 best practices
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    return PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=torch.device("cuda"),
        tensorboard_log=tensorboard_log,
        # Atari-tuned PPO settings (from SB3 RL Zoo / OpenAI baselines / CleanRL)
        learning_rate=linear_schedule(2.5e-4),
        n_steps=128,                 # per-env rollout length
        batch_size=256,              # minibatch size
        n_epochs=4,                  # PPO epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.1),
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01
    )
