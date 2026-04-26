from __future__ import annotations

import unittest

import numpy as np
import torch

from custom_breakout import BreakoutEnv
from custom_pong import PongEnv
from data.gns_shared_dataset import GNSNormalizationStats
from data.pong_common import GAME_TO_ID, flat_pong_state_to_slots
from models.gns_shared_simulator import GNSSharedSimulator


class GNSSharedSimulatorTests(unittest.TestCase):
    def _build_model(self) -> GNSSharedSimulator:
        stats = GNSNormalizationStats(
            pos_mean=np.zeros(2, dtype=np.float32),
            pos_std=np.ones(2, dtype=np.float32),
            vel_mean=np.zeros(2, dtype=np.float32),
            vel_std=np.ones(2, dtype=np.float32),
        )
        return GNSSharedSimulator(stats=stats, history_length=6, latent_dim=32, hidden_dim=64, message_passing_steps=2)

    def test_analytic_paddle_update_matches_pong_env(self) -> None:
        env = PongEnv(render_mode=None)
        obs, _ = env.reset(seed=3)
        slots, mask = flat_pong_state_to_slots(obs)
        env.step(1)
        expected = env.get_state()
        model = self._build_model()
        next_slots, _ = model.analytic_paddle_update(
            torch.as_tensor(slots[None], dtype=torch.float32),
            torch.as_tensor(mask[None], dtype=torch.float32),
            torch.as_tensor([1], dtype=torch.long),
            torch.as_tensor([GAME_TO_ID["pong"]], dtype=torch.long),
        )
        self.assertAlmostEqual(float(next_slots[0, 1, 1]), expected.paddle.y, places=4)
        self.assertAlmostEqual(float(next_slots[0, 1, 3]), expected.paddle.vy, places=4)

    def test_analytic_paddle_update_matches_breakout_env(self) -> None:
        env = BreakoutEnv(render_mode=None)
        env.reset(seed=3)
        slots, mask = env.state_to_slots()
        env.step(2)
        expected = env.get_state()
        model = self._build_model()
        next_slots, _ = model.analytic_paddle_update(
            torch.as_tensor(slots[None], dtype=torch.float32),
            torch.as_tensor(mask[None], dtype=torch.float32),
            torch.as_tensor([2], dtype=torch.long),
            torch.as_tensor([GAME_TO_ID["breakout"]], dtype=torch.long),
        )
        self.assertAlmostEqual(float(next_slots[0, 1, 0]), expected.paddle.x, places=4)
        self.assertAlmostEqual(float(next_slots[0, 1, 2]), expected.paddle.vx, places=4)


if __name__ == "__main__":
    unittest.main()
