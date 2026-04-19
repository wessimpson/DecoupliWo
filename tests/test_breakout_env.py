from __future__ import annotations

import tempfile
import unittest

from custom_breakout import BallState, BreakoutEnv, BreakoutState, PaddleState
from data.pong_common import GAME_TO_ID, TransitionShardWriter, load_shards


class BreakoutEnvTests(unittest.TestCase):
    def test_reset_returns_flat_observation_and_slots(self) -> None:
        env = BreakoutEnv(render_mode=None)
        obs, info = env.reset(seed=123)
        slots, mask = env.state_to_slots()
        self.assertEqual(obs.shape, (6,))
        self.assertEqual(info["event"], "reset")
        self.assertEqual(slots.shape, (10, 7))
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(float(mask.sum()), 10.0)

    def test_block_hit_deactivates_block(self) -> None:
        env = BreakoutEnv(render_mode=None, dt=1.0 / 60.0)
        env.reset(seed=1)
        state = env.get_state()
        block = state.blocks[0]
        env.set_state(
            BreakoutState(
                ball=BallState(
                    x=block.x + block.width / 2.0,
                    y=block.y + block.height + env.config.ball_radius + 1.0,
                    vx=0.0,
                    vy=-240.0,
                    radius=env.config.ball_radius,
                ),
                paddle=state.paddle,
                blocks=state.blocks,
            )
        )
        _, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreaterEqual(reward, 1.0)
        self.assertTrue(info["block_hit"])
        self.assertEqual(sum(1 for block in env.get_state().blocks if block.active), 7)

    def test_transition_writer_accepts_breakout_slots(self) -> None:
        env = BreakoutEnv(render_mode=None)
        obs, _ = env.reset(seed=1)
        slots, mask = env.state_to_slots()
        next_obs, reward, terminated, truncated, _ = env.step(0)
        next_slots, next_mask = env.state_to_slots()
        with tempfile.TemporaryDirectory() as tmp:
            writer = TransitionShardWriter(tmp, "train", chunk_size=1)
            writer.append(
                obs,
                0,
                next_obs,
                reward,
                terminated,
                truncated,
                0,
                0,
                0,
                0,
                game_id=GAME_TO_ID["breakout"],
                object_slots=slots,
                next_object_slots=next_slots,
                object_mask=mask,
                next_object_mask=next_mask,
            )
            writer.flush()
            data = load_shards(tmp, "train")
        self.assertEqual(int(data["game_id"][0]), GAME_TO_ID["breakout"])
        self.assertEqual(data["object_slots"].shape, (1, 10, 7))
        self.assertEqual(float(data["object_mask"][0].sum()), 10.0)


if __name__ == "__main__":
    unittest.main()
