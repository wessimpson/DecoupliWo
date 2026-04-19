from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_pong import BallState, GameState, PaddleState, PongEnv
from data.pong_common import (
    GAME_TO_ID,
    OBJECT_TYPE_TO_ID,
    TransitionShardWriter,
    flat_pong_state_to_slots,
    load_shards,
    slot_config_from_env,
)


class PongEnvTests(unittest.TestCase):
    def test_reset_returns_flat_observation(self) -> None:
        env = PongEnv(render_mode=None)
        obs, info = env.reset(seed=123)
        self.assertEqual(obs.shape, (6,))
        self.assertEqual(info["event"], "reset")

    def test_left_wall_reflects(self) -> None:
        env = PongEnv(mode="normal", render_mode=None)
        env.reset(seed=1)
        env.set_state(
            GameState(
                ball=BallState(x=10.0, y=120.0, vx=-240.0, vy=0.0, radius=8.0),
                paddle=PaddleState(
                    x=env.config.width - env.config.paddle_margin - env.config.paddle_width,
                    y=160.0,
                    width=env.config.paddle_width,
                    height=env.config.paddle_height,
                ),
            )
        )
        _, _, terminated, truncated, _ = env.step(0)
        state = env.get_state()
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreater(state.ball.vx, 0.0)

    def test_gravity_mode_floor_bounce_stays_active(self) -> None:
        env = PongEnv(mode="gravity", render_mode=None, gravity=900.0, dt=1.0 / 30.0)
        env.reset(seed=1)
        env.set_state(
            GameState(
                ball=BallState(
                    x=200.0,
                    y=env.config.height - env.config.ball_radius - 2.0,
                    vx=-180.0,
                    vy=220.0,
                    radius=env.config.ball_radius,
                ),
                paddle=PaddleState(
                    x=env.config.width - env.config.paddle_margin - env.config.paddle_width,
                    y=170.0,
                    width=env.config.paddle_width,
                    height=env.config.paddle_height,
                ),
            )
        )
        _, _, terminated, truncated, _ = env.step(0)
        state = env.get_state()
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLessEqual(state.ball.y, env.config.height - env.config.ball_radius + 1e-6)
        self.assertLess(state.ball.vy, 0.0)

    def test_teleport_mode_wraps_vertical_position(self) -> None:
        env = PongEnv(mode="teleport", render_mode=None)
        env.reset(seed=1)
        env.set_state(
            GameState(
                ball=BallState(
                    x=250.0,
                    y=env.config.height + env.config.ball_radius + 4.0,
                    vx=-120.0,
                    vy=60.0,
                    radius=env.config.ball_radius,
                ),
                paddle=PaddleState(
                    x=env.config.width - env.config.paddle_margin - env.config.paddle_width,
                    y=150.0,
                    width=env.config.paddle_width,
                    height=env.config.paddle_height,
                ),
            )
        )
        _, _, _, _, info = env.step(0)
        state = env.get_state()
        self.assertTrue(info["wrapped"])
        self.assertGreaterEqual(state.ball.y, -env.config.ball_radius)
        self.assertLessEqual(state.ball.y, env.config.height + env.config.ball_radius)
        self.assertAlmostEqual(state.ball.vy, 60.0)

    def test_missing_the_ball_terminates_episode(self) -> None:
        env = PongEnv(mode="normal", render_mode=None)
        env.reset(seed=1)
        env.set_state(
            GameState(
                ball=BallState(
                    x=env.config.width + env.config.ball_radius + 1.0,
                    y=env.config.height / 2.0,
                    vx=120.0,
                    vy=0.0,
                    radius=env.config.ball_radius,
                ),
                paddle=PaddleState(
                    x=env.config.width - env.config.paddle_margin - env.config.paddle_width,
                    y=150.0,
                    width=env.config.paddle_width,
                    height=env.config.paddle_height,
                ),
            )
        )
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertTrue(info["miss"])

    def test_flat_state_to_slots_uses_ball_and_paddle_slots(self) -> None:
        env = PongEnv(render_mode=None)
        obs, _ = env.reset(seed=1)
        slots, mask = flat_pong_state_to_slots(obs, slot_config_from_env(env))
        self.assertEqual(slots.shape, (10, 7))
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(float(mask.sum()), 2.0)
        self.assertEqual(int(slots[0, 6]), OBJECT_TYPE_TO_ID["ball"])
        self.assertEqual(int(slots[1, 6]), OBJECT_TYPE_TO_ID["paddle"])
        self.assertAlmostEqual(float(slots[0, 0]), float(obs[0]))
        self.assertAlmostEqual(float(slots[1, 1]), float(obs[4]))

    def test_transition_writer_saves_object_slots_and_game_id(self) -> None:
        env = PongEnv(render_mode=None)
        obs, _ = env.reset(seed=1)
        next_obs, reward, terminated, truncated, info = env.step(0)
        with tempfile.TemporaryDirectory() as tmp:
            writer = TransitionShardWriter(tmp, "train", chunk_size=1, slot_config=slot_config_from_env(env))
            writer.append(obs, 0, next_obs, reward, terminated, truncated, 0, 0, 0, 0)
            writer.flush()
            data = load_shards(tmp, "train")
        self.assertEqual(data["object_slots"].shape, (1, 10, 7))
        self.assertEqual(data["next_object_slots"].shape, (1, 10, 7))
        self.assertEqual(data["object_mask"].shape, (1, 10))
        self.assertEqual(int(data["game_id"][0]), GAME_TO_ID["pong"])
        np.testing.assert_allclose(data["state"][0], obs)

    def test_load_shards_synthesizes_slots_for_legacy_flat_shards(self) -> None:
        env = PongEnv(render_mode=None)
        obs, _ = env.reset(seed=2)
        next_obs, reward, terminated, truncated, _ = env.step(0)
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "train"
            split.mkdir()
            np.savez_compressed(
                split / "shard_00000.npz",
                state=obs[None],
                action=np.asarray([0], dtype=np.int64),
                next_state=next_obs[None],
                reward=np.asarray([reward], dtype=np.float32),
                terminated=np.asarray([terminated], dtype=np.bool_),
                truncated=np.asarray([truncated], dtype=np.bool_),
                rule_id=np.asarray([0], dtype=np.int64),
                event_id=np.asarray([0], dtype=np.int64),
                episode_id=np.asarray([0], dtype=np.int64),
                step=np.asarray([0], dtype=np.int64),
            )
            data = load_shards(tmp, "train")
        self.assertIn("object_slots", data)
        self.assertIn("game_id", data)
        self.assertEqual(data["object_slots"].shape, (1, 10, 7))
        self.assertEqual(float(data["object_mask"][0].sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
