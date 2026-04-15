from __future__ import annotations

import unittest

from custom_pong import BallState, GameState, PaddleState, PongEnv


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


if __name__ == "__main__":
    unittest.main()
