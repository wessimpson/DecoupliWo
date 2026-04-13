from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
from ocatari.core import OCAtari

from env.wrappers import HoldActionBetweenDecisions

# Map short names to ALE env IDs

#air_raid, space_invaders
#air_raid, assault
#assault, space_invaders
GAME_TO_ENV_ID: dict[str, str] = {
	"space_invaders": "ALE/SpaceInvaders-v5",
	"air_raid": "ALE/AirRaid-v5",
	"assault": "ALE/Assault-v5",
	"beam_rider": "ALE/BeamRider-v5",
	"breakout": "ALE/Breakout-v5",
	"carnival": "ALE/Carnival-v5",
	"centipede": "ALE/Centipede-v5",
	"galaxian": "ALE/Galaxian-v5",
}


def make_atari_env(
	game: str,
	render_mode: Optional[str] = "rgb_array",
	mode: str = "ram",
	obs_mode: str = "ori",
	hud: bool = True,
	frameskip: int = 1,
	decision_interval: Optional[int] = None,
) -> gym.Env:
	env_id = GAME_TO_ENV_ID.get(game)
	if env_id is None:
		available = ", ".join(sorted(GAME_TO_ENV_ID.keys()))
		raise ValueError(f"Unknown game '{game}'. Available: {available}")

	# Workaround: OCAtari assault RAM extractor crashes with HUD enabled.
	if game == "assault":
		hud = False

	env = OCAtari(
		env_id,
		mode=mode,
		hud=hud,
		obs_mode=obs_mode,
		render_mode=render_mode,
		frameskip=frameskip,
	)
	if decision_interval and decision_interval > 1:
		env = HoldActionBetweenDecisions(env, interval=decision_interval)
	return env


def make_env_factory(
	game: str,
	render_mode: Optional[str],
	mode: str,
	obs_mode: str,
	hud: bool,
	frameskip: int,
	decision_interval: Optional[int],
) -> Callable[[], gym.Env]:
	def _factory() -> gym.Env:
		return make_atari_env(
			game=game,
			render_mode=render_mode,
			mode=mode,
			obs_mode=obs_mode,
			hud=hud,
			frameskip=frameskip,
			decision_interval=decision_interval,
		)
	return _factory

def main() -> None:
	# Simple smoke test using Space Invaders
	env = make_atari_env(
		game="space_invaders",
		render_mode=None,
		mode="ram",
		obs_mode="ori",
		hud=True,
		frameskip=1,
		decision_interval=1,
	)
	obs, info = env.reset()
	print("reset obs shape:", getattr(obs, "shape", None))
	obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
	print("step: reward=", reward, "terminated=", terminated, "truncated=", truncated)
	env.close()

if __name__ == "__main__":
	main()

