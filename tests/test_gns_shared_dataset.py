from __future__ import annotations

import tempfile
import unittest

import numpy as np

from custom_pong import PongEnv
from data.gns_shared_dataset import GNSTrajectoryWindowDataset
from data.pong_common import GAME_TO_ID, TransitionShardWriter, flat_pong_state_to_slots, slot_config_from_env, write_metadata


class GNSSharedDatasetTests(unittest.TestCase):
    def test_window_dataset_builds_consecutive_histories(self) -> None:
        env = PongEnv(render_mode=None)
        obs, _ = env.reset(seed=7)
        slot_config = slot_config_from_env(env)
        with tempfile.TemporaryDirectory() as tmp:
            writer = TransitionShardWriter(tmp, "train", chunk_size=100, slot_config=slot_config)
            state = obs
            for step in range(8):
                slots, mask = flat_pong_state_to_slots(state, slot_config)
                next_state, reward, terminated, truncated, info = env.step(0)
                next_slots, next_mask = flat_pong_state_to_slots(next_state, slot_config)
                writer.append(
                    state,
                    0,
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    0,
                    1,
                    0,
                    step,
                    game_id=GAME_TO_ID["pong"],
                    object_slots=slots,
                    next_object_slots=next_slots,
                    object_mask=mask,
                    next_object_mask=next_mask,
                )
                state = next_state
            writer.flush()
            val_writer = TransitionShardWriter(tmp, "val", chunk_size=1, slot_config=slot_config)
            val_writer.flush()
            write_metadata(tmp, {"env_config": {"width": env.config.width, "height": env.config.height}})
            dataset = GNSTrajectoryWindowDataset(tmp, "train", history_length=6)
            sample = dataset[0]
        self.assertEqual(sample["history_slots"].shape, (6, 10, 7))
        self.assertEqual(sample["history_mask"].shape, (6, 10))
        self.assertEqual(sample["target_next_slots"].shape, (10, 7))
        self.assertEqual(float(sample["dynamic_pos_mask"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
