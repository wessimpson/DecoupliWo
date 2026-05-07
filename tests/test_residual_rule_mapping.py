from __future__ import annotations

import os
import unittest

import torch

from world_model.dataset import (
	correction_rule_tensor_from_name,
	correction_rule_tuple_from_env_name,
	is_rule_variant_name,
)


class ResidualRuleMappingTest(unittest.TestCase):
	def test_folder_rule_vectors(self) -> None:
		self.assertEqual(correction_rule_tuple_from_env_name("aliens"), (0.0, 0.0, 0.0))
		self.assertEqual(correction_rule_tuple_from_env_name("aliens_rules_fast"), (1.0, 0.0, 0.0))
		self.assertEqual(correction_rule_tuple_from_env_name("aliens_rules_multishot"), (0.0, 1.0, 0.0))
		self.assertEqual(correction_rule_tuple_from_env_name("aliens_rules_ricochet"), (0.0, 0.0, 1.0))
		self.assertFalse(is_rule_variant_name("aliens"))
		self.assertTrue(is_rule_variant_name("aliens_rules_fast"))

	def test_inference_rule_vectors(self) -> None:
		self.assertTrue(torch.equal(correction_rule_tensor_from_name("normal"), torch.tensor([0.0, 0.0, 0.0])))
		self.assertTrue(torch.equal(correction_rule_tensor_from_name("fast"), torch.tensor([1.0, 0.0, 0.0])))
		self.assertTrue(torch.equal(correction_rule_tensor_from_name("multishot"), torch.tensor([0.0, 1.0, 0.0])))
		self.assertTrue(torch.equal(correction_rule_tensor_from_name("ricochet"), torch.tensor([0.0, 0.0, 1.0])))
		self.assertTrue(torch.equal(correction_rule_tensor_from_name("multishot+ricochet"), torch.tensor([0.0, 1.0, 1.0])))

	def test_residual_target_zero_anchor_formula(self) -> None:
		full_target = torch.randn(3, 4, 2, 2)
		base_pred = torch.randn_like(full_target)
		mask = torch.tensor([False, True, False]).view(-1, 1, 1, 1)
		delta_target = full_target - base_pred
		delta_target = torch.where(mask, torch.zeros_like(delta_target), delta_target)
		self.assertTrue(torch.equal(delta_target[1], torch.zeros_like(delta_target[1])))
		self.assertTrue(torch.equal(delta_target[0], full_target[0] - base_pred[0]))

	@unittest.skipUnless(os.environ.get("RUN_WORLD_MODEL_HEAVY_TESTS") == "1", "heavy UNet smoke test")
	def test_diffuser_forward_shapes(self) -> None:
		from world_model.model.net.diffuser import Diffuser

		for num_rules in (0, 3):
			diff = Diffuser(
				num_actions=7,
				latent_channels=4,
				cross_attention_dim=32,
				history_len=2,
				num_rules=num_rules,
				pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
			)
			x = torch.randn(2, 3, 4, 16, 16)
			t = torch.randint(0, diff.noise_scheduler.config.num_train_timesteps, (2,))
			a = torch.tensor([0, 1])
			r = torch.zeros(2, num_rules) if num_rules else None
			y = diff(x, t, a, r)
			self.assertEqual(tuple(y.shape), (2, 4, 16, 16))


if __name__ == "__main__":
	unittest.main()
