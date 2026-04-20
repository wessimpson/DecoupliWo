from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise unittest.SkipTest("torch is not installed in this Python environment") from exc

from data.pong_common import OBJECT_TYPE_TO_ID
from models.object_graph import ObjectGraphBuilder
from models.object_losses import LossWeights, compute_object_centric_losses
from models.rule_conditioned_gnn import PongObjectConstants, RuleConditionedPongGNN


class ObjectWorldModelTests(unittest.TestCase):
    def test_graph_builder_uses_relative_features_and_hybrid_edges(self) -> None:
        constants = PongObjectConstants()
        builder = ObjectGraphBuilder(constants.slot_scales, num_object_types=4, edge_mode="hybrid", distance_threshold=0.05)
        slots = torch.zeros(1, 4, 7)
        slots[0, 0, :7] = torch.tensor([100.0, 120.0, 10.0, -5.0, 16.0, 16.0, OBJECT_TYPE_TO_ID["ball"]])
        slots[0, 1, :7] = torch.tensor([500.0, 120.0, 0.0, 0.0, 12.0, 88.0, OBJECT_TYPE_TO_ID["paddle"]])
        slots[0, 2, :7] = torch.tensor([110.0, 120.0, 0.0, 0.0, 30.0, 20.0, OBJECT_TYPE_TO_ID["block"]])
        slots[0, 3, :7] = torch.tensor([400.0, 400.0, 0.0, 0.0, 30.0, 20.0, OBJECT_TYPE_TO_ID["block"]])
        mask = torch.ones(1, 4)

        graph = builder(slots, mask)

        self.assertEqual(graph.edge_features.shape, (1, 4, 4, builder.edge_feature_dim))
        self.assertEqual(float(graph.edge_mask[0, 0, 0]), 0.0)
        self.assertEqual(float(graph.edge_mask[0, 1, 0]), 1.0)
        self.assertEqual(float(graph.edge_mask[0, 2, 3]), 0.0)
        self.assertAlmostEqual(float(graph.relative_position[0, 1, 0, 0]), (100.0 - 500.0) / constants.width)
        self.assertAlmostEqual(float(graph.relative_velocity[0, 1, 0, 1]), (-5.0 - 0.0) / constants.max_ball_speed)

    def test_model_forward_and_losses_are_slot_based(self) -> None:
        model = RuleConditionedPongGNN(latent_dim=16, hidden_dim=32, rule_dim=8, type_dim=4, message_passing_steps=2)
        state = torch.tensor([[100.0, 120.0, 240.0, 30.0, 160.0, 0.0]], dtype=torch.float32)
        next_state = torch.tensor([[104.0, 120.5, 240.0, 30.0, 160.0, 0.0]], dtype=torch.float32)
        slots, mask = model.state_to_slots(state)
        next_slots, next_mask = model.state_to_slots(next_state)
        batch = {
            "state": state,
            "action": torch.tensor([0], dtype=torch.long),
            "rule_id": torch.tensor([0], dtype=torch.long),
            "event_id": torch.tensor([1], dtype=torch.long),
            "game_id": torch.tensor([0], dtype=torch.long),
            "object_slots": slots,
            "next_object_slots": next_slots,
            "object_mask": mask,
            "next_object_mask": next_mask,
        }

        out = model(state, batch["action"], batch["rule_id"], object_slots=slots, object_mask=mask, game_id=batch["game_id"])
        losses = compute_object_centric_losses(model, batch, out, LossWeights(alpha_rel=0.1, gamma_event=0.05))

        self.assertEqual(out["pred_next_slots"].shape, slots.shape)
        self.assertEqual(out["pred_event_logits"].shape[-1], model.num_events)
        self.assertIn("rel_loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
