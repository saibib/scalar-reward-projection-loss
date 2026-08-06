import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_projection_diagnostics import (
    edge_lookup_predictions,
    projection_diagnostics,
)
from run_multipref_text_reward_downstream import sorted_edge_observations
from run_multipref_lora_reward_downstream import analyze_triplets


def edge_frame(probabilities, support=100.0):
    rows = []
    for (i, j), p_i in probabilities.items():
        rows.append(
            {
                "i": i,
                "j": j,
                "support": support,
                "weighted_margin": support * (2.0 * p_i - 1.0),
                "margin": 2.0 * p_i - 1.0,
            }
        )
    return pd.DataFrame(rows)


class RewardProjectionDiagnosticsTest(unittest.TestCase):
    def test_exact_bradley_terry_flow_has_zero_loss_matched_residual(self) -> None:
        scores = {"A": 0.8, "B": 0.1, "C": -0.9}
        probabilities = {}
        for i, j in [("A", "B"), ("A", "C"), ("B", "C")]:
            probabilities[(i, j)] = 1.0 / (1.0 + math.exp(-(scores[i] - scores[j])))

        result = projection_diagnostics(
            edge_frame(probabilities),
            ("A", "B", "C"),
            logit_smoothing=0.0,
        )

        self.assertEqual(result["projection_fit_ok"], 1.0)
        self.assertLess(result["kl_projection_per_weight"], 1e-12)
        self.assertLess(result["brier_projection_per_weight"], 1e-12)
        self.assertLess(result["logit_hodge_fraction"], 1e-12)

    def test_logit_cycle_has_positive_scalar_projection_regret(self) -> None:
        # A > B, B > C, C > A in the stored A-B, A-C, B-C orientation.
        probabilities = {("A", "B"): 0.8, ("A", "C"): 0.2, ("B", "C"): 0.8}
        result = projection_diagnostics(edge_frame(probabilities), ("A", "B", "C"))

        self.assertGreater(result["kl_projection_per_weight"], 0.05)
        self.assertGreater(result["brier_projection_per_weight"], 0.02)
        self.assertGreater(result["logit_hodge_fraction"], 0.99)

    def test_projection_is_invariant_to_consistent_node_relabeling(self) -> None:
        original = edge_frame({("A", "B"): 0.72, ("A", "C"): 0.31, ("B", "C"): 0.64})
        renamed = original.replace({"A": "X", "B": "Y", "C": "Z"})
        first = projection_diagnostics(original, ("A", "B", "C"))
        second = projection_diagnostics(renamed, ("X", "Y", "Z"))

        for key in [
            "kl_projection_per_weight",
            "kl_projection_fraction",
            "brier_projection_per_weight",
            "brier_projection_fraction",
            "logit_hodge_fraction",
        ]:
            self.assertTrue(math.isclose(first[key], second[key], rel_tol=1e-10, abs_tol=1e-12), key)

    def test_edge_lookup_predictions_respect_row_orientation(self) -> None:
        edges = edge_frame({("A", "B"): 0.75}, support=10.0)
        obs = pd.DataFrame(
            [
                {"model_a": "A", "model_b": "B", "i": "A", "j": "B"},
                {"model_a": "B", "model_b": "A", "i": "A", "j": "B"},
            ]
        )
        predictions = edge_lookup_predictions(edges, obs, smoothing=0.0)
        np.testing.assert_allclose(predictions, np.array([0.75, 0.25]))

    def test_same_model_comparisons_can_train_rm_without_entering_graph(self) -> None:
        flat = pd.DataFrame(
            [
                {
                    "raw_idx": 0,
                    "comparison_id": "same",
                    "prompt_id": "p0",
                    "annotation_group": "normal",
                    "evaluator": "u0",
                    "model_a": "A",
                    "model_b": "A",
                    "overall_sign": 1,
                    "overall_weight": 1.0,
                },
                {
                    "raw_idx": 1,
                    "comparison_id": "cross",
                    "prompt_id": "p1",
                    "annotation_group": "normal",
                    "evaluator": "u1",
                    "model_a": "A",
                    "model_b": "B",
                    "overall_sign": -1,
                    "overall_weight": 1.0,
                },
            ]
        )
        graph = sorted_edge_observations(flat, "overall", "all")
        reward_train = sorted_edge_observations(flat, "overall", "all", include_same_model=True)

        self.assertEqual(len(graph), 1)
        self.assertEqual(len(reward_train), 2)
        self.assertEqual(int(reward_train.iloc[0]["row_sign"]), 1)
        self.assertEqual(reward_train.iloc[0]["i"], reward_train.iloc[0]["j"])

    def test_ties_are_optional_half_probability_training_targets(self) -> None:
        flat = pd.DataFrame(
            [
                {
                    "raw_idx": 0,
                    "comparison_id": "tie",
                    "prompt_id": "p0",
                    "annotation_group": "normal",
                    "evaluator": "u0",
                    "model_a": "A",
                    "model_b": "B",
                    "overall_sign": 0,
                    "overall_weight": 0.0,
                    "overall_is_tie": True,
                }
            ]
        )
        dropped = sorted_edge_observations(flat, "overall", "all")
        retained = sorted_edge_observations(
            flat,
            "overall",
            "all",
            include_ties=True,
            tie_weight=2.0,
        )

        self.assertTrue(dropped.empty)
        self.assertEqual(len(retained), 1)
        self.assertEqual(float(retained.iloc[0]["target"]), 0.5)
        self.assertEqual(int(retained.iloc[0]["sign"]), 0)
        self.assertEqual(float(retained.iloc[0]["weight"]), 2.0)

    def test_lora_region_analysis_emits_projection_and_uplift_metrics(self) -> None:
        train = pd.DataFrame(
            [
                {"model_a": "A", "model_b": "B", "i": "A", "j": "B", "sign": 1, "row_sign": 1, "weight": 10.0},
                {"model_a": "A", "model_b": "C", "i": "A", "j": "C", "sign": -1, "row_sign": -1, "weight": 10.0},
                {"model_a": "B", "model_b": "C", "i": "B", "j": "C", "sign": 1, "row_sign": 1, "weight": 10.0},
            ]
        )
        test = train.copy()
        test["neural_p_a"] = np.array([0.8, 0.2, 0.8])
        args = SimpleNamespace(
            max_triplets=None,
            min_train_edge_weight=5.0,
            min_test_weight=5.0,
            min_test_edges=2,
            tau=0.0,
        )
        fit_meta = {
            "fit_ok": True,
            "n_train_obs": 3,
            "train_weight": 30.0,
            "final_train_loss": 0.5,
            "train_seconds": 1.0,
        }

        rows = analyze_triplets(0, "all", "overall", train, test, args, fit_meta)

        self.assertEqual(len(rows), 1)
        self.assertIn("train_kl_projection_per_weight", rows[0])
        self.assertIn("edge_lookup_test_log_loss", rows[0])
        self.assertIn("uplift_test_log_loss", rows[0])


if __name__ == "__main__":
    unittest.main()
