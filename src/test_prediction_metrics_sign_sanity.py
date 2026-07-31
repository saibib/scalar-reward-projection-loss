import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multipref_routing_uplift import prediction_metrics


class PredictionMetricsSignSanityTest(unittest.TestCase):
    def test_pair_direction_matches_row_direction_when_a_is_sorted_i(self) -> None:
        obs = pd.DataFrame(
            [
                {
                    "model_a": "A",
                    "model_b": "B",
                    "i": "A",
                    "j": "B",
                    "sign": 1,
                    "row_sign": 1,
                    "weight": 1.0,
                }
            ]
        )
        metrics = prediction_metrics(obs, np.array([0.9]))
        self.assertEqual(metrics["test_error_rate"], 0.0)
        self.assertEqual(metrics["test_majority_edge_error"], 0.0)
        self.assertEqual(metrics["test_flipped_majority_edge_error"], 1.0)
        self.assertLess(metrics["test_flip_majority_edge_error_gain"], 0.0)
        self.assertEqual(metrics["test_row_edge_target_mismatch_rate"], 0.0)

    def test_pair_direction_matches_row_direction_when_b_is_sorted_i(self) -> None:
        obs = pd.DataFrame(
            [
                {
                    "model_a": "B",
                    "model_b": "A",
                    "i": "A",
                    "j": "B",
                    "sign": 1,
                    "row_sign": -1,
                    "weight": 1.0,
                }
            ]
        )
        metrics = prediction_metrics(obs, np.array([0.1]))
        self.assertEqual(metrics["test_error_rate"], 0.0)
        self.assertEqual(metrics["test_majority_edge_error"], 0.0)
        self.assertEqual(metrics["test_flipped_majority_edge_error"], 1.0)
        self.assertLess(metrics["test_flip_majority_edge_error_gain"], 0.0)
        self.assertEqual(metrics["test_row_edge_target_mismatch_rate"], 0.0)

    def test_tied_predicted_edge_counts_as_half_error(self) -> None:
        obs = pd.DataFrame(
            [
                {
                    "model_a": "A",
                    "model_b": "B",
                    "i": "A",
                    "j": "B",
                    "sign": 1,
                    "row_sign": 1,
                    "weight": 1.0,
                }
            ]
        )
        metrics = prediction_metrics(obs, np.array([0.5]))
        self.assertTrue(math.isclose(metrics["test_majority_edge_error"], 0.5))


if __name__ == "__main__":
    unittest.main()
