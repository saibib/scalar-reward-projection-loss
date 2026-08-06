import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_multipref_reward_projection_v2 import network_permutation_pvalues


class NetworkPermutationInferenceTest(unittest.TestCase):
    def test_exact_six_node_permutation_detects_injected_network_effect(self) -> None:
        nodes = list("ABCDEF")
        aspects = ["overall", "helpful", "truthful", "harmless"]
        rng = np.random.default_rng(17)
        triplet_values = {
            (aspect, triplet): float(rng.normal())
            for aspect in aspects
            for triplet in itertools.combinations(nodes, 3)
        }
        rows = []
        for aspect_idx, aspect in enumerate(aspects):
            for triplet in itertools.combinations(nodes, 3):
                predictor = triplet_values[(aspect, triplet)]
                rows.append(
                    {
                        "group": "all",
                        "aspect": aspect,
                        "triplet": " | ".join(triplet),
                        "predictor": predictor,
                        "outcome": predictor + 0.01 * float(rng.normal()),
                        "train_mean_abs_margin": 0.2 + 0.01 * aspect_idx + 0.01 * float(rng.random()),
                        "train_min_edge_support": 50.0 + float(rng.integers(0, 20)),
                        "train_total_weight": 200.0 + float(rng.integers(0, 50)),
                    }
                )
        data = pd.DataFrame(rows)

        p_two, p_one, n_permutations = network_permutation_pvalues(
            data,
            predictor="predictor",
            outcome="outcome",
            max_permutations=0,
            seed=123,
        )

        self.assertEqual(n_permutations, 720)
        self.assertLess(p_two, 0.01)
        self.assertLess(p_one, 0.01)


if __name__ == "__main__":
    unittest.main()
