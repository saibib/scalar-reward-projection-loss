#!/usr/bin/env python3
"""Sanity tests for the matched scalar-versus-interaction experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_multipref_nested_reward_results import audit_predictions
from run_multipref_downstream_alignment import stable_prompt_kfold_split


class PromptFoldTests(unittest.TestCase):
    def test_folds_are_disjoint_and_exhaustive(self) -> None:
        prompts = [f"prompt-{index}" for index in range(103)]
        tests = []
        for fold in range(5):
            train, test = stable_prompt_kfold_split(prompts, seed=17, fold_index=fold, n_folds=5)
            self.assertFalse(train.intersection(test))
            self.assertEqual(set(prompts), train.union(test))
            tests.append(test)
        for left in range(5):
            for right in range(left + 1, 5):
                self.assertFalse(tests[left].intersection(tests[right]))
        self.assertEqual(set(prompts), set().union(*tests))


class AntisymmetricHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("torch is not installed") from exc
        cls.torch = torch

    def test_rank_zero_and_interaction_heads_are_antisymmetric(self) -> None:
        from antisymmetric_reward_head import make_antisymmetric_reward_head

        torch = self.torch
        torch.manual_seed(3)
        hidden_a = torch.randn(7, 11)
        hidden_b = torch.randn(7, 11)
        for rank in [0, 4]:
            head = make_antisymmetric_reward_head(11, rank, dropout=0.0)
            head.eval()
            forward = head(hidden_a, hidden_b)
            reverse = head(hidden_b, hidden_a)
            diagonal = head(hidden_a, hidden_a)
            self.assertTrue(torch.allclose(forward, -reverse, atol=1e-7, rtol=1e-7))
            self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-7))

    def test_interaction_branch_receives_gradient_from_nested_initialization(self) -> None:
        from antisymmetric_reward_head import make_antisymmetric_reward_head

        torch = self.torch
        torch.manual_seed(5)
        head = make_antisymmetric_reward_head(9, 3, dropout=0.0)
        hidden_a = torch.randn(8, 9)
        hidden_b = torch.randn(8, 9)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(hidden_a, hidden_b),
            torch.ones(8),
        )
        loss.backward()
        self.assertGreater(float(head.right.weight.grad.abs().sum()), 0.0)


class NestedRunnerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("torch is not installed") from exc
        cls.torch = torch

    def test_last_token_uses_each_sequence_length(self) -> None:
        from run_multipref_lora_nested_reward_downstream import _last_token

        torch = self.torch
        hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        expected = torch.stack([hidden[0, 1], hidden[1, 2]])
        self.assertTrue(torch.equal(_last_token(hidden, attention_mask), expected))

    def test_mocked_nested_model_forward_and_backward(self) -> None:
        import torch.nn as nn
        from run_multipref_lora_nested_reward_downstream import make_nested_model

        torch = self.torch

        class TinyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(hidden_size=5)
                self.embedding = nn.Embedding(16, 5)

            def forward(self, input_ids, attention_mask, **_kwargs):
                del attention_mask
                return SimpleNamespace(hidden_states=[self.embedding(input_ids)])

        fake_stack = (torch, None, None, None, None, None, None, None, None)
        args = SimpleNamespace(interaction_dropout=0.0, precision="fp32")
        with (
            patch(
                "run_multipref_lora_nested_reward_downstream.import_lora_stack",
                return_value=fake_stack,
            ),
            patch(
                "run_multipref_lora_nested_reward_downstream.make_lora_reward_model",
                return_value=TinyBackbone(),
            ),
        ):
            model = make_nested_model(args, torch.device("cpu"), pad_token_id=0, rank=3)

        batch = {
            "input_ids_a": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask_a": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "input_ids_b": torch.tensor([[6, 7, 8], [9, 0, 0]]),
            "attention_mask_b": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        }
        swapped = {
            "input_ids_a": batch["input_ids_b"],
            "attention_mask_a": batch["attention_mask_b"],
            "input_ids_b": batch["input_ids_a"],
            "attention_mask_b": batch["attention_mask_a"],
        }
        logits = model(batch)
        reverse = model(swapped)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.allclose(logits, -reverse, atol=1e-7, rtol=1e-7))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(2)).backward()
        self.assertGreater(float(model.reward_head.right.weight.grad.abs().sum()), 0.0)


class PredictionAuditTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "split_index": [0, 1],
                "group": ["all", "all"],
                "aspect": ["overall", "overall"],
                "prompt_id": ["p0", "p1"],
                "raw_idx": [0, 1],
                "annotation_group": ["normal", "normal"],
                "evaluator": ["e0", "e1"],
                "target": [1.0, 0.0],
                "weight": [1.0, 1.0],
                "neural_p_a": [0.7, 0.3],
                "interaction_p_a": [0.8, 0.2],
            }
        )

    def test_disjoint_predictions_pass(self) -> None:
        audit = audit_predictions(self.frame())
        self.assertEqual(audit["max_folds_per_prompt"], 1)

    def test_prompt_repeated_across_folds_fails(self) -> None:
        frame = self.frame()
        frame.loc[1, "prompt_id"] = "p0"
        with self.assertRaisesRegex(ValueError, "multiple held-out folds"):
            audit_predictions(frame)


if __name__ == "__main__":
    unittest.main()
