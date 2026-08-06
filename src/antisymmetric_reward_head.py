#!/usr/bin/env python3
"""Low-rank antisymmetric reward head used by matched downstream comparators."""

from __future__ import annotations

import math
from typing import Any


def make_antisymmetric_reward_head(hidden_size: int, rank: int, dropout: float = 0.1) -> Any:
    """Build a nested scalar-plus-skew interaction head.

    The returned module maps two response representations to

        r(a) - r(b) + <U a, V b> - <U b, V a>,

    which is exactly antisymmetric under swapping the responses. Rank zero
    recovers the scalar submodel.
    """
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive.")
    if rank < 0:
        raise ValueError("rank must be non-negative.")

    import torch
    import torch.nn as nn

    class AntisymmetricRewardHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.rank = int(rank)
            self.dropout = nn.Dropout(float(dropout))
            self.scalar = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.scalar.weight)
            nn.init.zeros_(self.scalar.bias)
            if rank:
                self.left = nn.Linear(hidden_size, rank, bias=False)
                self.right = nn.Linear(hidden_size, rank, bias=False)
                nn.init.normal_(self.left.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
                nn.init.zeros_(self.right.weight)
            else:
                self.left = None
                self.right = None

        def forward(self, hidden_a: Any, hidden_b: Any) -> Any:
            hidden_a = self.dropout(hidden_a)
            hidden_b = self.dropout(hidden_b)
            scalar = self.scalar(hidden_a).squeeze(-1) - self.scalar(hidden_b).squeeze(-1)
            if self.rank == 0:
                return scalar
            left_a = self.left(hidden_a)
            left_b = self.left(hidden_b)
            right_a = self.right(hidden_a)
            right_b = self.right(hidden_b)
            interaction = (left_a * right_b).sum(dim=-1) - (left_b * right_a).sum(dim=-1)
            return scalar + interaction / math.sqrt(self.rank)

    return AntisymmetricRewardHead()
