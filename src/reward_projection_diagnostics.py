#!/usr/bin/env python3
"""Loss-matched scalar projection diagnostics for pairwise preferences.

The linear Hodge residual asks whether observed margins are additive score
differences. Reward models normally use a nonlinear Bradley--Terry link,

    P(i > j) = sigmoid(r_i - r_j).

This module measures the population proper-loss regret induced by that scalar
restriction. The KL and Brier projection losses are directly comparable to
held-out log-loss and Brier gaps between scalar and saturated pairwise models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy


COMPARISON_METRICS = (
    "test_log_loss",
    "test_brier",
    "test_error_rate",
    "test_calibration_abs_error",
    "test_pair_calibration_mae",
    "test_majority_edge_error",
)


def _edge_arrays(
    edge_df: pd.DataFrame,
    nodes: Sequence[str],
    min_support: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    required = {"i", "j", "support", "weighted_margin"}
    missing = sorted(required.difference(edge_df.columns))
    if missing:
        raise ValueError(f"Missing edge columns: {missing}")

    node_to_idx = {str(node): idx for idx, node in enumerate(nodes)}
    rows = edge_df[edge_df["support"] >= min_support].copy()
    rows = rows[rows["i"].astype(str).isin(node_to_idx) & rows["j"].astype(str).isin(node_to_idx)]
    if rows.empty:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return rows, empty_i, empty_i, empty_f, empty_f, node_to_idx

    left = rows["i"].astype(str).map(node_to_idx).to_numpy(dtype=int)
    right = rows["j"].astype(str).map(node_to_idx).to_numpy(dtype=int)
    support = rows["support"].to_numpy(dtype=float)
    weighted_margin = rows["weighted_margin"].to_numpy(dtype=float)
    if np.any(~np.isfinite(support)) or np.any(support <= 0):
        raise ValueError("Edge support must be finite and positive.")
    if np.any(~np.isfinite(weighted_margin)) or np.any(np.abs(weighted_margin) > support + 1e-8):
        raise ValueError("Weighted margins must be finite and bounded by support.")
    return rows, left, right, support, weighted_margin, node_to_idx


def fit_bradley_terry_edges(
    edge_df: pd.DataFrame,
    nodes: Sequence[str],
    min_support: float = 0.0,
    l2: float = 1e-8,
) -> Dict[str, Any]:
    """Fit a scalar Bradley--Terry projection to aggregate edge counts."""
    rows, left, right, support, weighted_margin, _ = _edge_arrays(edge_df, nodes, min_support)
    n_nodes = len(nodes)
    if rows.empty or n_nodes < 2:
        return {
            "fit_ok": False,
            "reason": "empty_or_single_node",
            "scores": np.zeros(n_nodes, dtype=float),
            "n_iter": 0,
            "objective": np.nan,
        }

    wins = 0.5 * (support + weighted_margin)
    losses = support - wins

    def unpack(x: np.ndarray) -> np.ndarray:
        scores = np.zeros(n_nodes, dtype=float)
        scores[: n_nodes - 1] = x
        return scores

    def objective(x: np.ndarray) -> float:
        scores = unpack(x)
        z = scores[left] - scores[right]
        nll = np.sum(wins * np.logaddexp(0.0, -z) + losses * np.logaddexp(0.0, z))
        return float(nll + 0.5 * l2 * np.sum(x * x))

    def gradient(x: np.ndarray) -> np.ndarray:
        scores = unpack(x)
        z = scores[left] - scores[right]
        dz = support * expit(z) - wins
        grad_scores = np.zeros(n_nodes, dtype=float)
        np.add.at(grad_scores, left, dz)
        np.add.at(grad_scores, right, -dz)
        return grad_scores[: n_nodes - 1] + l2 * x

    result = minimize(
        objective,
        np.zeros(n_nodes - 1, dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 1000},
    )
    scores = unpack(result.x if np.all(np.isfinite(result.x)) else np.zeros(n_nodes - 1))
    scores -= float(np.mean(scores))
    return {
        "fit_ok": bool(result.success and np.all(np.isfinite(scores))),
        "reason": "" if result.success else str(result.message),
        "scores": scores,
        "n_iter": int(result.nit),
        "objective": float(objective(scores[: n_nodes - 1])),
    }


def _bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0 - 1e-12)
    return xlogy(p, p / q) + xlogy(1.0 - p, (1.0 - p) / (1.0 - q))


def _weighted_linear_residual(y: np.ndarray, w: np.ndarray, left: np.ndarray, right: np.ndarray, n: int) -> Dict[str, float]:
    design = np.zeros((len(y), n - 1), dtype=float)
    for row_idx, (ii, jj) in enumerate(zip(left, right)):
        if ii < n - 1:
            design[row_idx, ii] += 1.0
        if jj < n - 1:
            design[row_idx, jj] -= 1.0
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    fitted = design @ coef
    total = float(np.sum(w * y * y))
    residual = float(np.sum(w * (y - fitted) ** 2))
    return {
        "logit_total_energy": total,
        "logit_residual_energy": residual,
        "logit_hodge_fraction": np.nan if total <= 0 else residual / total,
    }


def projection_diagnostics(
    edge_df: pd.DataFrame,
    nodes: Sequence[str],
    min_support: float = 0.0,
    l2: float = 1e-8,
    logit_smoothing: float = 0.5,
) -> Dict[str, float]:
    """Compute loss-matched scalar projection regret on an edge graph.

    ``*_per_weight`` quantities are the theoretically direct population excess
    risks. ``*_fraction`` normalizes them by the risk improvement of the
    saturated edge probabilities over the zero-reward predictor q=0.5.
    """
    rows, left, right, support, weighted_margin, _ = _edge_arrays(edge_df, nodes, min_support)
    if rows.empty or len(nodes) < 2:
        return {
            "projection_fit_ok": 0.0,
            "projection_n_iter": 0.0,
            "projection_total_weight": 0.0,
            "kl_projection_loss": np.nan,
            "kl_projection_per_weight": np.nan,
            "kl_null_loss": np.nan,
            "kl_projection_fraction": np.nan,
            "brier_projection_loss": np.nan,
            "brier_projection_per_weight": np.nan,
            "brier_null_loss": np.nan,
            "brier_projection_fraction": np.nan,
            "logit_total_energy": np.nan,
            "logit_residual_energy": np.nan,
            "logit_hodge_fraction": np.nan,
        }

    fit = fit_bradley_terry_edges(edge_df, nodes, min_support=min_support, l2=l2)
    scores = np.asarray(fit["scores"], dtype=float)
    observed_p = np.clip(0.5 * (1.0 + weighted_margin / support), 0.0, 1.0)
    fitted_p = expit(scores[left] - scores[right])
    total_weight = float(np.sum(support))

    kl_projection = float(np.sum(support * _bernoulli_kl(observed_p, fitted_p)))
    kl_null = float(np.sum(support * _bernoulli_kl(observed_p, np.full_like(observed_p, 0.5))))
    brier_projection = float(np.sum(support * (observed_p - fitted_p) ** 2))
    brier_null = float(np.sum(support * (observed_p - 0.5) ** 2))

    if logit_smoothing < 0:
        raise ValueError("logit_smoothing must be non-negative.")
    wins = 0.5 * (support + weighted_margin)
    smoothed_p = (wins + logit_smoothing) / (support + 2.0 * logit_smoothing)
    smoothed_p = np.clip(smoothed_p, 1e-12, 1.0 - 1e-12)
    logits = np.log(smoothed_p) - np.log1p(-smoothed_p)
    logit_diag = _weighted_linear_residual(logits, support, left, right, len(nodes))

    return {
        "projection_fit_ok": float(bool(fit["fit_ok"])),
        "projection_n_iter": float(fit["n_iter"]),
        "projection_total_weight": total_weight,
        "kl_projection_loss": kl_projection,
        "kl_projection_per_weight": kl_projection / total_weight,
        "kl_null_loss": kl_null,
        "kl_projection_fraction": np.nan if kl_null <= 0 else kl_projection / kl_null,
        "brier_projection_loss": brier_projection,
        "brier_projection_per_weight": brier_projection / total_weight,
        "brier_null_loss": brier_null,
        "brier_projection_fraction": np.nan if brier_null <= 0 else brier_projection / brier_null,
        **logit_diag,
    }


def edge_lookup_predictions(
    train_edges: pd.DataFrame,
    obs: pd.DataFrame,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Predict row-local P(A wins) with saturated train-edge probabilities."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    required = {"model_a", "i", "j"}
    missing = sorted(required.difference(obs.columns))
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")

    edge_prob: Dict[Tuple[str, str], float] = {}
    for i, j, support, weighted_margin in train_edges[
        ["i", "j", "support", "weighted_margin"]
    ].itertuples(index=False, name=None):
        wins_i = 0.5 * (float(support) + float(weighted_margin))
        p_i = (wins_i + smoothing) / (float(support) + 2.0 * smoothing)
        edge_prob[(str(i), str(j))] = float(np.clip(p_i, 1e-12, 1.0 - 1e-12))

    predictions = []
    for model_a, i, j in obs[["model_a", "i", "j"]].itertuples(index=False, name=None):
        key = (str(i), str(j))
        if key not in edge_prob:
            raise KeyError(f"No training edge probability for {key}.")
        p_i = edge_prob[key]
        predictions.append(p_i if str(model_a) == str(i) else 1.0 - p_i)
    return np.asarray(predictions, dtype=float)
