#!/usr/bin/env python3
"""CPU-only routing-uplift experiment for MultiPref.

This script tests the operational recommendation directly:

    Use a scalar reward model in low-cyclic-residual regions, but route
    high-rho_cyc regions to a non-scalar pairwise/pluralistic mechanism.

Baseline
--------
For each split and aspect, train one text-conditioned scalar reward model on
all training annotations:

    r_theta(prompt, completion) = theta^T TFIDF(prompt, completion)
    P(A > B) = sigmoid(r(A) - r(B)).

Routed alternative
------------------
For each group/aspect/model-triplet region, estimate held-in pairwise
probabilities for the three model-pair edges. This mechanism is not constrained
to come from one scalar reward score, so it can represent cyclic edge patterns.

Outputs
-------
The region table contains both scalar and pairwise held-out metrics, plus
uplift = scalar_metric - pairwise_metric for loss/error metrics. Positive uplift
means routing to the pairwise mechanism improved the metric.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier

from run_multipref_cycles_v4 import ASPECTS
from run_multipref_downstream_alignment import (
    aggregate_edges,
    hodge_residual,
    region_mask,
    stable_prompt_split,
    triplet_cycle_stats,
)
from run_multipref_text_reward_downstream import (
    DEFAULT_ARROW,
    build_pair_features,
    flatten_annotations,
    load_raw,
    sigmoid,
    sorted_edge_observations,
    train_reward_model,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)

DEFAULT_OUTDIR = Path("src/results/multipref_routing_uplift")
LOSS_METRICS = [
    "log_loss",
    "brier",
    "error_rate",
    "calibration_abs_error",
    "pair_calibration_mae",
    "majority_edge_error",
]

EMPTY_SIGN_SANITY_METRICS = {
    "test_flipped_log_loss": np.nan,
    "test_flipped_brier": np.nan,
    "test_flipped_error_rate": np.nan,
    "test_flipped_pair_calibration_mae": np.nan,
    "test_flipped_majority_edge_error": np.nan,
    "test_flip_error_rate_gain": np.nan,
    "test_flip_majority_edge_error_gain": np.nan,
    "test_majority_minus_row_error": np.nan,
    "test_mean_target_a": np.nan,
    "test_mean_p_a": np.nan,
    "test_pair_mean_obs_i": np.nan,
    "test_pair_mean_pred_i": np.nan,
    "test_pair_obs_majority_i_rate": np.nan,
    "test_pair_pred_majority_i_rate": np.nan,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groups", nargs="+", default=["all", "normal", "expert"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-max-iter", type=int, default=20)
    parser.add_argument("--edge-smoothing", type=float, default=1.0)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-raw-rows", type=int, default=None, help="Optional raw row cap for smoke tests.")
    return parser.parse_args()


def prediction_metrics(
    obs: pd.DataFrame,
    p_a: np.ndarray,
) -> Dict[str, float]:
    """Evaluate row-local P(A wins) predictions on a region."""
    if obs.empty:
        return {
            "test_weight": 0.0,
            "test_n_obs": 0,
            "test_edges": 0,
            "test_log_loss": np.nan,
            "test_brier": np.nan,
            "test_accuracy": np.nan,
            "test_error_rate": np.nan,
            "test_calibration_abs_error": np.nan,
            "test_pair_calibration_mae": np.nan,
            "test_majority_edge_error": np.nan,
            **EMPTY_SIGN_SANITY_METRICS,
        }

    target_a = (obs["row_sign"].to_numpy(dtype=int) > 0).astype(float)
    w = obs["weight"].to_numpy(dtype=float)
    p_a = np.clip(np.asarray(p_a, dtype=float), 1e-12, 1.0 - 1e-12)
    log_loss = -target_a * np.log(p_a) - (1.0 - target_a) * np.log(1.0 - p_a)
    brier = (p_a - target_a) ** 2
    correct = ((p_a >= 0.5) == (target_a > 0.5)).astype(float)

    def pairwise_summaries(edge_p_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eval_df = obs.copy()
        eval_df["p_i"] = np.where(eval_df["model_a"] == eval_df["i"], edge_p_a, 1.0 - edge_p_a)
        eval_df["target_i"] = (eval_df["sign"] > 0).astype(float)

        pair_rows = []
        for (_i, _j), g in eval_df.groupby(["i", "j"], sort=True):
            gw = g["weight"].to_numpy(dtype=float)
            pair_weight = float(gw.sum())
            obs_rate = float(np.average(g["target_i"], weights=gw))
            pred_rate = float(np.average(g["p_i"], weights=gw))
            heldout_margin = 2.0 * obs_rate - 1.0
            pred_margin = 2.0 * pred_rate - 1.0
            if math.isclose(heldout_margin, 0.0) or math.isclose(pred_margin, 0.0):
                majority_error = 0.5
            else:
                majority_error = float(np.sign(heldout_margin) != np.sign(pred_margin))
            pair_rows.append((pair_weight, obs_rate, pred_rate, abs(obs_rate - pred_rate), majority_error))
        if not pair_rows:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty
        return tuple(np.array([r[idx] for r in pair_rows], dtype=float) for idx in range(5))  # type: ignore[return-value]

    pair_weights, pair_obs_rates, pair_pred_rates, pair_cal_errors, pair_majority_errors = pairwise_summaries(p_a)
    flipped_p_a = 1.0 - p_a
    flipped_log_loss = -target_a * np.log(flipped_p_a) - (1.0 - target_a) * np.log(1.0 - flipped_p_a)
    flipped_brier = (flipped_p_a - target_a) ** 2
    flipped_correct = ((flipped_p_a >= 0.5) == (target_a > 0.5)).astype(float)
    (
        flipped_pair_weights,
        _flipped_pair_obs_rates,
        flipped_pair_pred_rates,
        flipped_pair_cal_errors,
        flipped_pair_majority_errors,
    ) = pairwise_summaries(flipped_p_a)

    error_rate = float(1.0 - np.average(correct, weights=w))
    flipped_error_rate = float(1.0 - np.average(flipped_correct, weights=w))
    majority_edge_error = float(np.average(pair_majority_errors, weights=pair_weights))
    flipped_majority_edge_error = float(np.average(flipped_pair_majority_errors, weights=flipped_pair_weights))

    return {
        "test_weight": float(w.sum()),
        "test_n_obs": int(len(obs)),
        "test_edges": int(len(pair_weights)),
        "test_log_loss": float(np.average(log_loss, weights=w)),
        "test_brier": float(np.average(brier, weights=w)),
        "test_accuracy": float(np.average(correct, weights=w)),
        "test_error_rate": error_rate,
        "test_calibration_abs_error": float(abs(np.average(target_a, weights=w) - np.average(p_a, weights=w))),
        "test_pair_calibration_mae": float(np.average(pair_cal_errors, weights=pair_weights)),
        "test_majority_edge_error": majority_edge_error,
        "test_flipped_log_loss": float(np.average(flipped_log_loss, weights=w)),
        "test_flipped_brier": float(np.average(flipped_brier, weights=w)),
        "test_flipped_error_rate": flipped_error_rate,
        "test_flipped_pair_calibration_mae": float(np.average(flipped_pair_cal_errors, weights=flipped_pair_weights)),
        "test_flipped_majority_edge_error": flipped_majority_edge_error,
        "test_flip_error_rate_gain": error_rate - flipped_error_rate,
        "test_flip_majority_edge_error_gain": majority_edge_error - flipped_majority_edge_error,
        "test_majority_minus_row_error": majority_edge_error - error_rate,
        "test_mean_target_a": float(np.average(target_a, weights=w)),
        "test_mean_p_a": float(np.average(p_a, weights=w)),
        "test_pair_mean_obs_i": float(np.average(pair_obs_rates, weights=pair_weights)),
        "test_pair_mean_pred_i": float(np.average(pair_pred_rates, weights=pair_weights)),
        "test_pair_obs_majority_i_rate": float(np.average(pair_obs_rates > 0.5, weights=pair_weights)),
        "test_pair_pred_majority_i_rate": float(np.average(pair_pred_rates > 0.5, weights=pair_weights)),
    }


def scalar_predictions(
    clf: SGDClassifier,
    xdiff_raw,
    obs: pd.DataFrame,
) -> np.ndarray:
    x = xdiff_raw[obs["raw_idx"].to_numpy(dtype=int)]
    return sigmoid(clf.decision_function(x))


def pairwise_predictions(
    train_edges: pd.DataFrame,
    obs: pd.DataFrame,
    smoothing: float,
) -> np.ndarray:
    edge_prob: Dict[Tuple[str, str], float] = {}
    for i, j, support, weighted_margin in train_edges[
        ["i", "j", "support", "weighted_margin"]
    ].itertuples(index=False, name=None):
        wins_i = 0.5 * (float(support) + float(weighted_margin))
        p_i = (wins_i + smoothing) / (float(support) + 2.0 * smoothing)
        edge_prob[(str(i), str(j))] = float(np.clip(p_i, 1e-12, 1.0 - 1e-12))

    preds: List[float] = []
    for model_a, i, j in obs[["model_a", "i", "j"]].itertuples(index=False, name=None):
        key = (str(i), str(j))
        p_i = edge_prob[key]
        preds.append(p_i if str(model_a) == str(i) else 1.0 - p_i)
    return np.array(preds, dtype=float)


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def analyze_region(
    split_index: int,
    group: str,
    aspect: str,
    triplet: Tuple[str, str, str],
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    scalar_clf: SGDClassifier,
    xdiff_raw,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    train_region = train_obs[region_mask(train_obs, triplet)].copy()
    test_region = test_obs[region_mask(test_obs, triplet)].copy()
    train_edges = aggregate_edges(train_region)
    test_edges = aggregate_edges(test_region)
    eligible_train_edges = train_edges[train_edges["support"] >= args.min_train_edge_weight]

    if len(eligible_train_edges) < 3:
        return None
    if float(test_edges["support"].sum()) < args.min_test_weight or len(test_edges) < args.min_test_edges:
        return None

    residual = hodge_residual(train_edges, triplet, min_support=args.min_train_edge_weight)
    if not np.isfinite(residual["rho_cyc"]):
        return None
    cycle = triplet_cycle_stats(train_edges, triplet, min_support=args.min_train_edge_weight, tau=args.tau)

    scalar_p = scalar_predictions(scalar_clf, xdiff_raw, test_region)
    pairwise_p = pairwise_predictions(train_edges, test_region, smoothing=args.edge_smoothing)
    scalar_metrics = prediction_metrics(test_region, scalar_p)
    pairwise_metrics = prediction_metrics(test_region, pairwise_p)

    edge_supports = eligible_train_edges["support"].to_numpy(dtype=float)
    edge_margins = eligible_train_edges["margin"].to_numpy(dtype=float)
    row: Dict[str, Any] = {
        "split_index": split_index,
        "group": group,
        "aspect": aspect,
        "region_type": "model_triplet",
        "triplet": " | ".join(triplet),
        "node_1": triplet[0],
        "node_2": triplet[1],
        "node_3": triplet[2],
        "train_n_obs": int(len(train_region)),
        "train_total_weight": float(train_region["weight"].sum()),
        "train_edges": int(len(train_edges)),
        "train_eligible_edges": int(len(eligible_train_edges)),
        "train_min_edge_support": float(np.min(edge_supports)),
        "train_mean_edge_support": float(np.mean(edge_supports)),
        "train_mean_abs_margin": float(np.mean(np.abs(edge_margins))),
        "train_max_abs_margin": float(np.max(np.abs(edge_margins))),
    }
    row.update({f"train_{k}": v for k, v in residual.items()})
    row.update({f"train_{k}": v for k, v in cycle.items()})
    row.update(prefix_metrics("scalar", scalar_metrics))
    row.update(prefix_metrics("pairwise", pairwise_metrics))

    for metric in LOSS_METRICS:
        row[f"uplift_{metric}"] = row[f"scalar_test_{metric}"] - row[f"pairwise_test_{metric}"]
    row["uplift_accuracy"] = row["pairwise_test_accuracy"] - row["scalar_test_accuracy"]
    return row


def finite_corr(x: pd.Series, y: pd.Series, kind: str = "pearson") -> Tuple[float, float]:
    xx = x.to_numpy(dtype=float)
    yy = y.to_numpy(dtype=float)
    keep = np.isfinite(xx) & np.isfinite(yy)
    if int(keep.sum()) < 3:
        return np.nan, np.nan
    xx = xx[keep]
    yy = yy[keep]
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return np.nan, np.nan
    result = pearsonr(xx, yy) if kind == "pearson" else spearmanr(xx, yy)
    return float(result.statistic), float(result.pvalue)


def summarize_uplift(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: List[Tuple[str, pd.DataFrame]] = [("split_region_rows", results)]

    agg = results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)
    scopes.append(("aggregate_region_rows", agg))

    for scope, d in scopes:
        for metric in LOSS_METRICS:
            col = f"uplift_{metric}"
            pearson, pearson_p = finite_corr(d["train_rho_cyc"], d[col], "pearson")
            spearman, spearman_p = finite_corr(d["train_rho_cyc"], d[col], "spearman")
            q25 = d["train_rho_cyc"].quantile(0.25)
            q75 = d["train_rho_cyc"].quantile(0.75)
            low = d[d["train_rho_cyc"] <= q25]
            high = d[d["train_rho_cyc"] >= q75]
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n_regions": int(len(d)),
                    "mean_scalar": float(d[f"scalar_test_{metric}"].mean()),
                    "mean_pairwise": float(d[f"pairwise_test_{metric}"].mean()),
                    "mean_uplift": float(d[col].mean()),
                    "pearson_rho_vs_uplift": pearson,
                    "pearson_p": pearson_p,
                    "spearman_rho_vs_uplift": spearman,
                    "spearman_p": spearman_p,
                    "low_rho_mean_uplift": float(low[col].mean()),
                    "high_rho_mean_uplift": float(high[col].mean()),
                    "high_minus_low_uplift": float(high[col].mean() - low[col].mean()),
                }
            )
    return pd.DataFrame(rows)


def weighted_metric(d: pd.DataFrame, metric_col: str) -> float:
    w = d["scalar_test_weight"].to_numpy(dtype=float)
    y = d[metric_col].to_numpy(dtype=float)
    keep = np.isfinite(w) & np.isfinite(y) & (w > 0)
    if int(keep.sum()) == 0:
        return np.nan
    return float(np.average(y[keep], weights=w[keep]))


def routing_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for scope_name, d in [
        ("split_region_rows", results),
        ("aggregate_region_rows", results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)),
    ]:
        d = d.copy()
        for metric in LOSS_METRICS:
            scalar_col = f"scalar_test_{metric}"
            pair_col = f"pairwise_test_{metric}"
            scalar_value = weighted_metric(d, scalar_col)
            pairwise_value = weighted_metric(d, pair_col)
            for frac in fractions:
                routed_col = f"routed_{metric}"
                if frac <= 0.0:
                    route = np.zeros(len(d), dtype=bool)
                    threshold = np.inf
                elif frac >= 1.0:
                    route = np.ones(len(d), dtype=bool)
                    threshold = -np.inf
                else:
                    threshold = float(d["train_rho_cyc"].quantile(1.0 - frac))
                    route = d["train_rho_cyc"].to_numpy(dtype=float) >= threshold
                d[routed_col] = np.where(route, d[pair_col], d[scalar_col])
                routed_value = weighted_metric(d, routed_col)
                rows.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "route_top_rho_fraction": frac,
                        "rho_threshold": threshold,
                        "scalar_value": scalar_value,
                        "all_pairwise_value": pairwise_value,
                        "routed_value": routed_value,
                        "improvement_vs_scalar": scalar_value - routed_value,
                        "all_pairwise_improvement_vs_scalar": scalar_value - pairwise_value,
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args.arrow, args.max_raw_rows)
    flat = flatten_annotations(raw)
    _vectorizer, xdiff_raw = build_pair_features(
        raw,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()

    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    skipped = {"empty_obs": 0, "fit_failed": 0, "ineligible_region": 0}

    for split_index in range(args.n_splits):
        train_prompts, test_prompts = stable_prompt_split(
            prompt_ids=prompt_ids,
            seed=args.seed,
            split_index=split_index,
            test_frac=args.test_frac,
        )
        for aspect in args.aspects:
            pooled_obs = sorted_edge_observations(flat, aspect=aspect, group="all")
            pooled_train = pooled_obs[pooled_obs["prompt_id"].isin(train_prompts)].copy()
            scalar_clf, fit_meta = train_reward_model(
                xdiff_raw=xdiff_raw,
                train_obs=pooled_train,
                alpha=args.sgd_alpha,
                max_iter=args.sgd_max_iter,
                seed=args.seed + split_index,
            )
            fit_rows.append({"split_index": split_index, "aspect": aspect, **fit_meta})
            if scalar_clf is None:
                skipped["fit_failed"] += len(args.groups)
                continue

            for group in args.groups:
                obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                if obs.empty:
                    skipped["empty_obs"] += 1
                    continue
                train_obs = obs[obs["prompt_id"].isin(train_prompts)].copy()
                test_obs = obs[obs["prompt_id"].isin(test_prompts)].copy()
                nodes = sorted(set(train_obs["i"]).union(set(train_obs["j"])))
                triplets = list(itertools.combinations(nodes, 3))
                if args.max_triplets is not None:
                    triplets = triplets[: args.max_triplets]
                for triplet in triplets:
                    row = analyze_region(
                        split_index=split_index,
                        group=group,
                        aspect=aspect,
                        triplet=tuple(triplet),
                        train_obs=train_obs,
                        test_obs=test_obs,
                        scalar_clf=scalar_clf,
                        xdiff_raw=xdiff_raw,
                        args=args,
                    )
                    if row is None:
                        skipped["ineligible_region"] += 1
                    else:
                        all_rows.append(row)

    results = pd.DataFrame(all_rows)
    fits = pd.DataFrame(fit_rows)
    uplift_summary = summarize_uplift(results) if not results.empty else pd.DataFrame()
    route_summary = routing_table(results) if not results.empty else pd.DataFrame()
    agg = (
        results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)
        if not results.empty
        else pd.DataFrame()
    )

    result_path = args.outdir / "multipref_routing_uplift_region_results.csv"
    aggregate_path = args.outdir / "multipref_routing_uplift_region_aggregated.csv"
    fit_path = args.outdir / "multipref_routing_uplift_fit_summary.csv"
    uplift_path = args.outdir / "multipref_routing_uplift_summary.csv"
    route_path = args.outdir / "multipref_routing_policy_summary.csv"
    metadata_path = args.outdir / "multipref_routing_uplift_metadata.json"

    results.to_csv(result_path, index=False)
    agg.to_csv(aggregate_path, index=False)
    fits.to_csv(fit_path, index=False)
    uplift_summary.to_csv(uplift_path, index=False)
    route_summary.to_csv(route_path, index=False)
    metadata = {
        "arrow": str(args.arrow),
        "n_raw_rows": int(len(raw)),
        "n_annotation_rows": int(len(flat)),
        "n_prompt_ids": int(len(prompt_ids)),
        "n_region_rows": int(len(results)),
        "n_splits": args.n_splits,
        "groups": args.groups,
        "aspects": args.aspects,
        "test_frac": args.test_frac,
        "max_features": args.max_features,
        "actual_features": int(xdiff_raw.shape[1]),
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
        "sgd_alpha": args.sgd_alpha,
        "sgd_max_iter": args.sgd_max_iter,
        "edge_smoothing": args.edge_smoothing,
        "min_train_edge_weight": args.min_train_edge_weight,
        "min_test_weight": args.min_test_weight,
        "min_test_edges": args.min_test_edges,
        "tau": args.tau,
        "skipped": skipped,
        "result_path": str(result_path),
        "aggregate_path": str(aggregate_path),
        "fit_path": str(fit_path),
        "uplift_summary_path": str(uplift_path),
        "route_summary_path": str(route_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {result_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {uplift_path}")
    print(f"Wrote {route_path}")
    print(f"Wrote {metadata_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
