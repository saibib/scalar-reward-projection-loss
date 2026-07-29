#!/usr/bin/env python3
"""Group-conditioned reward-model routing uplift for MultiPref.

This script tests a pluralistic routing mechanism that is closer to alignment
practice than an empirical edge table:

Baseline:
    one pooled scalar text reward model per aspect, trained on normal+expert
    annotations together.

Pluralistic alternative:
    separate normal-worker and expert-worker text reward models for the same
    aspect. Evaluation routes each held-out group to its matching group head.

The key question is whether high-rho_cyc regions get larger held-out uplift
from the group-conditioned reward model.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from run_multipref_routing_uplift import LOSS_METRICS, prediction_metrics, scalar_predictions, weighted_metric
from run_multipref_text_reward_downstream import (
    DEFAULT_ARROW,
    build_pair_features,
    flatten_annotations,
    load_raw,
    sorted_edge_observations,
    train_reward_model,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)

DEFAULT_OUTDIR = Path("src/results/multipref_group_routing_uplift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groups", nargs="+", default=["normal", "expert"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-max-iter", type=int, default=20)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None)
    parser.add_argument("--max-raw-rows", type=int, default=None)
    return parser.parse_args()


def analyze_region(
    split_index: int,
    group: str,
    aspect: str,
    triplet: Tuple[str, str, str],
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    pooled_clf: SGDClassifier,
    group_clf: SGDClassifier,
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

    pooled_metrics = prediction_metrics(test_region, scalar_predictions(pooled_clf, xdiff_raw, test_region))
    group_metrics = prediction_metrics(test_region, scalar_predictions(group_clf, xdiff_raw, test_region))

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
    row.update({f"pooled_{k}": v for k, v in pooled_metrics.items()})
    row.update({f"group_head_{k}": v for k, v in group_metrics.items()})
    for metric in LOSS_METRICS:
        row[f"uplift_{metric}"] = row[f"pooled_test_{metric}"] - row[f"group_head_test_{metric}"]
    row["uplift_accuracy"] = row["group_head_test_accuracy"] - row["pooled_test_accuracy"]
    return row


def finite_corr(x: pd.Series, y: pd.Series, kind: str) -> Tuple[float, float]:
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
    scopes = [
        ("split_region_rows", results),
        ("aggregate_region_rows", results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)),
    ]
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
                    "mean_pooled": float(d[f"pooled_test_{metric}"].mean()),
                    "mean_group_head": float(d[f"group_head_test_{metric}"].mean()),
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


def routing_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    scopes = [
        ("split_region_rows", results),
        ("aggregate_region_rows", results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)),
    ]
    for scope, d0 in scopes:
        for metric in LOSS_METRICS:
            d = d0.copy()
            pooled_col = f"pooled_test_{metric}"
            group_col = f"group_head_test_{metric}"
            pooled_value = weighted_metric(d.rename(columns={"pooled_test_weight": "scalar_test_weight"}), pooled_col)
            group_value = weighted_metric(d.rename(columns={"pooled_test_weight": "scalar_test_weight"}), group_col)
            for frac in fractions:
                if frac <= 0.0:
                    route = np.zeros(len(d), dtype=bool)
                    threshold = np.inf
                elif frac >= 1.0:
                    route = np.ones(len(d), dtype=bool)
                    threshold = -np.inf
                else:
                    threshold = float(d["train_rho_cyc"].quantile(1.0 - frac))
                    route = d["train_rho_cyc"].to_numpy(dtype=float) >= threshold
                d["routed"] = np.where(route, d[group_col], d[pooled_col])
                routed_value = weighted_metric(
                    d.rename(columns={"pooled_test_weight": "scalar_test_weight"}),
                    "routed",
                )
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "route_top_rho_fraction": frac,
                        "rho_threshold": threshold,
                        "pooled_value": pooled_value,
                        "all_group_head_value": group_value,
                        "routed_value": routed_value,
                        "improvement_vs_pooled": pooled_value - routed_value,
                        "all_group_head_improvement_vs_pooled": pooled_value - group_value,
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args.arrow, args.max_raw_rows)
    flat = flatten_annotations(raw)
    _vectorizer, xdiff_raw = build_pair_features(raw, args.max_features, args.min_df, args.ngram_max)
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()

    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    skipped = {"empty_obs": 0, "fit_failed": 0, "ineligible_region": 0}

    for split_index in range(args.n_splits):
        train_prompts, test_prompts = stable_prompt_split(prompt_ids, args.seed, split_index, args.test_frac)
        for aspect in args.aspects:
            pooled_obs = sorted_edge_observations(flat, aspect=aspect, group="all")
            pooled_train = pooled_obs[pooled_obs["prompt_id"].isin(train_prompts)].copy()
            pooled_clf, pooled_meta = train_reward_model(
                xdiff_raw, pooled_train, args.sgd_alpha, args.sgd_max_iter, args.seed + split_index
            )
            fit_rows.append({"split_index": split_index, "group": "all", "aspect": aspect, "model": "pooled", **pooled_meta})
            if pooled_clf is None:
                skipped["fit_failed"] += len(args.groups)
                continue

            group_clfs: Dict[str, SGDClassifier] = {}
            for group in args.groups:
                obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                train_obs = obs[obs["prompt_id"].isin(train_prompts)].copy()
                clf, meta = train_reward_model(
                    xdiff_raw, train_obs, args.sgd_alpha, args.sgd_max_iter, args.seed + split_index
                )
                fit_rows.append({"split_index": split_index, "group": group, "aspect": aspect, "model": "group_head", **meta})
                if clf is None:
                    skipped["fit_failed"] += 1
                else:
                    group_clfs[group] = clf

            for group, group_clf in group_clfs.items():
                obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                if obs.empty:
                    skipped["empty_obs"] += 1
                    continue
                train_obs = obs[obs["prompt_id"].isin(train_prompts)].copy()
                test_obs = obs[obs["prompt_id"].isin(test_prompts)].copy()
                nodes = sorted(set(train_obs["i"]).union(set(train_obs["j"])))
                triplets = list(__import__("itertools").combinations(nodes, 3))
                if args.max_triplets is not None:
                    triplets = triplets[: args.max_triplets]
                for triplet in triplets:
                    row = analyze_region(
                        split_index,
                        group,
                        aspect,
                        tuple(triplet),
                        train_obs,
                        test_obs,
                        pooled_clf,
                        group_clf,
                        xdiff_raw,
                        args,
                    )
                    if row is None:
                        skipped["ineligible_region"] += 1
                    else:
                        all_rows.append(row)

    results = pd.DataFrame(all_rows)
    agg = (
        results.groupby(["group", "aspect", "triplet"], as_index=False).mean(numeric_only=True)
        if not results.empty
        else pd.DataFrame()
    )
    fits = pd.DataFrame(fit_rows)
    uplift_summary = summarize_uplift(results) if not results.empty else pd.DataFrame()
    route_summary = routing_table(results) if not results.empty else pd.DataFrame()

    result_path = args.outdir / "multipref_group_routing_region_results.csv"
    aggregate_path = args.outdir / "multipref_group_routing_region_aggregated.csv"
    fit_path = args.outdir / "multipref_group_routing_fit_summary.csv"
    uplift_path = args.outdir / "multipref_group_routing_uplift_summary.csv"
    route_path = args.outdir / "multipref_group_routing_policy_summary.csv"
    metadata_path = args.outdir / "multipref_group_routing_metadata.json"

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
