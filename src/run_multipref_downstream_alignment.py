#!/usr/bin/env python3
"""CPU-only downstream diagnostic experiment for MultiPref.

This script tests whether a cyclic-residual diagnostic measured on training
preferences predicts downstream scalar-model failure on held-out preferences.
It is intentionally cluster-friendly: each SLURM array task can run one random
split, then a final combine pass summarizes all split files.

Default experiment
------------------
For each random prompt-level split, annotation group, preference aspect, and
3-model region:

1. compute the training Hodge/scalar residual rho_cyc for that triplet;
2. fit a tiny regularized Bradley-Terry scalar reward model on training rows;
3. evaluate held-out log loss, Brier score, accuracy, and calibration errors.

The result is not full RLHF/DPO. It is the CPU-only predictive-validity test
needed before making the stronger claim that rho_cyc identifies regions where
single-scalar alignment models should fail.

Examples
--------
Run a local smoke test:

    python src/run_multipref_downstream_alignment.py --n-splits 1 --max-triplets 3

Run 100 splits as a SLURM array, using array ids 0..99:

    python src/run_multipref_downstream_alignment.py \
      --n-splits 100 \
      --split-index "$SLURM_ARRAY_TASK_ID" \
      --outdir src/results/multipref_downstream

Combine split outputs after the array finishes:

    python src/run_multipref_downstream_alignment.py \
      --combine-glob 'src/results/multipref_downstream/*_split_*_region_results.csv' \
      --outdir src/results/multipref_downstream
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr


ASPECTS = ["overall", "helpful", "truthful", "harmless"]
DEFAULT_FLAT = Path("src/results/multipref_v4/multipref_flat_annotations.csv")
DEFAULT_OUTDIR = Path("src/results/multipref_downstream")


@dataclass(frozen=True)
class RegionConfig:
    split_index: int
    group: str
    aspect: str
    nodes: Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flat",
        type=Path,
        default=DEFAULT_FLAT,
        help="Flattened MultiPref annotations produced by run_multipref_cycles_v4.py.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for downstream experiment outputs.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all", "normal", "expert"],
        help="Annotation groups. Use 'all' for pooled normal+expert rows.",
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=ASPECTS,
        choices=ASPECTS,
        help="Preference aspects to evaluate.",
    )
    parser.add_argument("--n-splits", type=int, default=20, help="Number of random prompt-level splits.")
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Run only one split. Useful for SLURM array jobs.",
    )
    parser.add_argument(
        "--split-index-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 1 if your cluster array ids are 1-based.",
    )
    parser.add_argument("--seed", type=int, default=20260622, help="Base random seed.")
    parser.add_argument("--test-frac", type=float, default=0.25, help="Fraction of prompts held out per split.")
    parser.add_argument(
        "--min-train-edge-weight",
        type=float,
        default=5.0,
        help="Minimum weighted training support required on each triplet edge.",
    )
    parser.add_argument(
        "--min-test-weight",
        type=float,
        default=5.0,
        help="Minimum total held-out support required in a region.",
    )
    parser.add_argument(
        "--min-test-edges",
        type=int,
        default=2,
        help="Minimum number of held-out model-pair edges required in a region.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Orientation threshold for triplet cycle counts.",
    )
    parser.add_argument(
        "--bt-l2",
        type=float,
        default=1e-2,
        help="L2 regularization for Bradley-Terry scalar scores.",
    )
    parser.add_argument(
        "--max-triplets",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for debugging only.",
    )
    parser.add_argument(
        "--combine-glob",
        type=str,
        default=None,
        help="Combine existing split CSVs matching this glob instead of rerunning the experiment.",
    )
    return parser.parse_args()


def stable_prompt_split(
    prompt_ids: Sequence[str],
    seed: int,
    split_index: int,
    test_frac: float,
) -> Tuple[set[str], set[str]]:
    if not 0.0 < test_frac < 1.0:
        raise ValueError("--test-frac must be between 0 and 1.")
    prompts = np.array(sorted({str(x) for x in prompt_ids}), dtype=object)
    rng = np.random.default_rng(seed + split_index * 1009)
    perm = rng.permutation(len(prompts))
    n_test = max(1, int(round(test_frac * len(prompts))))
    test = set(prompts[perm[:n_test]].tolist())
    train = set(prompts[perm[n_test:]].tolist())
    return train, test


def sorted_edge_observations(
    flat: pd.DataFrame,
    aspect: str,
    group: str,
) -> pd.DataFrame:
    sign_col = f"{aspect}_sign"
    weight_col = f"{aspect}_weight"
    required = ["prompt_id", "annotation_group", "alt_a", "alt_b", sign_col, weight_col]
    missing = [c for c in required if c not in flat.columns]
    if missing:
        raise ValueError(f"Missing columns for {aspect}: {missing}")

    if group == "all":
        use = flat
    else:
        use = flat[flat["annotation_group"].astype(str) == group]

    use = use[(use[sign_col] != 0) & (use[weight_col] > 0)].copy()
    if use.empty:
        return pd.DataFrame(columns=["prompt_id", "annotation_group", "i", "j", "sign", "weight"])

    rows = []
    for prompt_id, ann_group, alt_a, alt_b, sign, weight in use[
        ["prompt_id", "annotation_group", "alt_a", "alt_b", sign_col, weight_col]
    ].itertuples(index=False, name=None):
        a = str(alt_a)
        b = str(alt_b)
        if not a or not b or a == b:
            continue
        i, j = sorted((a, b))
        winner = a if int(sign) > 0 else b
        sorted_sign = 1 if winner == i else -1
        rows.append((str(prompt_id), str(ann_group), i, j, sorted_sign, float(weight)))

    return pd.DataFrame(rows, columns=["prompt_id", "annotation_group", "i", "j", "sign", "weight"])


def region_mask(obs: pd.DataFrame, nodes: Sequence[str]) -> pd.Series:
    node_set = set(nodes)
    return obs["i"].isin(node_set) & obs["j"].isin(node_set)


def aggregate_edges(obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame(columns=["i", "j", "support", "weighted_margin", "n_obs", "margin"])

    grouped = (
        obs.assign(weighted_signed=obs["sign"] * obs["weight"])
        .groupby(["i", "j"], sort=True)
        .agg(
            support=("weight", "sum"),
            weighted_margin=("weighted_signed", "sum"),
            n_obs=("sign", "size"),
        )
        .reset_index()
    )
    grouped["margin"] = grouped["weighted_margin"] / grouped["support"]
    return grouped


def edge_margin(edge_df: pd.DataFrame, u: str, v: str, min_support: float, tau: float = 0.0) -> Optional[float]:
    if u == v:
        return None
    i, j = sorted((u, v))
    hit = edge_df[(edge_df["i"] == i) & (edge_df["j"] == j)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    if float(row["support"]) < min_support:
        return None
    val = float(row["margin"])
    if abs(val) <= tau:
        return None
    return val if i == u else -val


def hodge_residual(edge_df: pd.DataFrame, nodes: Sequence[str], min_support: float) -> Dict[str, float]:
    rows = edge_df[edge_df["support"] >= min_support].copy()
    if rows.empty or len(nodes) < 2:
        return {"gradient_energy": np.nan, "residual_energy": np.nan, "rho_cyc": np.nan}

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    design = np.zeros((len(rows), max(n - 1, 1)))
    y = rows["margin"].to_numpy(dtype=float)
    w = rows["support"].to_numpy(dtype=float)

    for r, (i, j) in enumerate(rows[["i", "j"]].itertuples(index=False, name=None)):
        ii = node_to_idx[i]
        jj = node_to_idx[j]
        if ii < n - 1:
            design[r, ii] += 1.0
        if jj < n - 1:
            design[r, jj] -= 1.0

    sw = np.sqrt(w)
    try:
        scores, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        fitted = design @ scores
    except np.linalg.LinAlgError:
        return {"gradient_energy": np.nan, "residual_energy": np.nan, "rho_cyc": np.nan}

    total = float(np.sum(w * y * y))
    grad = float(np.sum(w * fitted * fitted))
    resid = float(np.sum(w * (y - fitted) ** 2))
    return {
        "gradient_energy": grad,
        "residual_energy": resid,
        "rho_cyc": np.nan if total <= 0 else resid / total,
    }


def triplet_cycle_stats(
    edge_df: pd.DataFrame,
    nodes: Sequence[str],
    min_support: float,
    tau: float,
) -> Dict[str, float]:
    i, j, k = nodes
    mij = edge_margin(edge_df, i, j, min_support=min_support, tau=tau)
    mjk = edge_margin(edge_df, j, k, min_support=min_support, tau=tau)
    mki = edge_margin(edge_df, k, i, min_support=min_support, tau=tau)
    eligible = int(mij is not None and mjk is not None and mki is not None)
    if not eligible:
        return {
            "eligible_triples": 0,
            "cyclic_triples": 0,
            "cycle_rate": np.nan,
            "weighted_cycle_intensity": np.nan,
        }

    forward = bool(mij > 0 and mjk > 0 and mki > 0)
    reverse = bool(mij < 0 and mjk < 0 and mki < 0)
    cyclic = int(forward or reverse)
    intensity = abs(float(mij * mjk * mki)) if cyclic else 0.0
    return {
        "eligible_triples": 1,
        "cyclic_triples": cyclic,
        "cycle_rate": float(cyclic),
        "weighted_cycle_intensity": intensity,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_bradley_terry(obs: pd.DataFrame, nodes: Sequence[str], l2: float) -> Tuple[np.ndarray, bool, int, float]:
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    if obs.empty or n < 2:
        return np.zeros(n), False, 0, np.nan

    left = obs["i"].map(node_to_idx).to_numpy(dtype=int)
    right = obs["j"].map(node_to_idx).to_numpy(dtype=int)
    y = obs["sign"].to_numpy(dtype=float)
    w = obs["weight"].to_numpy(dtype=float)

    def unpack(x: np.ndarray) -> np.ndarray:
        scores = np.zeros(n, dtype=float)
        scores[: n - 1] = x
        return scores

    def objective(x: np.ndarray) -> float:
        scores = unpack(x)
        z = scores[left] - scores[right]
        loss = np.sum(w * np.logaddexp(0.0, -y * z))
        penalty = 0.5 * l2 * float(np.sum(x * x))
        return float(loss + penalty)

    def gradient(x: np.ndarray) -> np.ndarray:
        scores = unpack(x)
        z = scores[left] - scores[right]
        dz = w * (-y) * sigmoid(-y * z)
        grad_scores = np.zeros(n, dtype=float)
        np.add.at(grad_scores, left, dz)
        np.add.at(grad_scores, right, -dz)
        return grad_scores[: n - 1] + l2 * x

    x0 = np.zeros(n - 1, dtype=float)
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-7, "ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    x = result.x if np.all(np.isfinite(result.x)) else x0
    scores = unpack(x)
    scores -= scores.mean()
    return scores, bool(result.success), int(result.nit), float(objective(x))


def evaluate_bradley_terry(obs: pd.DataFrame, nodes: Sequence[str], scores: np.ndarray) -> Dict[str, float]:
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
        }

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    left = obs["i"].map(node_to_idx).to_numpy(dtype=int)
    right = obs["j"].map(node_to_idx).to_numpy(dtype=int)
    y = obs["sign"].to_numpy(dtype=float)
    w = obs["weight"].to_numpy(dtype=float)
    z = scores[left] - scores[right]
    p = sigmoid(z)
    target = (y > 0).astype(float)
    eps = 1e-12
    log_loss = -target * np.log(np.clip(p, eps, 1.0)) - (1.0 - target) * np.log(np.clip(1.0 - p, eps, 1.0))
    brier = (p - target) ** 2
    pred_sign = np.sign(z)
    correct = np.where(np.isclose(z, 0.0), 0.5, (pred_sign == y).astype(float))
    total_weight = float(np.sum(w))

    pair_rows = []
    for (i, j), g in obs.assign(pred=p, target=target).groupby(["i", "j"], sort=True):
        gw = g["weight"].to_numpy(dtype=float)
        pair_weight = float(np.sum(gw))
        obs_rate = float(np.average(g["target"], weights=gw))
        pred_rate = float(np.average(g["pred"], weights=gw))
        score_margin = float(scores[node_to_idx[i]] - scores[node_to_idx[j]])
        heldout_margin = float(np.sum(g["sign"] * g["weight"]) / pair_weight)
        if math.isclose(score_margin, 0.0) or math.isclose(heldout_margin, 0.0):
            majority_error = 0.5
        else:
            majority_error = float(np.sign(score_margin) != np.sign(heldout_margin))
        pair_rows.append((pair_weight, abs(obs_rate - pred_rate), majority_error))

    pair_weights = np.array([r[0] for r in pair_rows], dtype=float)
    pair_cal_errors = np.array([r[1] for r in pair_rows], dtype=float)
    pair_majority_errors = np.array([r[2] for r in pair_rows], dtype=float)

    return {
        "test_weight": total_weight,
        "test_n_obs": int(len(obs)),
        "test_edges": int(len(pair_rows)),
        "test_log_loss": float(np.average(log_loss, weights=w)),
        "test_brier": float(np.average(brier, weights=w)),
        "test_accuracy": float(np.average(correct, weights=w)),
        "test_error_rate": float(1.0 - np.average(correct, weights=w)),
        "test_calibration_abs_error": float(abs(np.average(target, weights=w) - np.average(p, weights=w))),
        "test_pair_calibration_mae": float(np.average(pair_cal_errors, weights=pair_weights)),
        "test_majority_edge_error": float(np.average(pair_majority_errors, weights=pair_weights)),
    }


def analyze_region(
    config: RegionConfig,
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    min_train_edge_weight: float,
    min_test_weight: float,
    min_test_edges: int,
    tau: float,
    bt_l2: float,
) -> Optional[Dict[str, Any]]:
    nodes = config.nodes
    train_region = train_obs[region_mask(train_obs, nodes)].copy()
    test_region = test_obs[region_mask(test_obs, nodes)].copy()
    train_edges = aggregate_edges(train_region)
    test_edges = aggregate_edges(test_region)

    eligible_train_edges = train_edges[train_edges["support"] >= min_train_edge_weight]
    if len(eligible_train_edges) < 3:
        return None
    if float(test_edges["support"].sum()) < min_test_weight or len(test_edges) < min_test_edges:
        return None

    residual = hodge_residual(train_edges, nodes, min_support=min_train_edge_weight)
    if not np.isfinite(residual["rho_cyc"]):
        return None

    cycle = triplet_cycle_stats(train_edges, nodes, min_support=min_train_edge_weight, tau=tau)
    scores, fit_ok, fit_n_iter, fit_loss = fit_bradley_terry(train_region, nodes, l2=bt_l2)
    evaluation = evaluate_bradley_terry(test_region, nodes, scores)

    edge_supports = eligible_train_edges["support"].to_numpy(dtype=float)
    edge_margins = eligible_train_edges["margin"].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "split_index": config.split_index,
        "group": config.group,
        "aspect": config.aspect,
        "region_type": "model_triplet",
        "triplet": " | ".join(nodes),
        "node_1": nodes[0],
        "node_2": nodes[1],
        "node_3": nodes[2],
        "train_n_obs": int(len(train_region)),
        "train_total_weight": float(train_region["weight"].sum()),
        "train_edges": int(len(train_edges)),
        "train_eligible_edges": int(len(eligible_train_edges)),
        "train_min_edge_support": float(np.min(edge_supports)),
        "train_mean_edge_support": float(np.mean(edge_supports)),
        "train_mean_abs_margin": float(np.mean(np.abs(edge_margins))),
        "train_max_abs_margin": float(np.max(np.abs(edge_margins))),
        "bt_fit_ok": fit_ok,
        "bt_fit_n_iter": fit_n_iter,
        "bt_train_objective": fit_loss,
        "score_node_1": float(scores[0]),
        "score_node_2": float(scores[1]),
        "score_node_3": float(scores[2]),
    }
    out.update({f"train_{k}": v for k, v in residual.items()})
    out.update({f"train_{k}": v for k, v in cycle.items()})
    out.update(evaluation)
    return out


def finite_corr_result(x: pd.Series, y: pd.Series, kind: str) -> Tuple[float, float]:
    keep = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(keep.sum()) < 3:
        return np.nan, np.nan
    xx = x.to_numpy(dtype=float)[keep]
    yy = y.to_numpy(dtype=float)[keep]
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return np.nan, np.nan
    if kind == "pearson":
        result = pearsonr(xx, yy)
        return float(result.statistic), float(result.pvalue)
    if kind == "spearman":
        result = spearmanr(xx, yy)
        return float(result.statistic), float(result.pvalue)
    raise ValueError(kind)


def standardized_partial_beta(df: pd.DataFrame, outcome: str) -> float:
    cols = [
        "train_rho_cyc",
        "train_mean_abs_margin",
        "train_min_edge_support",
        "train_total_weight",
    ]
    keep_cols = [outcome] + cols
    d = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < len(cols) + 3:
        return np.nan

    y = d[outcome].to_numpy(dtype=float)
    x_cols = []
    for col in cols:
        x = d[col].to_numpy(dtype=float)
        if col == "train_total_weight":
            x = np.log1p(x)
        sd = float(np.std(x))
        if sd <= 0.0:
            return np.nan
        x_cols.append((x - float(np.mean(x))) / sd)

    y_sd = float(np.std(y))
    if y_sd <= 0.0:
        return np.nan
    y_std = (y - float(np.mean(y))) / y_sd
    xmat = np.column_stack([np.ones(len(d)), *x_cols])
    try:
        beta, *_ = np.linalg.lstsq(xmat, y_std, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])


def summarize_results(rows: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "test_log_loss",
        "test_brier",
        "test_error_rate",
        "test_calibration_abs_error",
        "test_pair_calibration_mae",
        "test_majority_edge_error",
    ]
    scopes: List[Tuple[str, pd.DataFrame]] = [("all", rows)]
    for group, gdf in rows.groupby("group", sort=True):
        scopes.append((f"group={group}", gdf))
    for aspect, adf in rows.groupby("aspect", sort=True):
        scopes.append((f"aspect={aspect}", adf))
    for (group, aspect), gadf in rows.groupby(["group", "aspect"], sort=True):
        scopes.append((f"group={group};aspect={aspect}", gadf))

    summary_rows = []
    for scope, d in scopes:
        for outcome in outcomes:
            if outcome not in d.columns:
                continue
            pearson_stat, pearson_p = finite_corr_result(d["train_rho_cyc"], d[outcome], "pearson")
            spearman_stat, spearman_p = finite_corr_result(d["train_rho_cyc"], d[outcome], "spearman")
            summary_rows.append(
                {
                    "scope": scope,
                    "outcome": outcome,
                    "n_regions": int(len(d)),
                    "n_splits": int(d["split_index"].nunique()) if "split_index" in d else np.nan,
                    "mean_train_rho_cyc": float(d["train_rho_cyc"].mean()),
                    "mean_outcome": float(d[outcome].mean()),
                    "pearson_rho_cyc": pearson_stat,
                    "pearson_p": pearson_p,
                    "spearman_rho_cyc": spearman_stat,
                    "spearman_p": spearman_p,
                    "partial_beta_rho_cyc": standardized_partial_beta(d, outcome),
                }
            )
    return pd.DataFrame(summary_rows)


def output_paths(outdir: Path, split_index: Optional[int]) -> Tuple[Path, Path, Path]:
    if split_index is None:
        stem = "multipref_downstream"
    else:
        stem = f"multipref_downstream_split_{split_index:04d}"
    return (
        outdir / f"{stem}_region_results.csv",
        outdir / f"{stem}_summary.csv",
        outdir / f"{stem}_metadata.json",
    )


def combine_outputs(pattern: str, outdir: Path) -> None:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region-result files matched --combine-glob: {pattern}")
    frames = [pd.read_csv(path) for path in matches]
    rows = pd.concat(frames, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path, summary_path, metadata_path = output_paths(outdir, split_index=None)
    rows.to_csv(result_path, index=False)
    summary = summarize_results(rows)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "combine",
        "combine_glob": pattern,
        "n_input_files": len(matches),
        "n_region_rows": int(len(rows)),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")


def run_experiment(args: argparse.Namespace) -> None:
    if args.combine_glob:
        combine_outputs(args.combine_glob, args.outdir)
        return

    if args.n_splits < 1:
        raise ValueError("--n-splits must be positive.")
    if args.split_index is None:
        split_indices = list(range(args.n_splits))
        output_split_index = None
    else:
        normalized = args.split_index - args.split_index_base
        if normalized < 0 or normalized >= args.n_splits:
            raise ValueError(
                f"Split index {args.split_index} with base {args.split_index_base} "
                f"maps to {normalized}, outside 0..{args.n_splits - 1}."
            )
        split_indices = [normalized]
        output_split_index = normalized

    flat = pd.read_csv(args.flat, nrows=args.max_rows)
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {"no_observations": 0, "ineligible_region": 0}

    for split_index in split_indices:
        train_prompts, test_prompts = stable_prompt_split(
            prompt_ids=prompt_ids,
            seed=args.seed,
            split_index=split_index,
            test_frac=args.test_frac,
        )
        for group in args.groups:
            for aspect in args.aspects:
                obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                if obs.empty:
                    skipped["no_observations"] += 1
                    continue

                nodes = sorted(set(obs["i"]).union(set(obs["j"])))
                triplets = list(itertools.combinations(nodes, 3))
                if args.max_triplets is not None:
                    triplets = triplets[: args.max_triplets]

                train_obs = obs[obs["prompt_id"].isin(train_prompts)].copy()
                test_obs = obs[obs["prompt_id"].isin(test_prompts)].copy()

                for triplet in triplets:
                    config = RegionConfig(
                        split_index=split_index,
                        group=group,
                        aspect=aspect,
                        nodes=tuple(triplet),
                    )
                    row = analyze_region(
                        config=config,
                        train_obs=train_obs,
                        test_obs=test_obs,
                        min_train_edge_weight=args.min_train_edge_weight,
                        min_test_weight=args.min_test_weight,
                        min_test_edges=args.min_test_edges,
                        tau=args.tau,
                        bt_l2=args.bt_l2,
                    )
                    if row is None:
                        skipped["ineligible_region"] += 1
                    else:
                        all_rows.append(row)

    rows = pd.DataFrame(all_rows)
    result_path, summary_path, metadata_path = output_paths(args.outdir, output_split_index)
    rows.to_csv(result_path, index=False)
    summary = summarize_results(rows) if not rows.empty else pd.DataFrame()
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "run",
        "flat": str(args.flat),
        "n_flat_rows": int(len(flat)),
        "n_prompt_ids": int(len(prompt_ids)),
        "n_region_rows": int(len(rows)),
        "n_splits_requested": int(args.n_splits),
        "split_indices_run": split_indices,
        "groups": args.groups,
        "aspects": args.aspects,
        "test_frac": args.test_frac,
        "min_train_edge_weight": args.min_train_edge_weight,
        "min_test_weight": args.min_test_weight,
        "min_test_edges": args.min_test_edges,
        "tau": args.tau,
        "bt_l2": args.bt_l2,
        "skipped": skipped,
        "result_path": str(result_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
