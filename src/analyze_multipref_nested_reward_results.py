#!/usr/bin/env python3
"""Inference and audits for the matched nested LoRA reward experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analyze_multipref_neural_paper_results import bh_fdr, simple_markdown_table
from analyze_multipref_reward_projection_v2 import (
    CONTROLS,
    aggregate_regions,
    leave_out_ranges,
    model_nodes,
    network_permutation_pvalues,
    regression_diagnostics,
)


PRIMARY_SPECS = [
    (
        "train_kl_projection_per_weight",
        "scalar_minus_interaction_test_log_loss",
    ),
    (
        "train_brier_projection_per_weight",
        "scalar_minus_interaction_test_brier",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-glob", required=True)
    parser.add_argument("--prediction-glob", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--max-network-permutations", type=int, default=0)
    return parser.parse_args()


def read_nonempty_csvs(pattern: str, suffix: str) -> Tuple[pd.DataFrame, List[str]]:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith(suffix))
    if not matches:
        raise SystemExit(f"No files matched {pattern!r} with suffix {suffix!r}.")
    frames = []
    kept = []
    for path in matches:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frames.append(frame)
        kept.append(path)
    if not frames:
        raise SystemExit(f"All matched {suffix} files were empty.")
    return pd.concat(frames, ignore_index=True), kept


def audit_regions(rows: pd.DataFrame) -> Dict[str, Any]:
    key = ["split_index", "group", "aspect", "triplet"]
    required = {
        *key,
        "train_kl_projection_per_weight",
        "train_brier_projection_per_weight",
        "scalar_minus_interaction_test_log_loss",
        "scalar_minus_interaction_test_brier",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing nested-experiment region columns: {missing}")
    duplicates = int(rows.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate split-region keys.")
    numeric = rows[list(required.difference(key))].replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Non-finite primary region values in: {bad}")
    return {
        "n_split_region_rows": int(len(rows)),
        "n_split_region_keys": int(rows[key].drop_duplicates().shape[0]),
        "n_splits": int(rows["split_index"].nunique()),
    }


def audit_predictions(predictions: pd.DataFrame) -> Dict[str, Any]:
    required = {
        "split_index",
        "group",
        "aspect",
        "prompt_id",
        "raw_idx",
        "target",
        "weight",
        "neural_p_a",
        "interaction_p_a",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing nested-experiment prediction columns: {missing}")
    finite_columns = ["target", "weight", "neural_p_a", "interaction_p_a"]
    finite = predictions[finite_columns].replace([np.inf, -np.inf], np.nan)
    if finite.isna().any().any():
        raise ValueError("Prediction files contain non-finite targets, weights, or probabilities.")
    if not predictions["neural_p_a"].between(0.0, 1.0).all():
        raise ValueError("Scalar predictions fall outside [0, 1].")
    if not predictions["interaction_p_a"].between(0.0, 1.0).all():
        raise ValueError("Interaction predictions fall outside [0, 1].")

    prompt_fold_counts = (
        predictions.groupby(["group", "aspect", "prompt_id"])["split_index"].nunique()
    )
    if int(prompt_fold_counts.max()) != 1:
        raise ValueError("At least one prompt appears in multiple held-out folds.")

    observation_key = ["group", "aspect", "raw_idx"]
    for candidate in ["annotation_group", "evaluator", "comparison_id"]:
        if candidate in predictions.columns:
            observation_key.append(candidate)
    cross_fold_duplicates = (
        predictions.groupby(observation_key)["split_index"].nunique().gt(1).sum()
    )
    if int(cross_fold_duplicates):
        raise ValueError(f"Found {int(cross_fold_duplicates)} observations in multiple held-out folds.")
    return {
        "n_prediction_rows": int(len(predictions)),
        "n_prediction_prompts": int(predictions["prompt_id"].nunique()),
        "n_prediction_splits": int(predictions["split_index"].nunique()),
        "max_folds_per_prompt": int(prompt_fold_counts.max()),
    }


def inference_table(regions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    secondary = [
        ("train_logit_hodge_fraction", "scalar_minus_interaction_test_log_loss"),
        ("train_logit_hodge_fraction", "scalar_minus_interaction_test_brier"),
    ]
    negative_controls = [
        ("train_kl_projection_per_weight", "test_log_loss"),
        ("train_kl_projection_per_weight", "interaction_test_log_loss"),
        ("train_rho_cyc", "test_majority_edge_error"),
    ]
    specs = (
        [("primary", predictor, outcome) for predictor, outcome in PRIMARY_SPECS]
        + [("secondary", predictor, outcome) for predictor, outcome in secondary]
        + [("negative_control", predictor, outcome) for predictor, outcome in negative_controls]
    )
    output = []
    for index, (family, predictor, outcome) in enumerate(specs):
        needed = [predictor, outcome, *CONTROLS, "group", "aspect", "triplet"]
        data = regions.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
        diagnostics = regression_diagnostics(data, predictor, outcome)
        p_two, p_positive, n_permutations = network_permutation_pvalues(
            data,
            predictor,
            outcome,
            args.max_network_permutations,
            args.seed + 7919 * index,
        )
        output.append(
            {
                "family": family,
                "predictor": predictor,
                "outcome": outcome,
                "n_regions": int(len(data)),
                "mean_predictor": float(data[predictor].mean()),
                "mean_outcome": float(data[outcome].mean()),
                "network_permutation_p_two_sided": p_two,
                "network_permutation_p_one_sided_positive": p_positive,
                "n_network_permutations": n_permutations,
                **diagnostics,
                **leave_out_ranges(data, predictor, outcome),
            }
        )
    table = pd.DataFrame(output)
    table["network_permutation_q_bh_within_family"] = np.nan
    for _family, indexes in table.groupby("family").groups.items():
        table.loc[indexes, "network_permutation_q_bh_within_family"] = bh_fdr(
            table.loc[indexes, "network_permutation_p_two_sided"].tolist()
        )
    return table


def _losses(frame: pd.DataFrame, probability_column: str) -> Tuple[np.ndarray, np.ndarray]:
    target = frame["target"].to_numpy(dtype=float)
    probability = np.clip(frame[probability_column].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7)
    log_loss = -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    brier = np.square(probability - target)
    return log_loss, brier


def _cluster_bootstrap_gap(
    frame: pd.DataFrame,
    loss_name: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    work = frame.copy()
    scalar_log, scalar_brier = _losses(work, "neural_p_a")
    interaction_log, interaction_brier = _losses(work, "interaction_p_a")
    if loss_name == "log_loss":
        scalar_loss, interaction_loss = scalar_log, interaction_log
    else:
        scalar_loss, interaction_loss = scalar_brier, interaction_brier
    weight = work["weight"].to_numpy(dtype=float)
    work["weighted_scalar"] = weight * scalar_loss
    work["weighted_interaction"] = weight * interaction_loss
    grouped = work.groupby("prompt_id", as_index=False).agg(
        scalar_sum=("weighted_scalar", "sum"),
        interaction_sum=("weighted_interaction", "sum"),
        weight_sum=("weight", "sum"),
    )
    scalar_mean = float(grouped["scalar_sum"].sum() / grouped["weight_sum"].sum())
    interaction_mean = float(grouped["interaction_sum"].sum() / grouped["weight_sum"].sum())
    observed = scalar_mean - interaction_mean
    rng = np.random.default_rng(seed)
    values = grouped[["scalar_sum", "interaction_sum", "weight_sum"]].to_numpy(dtype=float)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sample = values[rng.integers(0, len(values), size=len(values))]
        bootstrap[draw] = (sample[:, 0].sum() - sample[:, 1].sum()) / sample[:, 2].sum()
    return {
        "scalar_mean": scalar_mean,
        "interaction_mean": interaction_mean,
        "scalar_minus_interaction": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_prompts": int(len(grouped)),
    }


def paired_loss_table(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output = []
    for group_index, ((group, aspect), frame) in enumerate(
        predictions.groupby(["group", "aspect"], sort=True)
    ):
        for loss_index, loss_name in enumerate(["log_loss", "brier"]):
            output.append(
                {
                    "group": group,
                    "aspect": aspect,
                    "loss": loss_name,
                    **_cluster_bootstrap_gap(
                        frame,
                        loss_name,
                        args.n_bootstrap,
                        args.seed + 1009 * group_index + loss_index,
                    ),
                }
            )
    return pd.DataFrame(output)


def report_markdown(
    inference: pd.DataFrame,
    paired: pd.DataFrame,
    metadata: Dict[str, Any],
) -> str:
    primary = inference[inference["family"] == "primary"].copy()
    supported = primary[
        (primary["partial_beta"] > 0)
        & (primary["network_permutation_q_bh_within_family"] < 0.05)
    ]
    conclusion = (
        "At least one loss-matched scalar-compression hypothesis is supported."
        if not supported.empty
        else "Neither loss-matched scalar-compression hypothesis is supported at network-aware FDR 0.05."
    )
    primary_display = primary[
        [
            "predictor",
            "outcome",
            "n_regions",
            "mean_outcome",
            "partial_beta",
            "network_permutation_p_two_sided",
            "network_permutation_q_bh_within_family",
            "hc3_p",
            "model_membership_hc3_p",
        ]
    ].copy()
    paired_display = paired.copy()
    for frame in [primary_display, paired_display]:
        for column in frame.select_dtypes(include=[np.number]).columns:
            frame[column] = frame[column].map(lambda value: f"{value:.4g}")
    return "\n".join(
        [
            "# Matched Nested LoRA Reward Results",
            "",
            f"**Primary conclusion:** {conclusion}",
            "",
            "Positive scalar-minus-interaction loss means the antisymmetric interaction model predicts held-out judgments better than its nested scalar submodel.",
            "",
            "## Projection-Loss Tests",
            "",
            simple_markdown_table(primary_display),
            "",
            "## Paired Held-Out Loss",
            "",
            simple_markdown_table(paired_display),
            "",
            "## Audit",
            "",
            f"- Split-region rows: {metadata['n_split_region_rows']}",
            f"- Aggregated regions: {metadata['n_regions']}",
            f"- Prediction rows: {metadata['n_prediction_rows']}",
            f"- Maximum held-out folds per prompt: {metadata['max_folds_per_prompt']}",
            f"- Network permutations per primary test: {metadata['n_network_permutations']}",
            "- Prompt-cluster bootstrap intervals preserve within-prompt dependence.",
            "- Node-label permutations preserve overlapping model triplets.",
            "- This analysis is confirmatory only if its specification was frozen before inspecting these results.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    split_rows, region_files = read_nonempty_csvs(args.region_glob, "_region_results.csv")
    predictions, prediction_files = read_nonempty_csvs(args.prediction_glob, "_predictions.csv")
    region_audit = audit_regions(split_rows)
    prediction_audit = audit_predictions(predictions)
    regions = aggregate_regions(split_rows)
    inference = inference_table(regions, args)
    paired = paired_loss_table(predictions, args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    region_path = args.outdir / "nested_reward_aggregated_regions.csv"
    inference_path = args.outdir / "nested_reward_primary_effects.csv"
    paired_path = args.outdir / "nested_reward_paired_losses.csv"
    metadata_path = args.outdir / "nested_reward_metadata.json"
    report_path = args.outdir / "nested_reward_report.md"
    regions.to_csv(region_path, index=False)
    inference.to_csv(inference_path, index=False)
    paired.to_csv(paired_path, index=False)
    metadata = {
        **region_audit,
        **prediction_audit,
        "n_regions": int(len(regions)),
        "n_nodes": int(len(model_nodes(regions))),
        "n_network_permutations": int(inference["n_network_permutations"].min()),
        "region_files": region_files,
        "prediction_files": prediction_files,
        "n_bootstrap": args.n_bootstrap,
        "primary_hypotheses": [f"{predictor} -> {outcome}" for predictor, outcome in PRIMARY_SPECS],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown(inference, paired, metadata), encoding="utf-8")
    for path in [region_path, inference_path, paired_path, metadata_path, report_path]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
