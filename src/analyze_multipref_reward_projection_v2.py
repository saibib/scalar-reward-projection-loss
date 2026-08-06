#!/usr/bin/env python3
"""Theory-aligned downstream analysis for MultiPref reward-model outputs.

This analysis replaces absolute reward-model error as the primary outcome with
the held-out loss gap between a scalar text reward model and a saturated
train-edge predictor. It also replaces raw-margin residuals with proper-loss
matched Bradley--Terry projection regret and uses node-label permutations that
preserve overlap among model triplets.

The saturated comparator is deliberately simple and should be treated as an
edge-level representation baseline, not as a matched neural pairwise model.
Its role is to determine whether an absolute-error association is specific to
the scalar restriction or merely reflects difficult/unstable regions.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analyze_multipref_neural_paper_results import bh_fdr, simple_markdown_table
from reward_projection_diagnostics import edge_lookup_predictions, projection_diagnostics
from run_multipref_downstream_alignment import (
    aggregate_edges,
    hodge_residual,
    region_mask,
    stable_prompt_split,
)
from run_multipref_routing_uplift import prediction_metrics
from run_multipref_text_reward_downstream import sorted_edge_observations


DEFAULT_FLAT = Path("src/results/multipref_v4/multipref_flat_annotations.csv")
DEFAULT_OUTDIR = Path("src/results/multipref_reward_projection_v2")
CORE_METRICS = [
    "test_log_loss",
    "test_brier",
    "test_error_rate",
    "test_calibration_abs_error",
    "test_pair_calibration_mae",
    "test_majority_edge_error",
]
CONTROLS = ["train_mean_abs_margin", "train_min_edge_support", "train_total_weight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-glob", required=True, help="Split-level reward-model region CSV glob.")
    parser.add_argument("--flat", type=Path, default=DEFAULT_FLAT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--edge-smoothing", type=float, default=1.0)
    parser.add_argument("--projection-l2", type=float, default=1e-8)
    parser.add_argument("--logit-smoothing", type=float, default=0.5)
    parser.add_argument(
        "--max-network-permutations",
        type=int,
        default=0,
        help="Zero enumerates every node-label permutation when there are at most eight nodes.",
    )
    return parser.parse_args()


def load_split_rows(pattern: str) -> Tuple[pd.DataFrame, List[str]]:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region result files matched: {pattern}")
    frames = []
    for path in matches:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_file"] = path
        frames.append(frame)
    if not frames:
        raise SystemExit("All matched region files were empty.")
    rows = pd.concat(frames, ignore_index=True)
    key = ["split_index", "group", "aspect", "triplet"]
    missing = [col for col in key if col not in rows]
    if missing:
        raise ValueError(f"Missing region key columns: {missing}")
    if rows.duplicated(key).any():
        raise ValueError("Duplicate split/group/aspect/triplet rows found.")
    return rows, matches


def load_flat(path: Path) -> pd.DataFrame:
    flat = pd.read_csv(path)
    if "raw_idx" not in flat and "row_id" in flat:
        flat = flat.rename(columns={"row_id": "raw_idx"})
    required = {"raw_idx", "prompt_id", "annotation_group", "model_a", "model_b"}
    missing = sorted(required.difference(flat.columns))
    if missing:
        raise ValueError(f"Missing flattened annotation columns: {missing}")
    return flat


def enrich_rows(rows: pd.DataFrame, flat: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()
    enriched: List[Dict[str, Any]] = []
    grouped_rows = rows.groupby(["split_index", "group", "aspect"], sort=True)

    for (split_index_raw, group, aspect), region_rows in grouped_rows:
        split_index = int(split_index_raw)
        train_prompts, test_prompts = stable_prompt_split(
            prompt_ids,
            seed=args.seed,
            split_index=split_index,
            test_frac=args.test_frac,
        )
        obs = sorted_edge_observations(flat, aspect=str(aspect), group=str(group))
        train_obs = obs[obs["prompt_id"].astype(str).isin(train_prompts)].copy()
        test_obs = obs[obs["prompt_id"].astype(str).isin(test_prompts)].copy()

        for original in region_rows.to_dict(orient="records"):
            triplet = tuple(str(original["triplet"]).split(" | "))
            if len(triplet) != 3:
                raise ValueError(f"Expected a three-node triplet, got: {triplet}")
            train_region = train_obs[region_mask(train_obs, triplet)].copy()
            test_region = test_obs[region_mask(test_obs, triplet)].copy()
            train_edges = aggregate_edges(train_region)
            if len(train_edges) != 3:
                raise ValueError(f"Incomplete training triplet at split={split_index}: {triplet}")

            linear = hodge_residual(train_edges, triplet, min_support=args.min_train_edge_weight)
            recorded_rho = float(original["train_rho_cyc"])
            recomputed_rho = float(linear["rho_cyc"])
            if not math.isclose(recorded_rho, recomputed_rho, rel_tol=1e-9, abs_tol=1e-11):
                raise ValueError(
                    f"Hodge reconstruction mismatch at split={split_index}, aspect={aspect}, "
                    f"triplet={triplet}: recorded={recorded_rho}, recomputed={recomputed_rho}"
                )

            diagnostics = projection_diagnostics(
                train_edges,
                triplet,
                min_support=args.min_train_edge_weight,
                l2=args.projection_l2,
                logit_smoothing=args.logit_smoothing,
            )
            lookup_p = edge_lookup_predictions(train_edges, test_region, smoothing=args.edge_smoothing)
            lookup_metrics = prediction_metrics(test_region, lookup_p)

            out = dict(original)
            out["train_rho_cyc_recomputed"] = recomputed_rho
            out["train_rho_cyc_abs_discrepancy"] = abs(recorded_rho - recomputed_rho)
            out.update({f"train_{name}": value for name, value in diagnostics.items()})
            for metric in CORE_METRICS:
                lookup_name = f"edge_lookup_{metric}"
                uplift_name = f"uplift_{metric}"
                out[lookup_name] = float(lookup_metrics[metric])
                out[uplift_name] = float(original[metric]) - float(lookup_metrics[metric])
            enriched.append(out)

    result = pd.DataFrame(enriched)
    if len(result) != len(rows):
        raise AssertionError(f"Enrichment changed row count: {len(rows)} -> {len(result)}")
    return result


def aggregate_regions(rows: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["group", "aspect", "triplet"]
    numeric = rows.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric = [col for col in numeric if col != "split_index"]
    agg = rows.groupby(identifiers, as_index=False)[numeric].mean()
    counts = (
        rows.groupby(identifiers, as_index=False)
        .agg(n_split_rows=("split_index", "size"), n_splits=("split_index", "nunique"))
    )
    return agg.merge(counts, on=identifiers, validate="one_to_one")


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(np.std(array, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("Cannot standardize a constant or non-finite variable.")
    return (array - float(np.mean(array))) / sd


def model_nodes(regions: pd.DataFrame) -> List[str]:
    nodes: set[str] = set()
    for triplet in regions["triplet"].astype(str):
        nodes.update(triplet.split(" | "))
    return sorted(nodes)


def design_frame(
    df: pd.DataFrame,
    predictor: str,
    include_model_membership: bool,
) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x[predictor] = zscore(df[predictor])
    for control in CONTROLS:
        values = np.log1p(df[control]) if control == "train_total_weight" else df[control]
        x[control] = zscore(values)
    for fixed_effect in ["group", "aspect"]:
        dummies = pd.get_dummies(df[fixed_effect].astype(str), prefix=fixed_effect, drop_first=True, dtype=float)
        dummies.index = df.index
        x = pd.concat([x, dummies], axis=1)
    if include_model_membership:
        nodes = model_nodes(df)
        triplets = df["triplet"].astype(str).str.split(" | ", regex=False)
        for node_idx, node in enumerate(nodes[:-1]):
            x[f"contains_model_{node_idx}"] = triplets.apply(lambda values: float(node in values))
    return sm.add_constant(x, has_constant="add")


def standardized_beta(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    include_model_membership: bool = False,
) -> float:
    x = design_frame(df, predictor, include_model_membership)
    y = zscore(df[outcome])
    coef, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
    return float(coef[list(x.columns).index(predictor)])


def regression_diagnostics(df: pd.DataFrame, predictor: str, outcome: str) -> Dict[str, Any]:
    x = design_frame(df, predictor, include_model_membership=False)
    y = zscore(df[outcome])
    fit = sm.OLS(y, x).fit()
    predictor_idx = list(fit.params.index).index(predictor)
    hc3 = fit.get_robustcov_results(cov_type="HC3")
    clustered = fit.get_robustcov_results(
        cov_type="cluster",
        groups=df["triplet"].astype(str),
        use_correction=True,
    )
    influence = fit.get_influence()
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    max_idx = int(np.nanargmax(cooks))
    max_row = df.iloc[max_idx]

    model_fit = sm.OLS(y, design_frame(df, predictor, include_model_membership=True)).fit()
    model_hc3 = model_fit.get_robustcov_results(cov_type="HC3")
    model_idx = list(model_fit.params.index).index(predictor)
    return {
        "partial_beta": float(fit.params[predictor]),
        "ols_p": float(fit.pvalues[predictor]),
        "hc3_p": float(hc3.pvalues[predictor_idx]),
        "triplet_cluster_p": float(clustered.pvalues[predictor_idx]),
        "model_membership_beta": float(model_fit.params[predictor]),
        "model_membership_hc3_p": float(model_hc3.pvalues[model_idx]),
        "max_cooks_d": float(cooks[max_idx]),
        "max_influence_aspect": str(max_row["aspect"]),
        "max_influence_triplet": str(max_row["triplet"]),
    }


def permutation_maps(nodes: Sequence[str], max_permutations: int, seed: int) -> Iterable[Dict[str, str]]:
    n_exact = math.factorial(len(nodes))
    if len(nodes) <= 8 and (max_permutations <= 0 or max_permutations >= n_exact):
        for permuted in itertools.permutations(nodes):
            yield dict(zip(nodes, permuted))
        return

    if max_permutations <= 0:
        raise ValueError("Set --max-network-permutations for graphs with more than eight nodes.")
    rng = np.random.default_rng(seed)
    for _ in range(max_permutations):
        yield dict(zip(nodes, rng.permutation(np.asarray(nodes, dtype=object)).tolist()))


def network_permutation_pvalues(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    max_permutations: int,
    seed: int,
) -> Tuple[float, float, int]:
    nodes = model_nodes(df)
    observed = standardized_beta(df, predictor, outcome)
    lookup = {
        (str(row.group), str(row.aspect), frozenset(str(row.triplet).split(" | "))): float(getattr(row, predictor))
        for row in df.itertuples(index=False)
    }
    parsed = [tuple(value.split(" | ")) for value in df["triplet"].astype(str)]
    null_values: List[float] = []
    for mapping in permutation_maps(nodes, max_permutations, seed):
        permuted = df.copy()
        permuted[predictor] = [
            lookup[(str(group), str(aspect), frozenset(mapping[node] for node in triplet))]
            for group, aspect, triplet in zip(df["group"], df["aspect"], parsed)
        ]
        null_values.append(standardized_beta(permuted, predictor, outcome))
    null = np.asarray(null_values, dtype=float)
    two_sided = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
    one_sided = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return two_sided, one_sided, int(len(null))


def leave_out_ranges(df: pd.DataFrame, predictor: str, outcome: str) -> Dict[str, float]:
    aspect_betas = []
    for aspect in sorted(df["aspect"].astype(str).unique()):
        subset = df[df["aspect"].astype(str) != aspect]
        if len(subset) >= 20:
            aspect_betas.append(standardized_beta(subset, predictor, outcome))
    model_betas = []
    for node in model_nodes(df):
        subset = df[~df["triplet"].astype(str).str.contains(node, regex=False)]
        if len(subset) >= 20:
            model_betas.append(standardized_beta(subset, predictor, outcome))
    return {
        "leave_one_aspect_beta_min": float(np.min(aspect_betas)) if aspect_betas else np.nan,
        "leave_one_aspect_beta_max": float(np.max(aspect_betas)) if aspect_betas else np.nan,
        "leave_one_model_beta_min": float(np.min(model_betas)) if model_betas else np.nan,
        "leave_one_model_beta_max": float(np.max(model_betas)) if model_betas else np.nan,
    }


def inference_table(regions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    specs = [
        ("primary", "train_kl_projection_per_weight", "uplift_test_log_loss"),
        ("primary", "train_brier_projection_per_weight", "uplift_test_brier"),
        ("secondary", "train_logit_hodge_fraction", "uplift_test_log_loss"),
        ("secondary", "train_logit_hodge_fraction", "uplift_test_brier"),
        ("secondary", "train_rho_cyc", "uplift_test_pair_calibration_mae"),
        ("secondary", "train_rho_cyc", "uplift_test_majority_edge_error"),
        ("negative_control", "train_rho_cyc", "test_log_loss"),
        ("negative_control", "train_rho_cyc", "test_pair_calibration_mae"),
        ("negative_control", "train_rho_cyc", "test_majority_edge_error"),
        ("negative_control", "train_kl_projection_per_weight", "edge_lookup_test_log_loss"),
    ]
    output = []
    for index, (family, predictor, outcome) in enumerate(specs):
        needed = [predictor, outcome, *CONTROLS, "group", "aspect", "triplet"]
        data = regions.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
        diagnostics = regression_diagnostics(data, predictor, outcome)
        p_two, p_one, n_perm = network_permutation_pvalues(
            data,
            predictor,
            outcome,
            args.max_network_permutations,
            args.seed + 7919 * index,
        )
        row = {
            "family": family,
            "predictor": predictor,
            "outcome": outcome,
            "n_regions": int(len(data)),
            "mean_predictor": float(data[predictor].mean()),
            "mean_outcome": float(data[outcome].mean()),
            "network_permutation_p_two_sided": p_two,
            "network_permutation_p_one_sided_positive": p_one,
            "n_network_permutations": n_perm,
            **diagnostics,
            **leave_out_ranges(data, predictor, outcome),
        }
        output.append(row)
    table = pd.DataFrame(output)
    table["network_permutation_q_bh_within_family"] = np.nan
    for family, indexes in table.groupby("family").groups.items():
        q_values = bh_fdr(table.loc[indexes, "network_permutation_p_two_sided"].tolist())
        table.loc[indexes, "network_permutation_q_bh_within_family"] = q_values
    return table


def report_markdown(inference: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    primary = inference[inference["family"] == "primary"].copy()
    columns = [
        "predictor",
        "outcome",
        "n_regions",
        "mean_outcome",
        "partial_beta",
        "network_permutation_p_two_sided",
        "network_permutation_q_bh_within_family",
        "hc3_p",
        "model_membership_hc3_p",
        "leave_one_model_beta_min",
        "leave_one_model_beta_max",
    ]
    display = primary[columns].copy()
    for column in display.columns:
        if column not in {"predictor", "outcome"}:
            display[column] = display[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
    supported = primary[
        (primary["partial_beta"] > 0)
        & (primary["network_permutation_q_bh_within_family"] < 0.05)
    ]
    conclusion = (
        "At least one preregistered loss-matched hypothesis is supported."
        if not supported.empty
        else "Neither preregistered loss-matched hypothesis is supported at network-aware FDR 0.05."
    )
    return "\n".join(
        [
            "# MultiPref Reward Projection V2 Results",
            "",
            "The primary outcome is scalar neural-RM loss minus saturated train-edge loss. Positive values mean the edge model performs better. The primary predictors are the population KL and Brier regret of the best scalar Bradley--Terry projection on training edges.",
            "",
            f"**Primary conclusion:** {conclusion}",
            "",
            "## Primary Tests",
            "",
            simple_markdown_table(display),
            "",
            "## Design Audit",
            "",
            f"- Split rows: {metadata['n_split_rows']}",
            f"- Aggregated model-triplet regions: {metadata['n_regions']}",
            f"- Model nodes: {metadata['n_nodes']}",
            f"- Network permutations per test: {metadata['n_network_permutations']}",
            f"- Maximum reconstructed raw-Hodge discrepancy: {metadata['max_rho_discrepancy']:.3g}",
            "- Network permutations relabel model nodes and therefore preserve triplet overlap.",
            "- The saturated edge comparator is a diagnostic baseline, not a matched neural pairwise model.",
            "- Absolute-error associations are reported only as negative controls.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    rows, matches = load_split_rows(args.region_glob)
    flat = load_flat(args.flat)
    enriched = enrich_rows(rows, flat, args)
    regions = aggregate_regions(enriched)
    inference = inference_table(regions, args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    split_path = args.outdir / "reward_projection_v2_split_rows.csv"
    region_path = args.outdir / "reward_projection_v2_aggregated_regions.csv"
    inference_path = args.outdir / "reward_projection_v2_inference.csv"
    metadata_path = args.outdir / "reward_projection_v2_metadata.json"
    report_path = args.outdir / "reward_projection_v2_report.md"
    enriched.to_csv(split_path, index=False)
    regions.to_csv(region_path, index=False)
    inference.to_csv(inference_path, index=False)

    metadata = {
        "region_glob": args.region_glob,
        "input_files": matches,
        "flat": str(args.flat),
        "n_split_rows": int(len(enriched)),
        "n_regions": int(len(regions)),
        "n_nodes": int(len(model_nodes(regions))),
        "n_network_permutations": int(inference["n_network_permutations"].min()),
        "max_rho_discrepancy": float(enriched["train_rho_cyc_abs_discrepancy"].max()),
        "seed": args.seed,
        "test_frac": args.test_frac,
        "edge_smoothing": args.edge_smoothing,
        "projection_l2": args.projection_l2,
        "logit_smoothing": args.logit_smoothing,
        "primary_hypotheses": [
            "train_kl_projection_per_weight -> uplift_test_log_loss",
            "train_brier_projection_per_weight -> uplift_test_brier",
        ],
        "split_path": str(split_path),
        "region_path": str(region_path),
        "inference_path": str(inference_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown(inference, metadata), encoding="utf-8")
    print(f"Wrote {split_path}")
    print(f"Wrote {region_path}")
    print(f"Wrote {inference_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
