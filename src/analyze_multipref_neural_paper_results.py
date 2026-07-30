#!/usr/bin/env python3
"""Paper-quality analysis for MultiPref neural reward-model outputs.

This script is the conservative inference layer for
run_multipref_neural_reward_downstream.py.  It reads split-level region result
CSVs, collapses repeated split rows to unique group/aspect/triplet regions, and
reports whether train-set rho_cyc predicts held-out neural reward-model failure.

The primary estimand is not row-level significance over repeated split rows.
Instead, the default analysis:

  1. aggregates by group/aspect/triplet;
  2. estimates standardized partial beta for rho_cyc with controls for margin,
     support, and group/aspect fixed effects;
  3. reports bootstrap confidence intervals;
  4. reports within-group/aspect permutation p-values;
  5. reports high-vs-low rho quartile effects.

This is intended for paper tables and reviewer-facing robustness checks.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError as exc:
    raise SystemExit("Missing scipy. Install with: pip install scipy") from exc


DEFAULT_REGION_GLOB = "src/results/multipref_neural_reward_roberta_large/*_split_*_region_results.csv"
DEFAULT_OUTDIR = Path("src/results/multipref_neural_reward_roberta_large/paper_quality")
DEFAULT_OUTCOMES = [
    "test_log_loss",
    "test_brier",
    "test_error_rate",
    "test_calibration_abs_error",
    "test_pair_calibration_mae",
    "test_majority_edge_error",
]
CONTROL_COLS = [
    "train_mean_abs_margin",
    "train_min_edge_support",
    "train_total_weight",
]
FIXED_EFFECTS = ["group", "aspect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-glob", default=DEFAULT_REGION_GLOB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--outcomes", nargs="+", default=DEFAULT_OUTCOMES)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--min-regions", type=int, default=30)
    parser.add_argument(
        "--weighting",
        choices=["none", "test_weight"],
        default="none",
        help="Primary inference is unweighted by default; test_weight is a sensitivity option.",
    )
    parser.add_argument(
        "--scope-summary",
        action="store_true",
        help="Also write exploratory group/aspect scope summaries.",
    )
    return parser.parse_args()


def finite_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    d = df.copy()
    for col in cols:
        if col not in d.columns:
            raise ValueError(f"Missing required column: {col}")
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna(subset=list(cols)).copy()


def load_region_results(pattern: str) -> Tuple[pd.DataFrame, List[str]]:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region-result files matched --region-glob: {pattern}")
    frames = []
    for path in matches:
        df = pd.read_csv(path)
        df["source_file"] = path
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    if "triplet" not in rows.columns:
        rows["triplet"] = rows[["node_1", "node_2", "node_3"]].astype(str).agg(" | ".join, axis=1)
    rows["region_id"] = (
        rows["group"].astype(str)
        + "||"
        + rows["aspect"].astype(str)
        + "||"
        + rows["triplet"].astype(str)
    )
    return rows, matches


def aggregate_regions(rows: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [
        "split_index",
        "train_rho_cyc",
        "train_residual_energy",
        "train_gradient_energy",
        "train_cycle_rate",
        "train_weighted_cycle_intensity",
        "train_n_obs",
        "train_total_weight",
        "train_edges",
        "train_eligible_edges",
        "train_min_edge_support",
        "train_mean_edge_support",
        "train_mean_abs_margin",
        "train_max_abs_margin",
        "test_weight",
        "test_n_obs",
        "test_edges",
        *outcomes,
    ]
    numeric_cols = [col for col in numeric_cols if col in rows.columns]
    agg_spec: Dict[str, Any] = {col: (col, "mean") for col in numeric_cols if col != "split_index"}
    agg_spec["n_split_rows"] = ("split_index", "size")
    agg_spec["n_splits"] = ("split_index", "nunique")
    out = (
        rows.groupby(["region_id", "group", "aspect", "triplet"], as_index=False)
        .agg(**agg_spec)
        .reset_index(drop=True)
    )
    return out


def zscore(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    return (x - float(np.mean(x))) / sd


def design_matrix(
    df: pd.DataFrame,
    outcome: str,
    include_fixed_effects: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    d = df.copy()
    y = zscore(d[outcome].to_numpy(dtype=float))
    rho = zscore(d["train_rho_cyc"].to_numpy(dtype=float))
    if y is None or rho is None:
        return None, None, []

    cols = [np.ones(len(d), dtype=float), rho]
    names = ["intercept", "train_rho_cyc"]
    for col in CONTROL_COLS:
        if col not in d.columns:
            continue
        x = np.log1p(d[col].to_numpy(dtype=float)) if col == "train_total_weight" else d[col].to_numpy(dtype=float)
        zx = zscore(x)
        if zx is None:
            continue
        cols.append(zx)
        names.append(col)

    if include_fixed_effects:
        for fe in FIXED_EFFECTS:
            if fe not in d.columns:
                continue
            dummies = pd.get_dummies(d[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
            for name in dummies.columns:
                vals = dummies[name].to_numpy(dtype=float)
                if np.std(vals) > 1e-12:
                    cols.append(vals)
                    names.append(name)

    xmat = np.column_stack(cols)
    keep = np.isfinite(y) & np.isfinite(xmat).all(axis=1)
    if int(keep.sum()) < len(names) + 3:
        return None, None, []
    return xmat[keep], y[keep], names


def partial_beta(
    df: pd.DataFrame,
    outcome: str,
    weights: Optional[np.ndarray] = None,
    include_fixed_effects: bool = True,
) -> float:
    xmat, y, names = design_matrix(df, outcome, include_fixed_effects=include_fixed_effects)
    if xmat is None or y is None or "train_rho_cyc" not in names:
        return np.nan
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w[np.isfinite(df[outcome].to_numpy(dtype=float))]
        if len(w) != len(y) or np.any(w < 0) or float(np.sum(w)) <= 0.0:
            return np.nan
        sw = np.sqrt(w / float(np.mean(w)))
        xfit = xmat * sw[:, None]
        yfit = y * sw
    else:
        xfit = xmat
        yfit = y
    try:
        beta, *_ = np.linalg.lstsq(xfit, yfit, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[names.index("train_rho_cyc")])


def spearman_stat(df: pd.DataFrame, outcome: str) -> Tuple[float, float]:
    x = df["train_rho_cyc"].to_numpy(dtype=float)
    y = df[outcome].to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return np.nan, np.nan
    if np.std(x[keep]) <= 1e-12 or np.std(y[keep]) <= 1e-12:
        return np.nan, np.nan
    result = spearmanr(x[keep], y[keep])
    return float(result.statistic), float(result.pvalue)


def quartile_effect(df: pd.DataFrame, outcome: str) -> Tuple[float, float]:
    d = finite_frame(df, ["train_rho_cyc", outcome])
    if len(d) < 8:
        return np.nan, np.nan
    q25 = float(d["train_rho_cyc"].quantile(0.25))
    q75 = float(d["train_rho_cyc"].quantile(0.75))
    low = d[d["train_rho_cyc"] <= q25][outcome].to_numpy(dtype=float)
    high = d[d["train_rho_cyc"] >= q75][outcome].to_numpy(dtype=float)
    if len(low) == 0 or len(high) == 0:
        return np.nan, np.nan
    raw = float(np.mean(high) - np.mean(low))
    sd = float(np.std(d[outcome].to_numpy(dtype=float)))
    std = np.nan if sd <= 1e-12 else raw / sd
    return raw, std


def bootstrap_ci(
    df: pd.DataFrame,
    outcome: str,
    stat_name: str,
    n_bootstrap: int,
    seed: int,
    weights_col: Optional[str],
) -> Tuple[float, float]:
    if n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(df)
    vals: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        if stat_name == "partial_beta":
            weights = sample[weights_col].to_numpy(dtype=float) if weights_col else None
            val = partial_beta(sample, outcome, weights=weights)
        elif stat_name == "spearman":
            val, _ = spearman_stat(sample, outcome)
        elif stat_name == "quartile_raw":
            val, _ = quartile_effect(sample, outcome)
        elif stat_name == "quartile_std":
            _, val = quartile_effect(sample, outcome)
        else:
            raise ValueError(stat_name)
        if np.isfinite(val):
            vals.append(float(val))
    if len(vals) < max(30, n_bootstrap // 20):
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permuted_rho(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    rho = df["train_rho_cyc"].to_numpy(dtype=float).copy()
    out = rho.copy()
    strata = df[["group", "aspect"]].astype(str).agg("||".join, axis=1)
    for stratum in strata.unique():
        idx = np.flatnonzero(strata.to_numpy() == stratum)
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def permutation_pvalue(
    df: pd.DataFrame,
    outcome: str,
    observed: float,
    n_permutations: int,
    seed: int,
    weights_col: Optional[str],
) -> float:
    if n_permutations <= 0 or not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_permutations):
        d = df.copy()
        d["train_rho_cyc"] = permuted_rho(d, rng)
        weights = d[weights_col].to_numpy(dtype=float) if weights_col else None
        val = partial_beta(d, outcome, weights=weights)
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return np.nan
    vals_arr = np.asarray(vals, dtype=float)
    return float((1.0 + np.sum(np.abs(vals_arr) >= abs(observed))) / (len(vals_arr) + 1.0))


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    keep = np.isfinite(p)
    if int(keep.sum()) == 0:
        return q.tolist()
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(p[keep])]
    m = len(order)
    running = 1.0
    for rank, original_idx in enumerate(order[::-1], start=1):
        true_rank = m - rank + 1
        running = min(running, p[original_idx] * m / true_rank)
        q[original_idx] = running
    return q.tolist()


def simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table.

    pandas.DataFrame.to_markdown requires the optional tabulate dependency.
    Keeping this local avoids another package requirement on GPU devboxes.
    """
    if df.empty:
        return ""
    text = df.astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[col_idx])) for row in rows))
        for col_idx, header in enumerate(headers)
    ]

    def render_row(values: Sequence[str]) -> str:
        cells = [str(value).ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def primary_table(
    regions: pd.DataFrame,
    outcomes: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    weights_col = "test_weight" if args.weighting == "test_weight" else None
    for outcome_idx, outcome in enumerate(outcomes):
        d = finite_frame(regions, ["train_rho_cyc", outcome, *[c for c in CONTROL_COLS if c in regions.columns]])
        if len(d) < args.min_regions:
            continue
        weights = d[weights_col].to_numpy(dtype=float) if weights_col else None
        beta = partial_beta(d, outcome, weights=weights)
        beta_lo, beta_hi = bootstrap_ci(
            d,
            outcome,
            "partial_beta",
            args.n_bootstrap,
            args.seed + 1009 * outcome_idx,
            weights_col,
        )
        perm_p = permutation_pvalue(
            d,
            outcome,
            beta,
            args.n_permutations,
            args.seed + 9173 * outcome_idx,
            weights_col,
        )
        spear, spear_p = spearman_stat(d, outcome)
        spear_lo, spear_hi = bootstrap_ci(
            d,
            outcome,
            "spearman",
            args.n_bootstrap,
            args.seed + 4073 * outcome_idx,
            weights_col=None,
        )
        q_raw, q_std = quartile_effect(d, outcome)
        q_raw_lo, q_raw_hi = bootstrap_ci(
            d,
            outcome,
            "quartile_raw",
            args.n_bootstrap,
            args.seed + 6131 * outcome_idx,
            weights_col=None,
        )
        q_std_lo, q_std_hi = bootstrap_ci(
            d,
            outcome,
            "quartile_std",
            args.n_bootstrap,
            args.seed + 7127 * outcome_idx,
            weights_col=None,
        )
        rows.append(
            {
                "outcome": outcome,
                "n_regions": int(len(d)),
                "n_split_rows_mean": float(d["n_split_rows"].mean()) if "n_split_rows" in d else np.nan,
                "mean_train_rho_cyc": float(d["train_rho_cyc"].mean()),
                "sd_train_rho_cyc": float(d["train_rho_cyc"].std(ddof=0)),
                "mean_outcome": float(d[outcome].mean()),
                "partial_beta_rho_cyc": beta,
                "partial_beta_ci_low": beta_lo,
                "partial_beta_ci_high": beta_hi,
                "permutation_p": perm_p,
                "spearman_rho_cyc": spear,
                "spearman_ci_low": spear_lo,
                "spearman_ci_high": spear_hi,
                "spearman_p_asymptotic": spear_p,
                "high_minus_low_quartile_raw": q_raw,
                "high_minus_low_quartile_raw_ci_low": q_raw_lo,
                "high_minus_low_quartile_raw_ci_high": q_raw_hi,
                "high_minus_low_quartile_std": q_std,
                "high_minus_low_quartile_std_ci_low": q_std_lo,
                "high_minus_low_quartile_std_ci_high": q_std_hi,
                "weighting": args.weighting,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["permutation_q_bh"] = bh_fdr(out["permutation_p"].tolist())
    return out


def scope_summary(regions: pd.DataFrame, outcomes: Sequence[str], min_regions: int) -> pd.DataFrame:
    scopes: List[Tuple[str, pd.DataFrame]] = [("all", regions)]
    for group, gdf in regions.groupby("group", sort=True):
        scopes.append((f"group={group}", gdf))
    for aspect, adf in regions.groupby("aspect", sort=True):
        scopes.append((f"aspect={aspect}", adf))
    for (group, aspect), gadf in regions.groupby(["group", "aspect"], sort=True):
        scopes.append((f"group={group};aspect={aspect}", gadf))

    rows: List[Dict[str, Any]] = []
    for scope, d0 in scopes:
        for outcome in outcomes:
            if outcome not in d0.columns:
                continue
            d = finite_frame(d0, ["train_rho_cyc", outcome])
            if len(d) < min_regions:
                continue
            spear, spear_p = spearman_stat(d, outcome)
            beta = partial_beta(d, outcome, include_fixed_effects=False)
            q_raw, q_std = quartile_effect(d, outcome)
            rows.append(
                {
                    "scope": scope,
                    "outcome": outcome,
                    "n_regions": int(len(d)),
                    "mean_train_rho_cyc": float(d["train_rho_cyc"].mean()),
                    "mean_outcome": float(d[outcome].mean()),
                    "spearman_rho_cyc": spear,
                    "spearman_p_asymptotic": spear_p,
                    "partial_beta_no_fe": beta,
                    "high_minus_low_quartile_raw": q_raw,
                    "high_minus_low_quartile_std": q_std,
                }
            )
    return pd.DataFrame(rows)


def markdown_report(primary: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    lines = [
        "# MultiPref Neural Reward Paper-Quality Analysis",
        "",
        "Primary inference aggregates repeated split rows to unique group/aspect/triplet regions.",
        "The partial beta controls for train-set margin, support, and group/aspect fixed effects.",
        "Permutation p-values shuffle rho_cyc within group/aspect strata.",
        "",
        "## Metadata",
        "",
        f"- Input files: {metadata['n_input_files']}",
        f"- Split-level rows: {metadata['n_split_rows']}",
        f"- Aggregated regions: {metadata['n_regions']}",
        f"- Weighting: {metadata['weighting']}",
        f"- Bootstrap draws: {metadata['n_bootstrap']}",
        f"- Permutations: {metadata['n_permutations']}",
        "",
        "## Primary Table",
        "",
    ]
    if primary.empty:
        lines.append("No primary rows passed the minimum-region threshold.")
        return "\n".join(lines) + "\n"

    display_cols = [
        "outcome",
        "n_regions",
        "partial_beta_rho_cyc",
        "partial_beta_ci_low",
        "partial_beta_ci_high",
        "permutation_p",
        "permutation_q_bh",
        "spearman_rho_cyc",
        "spearman_ci_low",
        "spearman_ci_high",
        "high_minus_low_quartile_std",
        "high_minus_low_quartile_std_ci_low",
        "high_minus_low_quartile_std_ci_high",
    ]
    table = primary[display_cols].copy()
    for col in table.columns:
        if col != "outcome":
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    lines.append(simple_markdown_table(table))
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "Positive coefficients mean higher train-set rho_cyc predicts worse held-out outcomes.",
            "For calibration outcomes, this is the theoretically direct scalar-compression failure mode.",
            "For log loss, Brier, and error rate, a null result should not be treated as a refutation of the residual diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, matches = load_region_results(args.region_glob)
    present_outcomes = [col for col in args.outcomes if col in rows.columns]
    if not present_outcomes:
        raise SystemExit("None of the requested outcome columns are present.")

    regions = aggregate_regions(rows, present_outcomes)
    primary = primary_table(regions, present_outcomes, args)

    split_rows_path = args.outdir / "neural_reward_split_rows.csv"
    aggregate_path = args.outdir / "neural_reward_aggregated_regions.csv"
    primary_path = args.outdir / "neural_reward_primary_effects.csv"
    metadata_path = args.outdir / "neural_reward_paper_metadata.json"
    report_path = args.outdir / "neural_reward_paper_report.md"

    rows.to_csv(split_rows_path, index=False)
    regions.to_csv(aggregate_path, index=False)
    primary.to_csv(primary_path, index=False)

    metadata = {
        "region_glob": args.region_glob,
        "input_files": matches,
        "n_input_files": len(matches),
        "n_split_rows": int(len(rows)),
        "n_regions": int(len(regions)),
        "outcomes": present_outcomes,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "min_regions": args.min_regions,
        "weighting": args.weighting,
        "split_rows_path": str(split_rows_path),
        "aggregate_path": str(aggregate_path),
        "primary_path": str(primary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(markdown_report(primary, metadata), encoding="utf-8")

    if args.scope_summary:
        scopes = scope_summary(regions, present_outcomes, min_regions=args.min_regions)
        scopes_path = args.outdir / "neural_reward_scope_summary.csv"
        scopes.to_csv(scopes_path, index=False)
        metadata["scope_summary_path"] = str(scopes_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote {scopes_path}")

    print(f"Wrote {split_rows_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {primary_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
