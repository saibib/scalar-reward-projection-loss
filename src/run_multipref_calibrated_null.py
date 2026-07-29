#!/usr/bin/env python3
"""Calibrated transitive-null tests for the MultiPref global tournaments.

The MultiPref v4 analysis reports non-zero Hodge/scalar residuals for pooled
model tournaments. This script asks whether those residuals are larger than
expected from finite annotation noise under a transitive scalar edge flow.

For each annotation group and aspect, the null:

1. keeps the observed model-pair support graph;
2. keeps the number and weights of non-tie annotations on each model pair;
3. fits the closest bounded additive margin flow, m_ij = r_i - r_j; and
4. resamples pairwise signs with E[sign_ij] = m_ij.

The resulting null is calibrated to the same support used by the observed
Hodge residual, without creating prompt-local agendas that MultiPref lacks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from run_multipref_cycles_v4 import ASPECTS, EdgeFlow, cycle_stats, scalar_projection_residual


@dataclass
class EdgeCell:
    i: str
    j: str
    support: float
    weighted_margin: float
    weight_counts: Dict[float, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flat",
        type=Path,
        default=Path("src/results/multipref_v4/multipref_flat_annotations.csv"),
        help="Flattened MultiPref annotations from run_multipref_cycles_v4.py.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("src/results/multipref_v4/multipref_cycle_projection_summary.csv"),
        help="Observed MultiPref summary table.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("src/results/multipref_v4"),
        help="Directory for calibrated null outputs.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all", "normal", "expert"],
        help="Annotation groups to test. Use all for the pooled tournament.",
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=ASPECTS,
        choices=ASPECTS,
        help="Preference aspects to test.",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=5000,
        help="Number of calibrated null simulations per group/aspect.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--min-support",
        type=float,
        default=1.0,
        help="Minimum edge support used by the observed global analysis.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Canonical orientation threshold for cycle counts.",
    )
    parser.add_argument(
        "--margin-bound",
        type=float,
        default=0.999,
        help="Absolute bound for fitted additive expected margins.",
    )
    return parser.parse_args()


def signed_edge_observations(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    """Return sorted-edge signed observations for a given aspect."""
    sign_col = f"{aspect}_sign"
    weight_col = f"{aspect}_weight"
    rows = []
    use = df[(df[sign_col] != 0) & (df[weight_col] > 0)].copy()
    for a, b, sign, weight in use[["alt_a", "alt_b", sign_col, weight_col]].itertuples(index=False, name=None):
        a = str(a)
        b = str(b)
        if not a or not b or a == b:
            continue
        i, j = sorted([a, b])
        winner = a if int(sign) > 0 else b
        sorted_sign = +1 if winner == i else -1
        rows.append((i, j, sorted_sign, float(weight)))
    return pd.DataFrame(rows, columns=["i", "j", "sign", "weight"])


def aggregate_edges(obs: pd.DataFrame) -> Tuple[List[str], List[EdgeCell]]:
    nodes = sorted(set(obs["i"]).union(set(obs["j"])))
    cells: List[EdgeCell] = []
    for (i, j), g in obs.groupby(["i", "j"], sort=True):
        support = float(g["weight"].sum())
        weighted_margin = float((g["sign"] * g["weight"]).sum())
        weight_counts = {float(w): int(n) for w, n in g["weight"].value_counts().sort_index().items()}
        cells.append(EdgeCell(i=i, j=j, support=support, weighted_margin=weighted_margin, weight_counts=weight_counts))
    return nodes, cells


def cells_to_flow(nodes: Sequence[str], cells: Sequence[EdgeCell]) -> EdgeFlow:
    edges = {}
    for cell in cells:
        edges[(cell.i, cell.j)] = {
            "support": cell.support,
            "weighted_margin": cell.weighted_margin,
            "i_wins": 0.0,
            "j_wins": 0.0,
        }
    return EdgeFlow(edges=edges, nodes=list(nodes))


def incidence(nodes: Sequence[str], cells: Sequence[EdgeCell]) -> np.ndarray:
    idx = {node: k for k, node in enumerate(nodes)}
    B = np.zeros((len(cells), len(nodes)))
    for r, cell in enumerate(cells):
        B[r, idx[cell.i]] = 1.0
        B[r, idx[cell.j]] = -1.0
    return B


def fit_bounded_additive_margins(
    nodes: Sequence[str],
    cells: Sequence[EdgeCell],
    margin_bound: float,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Fit closest additive margins under |r_i-r_j| <= margin_bound."""
    if len(nodes) < 2 or not cells:
        return np.array([]), np.array([]), False, 0

    B = incidence(nodes, cells)
    y = np.array([cell.weighted_margin / cell.support for cell in cells], dtype=float)
    w = np.array([cell.support for cell in cells], dtype=float)
    sw = np.sqrt(w)

    # Unconstrained weighted least-squares fit supplies a stable initializer.
    B_gauge = B[:, :-1]
    init_reduced, *_ = np.linalg.lstsq(B_gauge * sw[:, None], y * sw, rcond=None)
    x0 = np.concatenate([init_reduced, [0.0]])
    x0 = x0 - x0.mean()

    def objective(r: np.ndarray) -> float:
        resid = B @ r - y
        return float(np.sum(w * resid * resid))

    def gradient(r: np.ndarray) -> np.ndarray:
        resid = B @ r - y
        return 2.0 * (B.T @ (w * resid))

    pair_constraint = LinearConstraint(B, -margin_bound, margin_bound)
    gauge_constraint = LinearConstraint(np.ones((1, len(nodes))), 0.0, 0.0)
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        constraints=[pair_constraint, gauge_constraint],
        bounds=Bounds(-np.inf, np.inf),
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    scores = result.x if result.success else x0
    scores = scores - scores.mean()
    fitted = np.clip(B @ scores, -margin_bound, margin_bound)
    n_at_bound = int(np.sum(np.isclose(np.abs(fitted), margin_bound, atol=1e-5)))
    return scores, fitted, bool(result.success), n_at_bound


def simulated_flow(
    rng: np.random.Generator,
    nodes: Sequence[str],
    cells: Sequence[EdgeCell],
    fitted_margins: np.ndarray,
) -> EdgeFlow:
    edges = {}
    for cell, margin in zip(cells, fitted_margins):
        p_i_wins = float((1.0 + margin) / 2.0)
        weighted_margin = 0.0
        support = 0.0
        for weight, n in cell.weight_counts.items():
            wins = int(rng.binomial(n, p_i_wins))
            weighted_margin += float(weight) * (2 * wins - n)
            support += float(weight) * n
        edges[(cell.i, cell.j)] = {
            "support": support,
            "weighted_margin": weighted_margin,
            "i_wins": 0.0,
            "j_wins": 0.0,
        }
    return EdgeFlow(edges=edges, nodes=list(nodes))


def null_draw_stats(flow: EdgeFlow, min_support: float, tau: float) -> Tuple[float, float, int]:
    residual = scalar_projection_residual(flow, min_support=min_support)
    cycles = cycle_stats(flow, min_support=min_support, tau=tau)
    return (
        float(residual["rho_cyc"]),
        float(cycles["cycle_rate"]),
        int(cycles["cyclic_triples"]),
    )


def p_ge(null_values: np.ndarray, observed: float) -> float:
    # Plus-one finite-simulation correction.
    return float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1))


def summarize_null(null_values: np.ndarray, observed: float, prefix: str) -> Dict[str, float]:
    mean = float(np.mean(null_values))
    sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else float("nan")
    z = float((observed - mean) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    return {
        f"observed_{prefix}": float(observed),
        f"null_mean_{prefix}": mean,
        f"null_sd_{prefix}": sd,
        f"z_{prefix}": z,
        f"p_ge_{prefix}": p_ge(null_values, observed),
        f"null_q025_{prefix}": float(np.quantile(null_values, 0.025)),
        f"null_q500_{prefix}": float(np.quantile(null_values, 0.5)),
        f"null_q975_{prefix}": float(np.quantile(null_values, 0.975)),
    }


def bh_qvalues(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted q-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(n, dtype=float)
    running = 1.0
    for k in range(n - 1, -1, -1):
        rank = k + 1
        running = min(running, ranked[k] * n / rank)
        q_ranked[k] = running
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q.tolist()


def observed_row(observed_summary: pd.DataFrame, group: str, aspect: str) -> pd.Series:
    key = "all" if group == "all" else f"group={group}"
    match = observed_summary[
        observed_summary["scope"].eq("global")
        & observed_summary["group"].eq(key)
        & observed_summary["aspect"].eq(aspect)
    ]
    if match.empty:
        raise ValueError(f"Missing observed summary for group={group}, aspect={aspect}")
    return match.iloc[0]


def group_filter(flat: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "all":
        return flat
    return flat[flat["annotation_group"].astype(str).eq(group)].copy()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    flat = pd.read_csv(args.flat)
    observed_summary = pd.read_csv(args.summary)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []

    for group, aspect in itertools.product(args.groups, args.aspects):
        sub = group_filter(flat, group)
        obs = signed_edge_observations(sub, aspect)
        nodes, cells = aggregate_edges(obs)
        if len(nodes) < 2 or not cells:
            continue

        flow = cells_to_flow(nodes, cells)
        obs_residual, obs_cycle_rate, obs_cyclic_triples = null_draw_stats(flow, args.min_support, args.tau)
        obs_table_row = observed_row(observed_summary, group, aspect)

        scores, fitted_margins, fit_ok, n_at_bound = fit_bounded_additive_margins(nodes, cells, args.margin_bound)
        for node, score in zip(nodes, scores):
            fit_rows.append({"group": group, "aspect": aspect, "model": node, "scalar_score": float(score)})

        rho_null = np.empty(args.n_null, dtype=float)
        cycle_rate_null = np.empty(args.n_null, dtype=float)
        cyclic_triples_null = np.empty(args.n_null, dtype=int)

        for sim in range(args.n_null):
            sim_flow = simulated_flow(rng, nodes, cells, fitted_margins)
            rho, cycle_rate, cyclic_triples = null_draw_stats(sim_flow, args.min_support, args.tau)
            rho_null[sim] = rho
            cycle_rate_null[sim] = cycle_rate
            cyclic_triples_null[sim] = cyclic_triples
            raw_rows.append(
                {
                    "group": group,
                    "aspect": aspect,
                    "sim": sim,
                    "rho_cyc": rho,
                    "cycle_rate": cycle_rate,
                    "cyclic_triples": cyclic_triples,
                }
            )

        row: Dict[str, object] = {
            "group": group,
            "aspect": aspect,
            "n_annotations": int(len(sub)),
            "n_non_tie_weighted_observations": float(obs["weight"].sum()),
            "nodes": int(len(nodes)),
            "edges": int(len(cells)),
            "eligible_triples": int(obs_table_row["eligible_triples"]),
            "observed_cyclic_triples": int(obs_cyclic_triples),
            "fit_success": fit_ok,
            "n_fitted_edges_at_bound": n_at_bound,
            "n_null": int(args.n_null),
            "seed": int(args.seed),
            "margin_bound": float(args.margin_bound),
        }
        row.update(summarize_null(rho_null, obs_residual, "rho_cyc"))
        row.update(summarize_null(cycle_rate_null, obs_cycle_rate, "cycle_rate"))
        row.update(summarize_null(cyclic_triples_null.astype(float), float(obs_cyclic_triples), "cyclic_triples"))
        summary_rows.append(row)
        print(
            f"[{group:>6} {aspect:>8}] "
            f"rho={obs_residual:.4f}, null={np.mean(rho_null):.4f}, p={p_ge(rho_null, obs_residual):.4g}; "
            f"cycles={obs_cyclic_triples}, null={np.mean(cyclic_triples_null):.2f}, "
            f"p={p_ge(cyclic_triples_null.astype(float), float(obs_cyclic_triples)):.4g}"
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_ge_rho_cyc_bh"] = bh_qvalues(summary["p_ge_rho_cyc"].to_numpy(float))
        summary["q_ge_cycle_rate_bh"] = bh_qvalues(summary["p_ge_cycle_rate"].to_numpy(float))
        summary["q_ge_cyclic_triples_bh"] = bh_qvalues(summary["p_ge_cyclic_triples"].to_numpy(float))
    raw = pd.DataFrame(raw_rows)
    fit = pd.DataFrame(fit_rows)
    summary_path = args.outdir / "multipref_calibrated_null_summary.csv"
    raw_path = args.outdir / "multipref_calibrated_null_raw.csv"
    fit_path = args.outdir / "multipref_calibrated_null_scalar_scores.csv"
    meta_path = args.outdir / "multipref_calibrated_null_metadata.json"
    summary.to_csv(summary_path, index=False)
    raw.to_csv(raw_path, index=False)
    fit.to_csv(fit_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "flat": str(args.flat),
                "observed_summary": str(args.summary),
                "groups": args.groups,
                "aspects": args.aspects,
                "n_null": args.n_null,
                "seed": args.seed,
                "min_support": args.min_support,
                "tau": args.tau,
                "margin_bound": args.margin_bound,
                "null_description": (
                    "Support-preserving bounded additive-margin null. For each group/aspect, "
                    "fit m_ij=r_i-r_j by weighted least squares with |m_ij| bounded, then "
                    "resample non-tie signs on the observed pair support while preserving "
                    "per-edge annotation weights."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {fit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
