"""
Experiment A orchestrator.

Computes, for each eligible country, the three orientation invariants of
its majority tournament on character types, then compares against a
permutation null that shuffles user-to-country assignments while keeping
each user's preference vector and each country's user count fixed. This
is the same null design used in the paper.

Outputs written to the results directory:

    per_country_observed.csv      one row per country, invariants.
    null_summary.json             null distribution statistics.
    aggregate_comparison.json     observed vs null, z-scores and p-values.
    tournaments/<COUNTRY>.npy     raw 20x20 tournaments (optional).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from moral_machine import (
    UserPreferences,
    country_majority_tournament_and_margins,
    eligible_countries,
    MIN_USERS_PER_COUNTRY,
    majority_tournament_and_margins_from_beta,
)
from topology import topological_invariants


# ---------------------------------------------------------------------------
# Observed invariants
# ---------------------------------------------------------------------------

@dataclass
class CountryResult:
    country: str
    n_users: int
    invariants: Dict[str, float]
    tournament: Optional[np.ndarray] = field(default=None, repr=False)
    margin: Optional[np.ndarray] = field(default=None, repr=False)
    support: Optional[np.ndarray] = field(default=None, repr=False)


def scalar_projection_stats(W: np.ndarray, support: np.ndarray) -> Dict[str, float]:
    """Weighted least-squares projection of a skew edge flow onto scores.

    This is the same scalar projection-loss diagnostic used for the LLM
    preference tournaments. Edges with zero support do not contribute.
    """
    if W.shape != support.shape or W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"W and support must be square matrices with the same shape; "
            f"got W={W.shape}, support={support.shape}"
        )

    K = W.shape[0]
    edges = [
        (i, j)
        for i in range(K)
        for j in range(i + 1, K)
        if support[i, j] > 0
    ]
    if not edges:
        return {
            "hodge_n_edges": 0,
            "hodge_gradient_energy": float("nan"),
            "hodge_cyclic_residual_energy": float("nan"),
            "hodge_total_energy": float("nan"),
            "hodge_rho_cyc": float("nan"),
        }

    A = np.zeros((len(edges), K), dtype=np.float64)
    y = np.zeros(len(edges), dtype=np.float64)
    weights = np.zeros(len(edges), dtype=np.float64)
    for row, (i, j) in enumerate(edges):
        A[row, i] = 1.0
        A[row, j] = -1.0
        y[row] = W[i, j]
        weights[row] = support[i, j]

    sqrt_w = np.sqrt(weights)
    Aw = A * sqrt_w[:, None]
    yw = y * sqrt_w
    scores, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    scores = scores - scores.mean()

    pred = A @ scores
    residual = y - pred
    total = float(np.sum(weights * y * y))
    resid = float(np.sum(weights * residual * residual))
    grad = float(np.sum(weights * pred * pred))

    return {
        "hodge_n_edges": int(len(edges)),
        "hodge_gradient_energy": grad,
        "hodge_cyclic_residual_energy": resid,
        "hodge_total_energy": total,
        "hodge_rho_cyc": resid / total if total > 0 else float("nan"),
    }


def compute_observed(
    prefs: UserPreferences,
    countries: Optional[List[str]] = None,
    save_tournaments: bool = False,
    progress: bool = True,
) -> List[CountryResult]:
    """Per-country tournament + invariants for the observed country assignment."""
    if countries is None:
        countries = eligible_countries(prefs)

    out: List[CountryResult] = []
    for idx, country in enumerate(countries):
        result = country_majority_tournament_and_margins(prefs, country)
        if result is None:
            continue
        T, W, support = result
        inv = topological_invariants(T)
        inv.update(scalar_projection_stats(W, support))
        n_users = int((prefs.countries == country).sum())
        if progress:
            print(
                f"[observed] {idx+1:>3}/{len(countries)}  {country}  "
                f"U={n_users:>6,}  cycles={inv['condorcet_3cycles']:>4}  "
                f"beta1={inv['beta_1_acyclic_complex']:>3}  "
                f"odd_parities={inv['odd_parity_4subsets']:>5}  "
                f"rho={inv['hodge_rho_cyc']:.4f}"
            )
        out.append(
            CountryResult(
                country=country,
                n_users=n_users,
                invariants=inv,
                tournament=T if save_tournaments else None,
                margin=W if save_tournaments else None,
                support=support if save_tournaments else None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------

def _tournament_from_beta(
    beta: np.ndarray,  # (U_c, K)
) -> np.ndarray:
    """Majority tournament for an arbitrary subset of users' beta vectors."""
    T, _, _ = majority_tournament_and_margins_from_beta(beta)
    return T


def run_permutation_null(
    prefs: UserPreferences,
    observed_countries: List[str],
    n_permutations: int = 500,
    seed: int = 0,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Permutation null: shuffle user-to-country labels, keep beta and sizes.

    For each permutation, we randomly reassign country labels to users
    preserving the original country size distribution, then recompute the
    invariants for each country. Aggregating across countries, we record the
    mean value of each invariant per permutation; this is compared against
    the same aggregation of the observed data.

    Returns a dict keyed by invariant name with shape (n_permutations,)
    arrays of per-permutation means.
    """
    rng = np.random.default_rng(seed)

    # Fixed country sizes (observed).
    country_sizes = {
        c: int((prefs.countries == c).sum()) for c in observed_countries
    }
    ordered_countries = list(observed_countries)
    sizes_list = [country_sizes[c] for c in ordered_countries]
    total_assignable = sum(sizes_list)

    # Pool of user indices (only those in eligible countries).
    eligible_mask = np.isin(prefs.countries, ordered_countries)
    pool_idx = np.where(eligible_mask)[0]
    if pool_idx.size < total_assignable:
        # If we ask for more than we have, cap to actual pool size and
        # scale sizes proportionally. Should not happen for the real data.
        scale = pool_idx.size / total_assignable
        sizes_list = [max(MIN_USERS_PER_COUNTRY, int(s * scale)) for s in sizes_list]

    metrics = [
        "condorcet_3cycles",
        "condorcet_3cycle_fraction",
        "odd_parity_4subsets",
        "odd_parity_fraction",
        "beta_1_acyclic_complex",
        "cohomology_class_nonzero",
        "hodge_rho_cyc",
    ]
    null_means: Dict[str, List[float]] = {m: [] for m in metrics}
    null_any_cohom: List[float] = []  # fraction of countries with nonzero class

    for p in range(n_permutations):
        shuffled = rng.permutation(pool_idx)
        cursor = 0
        per_country_vals: Dict[str, List[float]] = {m: [] for m in metrics}
        cohom_nonzero_count = 0
        for c, sz in zip(ordered_countries, sizes_list):
            user_idx = shuffled[cursor : cursor + sz]
            cursor += sz
            beta_c = prefs.beta[user_idx]
            T, W, support = majority_tournament_and_margins_from_beta(beta_c)
            inv = topological_invariants(T)
            inv.update(scalar_projection_stats(W, support))
            for m in metrics:
                per_country_vals[m].append(float(inv[m]))
            cohom_nonzero_count += int(inv["cohomology_class_nonzero"])
        for m in metrics:
            null_means[m].append(float(np.mean(per_country_vals[m])))
        null_any_cohom.append(cohom_nonzero_count / len(ordered_countries))

        if progress and (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"[null] {p+1}/{n_permutations} permutations complete")

    out = {m: np.asarray(v, dtype=np.float64) for m, v in null_means.items()}
    out["fraction_countries_cohom_nonzero"] = np.asarray(null_any_cohom, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

def compare_observed_to_null(
    observed: List[CountryResult],
    null_distributions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """One-sided z-scores and empirical p-values for each invariant."""
    metrics = [
        "condorcet_3cycles",
        "condorcet_3cycle_fraction",
        "odd_parity_4subsets",
        "odd_parity_fraction",
        "beta_1_acyclic_complex",
        "cohomology_class_nonzero",
        "hodge_rho_cyc",
    ]
    report: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        obs_vals = np.asarray([float(r.invariants[m]) for r in observed])
        obs_mean = float(obs_vals.mean())
        null = null_distributions[m]
        null_mean = float(null.mean())
        null_std = float(null.std())
        if null_std == 0:
            z = float("nan")
        else:
            z = (obs_mean - null_mean) / null_std
        # One-sided: observed exceeds null.
        p_one = float((null >= obs_mean).mean())
        report[m] = {
            "observed_mean": obs_mean,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "p_one_sided": p_one,
        }

    # Also report the cross-country fraction with nonzero cohomology class.
    obs_frac = float(np.mean([r.invariants["cohomology_class_nonzero"] for r in observed]))
    null = null_distributions["fraction_countries_cohom_nonzero"]
    null_std = float(null.std())
    z = (obs_frac - float(null.mean())) / null_std if null_std > 0 else float("nan")
    report["fraction_countries_cohom_nonzero"] = {
        "observed_mean": obs_frac,
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "z_score": z,
        "p_one_sided": float((null >= obs_frac).mean()),
    }
    return report


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_results(
    observed: List[CountryResult],
    null_distributions: Dict[str, np.ndarray],
    comparison: Dict[str, Dict[str, float]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Per-country observed.
    rows = []
    for r in observed:
        row = {"country": r.country, "n_users": r.n_users}
        row.update(r.invariants)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "per_country_observed.csv"), index=False
    )

    # Null summary.
    null_summary = {
        k: {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "n": int(v.size),
        }
        for k, v in null_distributions.items()
    }
    with open(os.path.join(out_dir, "null_summary.json"), "w") as f:
        json.dump(null_summary, f, indent=2)

    # Full null samples (for plotting).
    np.savez_compressed(
        os.path.join(out_dir, "null_samples.npz"),
        **{k: v for k, v in null_distributions.items()},
    )

    # Comparison.
    with open(os.path.join(out_dir, "aggregate_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # Raw tournaments (optional).
    tourn_dir = os.path.join(out_dir, "tournaments")
    have_tournaments = any(r.tournament is not None for r in observed)
    if have_tournaments:
        os.makedirs(tourn_dir, exist_ok=True)
        margin_dir = os.path.join(out_dir, "margins")
        support_dir = os.path.join(out_dir, "supports")
        os.makedirs(margin_dir, exist_ok=True)
        os.makedirs(support_dir, exist_ok=True)
        for r in observed:
            if r.tournament is not None:
                np.save(os.path.join(tourn_dir, f"{r.country}.npy"), r.tournament)
            if r.margin is not None:
                np.save(os.path.join(margin_dir, f"{r.country}.npy"), r.margin)
            if r.support is not None:
                np.save(os.path.join(support_dir, f"{r.country}.npy"), r.support)
