#!/usr/bin/env python3
"""Global Moral Machine direct-tournament check.

The main Moral Machine analysis computes one tournament per eligible country.
This script adds the complementary global check: pool all retained respondents
and build one 20-character-type majority tournament.

It also reports a size-matched random-partition baseline. For each null draw,
respondents from eligible countries are randomly repartitioned into groups with
the observed country sizes. The global pooled tournament is unchanged by such a
shuffle, so the null is only used to contextualize the country-level aggregate
already reported in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_A = ROOT / "src" / "experiment_a"
sys.path.insert(0, str(EXPERIMENT_A))

from experiment import scalar_projection_stats  # noqa: E402
from moral_machine import (  # noqa: E402
    MIN_USERS_PER_COUNTRY,
    UserPreferences,
    eligible_countries,
    majority_tournament_and_margins_from_beta,
)
from topology import topological_invariants  # noqa: E402


def load_user_preferences_cache(path: Path) -> UserPreferences:
    with np.load(path, allow_pickle=True) as data:
        return UserPreferences(
            user_ids=data["user_ids"],
            countries=data["countries"],
            beta=data["beta"],
            num_scenarios=data["num_scenarios"],
            character_names=list(data["character_names"].tolist()),
        )


def strict_cycle_count_from_margins(W: np.ndarray) -> int:
    cycles = 0
    for a, b, c in combinations(range(W.shape[0]), 3):
        if W[a, b] > 0 and W[b, c] > 0 and W[c, a] > 0:
            cycles += 1
        elif W[a, c] > 0 and W[c, b] > 0 and W[b, a] > 0:
            cycles += 1
    return cycles


def tournament_stats(beta: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    T, W, support = majority_tournament_and_margins_from_beta(beta)
    out = topological_invariants(T)
    out.update(scalar_projection_stats(W, support))
    margins = np.asarray(
        [W[i, j] for i in range(W.shape[0]) for j in range(i + 1, W.shape[1]) if support[i, j] > 0],
        dtype=float,
    )
    out["strict_condorcet_3cycles"] = strict_cycle_count_from_margins(W)
    out["majority_tie_edges"] = int(np.sum(np.isclose(margins, 0.0)))
    out["min_abs_edge_margin"] = float(np.min(np.abs(margins))) if margins.size else float("nan")
    return out, T


def summarize_country_observed(per_country_path: Path) -> Dict[str, float]:
    df = pd.read_csv(per_country_path)
    metrics = [
        "condorcet_3cycles",
        "condorcet_3cycle_fraction",
        "odd_parity_4subsets",
        "odd_parity_fraction",
        "cohomology_class_nonzero",
        "hodge_rho_cyc",
    ]
    return {f"country_mean_{m}": float(df[m].mean()) for m in metrics}


def random_partition_country_means(
    prefs: UserPreferences,
    countries: List[str],
    n_null: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = [int((prefs.countries == c).sum()) for c in countries]
    pool_idx = np.where(np.isin(prefs.countries, countries))[0]
    rows = []
    metrics = [
        "condorcet_3cycles",
        "condorcet_3cycle_fraction",
        "odd_parity_4subsets",
        "odd_parity_fraction",
        "cohomology_class_nonzero",
        "hodge_rho_cyc",
    ]

    for sim in range(n_null):
        shuffled = rng.permutation(pool_idx)
        cursor = 0
        vals = {m: [] for m in metrics}
        for size in sizes:
            idx = shuffled[cursor : cursor + size]
            cursor += size
            stats, _ = tournament_stats(prefs.beta[idx])
            for m in metrics:
                vals[m].append(float(stats[m]))
        row = {"sim": sim}
        for m in metrics:
            row[f"country_mean_{m}"] = float(np.mean(vals[m]))
        rows.append(row)
        if (sim + 1) % max(1, n_null // 10) == 0:
            print(f"[null] {sim + 1}/{n_null} random partitions")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefs-cache",
        type=Path,
        default=Path("results_moral_machine_projection/prefs_cache.npz"),
        help="Cached per-user direct Moral Machine preference vectors.",
    )
    parser.add_argument(
        "--per-country",
        type=Path,
        default=Path("results_moral_machine_projection/per_country_observed.csv"),
        help="Existing per-country observed table.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_moral_machine_projection"),
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=0,
        help=(
            "Optional random country-size partitions for context. Default 0 "
            "because the main country-level null is already produced by run.py."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefs = load_user_preferences_cache(args.prefs_cache)
    countries = eligible_countries(prefs)
    eligible_mask = np.isin(prefs.countries, countries)
    beta_eligible = prefs.beta[eligible_mask]

    print(
        f"[load] {prefs.beta.shape[0]:,} retained users; "
        f"{beta_eligible.shape[0]:,} in {len(countries)} countries with >= {MIN_USERS_PER_COUNTRY} users"
    )

    observed_rows = []
    for scope, beta_block, n_countries in [
        ("global_direct_all_retained", prefs.beta, int(pd.Series(prefs.countries).nunique())),
        ("global_direct_eligible_countries", beta_eligible, len(countries)),
    ]:
        global_stats, _ = tournament_stats(beta_block)
        observed_rows.append(
            {
                "scope": scope,
                "n_users": int(beta_block.shape[0]),
                "n_countries": int(n_countries),
                **global_stats,
            }
        )
    global_row = observed_rows[1]
    pd.DataFrame(observed_rows).to_csv(args.outdir / "global_direct_observed.csv", index=False)

    country_summary = summarize_country_observed(args.per_country)
    (args.outdir / "global_direct_summary.json").write_text(
        json.dumps(
            {
                "global_direct": observed_rows,
                "country_observed_means": country_summary,
                "interpretation": (
                    "The global direct tournament is reported both for all retained respondents and "
                    "for respondents in countries eligible for the country-level analysis. Country means "
                    "keep the cultural grouping used in the main Moral Machine analysis."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    if args.n_null > 0:
        null = random_partition_country_means(prefs, countries, args.n_null, args.seed)
        null.to_csv(args.outdir / "global_direct_country_partition_null.csv", index=False)

    print("\n================ GLOBAL DIRECT CHECK ================")
    print(f"users={global_row['n_users']:,}; countries={global_row['n_countries']}")
    print(
        f"eligible-country global cycles={global_row['condorcet_3cycles']} "
        f"({global_row['condorcet_3cycle_fraction']:.6g}); "
        f"strict cycles={global_row['strict_condorcet_3cycles']}; "
        f"tie edges={global_row['majority_tie_edges']}; "
        f"rho={global_row['hodge_rho_cyc']:.6g}; "
        f"odd parity={global_row['odd_parity_4subsets']}"
    )
    print(
        "country mean cycles="
        f"{country_summary['country_mean_condorcet_3cycles']:.6g}; "
        f"country mean rho={country_summary['country_mean_hodge_rho_cyc']:.6g}"
    )
    print("=====================================================\n")
    print(f"Wrote {args.outdir / 'global_direct_observed.csv'}")
    print(f"Wrote {args.outdir / 'global_direct_summary.json'}")
    if args.n_null > 0:
        print(f"Wrote {args.outdir / 'global_direct_country_partition_null.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
