"""
Command-line entry point for Experiment A.

Typical usage:

    # End-to-end test on auto-generated synthetic data (no download needed):
    python run.py --synthetic --out results_synthetic

    # Real Moral Machine data:
    python run.py --csv path/to/SharedResponsesFull.csv \
                  --out results_moral_machine \
                  --permutations 500

See `python run.py --help` for all options.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

from experiment import (
    compare_observed_to_null,
    compute_observed,
    run_permutation_null,
    write_results,
)
from moral_machine import (
    MIN_USERS_PER_COUNTRY,
    UserPreferences,
    aggregate_user_preferences,
    eligible_countries,
    stream_moral_machine,
    synthesise_moral_machine,
)


def load_user_preferences_cache(path: str) -> UserPreferences:
    """Load cached per-user preferences written by this script."""
    with np.load(path, allow_pickle=True) as data:
        return UserPreferences(
            user_ids=data["user_ids"],
            countries=data["countries"],
            beta=data["beta"],
            num_scenarios=data["num_scenarios"],
            character_names=list(data["character_names"].tolist()),
        )


def save_user_preferences_cache(path: str, prefs: UserPreferences) -> None:
    """Save the expensive streamed aggregation for later reruns."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        user_ids=prefs.user_ids,
        countries=prefs.countries,
        beta=prefs.beta,
        num_scenarios=prefs.num_scenarios,
        character_names=np.asarray(prefs.character_names, dtype=object),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment A: cycle/parity invariants and scalar projection loss "
            "in Moral Machine majority tournaments."
        )
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--csv",
        type=str,
        help="Path to SharedResponsesFull.csv (the Moral Machine dataset).",
    )
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic Moral Machine-format data and run end-to-end.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results_experiment_a",
        help="Directory to write results into.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of permutation-null draws (default: 500).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="CSV chunk size for streaming (rows).",
    )
    parser.add_argument(
        "--min-scenarios",
        type=int,
        default=8,
        help="Min number of scenarios per user to include (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for null-model permutations.",
    )
    parser.add_argument(
        "--save-tournaments",
        action="store_true",
        help="Save raw per-country tournament, margin, and support matrices.",
    )
    parser.add_argument(
        "--prefs-cache",
        type=str,
        default=None,
        help=(
            "Optional cache for aggregated per-user preferences. If the file "
            "exists it is loaded; otherwise it is written after streaming."
        ),
    )
    parser.add_argument(
        "--user-id-col",
        type=str,
        default="UserID",
        help="Column name identifying respondents "
             "(public Moral Machine release: UserID; some mirrors: ExtendedSessionID).",
    )
    parser.add_argument(
        "--country-col",
        type=str,
        default="UserCountry3",
        help="Column name for ISO3 country code.",
    )
    parser.add_argument(
        "--saved-col",
        type=str,
        default="Saved",
        help="Column name for the 0/1 saved indicator.",
    )
    parser.add_argument(
        "--synthetic-users",
        type=int,
        default=5_000,
        help="Number of synthetic users when --synthetic.",
    )
    parser.add_argument(
        "--synthetic-countries",
        type=int,
        default=40,
        help="Number of synthetic countries when --synthetic.",
    )
    parser.add_argument(
        "--synthetic-scenarios",
        type=int,
        default=12,
        help="Scenarios per synthetic user when --synthetic.",
    )
    args = parser.parse_args()

    # --- Resolve input CSV --------------------------------------------------
    if args.synthetic:
        tmp_dir = tempfile.mkdtemp(prefix="mm_synth_")
        csv_path = os.path.join(tmp_dir, "synthetic_moral_machine.csv")
        print(f"[synth] generating {args.synthetic_users:,} users in {args.synthetic_countries} "
              f"countries at {csv_path}")
        synthesise_moral_machine(
            csv_path,
            n_users=args.synthetic_users,
            n_scenarios_per_user=args.synthetic_scenarios,
            n_countries=args.synthetic_countries,
            seed=args.seed,
        )
    else:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
            return 2

    # --- Load and aggregate -------------------------------------------------
    t0 = time.time()
    if args.prefs_cache and os.path.exists(args.prefs_cache):
        print(f"[cache] loading aggregated user preferences from {args.prefs_cache}")
        prefs = load_user_preferences_cache(args.prefs_cache)
    else:
        chunks = stream_moral_machine(
            csv_path,
            chunk_size=args.chunk_size,
            user_id_col=args.user_id_col,
            country_col=args.country_col,
            saved_col=args.saved_col,
            progress=True,
        )
        prefs = aggregate_user_preferences(
            chunks,
            user_id_col=args.user_id_col,
            country_col=args.country_col,
            saved_col=args.saved_col,
            min_scenarios=args.min_scenarios,
            progress=True,
        )
        if args.prefs_cache:
            print(f"[cache] saving aggregated user preferences to {args.prefs_cache}")
            save_user_preferences_cache(args.prefs_cache, prefs)
    t1 = time.time()
    print(
        f"[load] {prefs.beta.shape[0]:,} users in {len(set(prefs.countries)):,} "
        f"raw country labels; elapsed {t1 - t0:.1f}s"
    )

    countries = eligible_countries(prefs)
    print(
        f"[load] {len(countries)} countries with >= {MIN_USERS_PER_COUNTRY} users"
    )
    if len(countries) < 3:
        print(
            "ERROR: fewer than 3 eligible countries. "
            "Reduce MIN_USERS_PER_COUNTRY in moral_machine.py or use more data.",
            file=sys.stderr,
        )
        return 3

    # --- Observed invariants ------------------------------------------------
    t2 = time.time()
    observed = compute_observed(
        prefs,
        countries=countries,
        save_tournaments=args.save_tournaments,
        progress=True,
    )
    t3 = time.time()
    print(f"[observed] elapsed {t3 - t2:.1f}s over {len(observed)} countries")

    # --- Permutation null ---------------------------------------------------
    t4 = time.time()
    null_distributions = run_permutation_null(
        prefs,
        observed_countries=[r.country for r in observed],
        n_permutations=args.permutations,
        seed=args.seed,
        progress=True,
    )
    t5 = time.time()
    print(f"[null] elapsed {t5 - t4:.1f}s over {args.permutations} permutations")

    # --- Compare ------------------------------------------------------------
    comparison = compare_observed_to_null(observed, null_distributions)

    print("\n================= SUMMARY =================")
    for key, stats in comparison.items():
        print(
            f"  {key:<42s}  obs={stats['observed_mean']:.4g}  "
            f"null={stats['null_mean']:.4g}+/-{stats['null_std']:.4g}  "
            f"z={stats['z_score']:+.2f}  p1={stats['p_one_sided']:.3g}"
        )
    print("===========================================\n")

    # --- Write --------------------------------------------------------------
    write_results(observed, null_distributions, comparison, args.out)
    print(f"[done] results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
