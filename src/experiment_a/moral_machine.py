"""
Moral Machine data loading and preference aggregation for Experiment A.

The public Moral Machine dataset (Awad et al. 2018; https://osf.io/3hvt2/)
distributes SharedResponsesFull.csv, ~15 GB uncompressed. Every scenario
produces two rows (one per option), with one row marked Saved=1.

This module:

    1. Streams the CSV in chunks (so the 15 GB file need not fit in memory).
    2. Aggregates each user's responses into a character-level preference
       vector beta_u in R^K, K = 20, using the paired-contrast estimator:

         beta_u[i] = (1/N_u) sum over scenarios s of user u:
            [ count_i(saved_side_s) - count_i(not_saved_side_s) ]

       where count_i is the number of characters of type i on the given
       side. This is the user-level AMCE regression coefficient under the
       simplifying assumption that sides contain independent character counts.
       The ordering of characters by beta_u is what we feed into the
       tournament-building step.

    3. Builds a per-country majority tournament T_c on K character types:
       T_c[i,j] = 1 iff more than half of users in country c have
       beta_u[i] > beta_u[j]. Ties are broken by character-type index.

We do NOT use Awad et al.'s nine aggregate moral dimensions; character
types give a finer 20-alternative tournament that carries more signal for
cycle detection. Users can swap in any alternative set by editing
CHARACTER_COLUMNS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# The 20 canonical Moral Machine character column names.
# (Column name in the Moral Machine CSV -> label used in output.)
CHARACTER_COLUMNS: List[str] = [
    "Man", "Woman",
    "OldMan", "OldWoman",
    "Boy", "Girl",
    "Pregnant", "Stroller",
    "LargeMan", "LargeWoman",
    "MaleExecutive", "FemaleExecutive",
    "MaleDoctor", "FemaleDoctor",
    "MaleAthlete", "FemaleAthlete",
    "Criminal", "Homeless",
    "Dog", "Cat",
]

# Minimum number of scenarios for a user to be included.
MIN_SCENARIOS_PER_USER = 8

# Minimum number of users for a country to be included.
MIN_USERS_PER_COUNTRY = 200


@dataclass
class UserPreferences:
    """Container for aggregated per-user preference vectors."""

    user_ids: np.ndarray           # (U,) string array
    countries: np.ndarray          # (U,) string array (ISO3)
    beta: np.ndarray               # (U, K) float array
    num_scenarios: np.ndarray      # (U,) int array
    character_names: List[str]     # length K


# ---------------------------------------------------------------------------
# Streaming load
# ---------------------------------------------------------------------------

def stream_moral_machine(
    path: str,
    chunk_size: int = 500_000,
    characters: List[str] = CHARACTER_COLUMNS,
    user_id_col: str = "UserID",
    country_col: str = "UserCountry3",
    saved_col: str = "Saved",
    extra_filter_cols: Optional[List[str]] = None,
    progress: bool = False,
) -> Iterable[pd.DataFrame]:
    """Yield chunks of the Moral Machine CSV with only the columns we need.

    The public SharedResponsesFull.csv contains occasional rows with empty
    cells in the character-count columns (bad data). We do not force int
    dtype at read time; instead we read permissively and coerce + drop bad
    rows chunk by chunk. The yielded chunk has character columns and the
    Saved column as int64.

    Parameters
    ----------
    path : path to SharedResponsesFull.csv (or a gzipped version).
    chunk_size : rows per chunk. Tune for available RAM.
    characters : which character-count columns to keep.
    user_id_col : column identifying the respondent. The Moral Machine
        public release uses "UserID"; some mirrors use "ExtendedSessionID".
    country_col : ISO3 country code column.
    saved_col : 0/1 indicator of which side was chosen.
    extra_filter_cols : any extra columns you want to retain.
    progress : if True, log per-chunk counts of dropped rows.
    """
    usecols = [user_id_col, country_col, saved_col, *characters]
    if extra_filter_cols:
        usecols = usecols + list(extra_filter_cols)

    # Only force string dtype on ID / country columns. Numeric columns are
    # read with pandas' default inference (tolerant of empty cells), then
    # coerced explicitly per-chunk below.
    dtype = {
        user_id_col: str,
        country_col: str,
    }

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        compression="infer",
        low_memory=False,
    )

    numeric_cols = [saved_col, *characters]
    for chunk_idx, chunk in enumerate(reader):
        n_raw = len(chunk)

        # Coerce numeric columns: bad cells (empty, non-numeric) become NaN.
        for c in numeric_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # Drop rows missing any essential field.
        chunk = chunk.dropna(subset=[user_id_col, country_col, *numeric_cols])

        # Only accept Saved in {0, 1}.
        chunk = chunk[chunk[saved_col].isin([0, 1])]

        if chunk.empty:
            if progress:
                print(f"[stream] chunk {chunk_idx}: 0 valid rows (raw={n_raw})")
            continue

        # Down-cast to compact integer dtypes for downstream efficiency.
        chunk[saved_col] = chunk[saved_col].astype(np.int8)
        for c in characters:
            chunk[c] = chunk[c].astype(np.int16)

        n_kept = len(chunk)
        if progress and n_kept != n_raw:
            print(
                f"[stream] chunk {chunk_idx}: kept {n_kept:,} / {n_raw:,} "
                f"(dropped {n_raw - n_kept:,} malformed rows)"
            )
        yield chunk


# ---------------------------------------------------------------------------
# Per-user preference vectors
# ---------------------------------------------------------------------------

def aggregate_user_preferences(
    chunks: Iterable[pd.DataFrame],
    characters: List[str] = CHARACTER_COLUMNS,
    user_id_col: str = "UserID",
    country_col: str = "UserCountry3",
    saved_col: str = "Saved",
    min_scenarios: int = MIN_SCENARIOS_PER_USER,
    progress: bool = True,
) -> UserPreferences:
    """Aggregate streamed chunks into per-user beta vectors.

    Implements the paired-contrast estimator. Each scenario contributes two
    rows, one saved and one not saved, so the per-scenario contribution is
    count_saved - count_not_saved. This can be computed row-wise as
    +count for Saved=1 rows and -count for Saved=0 rows; no within-chunk row
    pairing is required, which is important because chunk boundaries can split
    scenario pairs.

    The aggregation carries two sums per (user, character):

        sum_signed[u, i]  = sum over scenarios of (count_i_saved - count_i_notsaved)
        sum_scenarios[u]  = number of saved-side rows for user u

    The beta vector is sum_signed[u] / sum_scenarios[u].
    """
    # Running accumulators, keyed by user_id.
    # We use dicts of numpy arrays; merge at the end.
    signed_sums: dict = {}          # user_id -> np.ndarray(K,)
    scenario_counts: dict = {}      # user_id -> int
    user_country: dict = {}         # user_id -> ISO3

    K = len(characters)
    total_rows = 0

    for chunk_idx, chunk in enumerate(chunks):
        total_rows += len(chunk)
        if progress:
            print(
                f"[load] chunk {chunk_idx}: {len(chunk):>8,} rows; "
                f"cumulative {total_rows:>12,}"
            )

        signs = np.where(chunk[saved_col].to_numpy(dtype=np.int8) == 1, 1, -1)
        signed_counts = chunk[characters].to_numpy(dtype=np.int64) * signs[:, None]

        # Aggregate by user via pandas groupby for speed.
        df_diff = pd.DataFrame(signed_counts, columns=characters)
        df_diff["__uid"] = chunk[user_id_col].to_numpy()
        df_diff["__country"] = chunk[country_col].to_numpy()
        df_diff["__saved"] = chunk[saved_col].to_numpy(dtype=np.int8)
        grouped_sum = df_diff.groupby("__uid", sort=False)[characters].sum()
        grouped_count = df_diff.groupby("__uid", sort=False)["__saved"].sum()
        grouped_country = df_diff.groupby("__uid", sort=False)["__country"].first()

        for uid in grouped_sum.index:
            diff_vec = grouped_sum.loc[uid].values.astype(np.int64)
            c = int(grouped_count.loc[uid])
            if c <= 0:
                continue
            signed_sums[uid] = signed_sums.get(uid, np.zeros(K, dtype=np.int64)) + diff_vec
            scenario_counts[uid] = scenario_counts.get(uid, 0) + c
            user_country.setdefault(uid, grouped_country.loc[uid])

    if not signed_sums:
        raise RuntimeError("No user data aggregated; check CSV format and column names.")

    # Filter users with insufficient scenarios.
    keep_uids = [uid for uid, cnt in scenario_counts.items() if cnt >= min_scenarios]
    keep_uids.sort()
    U = len(keep_uids)
    if progress:
        print(f"[load] kept {U:,} users with >= {min_scenarios} scenarios")

    beta = np.zeros((U, K), dtype=np.float64)
    num_scen = np.zeros(U, dtype=np.int64)
    countries_out = np.empty(U, dtype=object)
    for i, uid in enumerate(keep_uids):
        cnt = scenario_counts[uid]
        beta[i] = signed_sums[uid].astype(np.float64) / cnt
        num_scen[i] = cnt
        countries_out[i] = user_country[uid]

    return UserPreferences(
        user_ids=np.asarray(keep_uids, dtype=object),
        countries=countries_out,
        beta=beta,
        num_scenarios=num_scen,
        character_names=list(characters),
    )


# ---------------------------------------------------------------------------
# Per-country majority tournaments
# ---------------------------------------------------------------------------

def majority_tournament_and_margins_from_beta(
    beta_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Majority tournament, signed margins, and edge supports for a user block.

    ``T`` is the binary tournament used for finite cycle/parity invariants.
    ``W`` is the skew-symmetric majority-margin edge flow used for scalar
    projection loss:

        W[i, j] = (wins_i - wins_j) / (wins_i + wins_j).

    Per-user ties are excluded from the denominator. If every user ties a pair,
    that edge receives zero support and zero margin. Binary tournament ties are
    still broken by character-type index so the topological routines receive a
    complete tournament, but those tie-broken edges carry no projection weight.
    """
    if beta_c.ndim != 2:
        raise ValueError(f"beta_c must be a 2D array, got shape {beta_c.shape}")
    K = beta_c.shape[1]
    T = np.zeros((K, K), dtype=np.int8)
    W = np.zeros((K, K), dtype=np.float64)
    support = np.zeros((K, K), dtype=np.float64)

    for i in range(K):
        for j in range(i + 1, K):
            diff = beta_c[:, i] - beta_c[:, j]
            wins_i = int((diff > 0).sum())
            wins_j = int((diff < 0).sum())
            edge_support = wins_i + wins_j

            if wins_i > wins_j:
                T[i, j] = 1
                T[j, i] = 0
            elif wins_j > wins_i:
                T[i, j] = 0
                T[j, i] = 1
            else:
                T[i, j] = 1
                T[j, i] = 0

            if edge_support > 0:
                margin = (wins_i - wins_j) / edge_support
                W[i, j] = margin
                W[j, i] = -margin
                support[i, j] = edge_support
                support[j, i] = edge_support

    return T, W, support


def country_majority_tournament_and_margins(
    prefs: UserPreferences,
    country: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Majority tournament, signed margins, and supports for one country.

    Returns None if the country has fewer users than MIN_USERS_PER_COUNTRY.
    """
    mask = prefs.countries == country
    U_c = int(mask.sum())
    if U_c < MIN_USERS_PER_COUNTRY:
        return None

    beta_c = prefs.beta[mask]                     # (U_c, K)
    return majority_tournament_and_margins_from_beta(beta_c)


def country_majority_tournament(
    prefs: UserPreferences,
    country: str,
) -> Optional[np.ndarray]:
    """Majority tournament on character types for users in one country.

    T[i,j] = 1 iff more users in this country have beta_u[i] > beta_u[j]
    than beta_u[j] > beta_u[i]. Exact ties are broken by character-type
    index. This wrapper is kept for older code that only needs directions.
    """
    result = country_majority_tournament_and_margins(prefs, country)
    if result is None:
        return None
    T, _, _ = result
    return T


def eligible_countries(prefs: UserPreferences) -> List[str]:
    """Countries with at least MIN_USERS_PER_COUNTRY respondents."""
    counts = pd.Series(prefs.countries).value_counts()
    keep = counts[counts >= MIN_USERS_PER_COUNTRY].index.tolist()
    # Drop missing / unknown country codes.
    keep = [c for c in keep if c and c != "NaN" and c != "UNK"]
    return sorted(keep)


# ---------------------------------------------------------------------------
# Synthetic data generator (for pipeline testing without the real dataset)
# ---------------------------------------------------------------------------

def synthesise_moral_machine(
    out_path: str,
    n_users: int = 5_000,
    n_scenarios_per_user: int = 12,
    n_countries: int = 40,
    seed: int = 0,
    characters: List[str] = CHARACTER_COLUMNS,
) -> str:
    """Generate a plausible Moral Machine-format CSV for pipeline testing.

    Each country has a random "latent preference profile" over characters;
    users draw perturbed versions. Scenarios are random 3-on-3 matchups; the
    saved side is the one with higher latent profile inner product plus
    noise. The generated CSV has exactly the schema our loader expects.

    Returns the path written.
    """
    rng = np.random.default_rng(seed)
    K = len(characters)
    # Country profiles, each a standard normal vector.
    country_codes = [f"C{i:03d}" for i in range(n_countries)]
    country_profiles = rng.standard_normal((n_countries, K))
    user_country_idx = rng.integers(0, n_countries, size=n_users)
    user_profiles = country_profiles[user_country_idx] + 0.5 * rng.standard_normal(
        (n_users, K)
    )

    rows = []
    for u in range(n_users):
        uid = f"U{u:06d}"
        country = country_codes[user_country_idx[u]]
        profile = user_profiles[u]
        for s in range(n_scenarios_per_user):
            # Two random 3-character sides.
            side_a = np.zeros(K, dtype=np.int8)
            side_b = np.zeros(K, dtype=np.int8)
            idx_a = rng.integers(0, K, size=3)
            idx_b = rng.integers(0, K, size=3)
            for k in idx_a:
                side_a[k] += 1
            for k in idx_b:
                side_b[k] += 1
            # Saved side determined by utility + noise.
            util_a = float(profile @ side_a)
            util_b = float(profile @ side_b)
            save_a = (util_a - util_b + 0.3 * rng.standard_normal()) > 0
            saved_a = 1 if save_a else 0
            saved_b = 1 - saved_a
            rows.append([uid, country, saved_a, *side_a.tolist()])
            rows.append([uid, country, saved_b, *side_b.tolist()])

    df = pd.DataFrame(
        rows, columns=["UserID", "UserCountry3", "Saved", *characters]
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
