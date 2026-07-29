"""
Moral Machine Condorcet Triple Analysis — FIXED for real data schema
====================================================================

Key fixes from v1:
  - ScenarioType "Social Status" not "Social"
  - Intervention field = which SIDE is the swerve outcome, not user action
  - User action is: which Saved=1 row's side they chose
  - Cross-chunk pairing: we DON'T pair rows. Instead, each row is one
    outcome side. Saved=1 means the user chose to save THIS side's characters.
    We only need the Saved=1 row to determine the user's choice along each
    dimension, using that row's AttributeLevel + structural fields.
  - This means we process ONLY Saved=1 rows (half the data), no pairing needed.
  - For Utilitarian (more lives), we DO need both rows to compare character
    counts. We handle this by a second pass or by using DiffNumberOFCharacters.

USAGE: python moral_machine_condorcet.py --data /path/to/SharedResponses.csv
"""

import os
import json
import argparse
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "./SharedResponses.csv"
OUTPUT_DIR = "./mm_output"

CHUNK_SIZE = 2_000_000
MIN_SCENARIOS_PER_USER = 8
MIN_DIMS_COVERED = 4
MIN_USERS_PER_GROUP = 100
N_ALTERNATIVES = 50
N_PERMUTATIONS = 500
RANDOM_SEED = 42

# 9 Moral Machine dimensions
DIMENSIONS = [
    "Species",        # humans vs pets
    "Utilitarian",    # more vs fewer lives
    "Age",            # young vs elderly
    "Law",            # lawful crossing vs unlawful
    "Social Status",  # high vs low status
    "Gender",         # female vs male
    "Fitness",        # fit vs large/fat
    "Pedestrian",     # pedestrians vs passengers
    "Action",         # inaction (stay) vs action (swerve)
]
N_DIM = len(DIMENSIONS)

# AttributeLevel values that map to the "+1" (preferred) direction per
# dimension, from the real data values shown above.
# The "+1" direction follows Awad et al. (2018) Fig. 2 sign convention.
PREFERRED_ATTR = {
    "Species":       "Hoomans",   # humans preferred over pets
    "Utilitarian":   "More",      # more lives preferred
    "Age":           "Young",     # young preferred
    "Social Status": "High",      # high status preferred
    "Gender":        "Female",    # female weakly preferred
    "Fitness":       "Fit",       # fit preferred
}
# The opposite levels:
OPPOSITE_ATTR = {
    "Species":       "Pets",
    "Utilitarian":   "Less",
    "Age":           "Old",
    "Social Status": "Low",
    "Gender":        "Male",
    "Fitness":       "Fat",
}

USECOLS = [
    'ResponseID', 'UserID', 'ScenarioType', 'ScenarioTypeStrict',
    'AttributeLevel', 'Intervention', 'PedPed', 'Barrier',
    'CrossingSignal', 'Saved', 'NumberOfCharacters',
    'DiffNumberOFCharacters', 'UserCountry3'
]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 200, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'serif',
})
COLORS = ['#2166ac', '#b2182b', '#4dac26', '#7570b3', '#d95f02']


# =============================================================================
# DATA INGESTION — single-row processing (no pairing needed)
# =============================================================================

def accumulate_user_choices(path):
    """
    Stream the CSV and accumulate per-user choice counts.

    KEY INSIGHT: We don't need to pair rows. Each row represents one outcome
    side of a scenario. Saved=1 means the user chose to save THAT side.
    So we only process Saved=1 rows: the AttributeLevel on that row tells
    us which attribute level the user chose to save.

    For the principal dimensions (Species, Age, Gender, Fitness, Social Status):
      - The Saved=1 row's AttributeLevel tells us the user's choice directly.
      - If AttributeLevel == preferred level → user chose +1 direction.

    For Utilitarian: The Saved=1 row's NumberOfCharacters vs the NOT-saved
    row's NumberOfCharacters tells us if more or fewer were saved. We use
    DiffNumberOFCharacters + NumberOfCharacters to infer this without
    needing the other row.

    For structural dimensions (Action, Pedestrian, Law):
      - Action: Intervention field on the Saved=1 row tells us if the saved
        side was the swerve side (Intervention=1) or stay side (Intervention=0).
        If user saved the stay side → inaction → +1.
      - Pedestrian: Barrier field. If Barrier=1 on Saved=1 row and PedPed=0,
        user saved the side behind the barrier (passengers). If Barrier=0,
        user saved the pedestrian side. Wait — need to check this logic.
        Actually: Barrier=1 means there IS a barrier. The barrier kills the
        passengers if the car stays. So if Saved=1 row has Barrier=0, the
        saved characters are pedestrians; Barrier=1, saved are passengers.
        Hmm, this needs careful checking against the MM documentation.
        Let's use PedPed: if PedPed=1, both sides are pedestrians (no
        passenger dimension). If PedPed=0, one side has passengers. The
        Saved=1 row represents who was saved. We need to know if they were
        pedestrians or passengers. Unfortunately this isn't directly in
        AttributeLevel for non-PedPed scenarios.
      - Law: CrossingSignal on the Saved=1 row: 1=lawful, 2=unlawful, 0=none.
    """
    # Accumulate into arrays indexed by user integer ID
    user_to_idx = {}
    user_country = {}
    capacity = 200_000
    pos = np.zeros((capacity, N_DIM), dtype=np.int32)
    tot = np.zeros((capacity, N_DIM), dtype=np.int32)

    def _ensure_capacity(needed):
        nonlocal capacity, pos, tot
        if needed > capacity:
            new_cap = max(needed, capacity * 2)
            new_pos = np.zeros((new_cap, N_DIM), dtype=np.int32)
            new_tot = np.zeros((new_cap, N_DIM), dtype=np.int32)
            new_pos[:capacity] = pos
            new_tot[:capacity] = tot
            pos, tot, capacity = new_pos, new_tot, new_cap

    print(f"Streaming {path} in chunks of {CHUNK_SIZE:,} rows...")

    for chunk_idx, chunk in enumerate(pd.read_csv(
            path, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False)):
        if chunk_idx % 5 == 0:
            print(f"  chunk {chunk_idx}: ~{(chunk_idx+1)*CHUNK_SIZE:,} rows; "
                  f"{len(user_to_idx):,} users so far")

        # ONLY process Saved=1 rows
        saved = chunk[chunk['Saved'] == 1].copy()
        if len(saved) == 0:
            continue

        # Get user indices
        user_arr = saved['UserID'].values
        new_users = set()
        idxs = np.empty(len(user_arr), dtype=np.int64)
        for i, u in enumerate(user_arr):
            if u in user_to_idx:
                idxs[i] = user_to_idx[u]
            else:
                idx = len(user_to_idx)
                user_to_idx[u] = idx
                idxs[i] = idx
                new_users.add(u)
        _ensure_capacity(len(user_to_idx))

        # Update country mapping for new users only
        if new_users:
            country_arr = saved['UserCountry3'].values
            for u, c in zip(user_arr, country_arr):
                if u in new_users and isinstance(c, str):
                    user_country[u] = c

        # ---- Principal dimensions: Species, Age, Gender, Fitness, Social Status ----
        stype_arr = saved['ScenarioType'].values
        sstrict_arr = saved['ScenarioTypeStrict'].values
        attr_arr = saved['AttributeLevel'].values

        for dim_name in ['Species', 'Age', 'Gender', 'Fitness', 'Social Status']:
            pref = PREFERRED_ATTR[dim_name]
            opp = OPPOSITE_ATTR[dim_name]
            k = DIMENSIONS.index(dim_name)

            # Match on ScenarioType (use both columns for robustness)
            mask = (stype_arr == dim_name) | (sstrict_arr == dim_name)
            # Further filter to rows where AttributeLevel is one of the
            # two known values for this dimension
            valid_attr = np.array([(a == pref or a == opp) if isinstance(a, str)
                                   else False for a in attr_arr])
            mask = mask & valid_attr
            if not mask.any():
                continue

            sub_users = idxs[mask]
            sub_attr = attr_arr[mask]
            chose_pref = (sub_attr == pref)

            np.add.at(tot, (sub_users, k), 1)
            np.add.at(pos, (sub_users[chose_pref], k), 1)

        # ---- Utilitarian: did user save more or fewer characters? ----
        # On the Saved=1 row: if DiffNumberOFCharacters > 0, the two sides
        # differ in count. But DiffNumberOFCharacters is |n1 - n2|, so we
        # need to know if the saved side had MORE. Use AttributeLevel:
        # if ScenarioType is Utilitarian, AttributeLevel is "More" or "Less".
        util_mask = (stype_arr == 'Utilitarian') | (sstrict_arr == 'Utilitarian')
        util_valid = np.array([(a == 'More' or a == 'Less') if isinstance(a, str)
                               else False for a in attr_arr])
        util_mask = util_mask & util_valid
        if util_mask.any():
            k = DIMENSIONS.index("Utilitarian")
            sub_users = idxs[util_mask]
            sub_attr = attr_arr[util_mask]
            chose_more = (sub_attr == 'More')
            np.add.at(tot, (sub_users, k), 1)
            np.add.at(pos, (sub_users[chose_more], k), 1)

        # ---- Action: did user choose to stay (inaction) or swerve? ----
        # Intervention on the Saved=1 row: 0 = this is the "stay" outcome,
        # 1 = this is the "swerve" outcome.
        # If user saved the stay side (Intervention=0) → chose inaction → +1
        # If user saved the swerve side (Intervention=1) → chose action → -1
        intervention = saved['Intervention'].values
        valid_int = ~pd.isna(intervention)
        if valid_int.any():
            k = DIMENSIONS.index("Action")
            sub_users = idxs[valid_int]
            sub_intv = intervention[valid_int].astype(np.int32)
            np.add.at(tot, (sub_users, k), 1)
            np.add.at(pos, (sub_users[sub_intv == 0], k), 1)

        # ---- Law: CrossingSignal on the saved row ----
        cs = saved['CrossingSignal'].values
        law_valid = (~pd.isna(cs)) & ((cs == 1) | (cs == 2))
        if law_valid.any():
            k = DIMENSIONS.index("Law")
            sub_users = idxs[law_valid]
            sub_cs = cs[law_valid].astype(np.int32)
            np.add.at(tot, (sub_users, k), 1)
            np.add.at(pos, (sub_users[sub_cs == 1], k), 1)

        # ---- Pedestrian vs Passenger ----
        # PedPed=0 means one side has passengers (behind barrier).
        # This is tricky: the Saved=1 row doesn't directly say "pedestrian"
        # or "passenger" in AttributeLevel for non-PedPed scenarios.
        # Skip this dimension for now (or use Barrier field).
        # If PedPed=0 and Barrier=1 on the Saved row: the saved characters
        # are the ones NOT hit by the barrier — i.e., pedestrians.
        # If Barrier=0 on the Saved row: saved characters are behind the
        # barrier — passengers.
        # ACTUALLY: in the MM design, Barrier=1 means there is a barrier
        # between the car and the pedestrians. If the car stays, it hits
        # the barrier and kills the passengers. If PedPed=0:
        #   Barrier on Saved=1 row: unclear without the full MM code.
        # For safety, let's use a conservative approach: only score this
        # dimension when ScenarioType explicitly involves PedPed variants.
        # We skip Pedestrian for now to avoid misclassification.

    # Trim
    n_users = len(user_to_idx)
    pos = pos[:n_users]
    tot = tot[:n_users]
    print(f"\n  Total unique users encountered: {n_users:,}")

    return pos, tot, user_to_idx, user_country


def estimate_betas(pos, tot, user_to_idx):
    """
    Per-user signed-AMCE vectors.
    beta_i^k = (pos_i^k / tot_i^k) - 0.5   [in [-0.5, 0.5]]
    NaN if tot_i^k == 0.
    """
    n_users = len(user_to_idx)
    betas = np.full((n_users, N_DIM), np.nan)
    total_scenarios = tot.sum(axis=1)
    dims_covered = (tot > 0).sum(axis=1)

    keep = (total_scenarios >= MIN_SCENARIOS_PER_USER) & \
           (dims_covered >= MIN_DIMS_COVERED)

    for k in range(N_DIM):
        has_data = (tot[:, k] > 0) & keep
        betas[has_data, k] = pos[has_data, k] / tot[has_data, k] - 0.5

    n_kept = keep.sum()
    print(f"  {n_kept:,} users retained (>= {MIN_SCENARIOS_PER_USER} scenarios, "
          f">= {MIN_DIMS_COVERED} dims)")
    return betas, keep


# =============================================================================
# CONDORCET DETECTION (unchanged from v1)
# =============================================================================

def generate_alternatives(M, dim, rng):
    return rng.choice([-1, +1], size=(M, dim))


def group_majorities_over_alternatives(betas_group, alternatives):
    M = alternatives.shape[0]
    n = betas_group.shape[0]
    if n == 0:
        return None
    B = np.where(np.isnan(betas_group), 0.0, betas_group)
    U = B @ alternatives.T  # (n, M)
    maj = np.zeros((M, M), dtype=np.int8)
    for i in range(M):
        diff = U[:, i:i+1] - U[:, i+1:]   # (n, M-i-1)
        wins_i = (diff > 0).sum(axis=0)
        wins_j = (diff < 0).sum(axis=0)
        for offset, (wi, wj) in enumerate(zip(wins_i, wins_j)):
            j = i + 1 + offset
            if wi > wj:
                maj[i, j] = 1; maj[j, i] = -1
            elif wj > wi:
                maj[i, j] = -1; maj[j, i] = 1
    return maj


def count_condorcet_triples(maj, M):
    cycles = 0
    total = 0
    for i, j, k in combinations(range(M), 3):
        ij, jk, ki = maj[i, j], maj[j, k], maj[k, i]
        if ij == 0 or jk == 0 or ki == 0:
            continue
        total += 1
        if (ij == 1 and jk == 1 and ki == 1) or \
           (ij == -1 and jk == -1 and ki == -1):
            cycles += 1
    return cycles, total


def run_condorcet_analysis(betas, keep, user_to_idx, user_country, args, rng):
    """Main Condorcet analysis after betas are estimated."""
    n_users = len(user_to_idx)
    idx_to_user = [None] * n_users
    for u, i in user_to_idx.items():
        idx_to_user[i] = u

    # Build country groups
    by_country = defaultdict(list)
    for i in range(n_users):
        if not keep[i]:
            continue
        u = idx_to_user[i]
        c = user_country.get(u)
        if isinstance(c, str):
            by_country[c].append(i)

    valid_countries = {c: idxs for c, idxs in by_country.items()
                       if len(idxs) >= args.min_users_per_group}
    print(f"\n  {len(valid_countries)} countries with >= {args.min_users_per_group} users")
    if len(valid_countries) == 0:
        print("  ERROR: No valid country groups. Try lowering --min-users-per-group.")
        return

    # Print top countries
    sorted_countries = sorted(valid_countries.items(), key=lambda x: -len(x[1]))
    print("  Top 10 countries:")
    for c, idxs in sorted_countries[:10]:
        print(f"    {c}: {len(idxs):,} users")

    # Generate alternatives
    alternatives = generate_alternatives(args.n_alternatives, N_DIM, rng)
    print(f"\n  Generated {args.n_alternatives} synthetic moral profiles")

    # Per-country majorities
    print("  Computing per-country majorities...")
    country_majorities = {}
    per_country_rates = {}
    total_cycles = 0
    total_triples = 0
    for c, idxs in valid_countries.items():
        grp_betas = betas[np.array(idxs)]
        maj = group_majorities_over_alternatives(grp_betas, alternatives)
        if maj is None:
            continue
        country_majorities[c] = maj
        cyc, tri = count_condorcet_triples(maj, args.n_alternatives)
        per_country_rates[c] = cyc / tri if tri > 0 else 0
        total_cycles += cyc
        total_triples += tri

    observed_freq = total_cycles / total_triples if total_triples > 0 else 0
    print(f"\n  Observed Condorcet frequency: {observed_freq:.4f} "
          f"({total_cycles} / {total_triples})")

    # Permutation null
    print(f"  Computing permutation null ({args.n_permutations} permutations)...")
    all_valid_idxs = []
    group_sizes = []
    for c in valid_countries:
        all_valid_idxs.extend(valid_countries[c])
        group_sizes.append(len(valid_countries[c]))
    all_valid_idxs = np.array(all_valid_idxs)

    null_freqs = np.zeros(args.n_permutations)
    for p in range(args.n_permutations):
        if p % max(1, args.n_permutations // 10) == 0:
            print(f"    permutation {p}/{args.n_permutations}")
        perm = rng.permutation(len(all_valid_idxs))
        shuffled = all_valid_idxs[perm]
        idx = 0
        t_cyc, t_tri = 0, 0
        for size in group_sizes:
            grp_idxs = shuffled[idx: idx + size]
            idx += size
            grp_betas = betas[grp_idxs]
            maj = group_majorities_over_alternatives(grp_betas, alternatives)
            if maj is None:
                continue
            c_, t_ = count_condorcet_triples(maj, args.n_alternatives)
            t_cyc += c_
            t_tri += t_
        null_freqs[p] = t_cyc / t_tri if t_tri > 0 else 0

    null_mean = null_freqs.mean()
    null_std = null_freqs.std()
    z_score = (observed_freq - null_mean) / max(null_std, 1e-10)
    p_value = max(np.mean(null_freqs >= observed_freq), 1.0 / args.n_permutations)

    print(f"\n  Null: {null_mean:.4f} ± {null_std:.4f}")
    print(f"  Z-score: {z_score:.2f}, p < {p_value:.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax = axes[0]
    ax.hist(null_freqs, bins=30, color='#cccccc', edgecolor='black',
            linewidth=0.4, alpha=0.85,
            label='Permutation null\n(country labels shuffled)')
    ax.axvline(observed_freq, color=COLORS[1], linewidth=2.5,
               label=f'Observed ({observed_freq:.4f})')
    ax.set_xlabel('Mean Condorcet triple frequency')
    ax.set_ylabel('Permutation count')
    ax.set_title(f'(a) Cross-country cycles vs. noise\n'
                 f'(z = {z_score:.1f}, p < {p_value:.3f})')
    ax.legend(fontsize=7)

    ax = axes[1]
    rates = sorted(per_country_rates.values(), reverse=True)
    ax.bar(range(len(rates)), rates, color=COLORS[0], alpha=0.85,
           edgecolor='black', linewidth=0.3)
    ax.axhline(y=null_mean, color='gray', linestyle=':', alpha=0.7,
               label=f'Null mean ({null_mean:.4f})')
    ax.set_xlabel('Country (sorted)')
    ax.set_ylabel('Within-country Condorcet freq')
    ax.set_title(f'(b) Per-country distribution ({len(rates)} countries)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_pdf = os.path.join(args.output, 'moral_machine_condorcet.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure to {out_pdf}")

    # Mean beta per country for diagnostics
    print("\n  Mean beta per country (top 5 by user count):")
    for c, idxs in sorted_countries[:5]:
        if c not in valid_countries:
            continue
        grp_betas = betas[np.array(valid_countries[c])]
        mean_b = np.nanmean(grp_betas, axis=0)
        print(f"    {c} (n={len(valid_countries[c])}): " +
              " ".join(f"{DIMENSIONS[k][:4]}={mean_b[k]:+.3f}" for k in range(N_DIM)))

    # Save JSON
    results = {
        'n_users_total': int(keep.sum()),
        'n_groups': len(valid_countries),
        'observed_freq': float(observed_freq),
        'total_cycles': int(total_cycles),
        'total_triples': int(total_triples),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'top_countries': [(c, float(r)) for c, r in
                          sorted(per_country_rates.items(), key=lambda x: -x[1])[:20]],
    }
    out_json = os.path.join(args.output, 'mm_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--n-alternatives', type=int, default=N_ALTERNATIVES)
    parser.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS)
    parser.add_argument('--min-users-per-group', type=int, default=MIN_USERS_PER_GROUP)
    parser.add_argument('--cache', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached data from {args.cache}")
        cache = np.load(args.cache, allow_pickle=True)
        pos_arr = cache['pos']
        tot_arr = cache['tot']
        user_to_idx = cache['user_to_idx'].item()
        user_country = cache['user_country'].item()
        betas, keep = estimate_betas(pos_arr, tot_arr, user_to_idx)
    else:
        pos_arr, tot_arr, user_to_idx, user_country = accumulate_user_choices(args.data)
        betas, keep = estimate_betas(pos_arr, tot_arr, user_to_idx)
        if args.cache:
            print(f"Caching to {args.cache}")
            np.savez(args.cache, pos=pos_arr, tot=tot_arr,
                     user_to_idx=user_to_idx, user_country=user_country)

    run_condorcet_analysis(betas, keep, user_to_idx, user_country, args, rng)


if __name__ == '__main__':
    main()