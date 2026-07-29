"""
Unit tests and end-to-end sanity checks.

Run with:
    python test_experiment.py

All tests should pass; any failure indicates a regression in the topology
computation or the data pipeline.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from topology import (
    betti_1_acyclic_complex,
    cohomology_class_nonzero,
    count_condorcet_cycles,
    is_cyclic_triple,
    tetrahedral_parities,
    topological_invariants,
    validate_tournament,
    _matrix_rank_f2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transitive_tournament(n: int) -> np.ndarray:
    """0 beats 1 beats 2 ... beats n-1. No cycles."""
    T = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i < j:
                T[i, j] = 1
    return T


def cyclic_triple_tournament() -> np.ndarray:
    """n=3 with a single Condorcet cycle 0->1->2->0."""
    T = np.zeros((3, 3), dtype=np.int8)
    T[0, 1] = 1
    T[1, 2] = 1
    T[2, 0] = 1
    return T


def random_tournament(n: int, seed: int) -> np.ndarray:
    """Uniformly random tournament on n vertices."""
    rng = np.random.default_rng(seed)
    T = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                T[i, j] = 1
            else:
                T[j, i] = 1
    return T


# ---------------------------------------------------------------------------
# Tests: F_2 linear algebra
# ---------------------------------------------------------------------------

def test_rank_f2_identity():
    assert _matrix_rank_f2(np.eye(5, dtype=np.int8)) == 5


def test_rank_f2_zero():
    assert _matrix_rank_f2(np.zeros((4, 6), dtype=np.int8)) == 0


def test_rank_f2_dependent_rows():
    M = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],  # = row0 + row1 (mod 2)
        ],
        dtype=np.int8,
    )
    assert _matrix_rank_f2(M) == 2


# ---------------------------------------------------------------------------
# Tests: basic tournament predicates
# ---------------------------------------------------------------------------

def test_validate_accepts_good():
    validate_tournament(transitive_tournament(5))
    validate_tournament(cyclic_triple_tournament())


def test_validate_rejects_bad():
    bad = np.array([[0, 1], [1, 0]], dtype=np.int8)  # both off-diagonals = 1
    try:
        validate_tournament(bad)
    except ValueError:
        return
    raise AssertionError("validate_tournament should have rejected a non-tournament")


def test_is_cyclic_triple_on_cycle():
    T = cyclic_triple_tournament()
    assert is_cyclic_triple(T, 0, 1, 2) is True
    # Any permutation of the same 3 elements should be cyclic
    for perm in [(1, 0, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)]:
        assert is_cyclic_triple(T, *perm) is True


def test_is_cyclic_triple_on_transitive():
    T = transitive_tournament(3)
    assert is_cyclic_triple(T, 0, 1, 2) is False


# ---------------------------------------------------------------------------
# Tests: invariant 1, Condorcet cycle count
# ---------------------------------------------------------------------------

def test_condorcet_cycles_transitive():
    for n in range(3, 8):
        assert count_condorcet_cycles(transitive_tournament(n)) == 0


def test_condorcet_cycles_single_3cycle():
    assert count_condorcet_cycles(cyclic_triple_tournament()) == 1


def test_condorcet_cycles_n4_one_triple():
    """n=4 with exactly one cyclic sub-triple {0,1,2} and 3 otherwise acyclic."""
    T = transitive_tournament(4)
    # Flip 2 -> 0 becomes 0 -> 2 originally. Need cycle 0->1->2->0.
    # transitive has 0->1, 1->2, 0->2. Flip 0->2 to 2->0 to create cycle.
    T[0, 2] = 0
    T[2, 0] = 1
    # This creates a cycle on {0,1,2}. Other triples: {0,1,3} is 0->1, 0->3, 1->3: acyclic.
    # {0,2,3}: 2->0, 0->3, 2->3: acyclic. {1,2,3}: 1->2, 1->3, 2->3: acyclic.
    assert count_condorcet_cycles(T) == 1


# ---------------------------------------------------------------------------
# Tests: invariant 2, tetrahedral parities and cohomology
# ---------------------------------------------------------------------------

def test_parities_transitive():
    for n in range(4, 8):
        T = transitive_tournament(n)
        assert tetrahedral_parities(T).sum() == 0
        assert cohomology_class_nonzero(T) is False


def test_parities_n4_single_cycle():
    """n=4 with one cyclic triangle: exactly one 4-subset, parity = 1."""
    T = transitive_tournament(4)
    T[0, 2] = 0
    T[2, 0] = 1
    parities = tetrahedral_parities(T)
    assert parities.size == 1
    assert parities[0] == 1
    assert cohomology_class_nonzero(T) is True


def test_cohomology_nonzero_agrees_with_parity():
    """Direct linear-algebra cohomology test must agree with any-nonzero-parity."""
    for seed in range(20):
        for n in [4, 5, 6, 7]:
            T = random_tournament(n, seed=seed)
            has_nz_parity = bool(tetrahedral_parities(T).any())
            has_nz_class = cohomology_class_nonzero(T)
            assert has_nz_parity == has_nz_class, (
                f"Mismatch at n={n}, seed={seed}: parity={has_nz_parity}, "
                f"class={has_nz_class}"
            )


# ---------------------------------------------------------------------------
# Tests: invariant 3, beta_1 of acyclic complex
# ---------------------------------------------------------------------------

def test_beta_1_transitive():
    """Transitive tournament fills in all triangles, so beta_1 of the
    2-skeleton of the (n-1)-simplex vanishes for n >= 4."""
    # n=3: all triangles acyclic means 1 triangle fills the unique loop.
    # K has 3 edges, 1 triangle, beta_1 = (3-3+1) - 1 = 0.
    b = betti_1_acyclic_complex(transitive_tournament(3))
    assert b["beta_1"] == 0
    # n=4: 6 edges, 4 triangles all present. dim ker d_1 = 3, rank d_2 = 3.
    # beta_1 = 0.
    b = betti_1_acyclic_complex(transitive_tournament(4))
    assert b["beta_1"] == 0
    # n=5: 10 edges, 10 triangles. beta_1 should be 0.
    b = betti_1_acyclic_complex(transitive_tournament(5))
    assert b["beta_1"] == 0


def test_beta_1_single_3cycle():
    """n=3 with a cyclic triple: the triangle is empty in the complex.
    1-skeleton is a single 3-cycle, no 2-simplices. beta_1 = 1."""
    b = betti_1_acyclic_complex(cyclic_triple_tournament())
    assert b["beta_1"] == 1
    assert b["num_acyclic_triples"] == 0


def test_beta_1_nonnegative_random():
    """Sanity: beta_1 must be >= 0 for any tournament."""
    for seed in range(30):
        for n in range(3, 10):
            T = random_tournament(n, seed=seed)
            b = betti_1_acyclic_complex(T)
            assert b["beta_1"] >= 0, f"beta_1 < 0 at n={n}, seed={seed}"


# ---------------------------------------------------------------------------
# Tests: aggregated invariants entry point
# ---------------------------------------------------------------------------

def test_topological_invariants_schema():
    inv = topological_invariants(random_tournament(6, seed=42))
    expected_keys = {
        "n",
        "condorcet_3cycles",
        "condorcet_3cycle_fraction",
        "num_4subsets",
        "odd_parity_4subsets",
        "odd_parity_fraction",
        "cohomology_class_nonzero",
        "beta_1_acyclic_complex",
        "num_acyclic_triples",
    }
    assert set(inv.keys()) >= expected_keys


def test_scalar_projection_residual_zero_for_gradient():
    """A margin field generated by scalar scores should have zero residual."""
    from experiment import scalar_projection_stats

    scores = np.array([1.5, 0.25, -0.5, -1.25])
    W = scores[:, None] - scores[None, :]
    support = np.ones_like(W)
    np.fill_diagonal(support, 0)
    stats = scalar_projection_stats(W, support)
    assert stats["hodge_n_edges"] == 6
    assert stats["hodge_rho_cyc"] < 1e-12


def test_scalar_projection_residual_positive_for_cycle():
    """A pure 3-cycle edge flow cannot be represented by scalar scores."""
    from experiment import scalar_projection_stats

    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = W[1, 2] = W[2, 0] = 1.0
    W[1, 0] = W[2, 1] = W[0, 2] = -1.0
    support = np.ones_like(W)
    np.fill_diagonal(support, 0)
    stats = scalar_projection_stats(W, support)
    assert 0 < stats["hodge_rho_cyc"] <= 1


# ---------------------------------------------------------------------------
# End-to-end pipeline on synthetic data
# ---------------------------------------------------------------------------

def test_stream_handles_malformed_rows():
    """stream_moral_machine must drop rows with empty/non-numeric cells
    in numeric columns instead of raising ValueError."""
    import pandas as pd
    from moral_machine import CHARACTER_COLUMNS, stream_moral_machine

    tmp = tempfile.mkdtemp(prefix="exp_a_malformed_")
    try:
        csv_path = os.path.join(tmp, "malformed.csv")
        # Build a tiny CSV with:
        #   - 4 valid rows (2 scenarios, 2 users)
        #   - 1 row with empty Saved cell
        #   - 1 row with empty character cell
        #   - 1 row with non-numeric character cell
        headers = ["UserID", "UserCountry3", "Saved", *CHARACTER_COLUMNS]
        good_row = ["U001", "USA", "1"] + ["0"] * len(CHARACTER_COLUMNS)
        good_row2 = ["U001", "USA", "0"] + ["1"] + ["0"] * (len(CHARACTER_COLUMNS) - 1)
        good_row3 = ["U002", "DEU", "1"] + ["0"] * len(CHARACTER_COLUMNS)
        good_row4 = ["U002", "DEU", "0"] + ["1"] + ["0"] * (len(CHARACTER_COLUMNS) - 1)
        bad_empty_saved = ["U003", "FRA", ""] + ["0"] * len(CHARACTER_COLUMNS)
        bad_empty_char = ["U003", "FRA", "1"] + [""] + ["0"] * (len(CHARACTER_COLUMNS) - 1)
        bad_nonnumeric = ["U003", "FRA", "1"] + ["x"] + ["0"] * (len(CHARACTER_COLUMNS) - 1)
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for row in [good_row, good_row2, good_row3, good_row4,
                        bad_empty_saved, bad_empty_char, bad_nonnumeric]:
                f.write(",".join(row) + "\n")

        # This previously raised ValueError; it must now succeed and yield
        # only the 4 good rows.
        total_kept = 0
        for chunk in stream_moral_machine(csv_path, chunk_size=100, progress=False):
            total_kept += len(chunk)
            # All returned cells must be valid integers.
            assert chunk["Saved"].isin([0, 1]).all()
            for c in CHARACTER_COLUMNS:
                assert pd.api.types.is_integer_dtype(chunk[c].dtype)
        assert total_kept == 4, f"Expected 4 valid rows, got {total_kept}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_synthetic():
    """Run the full pipeline on a small synthetic Moral Machine-format CSV."""
    from moral_machine import (
        aggregate_user_preferences,
        eligible_countries,
        stream_moral_machine,
        synthesise_moral_machine,
    )
    from experiment import (
        compare_observed_to_null,
        compute_observed,
        run_permutation_null,
    )

    tmp = tempfile.mkdtemp(prefix="exp_a_test_")
    try:
        csv_path = os.path.join(tmp, "synth.csv")
        synthesise_moral_machine(
            csv_path,
            n_users=1500,
            n_scenarios_per_user=10,
            n_countries=6,
            seed=0,
        )
        # The test uses a permissive min-users threshold so small synth data
        # produces enough countries.
        import moral_machine
        original_min_users = moral_machine.MIN_USERS_PER_COUNTRY
        moral_machine.MIN_USERS_PER_COUNTRY = 100
        try:
            chunks = stream_moral_machine(csv_path, chunk_size=5000)
            prefs = aggregate_user_preferences(chunks, min_scenarios=5, progress=False)
            countries = eligible_countries(prefs)
            assert len(countries) >= 3, f"Only {len(countries)} eligible countries"

            observed = compute_observed(prefs, countries=countries, progress=False)
            assert len(observed) == len(countries)
            for r in observed:
                assert r.invariants["n"] == 20
                # All invariants finite and valid.
                assert 0 <= r.invariants["condorcet_3cycles"] <= 1140
                assert 0 <= r.invariants["odd_parity_fraction"] <= 1
                assert r.invariants["beta_1_acyclic_complex"] >= 0
                assert 0 <= r.invariants["hodge_rho_cyc"] <= 1

            null = run_permutation_null(
                prefs,
                observed_countries=[r.country for r in observed],
                n_permutations=20,
                seed=0,
                progress=False,
            )
            report = compare_observed_to_null(observed, null)
            # The report must contain our key invariant.
            assert "beta_1_acyclic_complex" in report
            assert "cohomology_class_nonzero" in report
            assert "hodge_rho_cyc" in report
            # z-score must be a real number.
            for k, stats in report.items():
                assert np.isfinite(stats["observed_mean"])
                assert np.isfinite(stats["null_mean"])
        finally:
            moral_machine.MIN_USERS_PER_COUNTRY = original_min_users
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_rank_f2_identity,
        test_rank_f2_zero,
        test_rank_f2_dependent_rows,
        test_validate_accepts_good,
        test_validate_rejects_bad,
        test_is_cyclic_triple_on_cycle,
        test_is_cyclic_triple_on_transitive,
        test_condorcet_cycles_transitive,
        test_condorcet_cycles_single_3cycle,
        test_condorcet_cycles_n4_one_triple,
        test_parities_transitive,
        test_parities_n4_single_cycle,
        test_cohomology_nonzero_agrees_with_parity,
        test_beta_1_transitive,
        test_beta_1_single_3cycle,
        test_beta_1_nonnegative_random,
        test_topological_invariants_schema,
        test_scalar_projection_residual_zero_for_gradient,
        test_scalar_projection_residual_positive_for_cycle,
        test_stream_handles_malformed_rows,
        test_end_to_end_synthetic,
    ]
    n_passed = 0
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            n_failed += 1
    print(f"\n{n_passed} passed, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
