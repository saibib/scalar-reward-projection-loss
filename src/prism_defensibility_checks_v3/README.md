# PRISM Condorcet Defensibility Checks

This add-on script is designed to make the PRISM Condorcet/model-tournament results more defensible before they are used in a paper or workshop draft.

It checks four things:

1. **Attribute coverage and missingness** for `choice_attributes` or `performance_attributes`.
2. **High-vs-low attribute-conditioned tournaments**, using winner-conditioned attributes correctly.
3. **Deduplicated strongest-cycle edge profiles**, so repeated cycle settings do not masquerade as independent evidence.
4. **Statistical uncertainty and nulls**, including user bootstraps and an edge-level transitive Bradley--Terry-style null.

The script uses the PRISM `conversations` subset and parses `conversation_history` directly. The opening-turn model responses are used for model tournaments. Later same-model A/B turns are not mixed into the model-source tournament.

## Important methodological guardrail

In PRISM, `choice_attributes` and `performance_attributes` are attached to the **highest-rated model in the first turn**, not to every model shown. Therefore, attribute-conditioned tournaments are built only from pairwise edges where the global opening-turn winner beats another displayed model. This avoids incorrectly treating the winner-conditioned attributes as per-model scores for losing models.

The resulting interpretation is:

> A high-attribute tournament is a tournament over comparisons where the selected winner was attributed high helpfulness/safety/factuality/etc.

It is **not** a direct tournament over objective model helpfulness/safety/factuality.

## Installation

```bash
unzip prism_defensibility_checks.zip
cd prism_defensibility_checks

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run checks

Recommended run:

```bash
python -m prism_checks.run_defensibility_checks \
  --dataset-name HannahRoseKirk/prism-alignment \
  --subset conversations \
  --split train \
  --output-dir outputs/defensibility_checks \
  --aggregation user \
  --score-deltas 0 2 5 10 \
  --n-min-values 10 25 50 100 \
  --tau-values 0 0.05 0.10 \
  --attribute-prefix choice \
  --bootstrap-reps 200 \
  --null-reps 200 \
  --null-score-delta 5 \
  --null-n-min 25 \
  --null-tau 0
```

For a quicker exploratory run:

```bash
python -m prism_checks.run_defensibility_checks \
  --dataset-name HannahRoseKirk/prism-alignment \
  --subset conversations \
  --split train \
  --output-dir outputs/defensibility_quick \
  --aggregation user \
  --score-deltas 5 \
  --n-min-values 25 \
  --tau-values 0 0.05 \
  --attribute-prefix choice \
  --bootstrap-reps 50 \
  --null-reps 50
```

## Main outputs

The script writes:

- `attribute_summary.csv`
- `pooled_cycle_stats.csv`
- `pooled_hodge_stats.csv`
- `pooled_strongest_cycles.csv`
- `dedup_cycle_edge_attribute_profiles.csv`
- `attribute_high_low_cycle_stats.csv`
- `attribute_high_low_hodge_stats.csv`
- `attribute_high_low_strongest_cycles.csv`
- `bootstrap_summary.csv`
- `transitive_null_summary.csv`
- `DEFENSIBILITY_CHECKS_SUMMARY.md`
- plots in `figures/`

## Figures

Open the figures folder after the run:

```bash
open outputs/defensibility_checks/figures
```

The most useful plots are:

- `rho_cyc_high_vs_low_attributes.png`
- `cycle_rate_high_vs_low_attributes.png`
- `bootstrap_rho_cyc_median.png`
- `null_test_observed_rho_cyc.png`
- `attribute_missing_rate.png`

## How to interpret the nulls

The presence of a strict cycle is already enough to show the **observed tournament** is not exactly scalar-representable. The nulls answer a different question: whether the observed cyclicity is large relative to what a finite sample from an approximately transitive edge process would produce.

The transitive null is edge-level and support-preserving. It is not a full generative model of PRISM participants, but it is useful as a conservative diagnostic for whether the observed cycle rate or cyclic residual mass is larger than expected under a scalar ranking with similar edge support.

## v2 patch notes

This patched version fixes a `KeyError: 'is_winner_edge'` that could occur when a comparison dataframe was empty or produced by a code path that did not materialize the winner-edge flag. The runner now normalizes the comparison table before all downstream filters.

The transitive-null summary now reports one-sided empirical Monte Carlo p-values with add-one smoothing:

`p_ge_observed_stat = (1 + # null_stat >= observed_stat) / (1 + null_reps)`

These p-values answer: under the fitted transitive edge-level null with the same support structure, how often do simulated tournaments produce cycle rate or cyclic residual mass at least as large as observed?
