# Scalar Projection Loss Paper

This repository contains the current manuscript and analysis outputs for the paper on scalar reward learning, Condorcet/Hodge diagnostics, and pluralistic preference aggregation.

## Current Manuscript

- `paper_cai/main5_llm_focused_revised_no_conceptual_figures.tex`
- `paper_cai/main5_llm_focused_revised_no_conceptual_figures.pdf`
- `paper_cai/figures_generated/`

The current TeX intentionally includes only the empirical figures. Earlier conceptual placeholder figures were removed from the manuscript and moved into the cleanup archive.

## Active Analysis Paths

- `src/write_paper_result_figures.py` writes the paper-facing PRISM, MultiPref, and Moral Machine plots from finished result tables.
- `src/prism_defensibility_checks_v3/` contains the PRISM defensibility-check pipeline and the pooled edge tables used in the PRISM figure.
- `src/run_multipref_cycles_v4.py` and `src/results/multipref_v4/` contain the current MultiPref analysis and outputs.
- `src/run_multipref_calibrated_null.py` runs the support-preserving transitive-null calibration for MultiPref global residuals.
- `src/experiment_a/` contains the direct Moral Machine projection/cycle pipeline.
- `src/moral_machine_condorcet.py`, `mm_results/`, and `results_moral_machine/` are retained while the Moral Machine latent-profile discrepancy is unresolved.

## Key Result Directories

- `results_prism_projection_5000/`
- `src/results/multipref_v4/`
- `results_moral_machine_projection/`
- `results_moral_machine/`
- `mm_results/`

`results_moral_machine_projection/` is the current direct Moral Machine pipeline used by the paper figure. `results_moral_machine/` and `mm_results/` are intentionally both kept for now because they encode different versions of the latent Moral Machine profile analysis.

## Data

- `data/original/SharedResponses.csv.tar.gz`

The full Moral Machine CSV is large and needed for a clean rerun of the Moral Machine pipeline. It has not been moved into the archive.

## Rebuilding

Regenerate paper figures:

```bash
python src/write_paper_result_figures.py
```

Rerun the calibrated MultiPref residual null:

```bash
python src/run_multipref_calibrated_null.py --n-null 5000 --seed 20260615
```

Build the manuscript:

```bash
cd paper_cai
latexmk -pdf -interaction=nonstopmode main5_llm_focused_revised_no_conceptual_figures.tex
latexmk -c main5_llm_focused_revised_no_conceptual_figures.tex
```

## Cleanup Archive

Old drafts, superseded analyses, exploratory scripts, notebooks, caches, and previous result directories were moved to:

```text
_archive_pre_paper_cleanup/
```

Nothing was hard-deleted during the cleanup. See `CLEANUP_AUDIT.md` for the rationale and the Moral Machine provenance warning.
