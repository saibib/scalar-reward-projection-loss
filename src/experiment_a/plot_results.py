"""
Publication figure for Experiment A results.

Reads the output directory produced by `run.py` and produces:

  fig_experiment_a.pdf / fig_experiment_a.png
      Four-panel figure:
        A. Null distribution of the mean Condorcet 3-cycle rate per country.
        B. Null distribution of the mean scalar projection residual.
        C. Null distribution of the fraction of countries with a finite
           parity obstruction.
        D. Comparison of the latent moral-profile construction with the
           direct character-type construction.

  fig_experiment_a_supplement.pdf / .png
      Two-panel supplement:
        E. Per-country Condorcet cycle count vs. odd-parity count.
        F. Null-sample histograms for all six aggregate invariants
           side-by-side.

Usage:
    python plot_results.py --input-dir results_moral_machine \
                           --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Minimal publication style; no seaborn dependency."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "DejaVu Sans",
        }
    )


# Colors chosen to be colorblind-friendly and print well in greyscale.
C_OBSERVED = "#c0392b"       # red-brown, observed statistic
C_NULL = "#95a5a6"           # mid-grey, null distribution
C_NULL_EDGE = "#2c3e50"      # dark slate, histogram edges
C_CYCLIC = "#e67e22"         # orange, countries with finite parity obstruction
C_ACYCLIC = "#3498db"        # blue, countries without finite parity obstruction
C_ANNOTATION = "#2c3e50"     # dark slate, text


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_results(input_dir: str):
    """Load all outputs produced by run.py."""
    per_country_path = os.path.join(input_dir, "per_country_observed.csv")
    null_samples_path = os.path.join(input_dir, "null_samples.npz")
    comparison_path = os.path.join(input_dir, "aggregate_comparison.json")
    for p in (per_country_path, null_samples_path, comparison_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing expected output file: {p}. "
                f"Run `python run.py --out {input_dir}` first."
            )
    per_country = pd.read_csv(per_country_path)
    with np.load(null_samples_path) as data:
        null_samples = {k: data[k].copy() for k in data.files}
    with open(comparison_path) as f:
        comparison = json.load(f)
    required_residual = "hodge_rho_cyc"
    missing = []
    if required_residual not in per_country.columns:
        missing.append(f"{per_country_path}:{required_residual}")
    if required_residual not in null_samples:
        missing.append(f"{null_samples_path}:{required_residual}")
    if required_residual not in comparison:
        missing.append(f"{comparison_path}:{required_residual}")
    if missing:
        raise ValueError(
            "Moral Machine scalar projection residuals are missing. "
            "Rerun `src/experiment_a/run.py` with the updated code before "
            "rendering paper figures. Missing: " + ", ".join(missing)
        )
    return per_country, null_samples, comparison


def load_optional_comparison(path: str | None) -> Dict[str, Dict[str, float]] | None:
    """Load an optional aggregate-comparison JSON file."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Null vs observed histogram panel
# ---------------------------------------------------------------------------

def _format_p_value(p: float) -> str:
    """Format a one-sided empirical p-value for display.

    If the observed value exceeded every null draw, p is reported as an upper
    bound 1/(N+1) using the Clopper-Pearson convention for permutation tests.
    """
    if p == 0:
        return "p < 0.002"  # caller should override with a tighter bound if known
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.3g}"


def plot_null_vs_observed(
    ax: plt.Axes,
    null_samples: np.ndarray,
    observed_value: float,
    z_score: float,
    p_value: float,
    n_permutations: int,
    xlabel: str,
    title: str,
) -> None:
    """Histogram of null distribution with observed value marked."""
    # Adaptive binning: Freedman-Diaconis, clamped.
    data = null_samples
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        bin_width = 2 * iqr * len(data) ** (-1 / 3)
        nbins = max(12, min(40, int(np.ceil((data.max() - data.min()) / bin_width))))
    else:
        nbins = 20

    ax.hist(
        data,
        bins=nbins,
        color=C_NULL,
        edgecolor=C_NULL_EDGE,
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(
        observed_value,
        color=C_OBSERVED,
        linewidth=2.2,
        zorder=5,
    )

    # Tighter p-value bound when observed exceeds all nulls.
    if p_value == 0:
        p_str = f"p < {1 / (n_permutations + 1):.3g}"
    else:
        p_str = _format_p_value(p_value)

    ax.text(
        0.97,
        0.97,
        f"Null n = {len(data)}\nObserved = {observed_value:.4g}\nz = {z_score:+.2f}\n{p_str}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=C_ANNOTATION,
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#bdc3c7",
            linewidth=0.5,
        ),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutation count")
    ax.set_title(title, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Scatter: sample size vs cycle count
# ---------------------------------------------------------------------------

def plot_sample_size_vs_cycles(
    ax: plt.Axes,
    per_country: pd.DataFrame,
    label_cyclic_countries: bool = True,
) -> None:
    """Per-country n_users vs Condorcet cycle count, colored by parity status."""
    cyclic_mask = per_country["cohomology_class_nonzero"].astype(bool)
    acyclic = per_country[~cyclic_mask]
    cyclic = per_country[cyclic_mask]

    ax.scatter(
        acyclic["n_users"],
        acyclic["condorcet_3cycles"],
        s=28,
        c=C_ACYCLIC,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.6,
        label=f"No parity obstruction  (n = {len(acyclic)})",
    )
    ax.scatter(
        cyclic["n_users"],
        cyclic["condorcet_3cycles"],
        s=42,
        c=C_CYCLIC,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.6,
        label=f"Parity obstruction  (n = {len(cyclic)})",
    )

    if label_cyclic_countries and len(cyclic) > 0:
        # Offset labels to reduce overlap; this is a quick heuristic.
        for _, row in cyclic.iterrows():
            ax.annotate(
                row["country"],
                xy=(row["n_users"], row["condorcet_3cycles"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color=C_ANNOTATION,
                alpha=0.85,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Users per country (log scale)")
    ax.set_ylabel("Condorcet 3-cycles")
    ax.set_title("D   Sample size vs. cycle count", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Per-country bar: cyclic countries
# ---------------------------------------------------------------------------

def plot_cyclic_country_bars(
    ax: plt.Axes,
    per_country: pd.DataFrame,
) -> None:
    """Bar chart of odd-parity counts for the cyclic subset, sorted desc."""
    cyclic = per_country[per_country["cohomology_class_nonzero"].astype(bool)].copy()
    cyclic = cyclic.sort_values("odd_parity_4subsets", ascending=False)

    if len(cyclic) == 0:
        ax.text(
            0.5, 0.5,
            "No countries with a finite parity obstruction",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=C_ANNOTATION,
        )
        ax.set_title(
            "D   Cyclic countries: odd-parity 4-subsets",
            loc="left",
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x = np.arange(len(cyclic))
    bars = ax.bar(
        x,
        cyclic["odd_parity_4subsets"].values,
        color=C_CYCLIC,
        edgecolor=C_NULL_EDGE,
        linewidth=0.4,
        width=0.75,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cyclic["country"].values, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Odd-parity 4-subsets")
    ax.set_title(
        "D   Countries with odd-parity 4-subsets",
        loc="left",
        fontweight="bold",
    )

    # Annotate each bar with the cycle count for interpretability.
    for bar, cycles, n_u in zip(
        bars,
        cyclic["condorcet_3cycles"].values,
        cyclic["n_users"].values,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{int(cycles)}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=C_ANNOTATION,
        )

    ax.set_ylim(0, cyclic["odd_parity_4subsets"].max() * 1.18)
    ax.margins(x=0.02)


def plot_cycles_vs_residual(ax: plt.Axes, per_country: pd.DataFrame) -> None:
    """Per-country finite cycles against scalar projection residual."""
    cyclic_mask = per_country["cohomology_class_nonzero"].astype(bool)
    acyclic = per_country[~cyclic_mask]
    cyclic = per_country[cyclic_mask]

    ax.scatter(
        acyclic["condorcet_3cycles"],
        acyclic["hodge_rho_cyc"],
        s=34,
        c=C_ACYCLIC,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.6,
        label=f"No parity obstruction (n={len(acyclic)})",
    )
    ax.scatter(
        cyclic["condorcet_3cycles"],
        cyclic["hodge_rho_cyc"],
        s=48,
        c=C_CYCLIC,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.6,
        label=f"Parity obstruction (n={len(cyclic)})",
    )

    if len(cyclic) > 0:
        label_rows = cyclic.sort_values(
            ["condorcet_3cycles", "hodge_rho_cyc"], ascending=False
        ).head(8)
        for _, row in label_rows.iterrows():
            ax.annotate(
                row["country"],
                xy=(row["condorcet_3cycles"], row["hodge_rho_cyc"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color=C_ANNOTATION,
                alpha=0.88,
            )

    ax.set_xlabel("Condorcet 3-cycles")
    ax.set_ylabel(r"Scalar projection residual $\rho_{\mathrm{cyc}}$")
    ax.set_title(
        "D   Country tournaments: cycles and residuals",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=8)


def plot_construction_comparison(
    ax: plt.Axes,
    direct_comparison: Dict[str, Dict[str, float]],
    latent_comparison: Dict[str, Dict[str, float]] | None,
) -> None:
    """Compare shared Moral Machine diagnostics across two constructions."""
    metrics = [
        ("condorcet_3cycle_fraction", "Cycle\nfraction"),
        ("fraction_countries_cohom_nonzero", "Parity-obstruction\ncountry fraction"),
    ]
    construction_rows = [
        ("Latent profiles", latent_comparison, "#4c78a8"),
        ("Direct character types", direct_comparison, "#59a14f"),
    ]

    x = np.arange(len(metrics))
    width = 0.34
    offsets = [-width / 2, width / 2]
    any_plotted = False
    for offset, (label, comp, color) in zip(offsets, construction_rows):
        if comp is None:
            continue
        z_vals = []
        p_vals = []
        for key, _ in metrics:
            stats = comp.get(key)
            if stats is None:
                z_vals.append(np.nan)
                p_vals.append(np.nan)
            else:
                z_vals.append(float(stats["z_score"]))
                p_vals.append(float(stats["p_one_sided"]))
        bars = ax.bar(
            x + offset,
            z_vals,
            width=width,
            color=color,
            edgecolor=C_NULL_EDGE,
            linewidth=0.45,
            label=label,
            alpha=0.9,
        )
        any_plotted = True
        for bar, z, p in zip(bars, z_vals, p_vals):
            if not np.isfinite(z):
                continue
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            dy = 0.12 if y >= 0 else -0.12
            p_label = _format_p_value(p) if np.isfinite(p) else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + dy,
                f"z={z:+.2f}\n{p_label}",
                ha="center",
                va=va,
                fontsize=7.2,
                color=C_ANNOTATION,
            )

    if not any_plotted:
        ax.text(
            0.5,
            0.5,
            "Latent-profile comparison not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=C_ANNOTATION,
        )

    ax.axhline(0, color=C_NULL_EDGE, linewidth=0.8)
    ax.axhline(1.96, color=C_ANNOTATION, linewidth=0.65, linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Observed - null mean (null SDs)")
    ax.set_title("D   Construction comparison", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-1.5, 5.5)


# ---------------------------------------------------------------------------
# Supplement Panel E: consistency check
# ---------------------------------------------------------------------------

def plot_cycles_vs_parity(ax: plt.Axes, per_country: pd.DataFrame) -> None:
    """Scatter of condorcet_3cycles vs odd_parity_4subsets per country.

    Since each cyclic triple appears in (K-3) 4-subsets (K=20: 17 subsets),
    the two counts have a loose but bounded relationship. This is a
    built-in consistency check: no country should have parity counts
    wildly out of proportion to its cycle count.
    """
    cyclic_mask = per_country["cohomology_class_nonzero"].astype(bool)
    acyclic = per_country[~cyclic_mask]
    cyclic = per_country[cyclic_mask]

    ax.scatter(
        acyclic["condorcet_3cycles"],
        acyclic["odd_parity_4subsets"],
        s=28,
        c=C_ACYCLIC,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.6,
        label="No parity obstruction",
    )
    ax.scatter(
        cyclic["condorcet_3cycles"],
        cyclic["odd_parity_4subsets"],
        s=42,
        c=C_CYCLIC,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.6,
        label="Parity obstruction",
    )
    # Reference line: parity count = (K-3) * cycles (upper bound when
    # cycles are disjoint) where K is the tournament size. K=20 gives 17.
    x_max = max(1, int(per_country["condorcet_3cycles"].max()))
    xs = np.array([0, x_max])
    K = 20
    ax.plot(
        xs,
        (K - 3) * xs,
        linestyle="--",
        linewidth=0.8,
        color=C_NULL_EDGE,
        alpha=0.5,
        label=f"$(K-3) \\cdot $ cycles  (K={K})",
    )

    ax.set_xlabel("Condorcet 3-cycles")
    ax.set_ylabel("Odd-parity 4-subsets")
    ax.set_title(
        "E   Internal consistency: cycles vs. parities",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)


# ---------------------------------------------------------------------------
# Supplement Panel F: all invariants at once
# ---------------------------------------------------------------------------

def plot_all_invariant_nulls(
    ax: plt.Axes,
    null_samples: Dict[str, np.ndarray],
    comparison: Dict[str, Dict[str, float]],
) -> None:
    """Forest plot: observed - null_mean (in null SD units), with CIs."""
    metrics = [
        ("condorcet_3cycles", "Condorcet cycles\n(per-country mean)"),
        ("condorcet_3cycle_fraction", "Cycle fraction\n(per-country mean)"),
        ("hodge_rho_cyc", "Scalar residual\n(per-country mean)"),
        ("odd_parity_4subsets", "Odd-parity count\n(per-country mean)"),
        ("odd_parity_fraction", "Odd-parity fraction\n(per-country mean)"),
        ("cohomology_class_nonzero", "Parity obstruction\n(country fraction)"),
    ]
    labels = []
    zs = []
    ps = []
    for key, label in metrics:
        if key not in comparison:
            continue
        stats = comparison[key]
        if not np.isfinite(stats["z_score"]):
            continue
        labels.append(label)
        zs.append(stats["z_score"])
        ps.append(stats["p_one_sided"])

    y_pos = np.arange(len(labels))
    bar_colors = [C_OBSERVED if z > 0 else C_NULL for z in zs]
    ax.barh(
        y_pos,
        zs,
        color=bar_colors,
        edgecolor=C_NULL_EDGE,
        linewidth=0.4,
        alpha=0.85,
    )
    ax.axvline(0, color=C_NULL_EDGE, linewidth=0.8)
    ax.axvline(
        1.96, color=C_ANNOTATION, linewidth=0.6, linestyle=":", alpha=0.7
    )
    ax.axvline(
        -1.96, color=C_ANNOTATION, linewidth=0.6, linestyle=":", alpha=0.7
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Observed $-$ null mean  (null SDs, one-sided)")
    ax.set_title(
        "F   All invariants: observed vs. null",
        loc="left",
        fontweight="bold",
    )

    # Annotate with z and p.
    for y, z, p in zip(y_pos, zs, ps):
        label = f"z={z:+.2f}"
        if p == 0:
            label += ", p<0.002"
        elif p < 0.001:
            label += ", p<0.001"
        else:
            label += f", p={p:.3f}"
        x_text = z + (0.15 if z >= 0 else -0.15)
        ha = "left" if z >= 0 else "right"
        ax.text(
            x_text, y, label, va="center", ha=ha, fontsize=7.5, color=C_ANNOTATION
        )
    # Give some headroom for annotations.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo * 1.25 if lo < 0 else lo - 1, hi * 1.25 if hi > 0 else hi + 1)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def make_main_figure(
    per_country: pd.DataFrame,
    null_samples: Dict[str, np.ndarray],
    comparison: Dict[str, Dict[str, float]],
    latent_comparison: Dict[str, Dict[str, float]] | None,
    n_permutations: int,
    out_path_pdf: str,
    out_path_png: str,
) -> None:
    """Four-panel publication figure."""
    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.5, 8.0),
        constrained_layout=True,
    )

    # A: Condorcet cycle rate null
    key = "condorcet_3cycle_fraction"
    stats = comparison[key]
    plot_null_vs_observed(
        axes[0, 0],
        null_samples[key],
        stats["observed_mean"],
        stats["z_score"],
        stats["p_one_sided"],
        n_permutations,
        xlabel="Mean Condorcet 3-cycle fraction",
        title="A   Cycle frequency (per-country mean)",
    )

    # B: scalar projection residual null
    key = "hodge_rho_cyc"
    stats = comparison[key]
    plot_null_vs_observed(
        axes[0, 1],
        null_samples[key],
        stats["observed_mean"],
        stats["z_score"],
        stats["p_one_sided"],
        n_permutations,
        xlabel=r"Mean scalar projection residual $\rho_{\mathrm{cyc}}$",
        title="B   Scalar projection residual",
    )

    # C: finite parity-obstruction fraction null
    key = "fraction_countries_cohom_nonzero"
    stats = comparison[key]
    plot_null_vs_observed(
        axes[1, 0],
        null_samples[key],
        stats["observed_mean"],
        stats["z_score"],
        stats["p_one_sided"],
        n_permutations,
        xlabel="Fraction of countries with a parity obstruction",
        title="C   Finite parity obstruction",
    )

    # D: latent moral-profile construction vs direct character-type construction.
    plot_construction_comparison(axes[1, 1], comparison, latent_comparison)

    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    plt.close(fig)


def make_supplement_figure(
    per_country: pd.DataFrame,
    null_samples: Dict[str, np.ndarray],
    comparison: Dict[str, Dict[str, float]],
    out_path_pdf: str,
    out_path_png: str,
) -> None:
    """Two-panel supplement figure."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.0, 4.2),
        constrained_layout=True,
    )
    plot_cycles_vs_parity(axes[0], per_country)
    plot_all_invariant_nulls(axes[1], null_samples, comparison)
    fig.suptitle(
        "Moral Machine  —  supplement: invariant and residual checks",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render publication figures from Experiment A results."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing run.py outputs "
             "(per_country_observed.csv, null_samples.npz, aggregate_comparison.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to write figures to (created if absent).",
    )
    parser.add_argument(
        "--latent-comparison",
        type=str,
        default="results_moral_machine/aggregate_comparison.json",
        help="Optional aggregate_comparison.json from the latent moral-profile construction.",
    )
    args = parser.parse_args()

    set_style()
    os.makedirs(args.output_dir, exist_ok=True)
    per_country, null_samples, comparison = load_results(args.input_dir)
    latent_comparison = load_optional_comparison(args.latent_comparison)

    # Number of permutations (for p-value bound formatting).
    n_permutations = int(
        min(len(v) for v in null_samples.values() if hasattr(v, "__len__"))
    )

    main_pdf = os.path.join(args.output_dir, "fig_experiment_a.pdf")
    main_png = os.path.join(args.output_dir, "fig_experiment_a.png")
    supp_pdf = os.path.join(args.output_dir, "fig_experiment_a_supplement.pdf")
    supp_png = os.path.join(args.output_dir, "fig_experiment_a_supplement.png")

    print(f"[plot] main figure -> {main_pdf}")
    make_main_figure(
        per_country,
        null_samples,
        comparison,
        latent_comparison,
        n_permutations,
        main_pdf,
        main_png,
    )
    print(f"[plot] supplement figure -> {supp_pdf}")
    make_supplement_figure(per_country, null_samples, comparison, supp_pdf, supp_png)
    print(f"[plot] done. {n_permutations} null permutations, "
          f"{len(per_country)} countries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
