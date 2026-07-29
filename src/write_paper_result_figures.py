#!/usr/bin/env python3
"""Write candidate paper figures from the current analysis outputs.

The script deliberately reads finished result tables rather than re-running
dataset analyses. Use the analysis scripts first, then use this writer to
produce manuscript-facing plots.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cai" / "figures_generated"

PRISM_SUMMARY = ROOT / "results_prism_projection_5000" / "pooled_summary.csv"
PRISM_NULL = ROOT / "results_prism_projection_5000" / "null_raw_pooled.csv"
PRISM_EDGES = (
    ROOT
    / "src"
    / "prism_defensibility_checks_v3"
    / "outputs"
    / "defensibility_checks"
    / "pooled_edges_delta5.csv"
)

MULTIPREF_SUMMARY = ROOT / "src" / "results" / "multipref_v4" / "multipref_cycle_projection_summary.csv"
MULTIPREF_SUPPORT = ROOT / "src" / "results" / "multipref_v4" / "multipref_prompt_support_audit.csv"
MULTIPREF_FLAT = ROOT / "src" / "results" / "multipref_v4" / "multipref_flat_annotations.csv"
MULTIPREF_NULL = ROOT / "src" / "results" / "multipref_v4" / "multipref_calibrated_null_summary.csv"

MORAL_FIGS = ROOT / "results_moral_machine_projection" / "figures"


COL_OBS = "#c0392b"
COL_NULL = "#a8b3b5"
COL_EDGE = "#34495e"
COL_BLUE = "#2c7fb8"
COL_ORANGE = "#f28e2b"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def p_text(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3g}"


def null_hist(ax: plt.Axes, values: np.ndarray, obs: float, xlabel: str, title: str, p: float) -> None:
    ax.hist(values, bins=32, color=COL_NULL, edgecolor=COL_EDGE, linewidth=0.45)
    ax.axvline(obs, color=COL_OBS, lw=2.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Null simulations")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.97,
        0.96,
        f"Observed = {obs:.4g}\n{p_text(p)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COL_EDGE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c9d0d3", linewidth=0.5),
    )


def short_model(name: str) -> str:
    replacements = {
        "HuggingFaceH4/zephyr-7b-beta": "Zephyr-7B",
        "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "OASST-12B",
        "meta-llama/Llama-2-13b-chat-hf": "Llama2-13B",
        "meta-llama/Llama-2-70b-chat-hf": "Llama2-70B",
        "meta-llama/Llama-2-7b-chat-hf": "Llama2-7B",
        "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B",
        "models/chat-bison-001": "Chat-Bison",
        "tiiuae/falcon-7b-instruct": "Falcon-7B",
        "timdettmers/guanaco-33b-merged": "Guanaco-33B",
        "google/flan-t5-xxl": "Flan-T5-XXL",
        "gpt-4-1106-preview": "GPT-4-1106",
        "gpt-3.5-turbo": "GPT-3.5",
        "allenai/tulu-2-7b": "Tulu2-7B",
        "allenai/tulu-2-70b": "Tulu2-70B",
        "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct": "Llama3-70B",
        "meta-llama/Llama-2-70b-chat-hf": "Llama2-70B",
    }
    return replacements.get(name, name.replace("-control", "").replace("luminous-", "lum-"))


def matrix_from_edges(edges: pd.DataFrame, a_col: str, b_col: str, margin_col: str) -> tuple[list[str], np.ndarray]:
    models = sorted(set(edges[a_col]).union(set(edges[b_col])))
    idx = {m: i for i, m in enumerate(models)}
    W = np.zeros((len(models), len(models)))
    for _, row in edges.iterrows():
        i, j = idx[row[a_col]], idx[row[b_col]]
        W[i, j] = float(row[margin_col])
        W[j, i] = -float(row[margin_col])
    return models, W


def make_prism() -> None:
    summary = pd.read_csv(PRISM_SUMMARY).iloc[0]
    null = pd.read_csv(PRISM_NULL)
    edges = pd.read_csv(PRISM_EDGES)
    models, W = matrix_from_edges(edges, "model_a", "model_b", "margin")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    im = axes[0].imshow(W, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("A   Pooled PRISM margins", loc="left", fontweight="bold")
    axes[0].set_xticks(range(len(models)))
    axes[0].set_yticks(range(len(models)))
    labels = [short_model(m) for m in models]
    axes[0].set_xticklabels(labels, rotation=90)
    axes[0].set_yticklabels(labels)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Pairwise margin")

    null_hist(
        axes[1],
        null["cycle_rate"].to_numpy(float),
        float(summary["observed_cycle_rate"]),
        "Condorcet 3-cycle fraction",
        "B   Condorcet 3-cycle fraction",
        float(summary["p_ge_observed_cycle_rate"]),
    )
    null_hist(
        axes[2],
        null["hodge_rho_cyc"].to_numpy(float),
        float(summary["observed_hodge_rho_cyc"]),
        r"Scalar projection residual $\rho_{\mathrm{cyc}}$",
        "C   Scalar projection residual",
        float(summary["p_ge_observed_hodge_rho_cyc"]),
    )
    save(fig, "prism_results")


def multipref_margin_matrix() -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(MULTIPREF_FLAT)
    df = df[df["overall_weight"] > 0].copy()
    rows = []
    for _, r in df.iterrows():
        a = str(r["model_a"])
        b = str(r["model_b"])
        sign = float(r["overall_sign"])
        weight = float(r["overall_weight"])
        if a == b or weight <= 0:
            continue
        if a <= b:
            rows.append((a, b, sign * weight, weight))
        else:
            rows.append((b, a, -sign * weight, weight))
    agg = pd.DataFrame(rows, columns=["model_a", "model_b", "signed", "weight"])
    agg = agg.groupby(["model_a", "model_b"], as_index=False).agg(signed=("signed", "sum"), weight=("weight", "sum"))
    agg["margin"] = agg["signed"] / agg["weight"]
    return matrix_from_edges(agg, "model_a", "model_b", "margin")


def make_multipref() -> None:
    summary = pd.read_csv(MULTIPREF_SUMMARY)
    null_summary = pd.read_csv(MULTIPREF_NULL) if MULTIPREF_NULL.exists() else pd.DataFrame()
    global_rows = summary[summary["scope"].eq("global")].copy()
    global_rows["group_label"] = global_rows["group"].str.replace("group=", "", regex=False)
    aspects = ["overall", "helpful", "truthful", "harmless"]
    groups = ["all", "normal", "expert"]
    heat = np.full((len(groups), len(aspects)), np.nan)
    sig_labels = [["" for _ in aspects] for _ in groups]
    for gi, g in enumerate(groups):
        key = "all" if g == "all" else f"group={g}"
        for ai, aspect in enumerate(aspects):
            row = global_rows[(global_rows["group"] == key) & (global_rows["aspect"] == aspect)]
            if len(row):
                heat[gi, ai] = float(row["rho_cyc"].iloc[0])
            if not null_summary.empty and "q_ge_rho_cyc_bh" in null_summary.columns:
                nrow = null_summary[(null_summary["group"] == g) & (null_summary["aspect"] == aspect)]
                if len(nrow):
                    q = float(nrow["q_ge_rho_cyc_bh"].iloc[0])
                    if q < 0.01:
                        sig_labels[gi][ai] = "**"
                    elif q < 0.05:
                        sig_labels[gi][ai] = "*"

    models, W = multipref_margin_matrix()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    im = axes[0].imshow(heat, vmin=0, vmax=np.nanmax(heat), cmap="YlOrRd")
    axes[0].set_title("A   Scalar residual by aspect", loc="left", fontweight="bold")
    axes[0].set_xticks(range(len(aspects)))
    axes[0].set_xticklabels(aspects, rotation=30, ha="right")
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(groups)
    for gi in range(len(groups)):
        for ai in range(len(aspects)):
            axes[0].text(
                ai,
                gi,
                f"{heat[gi, ai]:.3f}{sig_labels[gi][ai]}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\rho_{\mathrm{cyc}}$")

    im = axes[1].imshow(W, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_title("B   Overall pooled margins", loc="left", fontweight="bold")
    labels = [short_model(m) for m in models]
    axes[1].set_xticks(range(len(models)))
    axes[1].set_yticks(range(len(models)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_yticklabels(labels)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            value = W[i, j]
            color = "white" if abs(value) >= 0.55 else "#111827"
            axes[1].text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.4, color=color)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Pairwise margin")
    save(fig, "multipref_results")


def copy_moral_machine() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        shutil.copyfile(MORAL_FIGS / f"fig_experiment_a.{ext}", OUT / f"moral_machine_results.{ext}")


def main() -> None:
    set_style()
    make_prism()
    make_multipref()
    copy_moral_machine()
    print(f"Wrote candidate figures to {OUT}")


if __name__ == "__main__":
    main()
