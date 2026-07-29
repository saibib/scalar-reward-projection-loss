#!/usr/bin/env python3
"""Text-conditioned reward-model downstream experiment for MultiPref.

This is the more realistic CPU-only companion to
run_multipref_downstream_alignment.py.  Instead of fitting scalar scores for
model IDs, it fits a text-conditioned reward model:

    r_theta(prompt, completion) = theta^T TFIDF(prompt, completion)
    P(A preferred to B) = sigmoid(r_theta(prompt, A) - r_theta(prompt, B)).

The model is trained on held-in prompt IDs and evaluated on held-out prompt IDs.
For each split, annotation group, aspect, and 3-model region, the script asks
whether train-set rho_cyc predicts held-out failures of this text reward model.

This is still not RLHF/DPO, but it is a genuine preference reward model over
prompt/response text rather than a score table over model names.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning

from run_multipref_cycles_v4 import ASPECTS, pref_to_sign_weight, safe_str
from run_multipref_downstream_alignment import (
    aggregate_edges,
    hodge_residual,
    region_mask,
    stable_prompt_split,
    summarize_results,
    triplet_cycle_stats,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/allenai___multipref/default/0.0.0/"
    / "12910233a0238a997ebe425656e9dfed7b0ff031/multipref-train.arrow"
)
DEFAULT_OUTDIR = Path("src/results/multipref_text_reward_downstream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrow",
        type=Path,
        default=DEFAULT_ARROW,
        help="Cached MultiPref Arrow file. Read directly to avoid Hugging Face cache locks.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groups", nargs="+", default=["all", "normal", "expert"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-max-iter", type=int, default=30)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-raw-rows", type=int, default=None, help="Optional raw row cap for smoke tests.")
    return parser.parse_args()


def iter_annotation_dicts(raw_row: pd.Series) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for group, col in [
        ("normal", "normal_worker_annotations"),
        ("expert", "expert_worker_annotations"),
    ]:
        anns = raw_row.get(col)
        if isinstance(anns, np.ndarray):
            anns = anns.tolist()
        if isinstance(anns, dict):
            anns = [anns]
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if isinstance(ann, dict):
                yield group, ann


def load_raw(arrow_path: Path, max_raw_rows: Optional[int]) -> pd.DataFrame:
    if not arrow_path.exists():
        raise FileNotFoundError(f"Could not find cached MultiPref Arrow file: {arrow_path}")
    ds = Dataset.from_file(str(arrow_path))
    raw = ds.to_pandas()
    if max_raw_rows is not None:
        raw = raw.iloc[:max_raw_rows].copy()
    raw = raw.reset_index(drop=True)
    raw["raw_idx"] = np.arange(len(raw), dtype=int)
    return raw


def flatten_annotations(raw: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        r = row._asdict()
        for group, ann in iter_annotation_dicts(pd.Series(r)):
            rec: Dict[str, Any] = {
                "raw_idx": int(r["raw_idx"]),
                "comparison_id": safe_str(r.get("comparison_id")),
                "prompt_id": safe_str(r.get("prompt_id")),
                "prompt_text": safe_str(r.get("text")),
                "annotation_group": group,
                "model_a": safe_str(r.get("model_a")),
                "model_b": safe_str(r.get("model_b")),
            }
            for aspect in ASPECTS:
                sign, weight = pref_to_sign_weight(ann.get(f"{aspect}_pref"))
                rec[f"{aspect}_sign"] = sign
                rec[f"{aspect}_weight"] = weight
            records.append(rec)

    flat = pd.DataFrame(records)
    if flat.empty:
        return flat
    sign_cols = [f"{aspect}_sign" for aspect in ASPECTS]
    return flat[(flat[sign_cols].abs().sum(axis=1) > 0)].reset_index(drop=True)


def response_docs(raw: pd.DataFrame) -> Tuple[List[str], List[str]]:
    prompts = raw["text"].fillna("").astype(str)
    completion_a = raw["completion_a"].fillna("").astype(str)
    completion_b = raw["completion_b"].fillna("").astype(str)
    docs_a = ("Prompt:\n" + prompts + "\n\nResponse:\n" + completion_a).tolist()
    docs_b = ("Prompt:\n" + prompts + "\n\nResponse:\n" + completion_b).tolist()
    return docs_a, docs_b


def build_pair_features(
    raw: pd.DataFrame,
    max_features: int,
    min_df: int,
    ngram_max: int,
) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    docs_a, docs_b = response_docs(raw)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(docs_a + docs_b)
    x_a = vectorizer.transform(docs_a)
    x_b = vectorizer.transform(docs_b)
    return vectorizer, (x_a - x_b).tocsr()


def sorted_edge_observations(flat: pd.DataFrame, aspect: str, group: str) -> pd.DataFrame:
    sign_col = f"{aspect}_sign"
    weight_col = f"{aspect}_weight"
    if group == "all":
        use = flat
    else:
        use = flat[flat["annotation_group"] == group]
    use = use[(use[sign_col] != 0) & (use[weight_col] > 0)].copy()

    rows = []
    for raw_idx, prompt_id, model_a, model_b, sign, weight in use[
        ["raw_idx", "prompt_id", "model_a", "model_b", sign_col, weight_col]
    ].itertuples(index=False, name=None):
        a = str(model_a)
        b = str(model_b)
        if not a or not b or a == b:
            continue
        i, j = sorted((a, b))
        winner = a if int(sign) > 0 else b
        sorted_sign = 1 if winner == i else -1
        rows.append(
            {
                "raw_idx": int(raw_idx),
                "prompt_id": str(prompt_id),
                "model_a": a,
                "model_b": b,
                "i": i,
                "j": j,
                "sign": sorted_sign,
                "row_sign": int(sign),
                "weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def train_reward_model(
    xdiff_raw: sparse.csr_matrix,
    train_obs: pd.DataFrame,
    alpha: float,
    max_iter: int,
    seed: int,
) -> Tuple[Optional[SGDClassifier], Dict[str, Any]]:
    if train_obs.empty:
        return None, {"fit_ok": False, "reason": "empty_train"}

    y = (train_obs["row_sign"].to_numpy(dtype=int) > 0).astype(int)
    if len(np.unique(y)) < 2:
        return None, {"fit_ok": False, "reason": "single_class"}

    x = xdiff_raw[train_obs["raw_idx"].to_numpy(dtype=int)]
    w = train_obs["weight"].to_numpy(dtype=float)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
        fit_intercept=True,
        class_weight=None,
    )
    clf.fit(x, y, sample_weight=w)
    return clf, {
        "fit_ok": True,
        "n_train_obs": int(len(train_obs)),
        "train_weight": float(w.sum()),
        "n_features": int(x.shape[1]),
        "n_iter": int(getattr(clf, "n_iter_", 0)),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def evaluate_reward_model(
    clf: SGDClassifier,
    xdiff_raw: sparse.csr_matrix,
    test_region: pd.DataFrame,
) -> Dict[str, float]:
    if test_region.empty:
        return {
            "test_weight": 0.0,
            "test_n_obs": 0,
            "test_edges": 0,
            "test_log_loss": np.nan,
            "test_brier": np.nan,
            "test_accuracy": np.nan,
            "test_error_rate": np.nan,
            "test_calibration_abs_error": np.nan,
            "test_pair_calibration_mae": np.nan,
            "test_majority_edge_error": np.nan,
        }

    x = xdiff_raw[test_region["raw_idx"].to_numpy(dtype=int)]
    logit_a = clf.decision_function(x)
    p_a = sigmoid(logit_a)
    target_a = (test_region["row_sign"].to_numpy(dtype=int) > 0).astype(float)
    w = test_region["weight"].to_numpy(dtype=float)
    eps = 1e-12

    log_loss = -target_a * np.log(np.clip(p_a, eps, 1.0)) - (1.0 - target_a) * np.log(np.clip(1.0 - p_a, eps, 1.0))
    brier = (p_a - target_a) ** 2
    pred_a = p_a >= 0.5
    correct = (pred_a == (target_a > 0.5)).astype(float)

    eval_df = test_region.copy()
    eval_df["p_i"] = np.where(eval_df["model_a"] == eval_df["i"], p_a, 1.0 - p_a)
    eval_df["target_i"] = (eval_df["sign"] > 0).astype(float)

    pair_rows = []
    for (i, j), g in eval_df.groupby(["i", "j"], sort=True):
        gw = g["weight"].to_numpy(dtype=float)
        pair_weight = float(gw.sum())
        obs_rate = float(np.average(g["target_i"], weights=gw))
        pred_rate = float(np.average(g["p_i"], weights=gw))
        heldout_margin = 2.0 * obs_rate - 1.0
        pred_margin = 2.0 * pred_rate - 1.0
        if math.isclose(heldout_margin, 0.0) or math.isclose(pred_margin, 0.0):
            majority_error = 0.5
        else:
            majority_error = float(np.sign(heldout_margin) != np.sign(pred_margin))
        pair_rows.append((pair_weight, abs(obs_rate - pred_rate), majority_error))

    pair_weights = np.array([r[0] for r in pair_rows], dtype=float)
    pair_cal_errors = np.array([r[1] for r in pair_rows], dtype=float)
    pair_majority_errors = np.array([r[2] for r in pair_rows], dtype=float)

    return {
        "test_weight": float(w.sum()),
        "test_n_obs": int(len(test_region)),
        "test_edges": int(len(pair_rows)),
        "test_log_loss": float(np.average(log_loss, weights=w)),
        "test_brier": float(np.average(brier, weights=w)),
        "test_accuracy": float(np.average(correct, weights=w)),
        "test_error_rate": float(1.0 - np.average(correct, weights=w)),
        "test_calibration_abs_error": float(abs(np.average(target_a, weights=w) - np.average(p_a, weights=w))),
        "test_pair_calibration_mae": float(np.average(pair_cal_errors, weights=pair_weights)),
        "test_majority_edge_error": float(np.average(pair_majority_errors, weights=pair_weights)),
    }


def analyze_triplets(
    split_index: int,
    group: str,
    aspect: str,
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    clf: SGDClassifier,
    xdiff_raw: sparse.csr_matrix,
    args: argparse.Namespace,
    fit_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    nodes = sorted(set(train_obs["i"]).union(set(train_obs["j"])))
    triplets = list(__import__("itertools").combinations(nodes, 3))
    if args.max_triplets is not None:
        triplets = triplets[: args.max_triplets]

    rows: List[Dict[str, Any]] = []
    for triplet in triplets:
        train_region = train_obs[region_mask(train_obs, triplet)].copy()
        test_region = test_obs[region_mask(test_obs, triplet)].copy()
        train_edges = aggregate_edges(train_region)
        test_edges = aggregate_edges(test_region)
        eligible_train_edges = train_edges[train_edges["support"] >= args.min_train_edge_weight]
        if len(eligible_train_edges) < 3:
            continue
        if float(test_edges["support"].sum()) < args.min_test_weight or len(test_edges) < args.min_test_edges:
            continue

        residual = hodge_residual(train_edges, triplet, min_support=args.min_train_edge_weight)
        if not np.isfinite(residual["rho_cyc"]):
            continue
        cycle = triplet_cycle_stats(train_edges, triplet, min_support=args.min_train_edge_weight, tau=args.tau)
        evaluation = evaluate_reward_model(clf, xdiff_raw, test_region)

        edge_supports = eligible_train_edges["support"].to_numpy(dtype=float)
        edge_margins = eligible_train_edges["margin"].to_numpy(dtype=float)
        row: Dict[str, Any] = {
            "split_index": split_index,
            "group": group,
            "aspect": aspect,
            "region_type": "model_triplet",
            "triplet": " | ".join(triplet),
            "node_1": triplet[0],
            "node_2": triplet[1],
            "node_3": triplet[2],
            "train_n_obs": int(len(train_region)),
            "train_total_weight": float(train_region["weight"].sum()),
            "train_edges": int(len(train_edges)),
            "train_eligible_edges": int(len(eligible_train_edges)),
            "train_min_edge_support": float(np.min(edge_supports)),
            "train_mean_edge_support": float(np.mean(edge_supports)),
            "train_mean_abs_margin": float(np.mean(np.abs(edge_margins))),
            "train_max_abs_margin": float(np.max(np.abs(edge_margins))),
            "reward_fit_ok": bool(fit_meta["fit_ok"]),
            "reward_fit_n_train_obs": int(fit_meta.get("n_train_obs", 0)),
            "reward_fit_train_weight": float(fit_meta.get("train_weight", 0.0)),
            "reward_fit_n_features": int(fit_meta.get("n_features", 0)),
            "reward_fit_n_iter": int(fit_meta.get("n_iter", 0)),
        }
        row.update({f"train_{k}": v for k, v in residual.items()})
        row.update({f"train_{k}": v for k, v in cycle.items()})
        row.update(evaluation)
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args.arrow, args.max_raw_rows)
    flat = flatten_annotations(raw)
    vectorizer, xdiff_raw = build_pair_features(
        raw,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )

    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()
    rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    skipped = {"empty_obs": 0, "fit_failed": 0}

    for split_index in range(args.n_splits):
        train_prompts, test_prompts = stable_prompt_split(
            prompt_ids=prompt_ids,
            seed=args.seed,
            split_index=split_index,
            test_frac=args.test_frac,
        )
        for group in args.groups:
            for aspect in args.aspects:
                obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                if obs.empty:
                    skipped["empty_obs"] += 1
                    continue
                train_obs = obs[obs["prompt_id"].isin(train_prompts)].copy()
                test_obs = obs[obs["prompt_id"].isin(test_prompts)].copy()
                clf, fit_meta = train_reward_model(
                    xdiff_raw=xdiff_raw,
                    train_obs=train_obs,
                    alpha=args.sgd_alpha,
                    max_iter=args.sgd_max_iter,
                    seed=args.seed + split_index,
                )
                fit_rows.append(
                    {
                        "split_index": split_index,
                        "group": group,
                        "aspect": aspect,
                        **fit_meta,
                    }
                )
                if clf is None:
                    skipped["fit_failed"] += 1
                    continue
                rows.extend(
                    analyze_triplets(
                        split_index=split_index,
                        group=group,
                        aspect=aspect,
                        train_obs=train_obs,
                        test_obs=test_obs,
                        clf=clf,
                        xdiff_raw=xdiff_raw,
                        args=args,
                        fit_meta=fit_meta,
                    )
                )

    results = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_rows)
    summary = summarize_results(results) if not results.empty else pd.DataFrame()

    result_path = args.outdir / "multipref_text_reward_region_results.csv"
    fit_path = args.outdir / "multipref_text_reward_fit_summary.csv"
    summary_path = args.outdir / "multipref_text_reward_summary.csv"
    metadata_path = args.outdir / "multipref_text_reward_metadata.json"
    results.to_csv(result_path, index=False)
    fits.to_csv(fit_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "arrow": str(args.arrow),
        "n_raw_rows": int(len(raw)),
        "n_annotation_rows": int(len(flat)),
        "n_prompt_ids": int(len(prompt_ids)),
        "n_region_rows": int(len(results)),
        "n_splits": args.n_splits,
        "groups": args.groups,
        "aspects": args.aspects,
        "test_frac": args.test_frac,
        "max_features": args.max_features,
        "actual_features": int(xdiff_raw.shape[1]),
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
        "sgd_alpha": args.sgd_alpha,
        "sgd_max_iter": args.sgd_max_iter,
        "min_train_edge_weight": args.min_train_edge_weight,
        "min_test_weight": args.min_test_weight,
        "min_test_edges": args.min_test_edges,
        "tau": args.tau,
        "skipped": skipped,
        "result_path": str(result_path),
        "fit_path": str(fit_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
