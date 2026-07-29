from __future__ import annotations

import argparse
import ast
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.optimize import minimize
from tqdm import tqdm

ATTRIBUTE_KEYS = [
    "values",
    "fluency",
    "factuality",
    "safety",
    "diversity",
    "creativity",
    "helpfulness",
]


def ensure_comparison_columns(comps: pd.DataFrame) -> pd.DataFrame:
    """Ensure downstream filters have expected columns even when extraction produced no rows
    or an older parquet/partial path is used. This prevents KeyError on is_winner_edge.
    """
    expected_defaults = {
        "is_winner_edge": False,
        "sign": 0,
        "winner_loser_gap": np.nan,
        "user_id": None,
        "conversation_id": None,
        "model_i": None,
        "model_j": None,
        "score_delta": np.nan,
    }
    out = comps.copy()
    for col, default in expected_defaults.items():
        if col not in out.columns:
            out[col] = default
    # Attribute columns should exist so summaries/subsets fail gracefully rather than raising.
    for prefix in ["choice", "performance"]:
        for key in ATTRIBUTE_KEYS:
            col = f"{prefix}_{key}"
            if col not in out.columns:
                out[col] = np.nan
    return out

USER_COL_CANDIDATES = [
    "user_id", "participant_id", "participant", "survey_id", "prolific_id",
    "user_hash", "worker_id", "id_user", "uid"
]
CONV_COL_CANDIDATES = [
    "conversation_id", "conversation_tree_id", "tree_id", "conversation_idx",
    "dialogue_id", "id", "conv_id"
]
TYPE_COL_CANDIDATES = [
    "conversation_type", "conversation_type_id", "conversation_type_name", "type", "condition"
]


def parse_obj(x: Any) -> Any:
    """Parse PRISM cells from Hugging Face/Pandas robustly.

    PRISM cells may arrive as native Python lists/dicts, numpy object arrays,
    pyarrow scalar-like objects, JSON strings, or Python-literal strings. The
    previous version returned None for numpy/Arrow-backed conversation_history
    cells, which yielded zero parsed comparisons.
    """
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass

    if isinstance(x, (list, dict, tuple)):
        return list(x) if isinstance(x, tuple) else x

    if hasattr(x, "as_py"):
        try:
            return x.as_py()
        except Exception:
            pass

    if hasattr(x, "tolist"):
        try:
            v = x.tolist()
            if isinstance(v, (list, dict, tuple)):
                return list(v) if isinstance(v, tuple) else v
        except Exception:
            pass

    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    return None


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def score_sign(a: float, b: float, delta: float) -> int:
    d = a - b
    if d > delta:
        return 1
    if d < -delta:
        return -1
    return 0


@dataclass
class OpeningComparison:
    user_id: str
    conversation_id: str
    conversation_type: str
    model_i: str
    model_j: str
    sign: int  # +1 means i beats j, -1 means j beats i, 0 tie/uncertain
    score_i: float
    score_j: float
    global_winner: str
    is_winner_edge: bool
    winner_loser_gap: Optional[float]
    choice_attrs: Dict[str, Optional[float]]
    performance_attrs: Dict[str, Optional[float]]


def normalize_attrs(raw: Any, prefix: str) -> Dict[str, Optional[float]]:
    obj = parse_obj(raw)
    out = {f"{prefix}_{k}": None for k in ATTRIBUTE_KEYS}
    if isinstance(obj, dict):
        for k in ATTRIBUTE_KEYS:
            out[f"{prefix}_{k}"] = safe_float(obj.get(k))
    return out


def extract_opening_records(
    df: pd.DataFrame,
    history_col: str,
    choice_attr_col: Optional[str],
    perf_attr_col: Optional[str],
    score_deltas: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one row per pairwise opening-turn model comparison per score delta.

    The main opening tournament uses all pairwise comparisons among opening-turn models.
    Attribute-conditioned comparisons use only edges where the pairwise winner is the
    unique highest-rated opening-turn model, because PRISM choice/performance attributes
    are winner-conditioned.
    """
    user_col = first_present(df.columns, USER_COL_CANDIDATES)
    conv_col = first_present(df.columns, CONV_COL_CANDIDATES)
    type_col = first_present(df.columns, TYPE_COL_CANDIDATES)

    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing PRISM conversation_history")
    for idx, row in iterator:
        hist = parse_obj(row.get(history_col))
        if not isinstance(hist, list):
            audit_rows.append({"row_index": idx, "status": "bad_history", "n_opening_models": 0})
            continue

        models = []
        for item in hist:
            if not isinstance(item, dict):
                continue
            try:
                turn = int(item.get("turn"))
            except Exception:
                turn = item.get("turn")
            role = str(item.get("role", "")).lower()
            if turn != 0:
                continue
            if role != "model":
                continue
            model_name = item.get("model_name")
            score = safe_float(item.get("score"))
            if model_name is None or score is None:
                continue
            models.append({
                "model_name": str(model_name),
                "model_provider": item.get("model_provider"),
                "score": score,
                "within_turn_id": item.get("within_turn_id"),
                "if_chosen": item.get("if_chosen"),
            })

        # De-duplicate repeated model names in opening turn by keeping the first occurrence.
        seen = set()
        unique_models = []
        for m in models:
            if m["model_name"] in seen:
                continue
            seen.add(m["model_name"])
            unique_models.append(m)
        models = unique_models

        n_models = len(models)
        if n_models < 2:
            audit_rows.append({"row_index": idx, "status": "too_few_models", "n_opening_models": n_models})
            continue

        scores = np.array([m["score"] for m in models], dtype=float)
        max_score = float(scores.max())
        winner_positions = np.flatnonzero(scores == max_score)
        unique_winner = len(winner_positions) == 1
        global_winner = models[int(winner_positions[0])]["model_name"] if unique_winner else None

        choice_attrs = normalize_attrs(row.get(choice_attr_col) if choice_attr_col else None, "choice")
        perf_attrs = normalize_attrs(row.get(perf_attr_col) if perf_attr_col else None, "performance")

        user_id = str(row.get(user_col)) if user_col else f"row_{idx}"
        conversation_id = str(row.get(conv_col)) if conv_col else f"row_{idx}"
        conversation_type = str(row.get(type_col)) if type_col else "all"

        for delta in score_deltas:
            for a, b in combinations(range(n_models), 2):
                mi, mj = models[a]["model_name"], models[b]["model_name"]
                si, sj = models[a]["score"], models[b]["score"]
                sign = score_sign(si, sj, delta)

                pair_winner = None
                pair_loser = None
                gap = None
                if sign == 1:
                    pair_winner, pair_loser = mi, mj
                    gap = si - sj
                elif sign == -1:
                    pair_winner, pair_loser = mj, mi
                    gap = sj - si

                # Attributes are only meaningful for edges where the unique global winner
                # is the pairwise winner. Otherwise, the observed attributes do not describe
                # the pairwise winner.
                is_winner_edge = bool(unique_winner and sign != 0 and pair_winner == global_winner)

                out = {
                    "score_delta": delta,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "conversation_type": conversation_type,
                    "model_i": mi,
                    "model_j": mj,
                    "sign": sign,
                    "score_i": si,
                    "score_j": sj,
                    "global_winner": global_winner,
                    "is_winner_edge": is_winner_edge,
                    "winner_loser_gap": gap if is_winner_edge else np.nan,
                }
                out.update(choice_attrs)
                out.update(perf_attrs)
                rows.append(out)

        audit_rows.append({
            "row_index": idx,
            "status": "ok",
            "n_opening_models": n_models,
            "unique_winner": unique_winner,
            "conversation_type": conversation_type,
        })

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def aggregate_edges(
    comps: pd.DataFrame,
    aggregation: str = "user",
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate signed pairwise comparisons into skew edge margins.

    Returns one row per unordered model pair with columns: model_a, model_b, margin,
    support, total_observations, wins_a, wins_b.
    margin > 0 means model_a beats model_b.
    """
    d = comps[comps["sign"] != 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["model_a", "model_b", "margin", "support", "total_observations", "wins_a", "wins_b"])

    # canonical unordered pair. y is from model_a to model_b.
    a_vals, b_vals, ys, weights = [], [], [], []
    for _, r in d.iterrows():
        mi, mj, s = str(r["model_i"]), str(r["model_j"]), int(r["sign"])
        if mi <= mj:
            a, b, y = mi, mj, s
        else:
            a, b, y = mj, mi, -s
        w = 1.0
        if weight_col is not None:
            wv = safe_float(r.get(weight_col))
            if wv is None:
                continue
            w = max(wv, 0.0) / 100.0
            if w <= 0:
                continue
        a_vals.append(a); b_vals.append(b); ys.append(y); weights.append(w)

    dd = pd.DataFrame({"model_a": a_vals, "model_b": b_vals, "y": ys, "weight": weights})
    if dd.empty:
        return pd.DataFrame(columns=["model_a", "model_b", "margin", "support", "total_observations", "wins_a", "wins_b"])

    if aggregation == "user":
        dd["user_id"] = d.loc[dd.index if False else d.index[:0], "user_id"] if False else None
        # Rebuild with user IDs robustly because we iterated above.
        records = []
        for _, r in d.iterrows():
            mi, mj, s = str(r["model_i"]), str(r["model_j"]), int(r["sign"])
            if mi <= mj:
                a, b, y = mi, mj, s
            else:
                a, b, y = mj, mi, -s
            w = 1.0
            if weight_col is not None:
                wv = safe_float(r.get(weight_col))
                if wv is None:
                    continue
                w = max(wv, 0.0) / 100.0
                if w <= 0:
                    continue
            records.append({"user_id": r["user_id"], "model_a": a, "model_b": b, "y": y, "weight": w})
        dd = pd.DataFrame(records)
        if dd.empty:
            return pd.DataFrame(columns=["model_a", "model_b", "margin", "support", "total_observations", "wins_a", "wins_b"])
        # Within user-edge, take weighted mean. Then average users equally.
        dd["wy"] = dd["weight"] * dd["y"]
        ue = dd.groupby(["user_id", "model_a", "model_b"], as_index=False).agg(
            wy=("wy", "sum"), weight=("weight", "sum"), n_obs=("y", "size")
        )
        ue = ue[ue["weight"] > 0]
        ue["user_margin"] = ue["wy"] / ue["weight"]
        out = ue.groupby(["model_a", "model_b"], as_index=False).agg(
            margin=("user_margin", "mean"),
            support=("user_id", "nunique"),
            total_observations=("n_obs", "sum"),
        )
        # Approximate wins from user margins for diagnostics.
        out["wins_a"] = np.nan
        out["wins_b"] = np.nan
        return out

    if aggregation == "conversation":
        dd["wy"] = dd["weight"] * dd["y"]
        out = dd.groupby(["model_a", "model_b"], as_index=False).agg(
            wy=("wy", "sum"), weight=("weight", "sum"), support=("y", "size")
        )
        out = out[out["weight"] > 0]
        out["margin"] = out["wy"] / out["weight"]
        out["total_observations"] = out["support"]
        out["wins_a"] = np.nan
        out["wins_b"] = np.nan
        return out[["model_a", "model_b", "margin", "support", "total_observations", "wins_a", "wins_b"]]

    raise ValueError("aggregation must be 'user' or 'conversation'")


def edge_lookup(edges: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    look = {}
    for _, r in edges.iterrows():
        a, b = str(r["model_a"]), str(r["model_b"])
        margin, support = float(r["margin"]), float(r["support"])
        look[(a, b)] = (margin, support)
        look[(b, a)] = (-margin, support)
    return look


def cycle_stats(edges: pd.DataFrame, n_min: int, tau: float, top_k: int = 100) -> Tuple[Dict[str, Any], pd.DataFrame]:
    models = sorted(set(edges["model_a"]).union(set(edges["model_b"]))) if not edges.empty else []
    look = edge_lookup(edges)
    eligible = 0
    cycles = []
    for a, b, c in combinations(models, 3):
        needed = [(a, b), (b, c), (c, a)]
        if not all(e in look and look[e][1] >= n_min for e in needed):
            continue
        eligible += 1
        mab, sab = look[(a, b)]
        mbc, sbc = look[(b, c)]
        mca, sca = look[(c, a)]
        positive = mab > tau and mbc > tau and mca > tau
        negative = mab < -tau and mbc < -tau and mca < -tau
        if positive or negative:
            if positive:
                edges_text = [f"{a} > {b}", f"{b} > {c}", f"{c} > {a}"]
                margins = [mab, mbc, mca]
                supports = [sab, sbc, sca]
                ordered = [a, b, c]
            else:
                edges_text = [f"{b} > {a}", f"{c} > {b}", f"{a} > {c}"]
                margins = [-mab, -mbc, -mca]
                supports = [sab, sbc, sca]
                ordered = [b, c, a]
            cycles.append({
                "model_1": ordered[0], "model_2": ordered[1], "model_3": ordered[2],
                "edge_1": edges_text[0], "edge_2": edges_text[1], "edge_3": edges_text[2],
                "margin_1": margins[0], "margin_2": margins[1], "margin_3": margins[2],
                "support_1": supports[0], "support_2": supports[1], "support_3": supports[2],
                "weighted_cycle_intensity": abs(margins[0] * margins[1] * margins[2]),
                "min_abs_margin": min(abs(x) for x in margins),
            })
    cyclic = len(cycles)
    stat = {
        "n_models": len(models),
        "n_edges": int((edges["support"] >= n_min).sum()) if not edges.empty else 0,
        "eligible_triples": eligible,
        "cyclic_triples": cyclic,
        "cycle_rate": cyclic / eligible if eligible else np.nan,
        "weighted_cycle_intensity_sum": float(sum(x["weighted_cycle_intensity"] for x in cycles)),
        "weighted_cycle_intensity_mean": float(np.mean([x["weighted_cycle_intensity"] for x in cycles])) if cycles else 0.0,
    }
    cdf = pd.DataFrame(cycles)
    if not cdf.empty:
        cdf = cdf.sort_values("weighted_cycle_intensity", ascending=False).head(top_k)
    return stat, cdf


def hodge_projection(edges: pd.DataFrame, n_min: int) -> Dict[str, Any]:
    e = edges[edges["support"] >= n_min].copy()
    if e.empty:
        return {"n_models": 0, "n_edges": 0, "gradient_energy": np.nan, "cyclic_residual_energy": np.nan, "total_energy": np.nan, "rho_cyc": np.nan}
    models = sorted(set(e["model_a"]).union(set(e["model_b"])))
    idx = {m: k for k, m in enumerate(models)}
    B = np.zeros((len(e), len(models)))
    y = e["margin"].to_numpy(float)
    w = e["support"].to_numpy(float)
    for row_idx, (_, r) in enumerate(e.iterrows()):
        B[row_idx, idx[str(r["model_a"])] ] = 1.0
        B[row_idx, idx[str(r["model_b"])] ] = -1.0
    sw = np.sqrt(np.maximum(w, 1e-12))
    Bw = B * sw[:, None]
    yw = y * sw
    # Add tiny centering ridge to improve numerical stability.
    u, *_ = np.linalg.lstsq(Bw, yw, rcond=None)
    u = u - u.mean()
    yhat = B @ u
    total = float(np.sum(w * y * y))
    resid = float(np.sum(w * (y - yhat) ** 2))
    grad = float(np.sum(w * yhat * yhat))
    return {
        "n_models": len(models),
        "n_edges": len(e),
        "gradient_energy": grad,
        "cyclic_residual_energy": resid,
        "total_energy": total,
        "rho_cyc": resid / total if total > 0 else np.nan,
    }


def fit_bt_from_edges(edges: pd.DataFrame, n_min: int) -> Tuple[List[str], np.ndarray]:
    e = edges[edges["support"] >= n_min].copy()
    models = sorted(set(e["model_a"]).union(set(e["model_b"])))
    if not models:
        return [], np.array([])
    idx = {m: k for k, m in enumerate(models)}
    # Convert margin/support into pseudo wins/losses. This is an edge-level transitive null,
    # not a claim about exact individual ballots.
    data = []
    for _, r in e.iterrows():
        n = max(int(round(float(r["support"]))), 1)
        p_a = (float(r["margin"]) + 1.0) / 2.0
        p_a = min(max(p_a, 1e-4), 1 - 1e-4)
        wins_a = p_a * n
        wins_b = (1 - p_a) * n
        data.append((idx[str(r["model_a"])], idx[str(r["model_b"])], wins_a, wins_b))

    K = len(models)

    def nll(theta_free: np.ndarray) -> float:
        theta = np.zeros(K)
        theta[1:] = theta_free
        val = 0.0
        for ia, ib, wa, wb in data:
            z = theta[ia] - theta[ib]
            # stable log-sigmoid terms
            val += wa * np.logaddexp(0, -z) + wb * np.logaddexp(0, z)
        return float(val)

    res = minimize(nll, np.zeros(K - 1), method="BFGS")
    theta = np.zeros(K)
    if res.success or res.x.size == K - 1:
        theta[1:] = res.x
    theta -= theta.mean()
    return models, theta


def simulate_transitive_null(edges: pd.DataFrame, n_min: int, reps: int, tau: float, rng: np.random.Generator) -> pd.DataFrame:
    e = edges[edges["support"] >= n_min].copy()
    models, theta = fit_bt_from_edges(edges, n_min)
    if not models:
        return pd.DataFrame()
    idx = {m: k for k, m in enumerate(models)}
    rows = []
    for rep in tqdm(range(reps), desc="Transitive edge-level null", leave=False):
        sim_rows = []
        for _, r in e.iterrows():
            a, b = str(r["model_a"]), str(r["model_b"])
            n = max(int(round(float(r["support"]))), 1)
            z = theta[idx[a]] - theta[idx[b]]
            p = 1.0 / (1.0 + np.exp(-z))
            wins_a = rng.binomial(n, p)
            margin = (wins_a - (n - wins_a)) / n
            sim_rows.append({"model_a": a, "model_b": b, "margin": margin, "support": n})
        sim_edges = pd.DataFrame(sim_rows)
        cstat, _ = cycle_stats(sim_edges, n_min=n_min, tau=tau, top_k=0)
        hstat = hodge_projection(sim_edges, n_min=n_min)
        rows.append({"rep": rep, **cstat, **{f"hodge_{k}": v for k, v in hstat.items()}})
    return pd.DataFrame(rows)


def bootstrap_users(comps: pd.DataFrame, n_min: int, tau: float, reps: int, aggregation: str, rng: np.random.Generator) -> pd.DataFrame:
    users = np.array(sorted(comps["user_id"].dropna().astype(str).unique()))
    if len(users) == 0:
        return pd.DataFrame()
    user_groups = {u: g for u, g in comps.groupby(comps["user_id"].astype(str))}
    rows = []
    for rep in tqdm(range(reps), desc="User bootstrap", leave=False):
        sampled = rng.choice(users, size=len(users), replace=True)
        parts = []
        for boot_idx, u in enumerate(sampled):
            g = user_groups[u].copy()
            # Preserve duplicate sampled users as independent bootstrap units.
            g["user_id"] = f"{u}__boot{boot_idx}"
            parts.append(g)
        bd = pd.concat(parts, ignore_index=True)
        edges = aggregate_edges(bd, aggregation=aggregation)
        cstat, _ = cycle_stats(edges, n_min=n_min, tau=tau, top_k=0)
        hstat = hodge_projection(edges, n_min=n_min)
        rows.append({"rep": rep, **cstat, **hstat})
    return pd.DataFrame(rows)


def summarize_bootstrap(df: pd.DataFrame, label: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(label)
    if df.empty:
        return {**out, "bootstrap_reps": 0}
    for col in ["cycle_rate", "cyclic_triples", "rho_cyc"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                out[f"{col}_mean"] = float(s.mean())
                out[f"{col}_q025"] = float(s.quantile(0.025))
                out[f"{col}_q500"] = float(s.quantile(0.5))
                out[f"{col}_q975"] = float(s.quantile(0.975))
    out["bootstrap_reps"] = len(df)
    return out


def edge_attribute_profiles(comps: pd.DataFrame, cycles: pd.DataFrame, attr_prefix: str = "choice") -> pd.DataFrame:
    comps = ensure_comparison_columns(comps)
    """Deduplicated profiles for edges appearing in strongest cycles.

    Uses only winner-edge comparisons where the pairwise winner is the global winner,
    because only those comparisons are described by PRISM choice/performance attributes.
    """
    if cycles.empty:
        return pd.DataFrame()
    attrs = [f"{attr_prefix}_{k}" for k in ATTRIBUTE_KEYS]
    edge_texts = sorted(set(cycles.get("edge_1", pd.Series(dtype=str))).union(cycles.get("edge_2", pd.Series(dtype=str))).union(cycles.get("edge_3", pd.Series(dtype=str))))
    records = []
    winner_edges = comps[(comps["is_winner_edge"] == True) & (comps["sign"] != 0)].copy()
    for edge in edge_texts:
        if " > " not in edge:
            continue
        winner, loser = edge.split(" > ", 1)
        mask = (
            ((winner_edges["model_i"].astype(str) == winner) & (winner_edges["model_j"].astype(str) == loser) & (winner_edges["sign"] == 1)) |
            ((winner_edges["model_j"].astype(str) == winner) & (winner_edges["model_i"].astype(str) == loser) & (winner_edges["sign"] == -1))
        )
        sub = winner_edges[mask]
        rec = {"edge": edge, "support": int(len(sub))}
        if len(sub) == 0:
            for a in attrs:
                rec[f"mean_{a}"] = np.nan
            rec["dominant_attribute"] = None
            rec["dominant_attribute_value"] = np.nan
            rec["mean_winner_loser_score_gap"] = np.nan
        else:
            means = {}
            for a in attrs:
                means[a] = pd.to_numeric(sub[a], errors="coerce").mean() if a in sub.columns else np.nan
                rec[f"mean_{a}"] = means[a]
            valid = {k: v for k, v in means.items() if not pd.isna(v)}
            if valid:
                dom = max(valid, key=valid.get)
                rec["dominant_attribute"] = dom
                rec["dominant_attribute_value"] = float(valid[dom])
            else:
                rec["dominant_attribute"] = None
                rec["dominant_attribute_value"] = np.nan
            rec["mean_winner_loser_score_gap"] = pd.to_numeric(sub["winner_loser_gap"], errors="coerce").mean()
        records.append(rec)
    return pd.DataFrame(records).sort_values(["support", "dominant_attribute_value"], ascending=False)


def attribute_summary(comps: pd.DataFrame, prefix: str = "choice") -> pd.DataFrame:
    comps = ensure_comparison_columns(comps)
    winner_edges = comps[comps["is_winner_edge"] == True].copy()
    # Conversation-level unique winner edges repeat up to 3 times per conversation. Summarize on conversations.
    conv = winner_edges.drop_duplicates(["conversation_id"])
    rows = []
    for k in ATTRIBUTE_KEYS:
        col = f"{prefix}_{k}"
        if col not in conv.columns:
            continue
        s = pd.to_numeric(conv[col], errors="coerce")
        non = s.dropna()
        rows.append({
            "attribute": col,
            "n_conversations": len(conv),
            "non_null": int(non.shape[0]),
            "missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()) if len(s) else np.nan,
            "mean": float(non.mean()) if len(non) else np.nan,
            "q25": float(non.quantile(0.25)) if len(non) else np.nan,
            "median": float(non.quantile(0.5)) if len(non) else np.nan,
            "q75": float(non.quantile(0.75)) if len(non) else np.nan,
            "n_high_q75": int((s >= non.quantile(0.75)).sum()) if len(non) else 0,
            "n_low_q25": int((s <= non.quantile(0.25)).sum()) if len(non) else 0,
            "n_users_high_q75": int(conv.loc[s >= non.quantile(0.75), "user_id"].nunique()) if len(non) else 0,
            "n_users_low_q25": int(conv.loc[s <= non.quantile(0.25), "user_id"].nunique()) if len(non) else 0,
        })
    return pd.DataFrame(rows)


def select_attribute_subset(comps: pd.DataFrame, attr_col: str, q: float, side: str) -> Tuple[pd.DataFrame, float]:
    comps = ensure_comparison_columns(comps)
    winner_edges = comps[comps["is_winner_edge"] == True].copy()
    s = pd.to_numeric(winner_edges[attr_col], errors="coerce")
    threshold = float(s.dropna().quantile(q)) if s.dropna().shape[0] else np.nan
    if side == "high":
        sub = winner_edges[s >= threshold].copy()
    elif side == "low":
        sub = winner_edges[s <= threshold].copy()
    else:
        raise ValueError(side)
    return sub, threshold


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str, outpath: Path) -> None:
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(max(8, 0.55 * df[x].nunique()), 5))
    if hue and hue in df.columns:
        # Basic grouped bar without seaborn.
        groups = list(df[hue].dropna().unique())
        cats = list(df[x].dropna().unique())
        width = 0.8 / max(len(groups), 1)
        xpos = np.arange(len(cats))
        for gi, g in enumerate(groups):
            vals = []
            for c in cats:
                sub = df[(df[x] == c) & (df[hue] == g)]
                vals.append(float(sub[y].iloc[0]) if len(sub) else np.nan)
            plt.bar(xpos + gi * width - 0.4 + width / 2, vals, width=width, label=str(g))
        plt.xticks(xpos, cats, rotation=45, ha="right")
        plt.legend()
    else:
        data = df[[x, y]].dropna().sort_values(y, ascending=False)
        plt.bar(data[x].astype(str), data[y])
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PRISM Condorcet defensibility checks.")
    ap.add_argument("--dataset-name", default="HannahRoseKirk/prism-alignment")
    ap.add_argument("--subset", default="conversations")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--history-col", default="conversation_history")
    ap.add_argument("--choice-attr-col", default="choice_attributes")
    ap.add_argument("--performance-attr-col", default="performance_attributes")
    ap.add_argument("--aggregation", choices=["user", "conversation"], default="user")
    ap.add_argument("--score-deltas", nargs="+", type=float, default=[0, 2, 5, 10])
    ap.add_argument("--n-min-values", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--tau-values", nargs="+", type=float, default=[0, 0.05, 0.10])
    ap.add_argument("--attribute-prefix", choices=["choice", "performance"], default="choice")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--null-score-delta", type=float, default=5)
    ap.add_argument("--null-n-min", type=int, default=25)
    ap.add_argument("--null-tau", type=float, default=0.0)
    ap.add_argument("--top-k-cycles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out = Path(args.output_dir)
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading dataset with datasets.load_dataset...")
    load_attempts = []
    df = None
    for desc, call in [
        ("load_dataset(name, subset, split)", lambda: load_dataset(args.dataset_name, args.subset, split=args.split)),
        ("load_dataset(name, split)", lambda: load_dataset(args.dataset_name, split=args.split)),
        ("load_dataset(name)", lambda: load_dataset(args.dataset_name)),
    ]:
        try:
            ds = call()
            if hasattr(ds, "to_pandas"):
                cand = ds.to_pandas()
            else:
                # DatasetDict fallback: prefer requested subset, then split, then train, then first key.
                keys = list(ds.keys())
                if args.subset in ds:
                    cand = ds[args.subset].to_pandas()
                elif args.split in ds:
                    cand = ds[args.split].to_pandas()
                elif "train" in ds:
                    cand = ds["train"].to_pandas()
                else:
                    cand = ds[keys[0]].to_pandas()
            load_attempts.append({"strategy": desc, "status": "ok", "rows": len(cand), "columns": list(cand.columns)})
            # Prefer a table that actually has the requested history column.
            if args.history_col in cand.columns:
                df = cand
                break
            if df is None:
                df = cand
        except Exception as exc:
            load_attempts.append({"strategy": desc, "status": "error", "error": repr(exc)})
    if df is None:
        raise RuntimeError(f"Could not load dataset. Attempts: {load_attempts}")
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    if args.history_col not in df.columns:
        raise ValueError(f"Missing history column {args.history_col!r}. Available columns: {list(df.columns)}")
    choice_col = args.choice_attr_col if args.choice_attr_col in df.columns else None
    perf_col = args.performance_attr_col if args.performance_attr_col in df.columns else None
    if choice_col is None:
        print(f"WARNING: choice attribute column {args.choice_attr_col!r} not found.")
    if perf_col is None:
        print(f"WARNING: performance attribute column {args.performance_attr_col!r} not found.")

    comps, parse_audit = extract_opening_records(
        df, args.history_col, choice_col, perf_col, args.score_deltas
    )
    comps = ensure_comparison_columns(comps)
    comps.to_parquet(out / "opening_pairwise_comparisons.parquet", index=False)
    parse_audit.to_csv(out / "parse_audit.csv", index=False)

    schema = {
        "dataset_name": args.dataset_name,
        "subset": args.subset,
        "split": args.split,
        "history_col": args.history_col,
        "choice_attr_col": choice_col,
        "performance_attr_col": perf_col,
        "columns": list(df.columns),
        "n_raw_rows": len(df),
        "n_pairwise_rows": len(comps),
        "load_attempts": load_attempts,
        "parse_status_counts": parse_audit["status"].value_counts(dropna=False).to_dict() if "status" in parse_audit.columns else {},
        "opening_model_count_summary": parse_audit["n_opening_models"].describe().to_dict() if "n_opening_models" in parse_audit.columns and len(parse_audit) else {},
    }
    (out / "schema_report.json").write_text(json.dumps(schema, indent=2))

    print("Writing attribute summary...")
    attr_sum = attribute_summary(comps, prefix=args.attribute_prefix)
    attr_sum.to_csv(out / "attribute_summary.csv", index=False)
    plot_bar(attr_sum, "attribute", "missing_rate", None, "Attribute missing rate", figs / "attribute_missing_rate.png")
    plot_bar(attr_sum, "attribute", "n_high_q75", None, "High-q75 conversations by attribute", figs / "attribute_high_q75_counts.png")

    all_stats = []
    hodge_rows = []
    strongest = []
    print("Computing pooled opening-score tournament diagnostics...")
    for delta in tqdm(args.score_deltas, desc="Pooled score_delta"):
        base = comps[comps["score_delta"] == delta]
        edges = aggregate_edges(base, aggregation=args.aggregation)
        edges.to_csv(out / f"pooled_edges_delta{delta:g}.csv", index=False)
        for n_min in args.n_min_values:
            h = hodge_projection(edges, n_min=n_min)
            hodge_rows.append({"stratum": "pooled", "score_delta": delta, "n_min": n_min, **h})
            for tau in args.tau_values:
                cs, cyc = cycle_stats(edges, n_min=n_min, tau=tau, top_k=args.top_k_cycles)
                all_stats.append({"stratum": "pooled", "score_delta": delta, "n_min": n_min, "tau": tau, **cs})
                if not cyc.empty:
                    cyc.insert(0, "tau", tau)
                    cyc.insert(0, "n_min", n_min)
                    cyc.insert(0, "score_delta", delta)
                    cyc.insert(0, "stratum", "pooled")
                    strongest.append(cyc)

    pooled_stats = pd.DataFrame(all_stats)
    pooled_stats.to_csv(out / "pooled_cycle_stats.csv", index=False)
    pd.DataFrame(hodge_rows).to_csv(out / "pooled_hodge_stats.csv", index=False)
    strongest_df = pd.concat(strongest, ignore_index=True) if strongest else pd.DataFrame()
    strongest_df.to_csv(out / "pooled_strongest_cycles.csv", index=False)

    print("Computing deduplicated strongest-cycle edge attribute profiles...")
    profiles = edge_attribute_profiles(comps, strongest_df, attr_prefix=args.attribute_prefix)
    profiles.to_csv(out / "dedup_cycle_edge_attribute_profiles.csv", index=False)

    print("Computing high-vs-low attribute-conditioned tournaments...")
    attr_stats = []
    attr_hodge = []
    attr_cycles = []
    attr_cols = [f"{args.attribute_prefix}_{k}" for k in ATTRIBUTE_KEYS if f"{args.attribute_prefix}_{k}" in comps.columns]
    for attr in tqdm(attr_cols, desc="Attributes"):
        for side, q in [("high_q75", 0.75), ("low_q25", 0.25)]:
            for delta in args.score_deltas:
                cd = comps[comps["score_delta"] == delta]
                sub, threshold = select_attribute_subset(cd, attr, q=0.75 if side == "high_q75" else 0.25, side="high" if side == "high_q75" else "low")
                edges = aggregate_edges(sub, aggregation=args.aggregation)
                for n_min in args.n_min_values:
                    h = hodge_projection(edges, n_min=n_min)
                    attr_hodge.append({
                        "attribute": attr, "conditioning": side, "threshold": threshold,
                        "score_delta": delta, "n_min": n_min, **h
                    })
                    for tau in args.tau_values:
                        cs, cyc = cycle_stats(edges, n_min=n_min, tau=tau, top_k=args.top_k_cycles)
                        attr_stats.append({
                            "attribute": attr, "conditioning": side, "threshold": threshold,
                            "score_delta": delta, "n_min": n_min, "tau": tau, **cs
                        })
                        if not cyc.empty:
                            cyc.insert(0, "conditioning", side)
                            cyc.insert(0, "attribute", attr)
                            cyc.insert(0, "tau", tau)
                            cyc.insert(0, "n_min", n_min)
                            cyc.insert(0, "score_delta", delta)
                            attr_cycles.append(cyc)

    attr_stats_df = pd.DataFrame(attr_stats)
    attr_hodge_df = pd.DataFrame(attr_hodge)
    attr_cycles_df = pd.concat(attr_cycles, ignore_index=True) if attr_cycles else pd.DataFrame()
    attr_stats_df.to_csv(out / "attribute_high_low_cycle_stats.csv", index=False)
    attr_hodge_df.to_csv(out / "attribute_high_low_hodge_stats.csv", index=False)
    attr_cycles_df.to_csv(out / "attribute_high_low_strongest_cycles.csv", index=False)

    # Plot high vs low for a canonical setting.
    canon = attr_hodge_df[(attr_hodge_df["score_delta"] == args.null_score_delta) & (attr_hodge_df["n_min"] == args.null_n_min)].copy()
    if not canon.empty:
        plot_bar(canon, "attribute", "rho_cyc", "conditioning", f"Cyclic residual mass, delta={args.null_score_delta}, n_min={args.null_n_min}", figs / "rho_cyc_high_vs_low_attributes.png")
    canon_c = attr_stats_df[(attr_stats_df["score_delta"] == args.null_score_delta) & (attr_stats_df["n_min"] == args.null_n_min) & (attr_stats_df["tau"] == args.null_tau)].copy()
    if not canon_c.empty:
        plot_bar(canon_c, "attribute", "cycle_rate", "conditioning", f"Cycle rate, delta={args.null_score_delta}, n_min={args.null_n_min}, tau={args.null_tau}", figs / "cycle_rate_high_vs_low_attributes.png")

    print("Running user bootstraps for pooled and high-q75 attribute settings...")
    boot_rows = []
    if args.bootstrap_reps > 0:
        # pooled canonical
        base = comps[comps["score_delta"] == args.null_score_delta]
        bdf = bootstrap_users(base, n_min=args.null_n_min, tau=args.null_tau, reps=args.bootstrap_reps, aggregation=args.aggregation, rng=rng)
        boot_rows.append(summarize_bootstrap(bdf, {"setting": "pooled", "attribute": None, "conditioning": "pooled", "score_delta": args.null_score_delta, "n_min": args.null_n_min, "tau": args.null_tau}))
        bdf.to_csv(out / "bootstrap_pooled_raw.csv", index=False)
        for attr in tqdm(attr_cols, desc="Attribute bootstraps"):
            sub, threshold = select_attribute_subset(base, attr, q=0.75, side="high")
            if sub["user_id"].nunique() < 5:
                continue
            bdf = bootstrap_users(sub, n_min=args.null_n_min, tau=args.null_tau, reps=args.bootstrap_reps, aggregation=args.aggregation, rng=rng)
            boot_rows.append(summarize_bootstrap(bdf, {"setting": "attribute_high_q75", "attribute": attr, "conditioning": "high_q75", "threshold": threshold, "score_delta": args.null_score_delta, "n_min": args.null_n_min, "tau": args.null_tau}))
            bdf.to_csv(out / f"bootstrap_{attr}_high_q75_raw.csv", index=False)
    boot_summary = pd.DataFrame(boot_rows)
    boot_summary.to_csv(out / "bootstrap_summary.csv", index=False)
    if not boot_summary.empty and "rho_cyc_q500" in boot_summary.columns:
        tmp = boot_summary.copy()
        tmp["label"] = tmp["attribute"].fillna("pooled")
        plot_bar(tmp, "label", "rho_cyc_q500", None, "Bootstrap median cyclic residual mass", figs / "bootstrap_rho_cyc_median.png")

    print("Running transitive edge-level nulls for pooled and high-q75 attribute settings...")
    null_rows = []
    if args.null_reps > 0:
        # Helper for observed/null comparison.
        def run_null_for(label: Dict[str, Any], edge_df: pd.DataFrame) -> None:
            obs_c, _ = cycle_stats(edge_df, n_min=args.null_n_min, tau=args.null_tau, top_k=0)
            obs_h = hodge_projection(edge_df, n_min=args.null_n_min)
            nd = simulate_transitive_null(edge_df, n_min=args.null_n_min, reps=args.null_reps, tau=args.null_tau, rng=rng)
            nd.to_csv(out / f"null_raw_{label['name']}.csv", index=False)
            row = dict(label)
            row.update({f"observed_{k}": v for k, v in obs_c.items()})
            row.update({f"observed_hodge_{k}": v for k, v in obs_h.items()})
            if not nd.empty:
                for col, obsval in [("cycle_rate", obs_c.get("cycle_rate")), ("cyclic_triples", obs_c.get("cyclic_triples")), ("hodge_rho_cyc", obs_h.get("rho_cyc"))]:
                    s = pd.to_numeric(nd[col], errors="coerce").dropna() if col in nd.columns else pd.Series(dtype=float)
                    if len(s):
                        row[f"null_{col}_mean"] = float(s.mean())
                        row[f"null_{col}_q95"] = float(s.quantile(0.95))
                        row[f"null_{col}_q975"] = float(s.quantile(0.975))
                        row[f"null_{col}_sd"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
                        row[f"z_observed_{col}"] = float((obsval - s.mean()) / s.std(ddof=1)) if len(s) > 1 and s.std(ddof=1) > 0 and obsval is not None and not pd.isna(obsval) else np.nan
                        # One-sided empirical Monte Carlo p-value: probability that a
                        # transitive edge-level null produces a statistic at least as large
                        # as observed. Add-one smoothing avoids impossible p=0 with finite reps.
                        row[f"p_ge_observed_{col}"] = float((1 + (s >= obsval).sum()) / (len(s) + 1)) if obsval is not None and not pd.isna(obsval) else np.nan
            null_rows.append(row)

        base = comps[comps["score_delta"] == args.null_score_delta]
        pooled_edges = aggregate_edges(base, aggregation=args.aggregation)
        run_null_for({"name": "pooled", "attribute": None, "conditioning": "pooled", "score_delta": args.null_score_delta, "n_min": args.null_n_min, "tau": args.null_tau}, pooled_edges)
        for attr in tqdm(attr_cols, desc="Attribute nulls"):
            sub, threshold = select_attribute_subset(base, attr, q=0.75, side="high")
            edges = aggregate_edges(sub, aggregation=args.aggregation)
            if edges.empty or (edges["support"] >= args.null_n_min).sum() < 3:
                continue
            run_null_for({"name": f"{attr}_high_q75", "attribute": attr, "conditioning": "high_q75", "threshold": threshold, "score_delta": args.null_score_delta, "n_min": args.null_n_min, "tau": args.null_tau}, edges)
    null_summary = pd.DataFrame(null_rows)
    null_summary.to_csv(out / "transitive_null_summary.csv", index=False)
    if not null_summary.empty and "observed_hodge_rho_cyc" in null_summary.columns:
        tmp = null_summary.copy()
        tmp["label"] = tmp["attribute"].fillna("pooled")
        plot_bar(tmp, "label", "observed_hodge_rho_cyc", None, "Observed rho_cyc for null-tested settings", figs / "null_test_observed_rho_cyc.png")

    # Human-readable summary.
    summary = ["# PRISM Condorcet Defensibility Checks Summary", ""]
    summary.append(f"Parsed pairwise comparison rows: {len(comps):,}")
    if not parse_audit.empty and "status" in parse_audit.columns:
        summary.append("")
        summary.append("## Parse audit status counts")
        summary.append(parse_audit["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count").to_markdown(index=False))
    summary.append(f"Aggregation: {args.aggregation}")
    summary.append(f"Canonical null/check setting: score_delta={args.null_score_delta}, n_min={args.null_n_min}, tau={args.null_tau}")
    summary.append("")
    if not attr_sum.empty:
        summary.append("## Attribute missingness/coverage")
        summary.append(attr_sum.to_markdown(index=False))
        summary.append("")
    if not canon.empty:
        summary.append("## High vs low attribute cyclic residuals, canonical setting")
        show = canon[["attribute", "conditioning", "n_models", "n_edges", "rho_cyc"]].sort_values("rho_cyc", ascending=False)
        summary.append(show.to_markdown(index=False))
        summary.append("")
    if not boot_summary.empty:
        summary.append("## Bootstrap summary")
        summary.append(boot_summary.to_markdown(index=False))
        summary.append("")
    if not null_summary.empty:
        summary.append("## Transitive null summary")
        cols = [c for c in ["name", "attribute", "conditioning", "observed_cycle_rate", "null_cycle_rate_mean", "null_cycle_rate_q95", "p_ge_observed_cycle_rate", "z_observed_cycle_rate", "observed_hodge_rho_cyc", "null_hodge_rho_cyc_mean", "null_hodge_rho_cyc_q95", "p_ge_observed_hodge_rho_cyc", "z_observed_hodge_rho_cyc"] if c in null_summary.columns]
        summary.append(null_summary[cols].to_markdown(index=False))
        summary.append("")
    (out / "DEFENSIBILITY_CHECKS_SUMMARY.md").write_text("\n".join(summary))
    print(f"Done. Wrote outputs to {out}")
    print(f"Summary: {out / 'DEFENSIBILITY_CHECKS_SUMMARY.md'}")
    print(f"Figures: {figs}")


if __name__ == "__main__":
    main()
