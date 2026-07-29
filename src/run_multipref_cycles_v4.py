#!/usr/bin/env python3
"""
Run MultiPref Condorcet-cycle and scalar-projection experiments.

This version is written for the actual fine-grained MultiPref schema where
normal_worker_annotations and expert_worker_annotations are nested lists of
annotation dictionaries. Each annotation dictionary has fields such as:

  evaluator, overall_pref, helpful_pref, truthful_pref, harmless_pref,
  *_confidence, *_checked_reasons, *_own_reason, time_spent, timestamp.

Important design choice
-----------------------
The row-local labels "A" and "B" are never treated as global alternatives.
The script maps A/B preferences onto either:

  1. actual source-model alternatives, if stable model_a/model_b columns exist; or
  2. prompt-local response identifiers, if source-model columns are absent or only
     contain row-local placeholders such as A/B.

For the paper, the main estimand should usually be the prompt-local tournament:
for each prompt/conversation, compare the set of model outputs/response IDs that
answer that same prompt.

Example
-------
pip install datasets pandas numpy matplotlib

python run_multipref_cycles_v4.py \
  --dataset allenai/multipref \
  --split train \
  --outdir results/multipref_v4 \
  --tournament-mode both \
  --make-plots

If load_dataset(dataset, split=...) fails because the dataset exposes a named
config, pass --config <name>.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("Missing dependency. Install with: pip install datasets pandas numpy matplotlib") from exc


# -----------------------------
# Schema detection
# -----------------------------

PROMPT_ID_CANDIDATES = [
    "prompt_id", "conversation_id", "example_id", "query_id", "instance_id",
    "comparison_id", "id",
]
PROMPT_TEXT_CANDIDATES = [
    "prompt", "text", "instruction", "input", "query", "question", "conversation", "messages",
]
MODEL_A_CANDIDATES = [
    "model_a", "model_A", "model_0", "model0", "model_a_id", "model_0_id",
    "response_a_model", "response_0_model", "response_a_model_id", "response_0_model_id",
    "answer_a_model", "answer_0_model", "completion_a_model", "completion_0_model",
    "generator_a", "source_a", "generator_0", "source_0", "source_model_a", "source_model_0",
]
MODEL_B_CANDIDATES = [
    "model_b", "model_B", "model_1", "model1", "model_b_id", "model_1_id",
    "response_b_model", "response_1_model", "response_b_model_id", "response_1_model_id",
    "answer_b_model", "answer_1_model", "completion_b_model", "completion_1_model",
    "generator_b", "source_b", "generator_1", "source_1", "source_model_b", "source_model_1",
]
RESPONSE_A_CANDIDATES = [
    "response_a", "response_A", "response_0", "response0", "completion_a", "completion_0",
    "answer_a", "answer_0", "text_a", "text_0", "output_a", "output_0",
]
RESPONSE_B_CANDIDATES = [
    "response_b", "response_B", "response_1", "response1", "completion_b", "completion_1",
    "answer_b", "answer_1", "text_b", "text_1", "output_b", "output_1",
]
NORMAL_ANN_CANDIDATES = ["normal_worker_annotations", "crowd_worker_annotations", "worker_annotations", "normal_annotations"]
EXPERT_ANN_CANDIDATES = ["expert_worker_annotations", "expert_annotations"]
DIRECT_EVALUATOR_CANDIDATES = ["evaluator", "annotator_id", "worker_id", "rater_id", "user_id", "worker"]

ASPECTS = ["overall", "helpful", "truthful", "harmless"]
PREF_KEYS = {
    "overall": ["overall_pref", "overall_preference", "preference", "pref", "winner"],
    "helpful": ["helpful_pref", "helpfulness_pref", "helpful_preference", "helpfulness_preference"],
    "truthful": ["truthful_pref", "truthfulness_pref", "truthful_preference", "truthfulness_preference"],
    "harmless": ["harmless_pref", "harmlessness_pref", "harmless_preference", "harmlessness_preference"],
}
CONF_KEYS = {
    "overall": ["overall_confidence"],
    "helpful": ["helpful_confidence", "helpfulness_confidence"],
    "truthful": ["truthful_confidence", "truthfulness_confidence"],
    "harmless": ["harmless_confidence", "harmlessness_confidence"],
}


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def stable_hash(x: Any, prefix: str = "r") -> str:
    s = safe_str(x)
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{h}"


def get_first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    lower = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        real = lower.get(k.lower())
        if real is not None:
            return d.get(real, default)
    return default


def get_role_model_from_response_obj(x: Any) -> str:
    """Extract a source model ID/name from a nested response/completion object, if present."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, dict):
        val = get_first(x, [
            "model", "model_id", "model_name", "source_model", "source_model_id",
            "generator", "generator_id", "source", "source_id", "provider",
        ], None)
        if val is not None:
            return safe_str(val)
        # Some HF rows place metadata one level down.
        for nested_key in ["metadata", "meta", "response", "completion", "answer", "output"]:
            nested = get_first(x, [nested_key], None)
            out = get_role_model_from_response_obj(nested)
            if out:
                return out
    return ""


def extract_model_id(row: pd.Series, explicit_col: Optional[str], response_col: Optional[str], role: str) -> str:
    """
    Extract the stable source model ID/name for role A or B.

    Priority:
      1. explicit model/source column detected or passed by CLI;
      2. model/source metadata nested inside the response/completion object;
      3. empty string, which forces response-level alternatives in auto mode.
    """
    if explicit_col:
        v = safe_str(row.get(explicit_col))
        if v:
            return v
    if response_col:
        v = get_role_model_from_response_obj(row.get(response_col))
        if v:
            return v
    return ""


def normalize_pref(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def pref_to_sign_weight(x: Any, slight_weight: float = 0.5, tie_weight: float = 0.0) -> Tuple[int, float]:
    """Return (+1,A wins), (-1,B wins), or (0,tie/missing)."""
    if x is None:
        return 0, 0.0
    try:
        if pd.isna(x):
            return 0, 0.0
    except Exception:
        pass

    if isinstance(x, (int, np.integer)):
        if int(x) == 0:
            return +1, 1.0
        if int(x) == 1:
            return -1, 1.0
    if isinstance(x, (float, np.floating)) and math.isfinite(float(x)):
        if float(x) == 0.0:
            return +1, 1.0
        if float(x) == 1.0:
            return -1, 1.0

    s = normalize_pref(x)
    if s in {"", "none", "nan", "null", "tie", "equal", "both", "neither", "no-preference"}:
        return 0, tie_weight

    # MultiPref-style labels shown in the user's schema screenshot.
    if s in {"a-is-clearly-better", "a-clearly-better", "clearly-a", "a"}:
        return +1, 1.0
    if s in {"a-is-slightly-better", "a-slightly-better", "slightly-a"}:
        return +1, slight_weight
    if s in {"b-is-clearly-better", "b-clearly-better", "clearly-b", "b"}:
        return -1, 1.0
    if s in {"b-is-slightly-better", "b-slightly-better", "slightly-b"}:
        return -1, slight_weight

    # Broader variants.
    if re.search(r"(^|-)a($|-)", s) and "better" in s:
        return (+1, slight_weight if "slight" in s else 1.0)
    if re.search(r"(^|-)b($|-)", s) and "better" in s:
        return (-1, slight_weight if "slight" in s else 1.0)
    if s in {"response-a", "option-a", "left", "0", "response-0", "response-1"}:
        return +1, 1.0
    if s in {"response-b", "option-b", "right", "1", "response-2"}:
        return -1, 1.0

    return 0, 0.0


@dataclass
class DetectedColumns:
    prompt_id: str
    prompt_text: Optional[str]
    model_a: Optional[str]
    model_b: Optional[str]
    response_a: Optional[str]
    response_b: Optional[str]
    normal_annotations: Optional[str]
    expert_annotations: Optional[str]
    direct_evaluator: Optional[str]


def detect_columns(df: pd.DataFrame, args: argparse.Namespace) -> DetectedColumns:
    cols = list(df.columns)
    prompt_id = args.prompt_id_col or args.prompt_col or first_existing(cols, PROMPT_ID_CANDIDATES)
    prompt_text = args.prompt_text_col or first_existing(cols, PROMPT_TEXT_CANDIDATES)
    model_a = args.model_a_col or first_existing(cols, MODEL_A_CANDIDATES)
    model_b = args.model_b_col or first_existing(cols, MODEL_B_CANDIDATES)
    response_a = args.response_a_col or first_existing(cols, RESPONSE_A_CANDIDATES)
    response_b = args.response_b_col or first_existing(cols, RESPONSE_B_CANDIDATES)
    normal = args.normal_annotations_col or first_existing(cols, NORMAL_ANN_CANDIDATES)
    expert = args.expert_annotations_col or first_existing(cols, EXPERT_ANN_CANDIDATES)
    evaluator = args.evaluator_col or first_existing(cols, DIRECT_EVALUATOR_CANDIDATES)

    if prompt_id is None:
        raise ValueError(f"Could not detect prompt/conversation ID column. Columns: {cols}")
    if normal is None and expert is None:
        # Direct flat annotation mode is possible, but at least one preference key must exist.
        has_direct_pref = any(first_existing(cols, keys) for keys in PREF_KEYS.values())
        if not has_direct_pref:
            raise ValueError(
                "Could not detect nested annotation columns or direct preference columns. "
                f"Columns: {cols}"
            )
    if (model_a is None or model_b is None) and (response_a is None or response_b is None):
        raise ValueError(
            "Need either stable model_a/model_b columns or response_a/response_b columns. "
            f"Columns: {cols}"
        )

    return DetectedColumns(prompt_id, prompt_text, model_a, model_b, response_a, response_b, normal, expert, evaluator)


def model_columns_are_placeholders(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    vals = set(df[col_a].dropna().astype(str).str.strip().str.lower().unique()) | set(
        df[col_b].dropna().astype(str).str.strip().str.lower().unique()
    )
    placeholders = {"a", "b", "model a", "model b", "model_a", "model_b", "response a", "response b", "0", "1"}
    return bool(vals) and vals.issubset(placeholders)


def choose_alt_mode(df: pd.DataFrame, dc: DetectedColumns, requested: str) -> str:
    if requested != "auto":
        return requested
    if dc.model_a and dc.model_b and not model_columns_are_placeholders(df, dc.model_a, dc.model_b):
        return "model"
    return "response"


# -----------------------------
# Flattening MultiPref annotations
# -----------------------------

def iter_annotation_dicts(row: pd.Series, dc: DetectedColumns) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (annotation_group, annotation_dict)."""
    yielded = False
    for group, col in [("normal", dc.normal_annotations), ("expert", dc.expert_annotations)]:
        if col is None:
            continue
        anns = row.get(col)
        if anns is None:
            continue
        # HF -> pandas may give np.ndarray, list, or object arrays.
        if isinstance(anns, np.ndarray):
            anns = anns.tolist()
        if isinstance(anns, dict):
            anns = [anns]
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if isinstance(ann, dict):
                yielded = True
                yield group, ann
    if not yielded:
        # Flat fallback: use the row itself as one annotation.
        d = row.to_dict()
        yield safe_str(d.get("annotation_group", "flat")) or "flat", d


def flatten_annotations(raw: pd.DataFrame, dc: DetectedColumns, alt_mode: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    direct_cols = list(raw.columns)

    # Direct flat preference columns, if nested columns are absent.
    direct_pref_cols = {aspect: first_existing(direct_cols, keys) for aspect, keys in PREF_KEYS.items()}

    for ridx, row in raw.iterrows():
        prompt_id = safe_str(row[dc.prompt_id])
        prompt_text = safe_str(row[dc.prompt_text]) if dc.prompt_text else ""

        response_a = row[dc.response_a] if dc.response_a else None
        response_b = row[dc.response_b] if dc.response_b else None
        model_a = extract_model_id(row, dc.model_a, dc.response_a, "A")
        model_b = extract_model_id(row, dc.model_b, dc.response_b, "B")

        if alt_mode == "model" and model_a and model_b:
            alt_a, alt_b = model_a, model_b
        else:
            # Response IDs must be prompt-local to avoid treating identical text hashes across
            # different prompts as the same alternative accidentally.
            base_a = response_a if response_a is not None else f"row={ridx}:A:model={model_a}"
            base_b = response_b if response_b is not None else f"row={ridx}:B:model={model_b}"
            alt_a = f"{prompt_id}::{stable_hash(base_a, 'A')}"
            alt_b = f"{prompt_id}::{stable_hash(base_b, 'B')}"

        for ann_group, ann in iter_annotation_dicts(row, dc):
            rec: Dict[str, Any] = {
                "row_id": ridx,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "annotation_group": ann_group,
                "evaluator": safe_str(get_first(ann, ["evaluator", "annotator_id", "worker_id", "rater_id", "user_id"], row.get(dc.direct_evaluator) if dc.direct_evaluator else "")),
                "user_id": safe_str(get_first(ann, ["evaluator", "annotator_id", "worker_id", "rater_id", "user_id"], row.get(dc.direct_evaluator) if dc.direct_evaluator else "")),
                "alt_a": alt_a,
                "alt_b": alt_b,
                "model_a": model_a,
                "model_b": model_b,
                "time_spent": get_first(ann, ["time_spent"], None),
                "timestamp": get_first(ann, ["timestamp"], None),
            }

            for aspect in ASPECTS:
                val = get_first(ann, PREF_KEYS[aspect], None)
                if val is None and direct_pref_cols.get(aspect):
                    val = row[direct_pref_cols[aspect]]
                conf = get_first(ann, CONF_KEYS[aspect], None)
                sign, weight = pref_to_sign_weight(val)
                rec[f"{aspect}_pref_raw"] = val
                rec[f"{aspect}_sign"] = sign
                rec[f"{aspect}_weight"] = weight
                rec[f"{aspect}_confidence"] = conf
            records.append(rec)

    flat = pd.DataFrame(records)
    # Drop empty annotations with no usable preference on any aspect.
    if not flat.empty:
        sign_cols = [f"{a}_sign" for a in ASPECTS]
        flat = flat[(flat[sign_cols].abs().sum(axis=1) > 0)].reset_index(drop=True)
    return flat


# -----------------------------
# Edge flow statistics
# -----------------------------

@dataclass
class EdgeFlow:
    edges: Dict[Tuple[str, str], Dict[str, float]]
    nodes: List[str]


def add_observation(edges: Dict[Tuple[str, str], Dict[str, float]], a: str, b: str, sign: int, weight: float) -> None:
    if sign == 0 or weight <= 0 or not a or not b or a == b:
        return
    i, j = sorted([str(a), str(b)])
    key = (i, j)
    if key not in edges:
        edges[key] = {"support": 0.0, "weighted_margin": 0.0, "i_wins": 0.0, "j_wins": 0.0}
    winner = a if sign > 0 else b
    sorted_sign = +1 if str(winner) == i else -1
    edges[key]["support"] += float(weight)
    edges[key]["weighted_margin"] += float(weight) * sorted_sign
    if sorted_sign > 0:
        edges[key]["i_wins"] += float(weight)
    else:
        edges[key]["j_wins"] += float(weight)


def build_edge_flow(df: pd.DataFrame, aspect: str) -> EdgeFlow:
    edges: Dict[Tuple[str, str], Dict[str, float]] = {}
    nodes = set()
    sign_col = f"{aspect}_sign"
    weight_col = f"{aspect}_weight"
    for a, b, sign, weight in df[["alt_a", "alt_b", sign_col, weight_col]].itertuples(index=False, name=None):
        nodes.add(str(a)); nodes.add(str(b))
        add_observation(edges, str(a), str(b), int(sign), float(weight))
    return EdgeFlow(edges=edges, nodes=sorted(nodes))


def margins(flow: EdgeFlow, min_support: float, tau: float = 0.0) -> Dict[Tuple[str, str], float]:
    out = {}
    for key, st in flow.edges.items():
        supp = st["support"]
        if supp >= min_support and supp > 0:
            m = st["weighted_margin"] / supp
            if abs(m) > tau:
                out[key] = float(m)
    return out


def get_margin(m: Dict[Tuple[str, str], float], i: str, j: str) -> Optional[float]:
    if i == j:
        return None
    key = tuple(sorted([i, j]))
    if key not in m:
        return None
    val = m[key]
    return val if key[0] == i else -val


def cycle_stats(flow: EdgeFlow, min_support: float, tau: float) -> Dict[str, Any]:
    m = margins(flow, min_support=min_support, tau=tau)
    nodes = flow.nodes
    eligible = 0
    cyclic = 0
    intensity_sum = 0.0
    strongest: List[Dict[str, Any]] = []

    for i, j, k in itertools.combinations(nodes, 3):
        mij = get_margin(m, i, j)
        mjk = get_margin(m, j, k)
        mki = get_margin(m, k, i)
        if mij is None or mjk is None or mki is None:
            continue
        eligible += 1
        forward = mij > 0 and mjk > 0 and mki > 0
        reverse = mij < 0 and mjk < 0 and mki < 0
        if forward or reverse:
            cyclic += 1
            intensity = abs(mij * mjk * mki)
            intensity_sum += intensity
            strongest.append({
                "i": i, "j": j, "k": k,
                "m_ij": mij, "m_jk": mjk, "m_ki": mki,
                "intensity": intensity,
                "orientation": "forward" if forward else "reverse",
            })

    strongest.sort(key=lambda x: x["intensity"], reverse=True)
    return {
        "nodes": len(nodes),
        "observed_edges": len(flow.edges),
        "eligible_edges": len(m),
        "eligible_triples": eligible,
        "cyclic_triples": cyclic,
        "cycle_rate": np.nan if eligible == 0 else cyclic / eligible,
        "weighted_cycle_intensity": np.nan if eligible == 0 else intensity_sum / eligible,
        "strongest_cycles": strongest,
    }


def scalar_projection_residual(flow: EdgeFlow, min_support: float) -> Dict[str, float]:
    edge_rows = []
    for (i, j), st in flow.edges.items():
        supp = st["support"]
        if supp >= min_support and supp > 0:
            edge_rows.append((i, j, st["weighted_margin"] / supp, supp))
    nodes = sorted({x for e in edge_rows for x in e[:2]})
    n = len(nodes)
    if len(edge_rows) == 0 or n < 2:
        return {"gradient_energy": np.nan, "residual_energy": np.nan, "rho_cyc": np.nan}

    idx = {node: ii for ii, node in enumerate(nodes)}
    A = np.zeros((len(edge_rows), max(n - 1, 1)))
    y = np.zeros(len(edge_rows))
    w = np.zeros(len(edge_rows))
    for r, (i, j, val, supp) in enumerate(edge_rows):
        ii, jj = idx[i], idx[j]
        if ii < n - 1:
            A[r, ii] += 1.0
        if jj < n - 1:
            A[r, jj] -= 1.0
        y[r] = val
        w[r] = supp
    sw = np.sqrt(w)
    try:
        u, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        fit = A @ u
    except np.linalg.LinAlgError:
        return {"gradient_energy": np.nan, "residual_energy": np.nan, "rho_cyc": np.nan}

    total = float(np.sum(w * y**2))
    grad = float(np.sum(w * fit**2))
    res = float(np.sum(w * (y - fit) ** 2))
    return {
        "gradient_energy": grad,
        "residual_energy": res,
        "rho_cyc": np.nan if total <= 0 else res / total,
    }


def analyze_subset(df: pd.DataFrame, aspect: str, min_support: float, tau: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flow = build_edge_flow(df, aspect)
    cs = cycle_stats(flow, min_support=min_support, tau=tau)
    pr = scalar_projection_residual(flow, min_support=min_support)
    out = {k: v for k, v in cs.items() if k != "strongest_cycles"}
    out.update(pr)
    return out, cs["strongest_cycles"]


# -----------------------------
# Experiment orchestration
# -----------------------------


def aspect_disagreement(flat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(ASPECTS, 2):
        sa = flat[f"{a}_sign"]
        sb = flat[f"{b}_sign"]
        mask = (sa != 0) & (sb != 0)
        n = int(mask.sum())
        d = int((sa[mask] != sb[mask]).sum()) if n else 0
        rows.append({
            "aspect_a": a,
            "aspect_b": b,
            "n_comparable": n,
            "n_disagree": d,
            "disagreement_rate": np.nan if n == 0 else d / n,
        })
    return pd.DataFrame(rows)


def run_global(flat: pd.DataFrame, aspects: Sequence[str], min_support: float, tau: float,
               min_group_rows: int, max_evaluators: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    masks: Dict[str, pd.Series] = {"all": pd.Series(True, index=flat.index)}
    for group in sorted(flat["annotation_group"].dropna().astype(str).unique()):
        mask = flat["annotation_group"].astype(str) == group
        if int(mask.sum()) >= min_group_rows:
            masks[f"group={group}"] = mask

    # Evaluator-level global analysis only where enough total rows exist.
    counts = flat.loc[flat["evaluator"].astype(str) != "", "evaluator"].astype(str).value_counts()
    for ev, count in counts.head(max_evaluators).items():
        if count >= min_group_rows:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ev))[:80]
            masks[f"evaluator={safe}"] = flat["evaluator"].astype(str) == str(ev)

    rows = []
    cycles = []
    for name, mask in masks.items():
        sub = flat.loc[mask]
        for aspect in aspects:
            stats, cyc = analyze_subset(sub, aspect, min_support, tau)
            rows.append({"scope": "global", "group": name, "prompt_id": "ALL", "aspect": aspect, "n_annotations": len(sub), **stats})
            for c in cyc[:50]:
                cycles.append({"scope": "global", "group": name, "prompt_id": "ALL", "aspect": aspect, **c})
    return pd.DataFrame(rows), pd.DataFrame(cycles)


def run_prompt_local(flat: pd.DataFrame, aspects: Sequence[str], min_support: float, tau: float,
                     prompt_group_col: str,
                     min_prompt_edges: int, min_prompt_nodes: int, max_prompts: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cycles = []
    grouped = list(flat.groupby(prompt_group_col, dropna=False))
    if max_prompts is not None:
        grouped = grouped[:max_prompts]

    for prompt_group, gdf in grouped:
        nodes = set(gdf["alt_a"].astype(str)) | set(gdf["alt_b"].astype(str))
        if len(nodes) < min_prompt_nodes:
            continue
        for group_name, sub in [("pooled", gdf)] + [(f"group={g}", x) for g, x in gdf.groupby("annotation_group")]:
            if len(set(sub["alt_a"].astype(str)) | set(sub["alt_b"].astype(str))) < min_prompt_nodes:
                continue
            for aspect in aspects:
                flow = build_edge_flow(sub, aspect)
                if len(flow.edges) < min_prompt_edges:
                    continue
                stats, cyc = analyze_subset(sub, aspect, min_support, tau)
                rows.append({
                    "scope": "prompt_local",
                    "group": group_name,
                    "prompt_id": safe_str(gdf["prompt_id"].iloc[0]) if "prompt_id" in gdf.columns and gdf["prompt_id"].nunique(dropna=False) == 1 else "",
                    "prompt_group": prompt_group,
                    "aspect": aspect,
                    "n_annotations": len(sub),
                    **stats,
                })
                for c in cyc[:20]:
                    cycles.append({
                        "scope": "prompt_local",
                        "group": group_name,
                        "prompt_id": safe_str(gdf["prompt_id"].iloc[0]) if "prompt_id" in gdf.columns and gdf["prompt_id"].nunique(dropna=False) == 1 else "",
                        "prompt_group": prompt_group,
                        "aspect": aspect,
                        **c,
                    })

    return pd.DataFrame(rows), pd.DataFrame(cycles)


def prompt_support_audit(flat: pd.DataFrame, prompt_group_col: str) -> pd.DataFrame:
    rows = []
    work = flat[flat["alt_a"].notna() & flat["alt_b"].notna()].copy()
    work = work[work["alt_a"].astype(str) != work["alt_b"].astype(str)]
    if work.empty:
        return pd.DataFrame(columns=[
            "prompt_group", "n_prompt_ids", "n_annotations", "n_models", "n_edges",
            "n_possible_edges", "n_supported_triangles", "models", "edges",
            "supported_triangles",
        ])

    work["edge"] = work.apply(lambda r: tuple(sorted([str(r["alt_a"]), str(r["alt_b"])])), axis=1)
    for prompt_group, g in work.groupby(prompt_group_col, dropna=False):
        edge_rows = g.drop_duplicates("edge")
        models = sorted(set(edge_rows["alt_a"].astype(str)) | set(edge_rows["alt_b"].astype(str)))
        edges = set(edge_rows["edge"])
        supported_triangles = []
        for tri in itertools.combinations(models, 3):
            required = {tuple(sorted(pair)) for pair in itertools.combinations(tri, 2)}
            if required.issubset(edges):
                supported_triangles.append(tri)

        rows.append({
            "prompt_group": prompt_group,
            "n_prompt_ids": int(g["prompt_id"].nunique(dropna=False)) if "prompt_id" in g.columns else 0,
            "n_annotations": int(len(g)),
            "n_models": int(len(models)),
            "n_edges": int(len(edges)),
            "n_possible_edges": int(len(models) * (len(models) - 1) // 2),
            "n_supported_triangles": int(len(supported_triangles)),
            "models": json.dumps(models, ensure_ascii=False),
            "edges": json.dumps(sorted([list(e) for e in edges]), ensure_ascii=False),
            "supported_triangles": json.dumps([list(t) for t in supported_triangles], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def write_prompt_support_outputs(flat: pd.DataFrame, prompt_group_col: str, outdir: Path) -> pd.DataFrame:
    audit = prompt_support_audit(flat, prompt_group_col)
    audit.to_csv(outdir / "multipref_prompt_support_audit.csv", index=False)

    summary: Dict[str, Any] = {
        "prompt_group_col": prompt_group_col,
        "n_prompt_groups": int(len(audit)),
        "n_groups_with_supported_triangles": int((audit["n_supported_triangles"] > 0).sum()) if not audit.empty else 0,
        "max_edges_per_prompt_group": int(audit["n_edges"].max()) if not audit.empty else 0,
        "max_supported_triangles_per_prompt_group": int(audit["n_supported_triangles"].max()) if not audit.empty else 0,
        "distributions": {},
    }
    for col in ["n_models", "n_edges", "n_supported_triangles"]:
        if audit.empty:
            summary["distributions"][col] = {}
        else:
            summary["distributions"][col] = {
                str(k): int(v) for k, v in audit[col].value_counts().sort_index().items()
            }
    with open(outdir / "multipref_prompt_support_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return audit


def model_edge_support(flat: pd.DataFrame, aspects: Sequence[str]) -> pd.DataFrame:
    work = flat[flat["alt_a"].notna() & flat["alt_b"].notna()].copy()
    work = work[work["alt_a"].astype(str) != work["alt_b"].astype(str)]
    if work.empty:
        cols = ["model_i", "model_j", "annotation_count", "normal_count", "expert_count"]
        cols.extend([f"{a}_non_tie_count" for a in aspects])
        return pd.DataFrame(columns=cols)

    work["model_i"] = work.apply(lambda r: sorted([str(r["alt_a"]), str(r["alt_b"])])[0], axis=1)
    work["model_j"] = work.apply(lambda r: sorted([str(r["alt_a"]), str(r["alt_b"])])[1], axis=1)

    rows = []
    for (model_i, model_j), g in work.groupby(["model_i", "model_j"], dropna=False):
        row = {
            "model_i": model_i,
            "model_j": model_j,
            "annotation_count": int(len(g)),
            "normal_count": int((g["annotation_group"].astype(str) == "normal").sum()),
            "expert_count": int((g["annotation_group"].astype(str) == "expert").sum()),
        }
        for aspect in aspects:
            row[f"{aspect}_non_tie_count"] = int((g[f"{aspect}_sign"] != 0).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("annotation_count", ascending=False).reset_index(drop=True)


def make_plots(summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    p = summary[(summary["scope"] == "prompt_local") & summary["rho_cyc"].notna()].copy()
    if not p.empty:
        for metric, ylabel, fname in [
            ("rho_cyc", r"Prompt-local cyclic residual mass, $\rho_{cyc}$", "multipref_prompt_local_rho_cyc"),
            ("cycle_rate", "Prompt-local Condorcet 3-cycle rate", "multipref_prompt_local_cycle_rate"),
        ]:
            aspects = [a for a in ASPECTS if a in set(p["aspect"])]
            data = [p.loc[p["aspect"] == a, metric].dropna().values for a in aspects]
            if aspects and any(len(x) for x in data):
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.boxplot(data, labels=aspects, showfliers=False)
                ax.set_ylabel(ylabel)
                ax.set_title("MultiPref prompt-local preference structure")
                plt.tight_layout()
                plt.savefig(figdir / f"{fname}.pdf")
                plt.savefig(figdir / f"{fname}.png", dpi=200)
                plt.close()

    g = summary[(summary["scope"] == "global") & summary["rho_cyc"].notna()].copy()
    if not g.empty:
        g0 = g[g["group"].isin(["all", "group=normal", "group=expert"])]
        if not g0.empty:
            pivot = g0.pivot_table(index="group", columns="aspect", values="rho_cyc", aggfunc="mean")
            if not pivot.empty:
                ax = pivot.plot(kind="bar", figsize=(8, 4))
                ax.set_ylabel(r"Global cyclic residual mass, $\rho_{cyc}$")
                ax.set_xlabel("")
                ax.set_title("MultiPref scalar-projection residual by annotator group")
                plt.tight_layout()
                plt.savefig(figdir / "multipref_global_rho_cyc.pdf")
                plt.savefig(figdir / "multipref_global_rho_cyc.png", dpi=200)
                plt.savefig(figdir / "multipref_global_rho_cyc_by_group.pdf")
                plt.savefig(figdir / "multipref_global_rho_cyc_by_group.png", dpi=200)
                plt.close()

            pivot_cycle = g0.pivot_table(index="group", columns="aspect", values="cycle_rate", aggfunc="mean")
            if not pivot_cycle.empty:
                ax = pivot_cycle.plot(kind="bar", figsize=(8, 4))
                ax.set_ylabel("Global Condorcet 3-cycle rate")
                ax.set_xlabel("")
                ax.set_title("MultiPref global cycle rate by annotator group")
                plt.tight_layout()
                plt.savefig(figdir / "multipref_global_cycle_rate.pdf")
                plt.savefig(figdir / "multipref_global_cycle_rate.png", dpi=200)
                plt.close()


def write_audit(raw: pd.DataFrame, flat: pd.DataFrame, dc: DetectedColumns, alt_mode: str,
                tournament_mode: str, prompt_group_col: str, outdir: Path) -> None:
    audit: Dict[str, Any] = {
        "raw_rows": int(len(raw)),
        "flattened_annotation_rows": int(len(flat)),
        "raw_columns": list(raw.columns),
        "detected_columns": dc.__dict__,
        "alt_mode": alt_mode,
        "tournament_mode": tournament_mode,
        "prompt_group_col": prompt_group_col,
        "preference_weighting": {
            "tie": 0.0,
            "slight": 0.5,
            "clear": 1.0,
            "ties_in_edges": "dropped because zero-weight observations do not add edge support",
            "status": "descriptive; bootstrap and transitive-null uncertainty are not implemented in this script",
        },
        "n_prompts": int(flat["prompt_id"].nunique(dropna=True)) if not flat.empty else 0,
        "n_prompt_texts": int(flat["prompt_text"].nunique(dropna=True)) if (not flat.empty and "prompt_text" in flat.columns) else 0,
        "annotation_group_counts": flat["annotation_group"].value_counts(dropna=False).to_dict() if not flat.empty else {},
        "n_evaluators": int(flat["evaluator"].nunique(dropna=True)) if not flat.empty else 0,
        "n_user_ids": int(flat["user_id"].nunique(dropna=True)) if (not flat.empty and "user_id" in flat.columns) else 0,
        "n_alternatives": int(pd.concat([flat["alt_a"], flat["alt_b"]]).nunique(dropna=True)) if not flat.empty else 0,
    }
    for aspect in ASPECTS:
        if not flat.empty:
            audit[f"{aspect}_raw_counts"] = flat[f"{aspect}_pref_raw"].astype(str).value_counts(dropna=False).to_dict()
            audit[f"{aspect}_sign_counts"] = flat[f"{aspect}_sign"].value_counts(dropna=False).to_dict()
    with open(outdir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Condorcet/Hodge-style experiments on MultiPref.")
    p.add_argument("--dataset", default="allenai/multipref")
    p.add_argument("--config", default=None, help="Optional HF config name. Omit if dataset has a default config.")
    p.add_argument("--split", default="train")
    p.add_argument("--outdir", default="results/multipref_v4")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--prompt-col", default=None, help="Backward-compatible alias for --prompt-id-col.")
    p.add_argument("--prompt-id-col", default=None)
    p.add_argument("--prompt-text-col", default=None)
    p.add_argument("--prompt-group-col", default="prompt_id",
                   help="Flattened column used for prompt-local grouping, e.g. prompt_id or prompt_text.")
    p.add_argument("--model-a-col", default=None)
    p.add_argument("--model-b-col", default=None)
    p.add_argument("--response-a-col", default=None)
    p.add_argument("--response-b-col", default=None)
    p.add_argument("--normal-annotations-col", default=None)
    p.add_argument("--expert-annotations-col", default=None)
    p.add_argument("--evaluator-col", default=None)

    p.add_argument("--alt-mode", choices=["auto", "model", "response"], default="auto",
                   help="model=use source model names; response=use prompt-local response IDs; auto=model unless model columns are placeholders/absent.")
    p.add_argument("--tournament-mode", choices=["global", "prompt-local", "both"], default="both")
    p.add_argument("--aspects", default="overall,helpful,truthful,harmless")
    p.add_argument("--min-edge-support", type=float, default=1.0)
    p.add_argument("--margin-threshold", type=float, default=0.0)
    p.add_argument("--min-prompt-edges", type=int, default=3)
    p.add_argument("--min-prompt-nodes", type=int, default=3)
    p.add_argument("--min-group-rows", type=int, default=20)
    p.add_argument("--max-evaluators", type=int, default=100)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--make-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}, config={args.config}, split={args.split}")
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)
    raw = ds.to_pandas()
    if args.sample is not None and args.sample < len(raw):
        raw = raw.sample(args.sample, random_state=args.seed).reset_index(drop=True)

    dc = detect_columns(raw, args)
    alt_mode = choose_alt_mode(raw, dc, args.alt_mode)
    print("Detected columns:")
    print(json.dumps(dc.__dict__, indent=2, default=str))
    print(f"Using alt_mode={alt_mode!r}; A/B are row-local roles, not global alternatives.")

    flat = flatten_annotations(raw, dc, alt_mode=alt_mode)
    if flat.empty:
        raise ValueError("Flattening produced no usable annotation rows. Check preference label keys/schema.")
    flat.to_csv(outdir / "multipref_flat_annotations.csv", index=False)

    requested = [x.strip() for x in args.aspects.split(",") if x.strip()]
    aspects = [a for a in requested if a in ASPECTS]
    print(f"Flattened {len(raw)} raw rows into {len(flat)} annotation rows.")
    print(f"Analyzing aspects: {aspects}")

    if args.prompt_group_col not in flat.columns:
        raise ValueError(
            f"--prompt-group-col {args.prompt_group_col!r} is not in flattened columns: {list(flat.columns)}"
        )

    write_audit(raw, flat, dc, alt_mode, args.tournament_mode, args.prompt_group_col, outdir)
    aspect_disagreement(flat).to_csv(outdir / "aspect_disagreement.csv", index=False)
    prompt_audit = write_prompt_support_outputs(flat, args.prompt_group_col, outdir)
    model_edge_support(flat, aspects).to_csv(outdir / "multipref_model_edge_support.csv", index=False)

    if args.tournament_mode in {"global", "both"}:
        global_summary, global_cycles = run_global(
            flat, aspects, args.min_edge_support, args.margin_threshold,
            min_group_rows=args.min_group_rows, max_evaluators=args.max_evaluators,
        )
    else:
        global_summary, global_cycles = pd.DataFrame(), pd.DataFrame()

    if args.tournament_mode in {"prompt-local", "both"}:
        prompt_summary, prompt_cycles = run_prompt_local(
            flat, aspects, args.min_edge_support, args.margin_threshold,
            prompt_group_col=args.prompt_group_col,
            min_prompt_edges=args.min_prompt_edges,
            min_prompt_nodes=args.min_prompt_nodes,
            max_prompts=args.max_prompts,
        )
    else:
        prompt_summary, prompt_cycles = pd.DataFrame(), pd.DataFrame()

    summary = pd.concat([global_summary, prompt_summary], ignore_index=True)
    cycles = pd.concat([global_cycles, prompt_cycles], ignore_index=True)
    if not cycles.empty:
        cycles = cycles.sort_values("intensity", ascending=False)

    summary.to_csv(outdir / "multipref_cycle_projection_summary.csv", index=False)
    cycles.to_csv(outdir / "multipref_strongest_cycles.csv", index=False)

    if not prompt_summary.empty:
        agg = prompt_summary.groupby(["group", "aspect"]).agg(
            n_prompt_rows=("prompt_group", "count"),
            mean_cycle_rate=("cycle_rate", "mean"),
            median_cycle_rate=("cycle_rate", "median"),
            mean_rho_cyc=("rho_cyc", "mean"),
            median_rho_cyc=("rho_cyc", "median"),
            mean_eligible_triples=("eligible_triples", "mean"),
            total_cyclic_triples=("cyclic_triples", "sum"),
            total_eligible_triples=("eligible_triples", "sum"),
        ).reset_index()
        agg["pooled_cycle_rate"] = agg["total_cyclic_triples"] / agg["total_eligible_triples"].replace(0, np.nan)
    else:
        agg = pd.DataFrame()
    agg.to_csv(outdir / "multipref_prompt_local_aggregate.csv", index=False)

    if args.make_plots:
        make_plots(summary, outdir)

    print("\nWrote:")
    for fn in [
        "dataset_audit.json",
        "multipref_flat_annotations.csv",
        "multipref_prompt_support_audit.csv",
        "multipref_prompt_support_summary.json",
        "multipref_model_edge_support.csv",
        "aspect_disagreement.csv",
        "multipref_cycle_projection_summary.csv",
        "multipref_prompt_local_aggregate.csv",
        "multipref_strongest_cycles.csv",
    ]:
        print(f"  {outdir / fn}")
    if args.make_plots:
        print(f"  {outdir / 'figures'}")

    print("\nPrompt-local aggregate:")
    if agg.empty:
        n_tri = int((prompt_audit["n_supported_triangles"] > 0).sum()) if not prompt_audit.empty else 0
        print(
            f"No prompt-local tournament rows met the analysis thresholds. "
            f"Prompt support audit found {n_tri} prompt groups with at least one supported triangle "
            f"using {args.prompt_group_col!r}."
        )
    else:
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
