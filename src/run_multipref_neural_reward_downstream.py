#!/usr/bin/env python3
"""Neural text reward-model downstream experiment for MultiPref.

This is the GPU companion to run_multipref_text_reward_downstream.py.  It trains
a transformer reward model over prompt/response text:

    r_theta(prompt, completion)
    P(A preferred to B) = sigmoid(r_theta(prompt, A) - r_theta(prompt, B)).

The model is trained on held-in prompt IDs and evaluated on held-out prompt IDs.
For each split, annotation group, aspect, and 3-model region, the script asks
whether train-set rho_cyc predicts held-out reward-model failures.

The script is intentionally single-process by default.  On a multi-GPU machine,
run different split indices with different CUDA_VISIBLE_DEVICES values.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from run_multipref_cycles_v4 import ASPECTS
from run_multipref_downstream_alignment import (
    aggregate_edges,
    hodge_residual,
    region_mask,
    stable_prompt_split,
    summarize_results,
    triplet_cycle_stats,
)
from run_multipref_routing_uplift import prediction_metrics
from run_multipref_text_reward_downstream import flatten_annotations, sorted_edge_observations


DEFAULT_OUTDIR = Path("src/results/multipref_neural_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="allenai/multipref", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Hugging Face split.")
    parser.add_argument(
        "--arrow",
        type=Path,
        default=None,
        help="Optional cached Arrow file. If omitted, load_dataset(dataset, split=...) is used.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--model-name", default="roberta-large")
    parser.add_argument("--groups", nargs="+", default=["all", "normal", "expert"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--split-index", type=int, default=None, help="Run one split only.")
    parser.add_argument("--split-index-base", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-raw-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--max-train-obs", type=int, default=None, help="Optional training observation cap.")
    parser.add_argument("--save-model", action="store_true", help="Save trained reward heads and tokenizers.")
    parser.add_argument(
        "--combine-glob",
        type=str,
        default=None,
        help="Combine existing split CSVs matching this glob instead of training.",
    )
    return parser.parse_args()


def import_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Missing neural RM dependencies. Install at least: "
            "pip install torch transformers datasets pandas scipy scikit-learn"
        ) from exc
    return torch, nn, F, DataLoader, Dataset, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def load_raw(args: argparse.Namespace) -> pd.DataFrame:
    if args.arrow is not None:
        from datasets import Dataset

        ds = Dataset.from_file(str(args.arrow))
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split=args.split)
    raw = ds.to_pandas()
    if args.max_raw_rows is not None:
        raw = raw.iloc[: args.max_raw_rows].copy()
    raw = raw.reset_index(drop=True)
    raw["raw_idx"] = np.arange(len(raw), dtype=int)
    return raw


def response_texts(raw: pd.DataFrame) -> Tuple[List[str], List[str]]:
    prompt = raw["text"].fillna("").astype(str)
    completion_a = raw["completion_a"].fillna("").astype(str)
    completion_b = raw["completion_b"].fillna("").astype(str)
    docs_a = ("Prompt:\n" + prompt + "\n\nResponse:\n" + completion_a).tolist()
    docs_b = ("Prompt:\n" + prompt + "\n\nResponse:\n" + completion_b).tolist()
    return docs_a, docs_b


def output_paths(outdir: Path, split_index: Optional[int]) -> Tuple[Path, Path, Path, Path]:
    if split_index is None:
        stem = "multipref_neural_reward"
    else:
        stem = f"multipref_neural_reward_split_{split_index:04d}"
    return (
        outdir / f"{stem}_region_results.csv",
        outdir / f"{stem}_fit_summary.csv",
        outdir / f"{stem}_summary.csv",
        outdir / f"{stem}_metadata.json",
    )


class RewardModelBase:
    pass


def make_reward_model(model_name: str, gradient_checkpointing: bool):
    torch, nn, _F, _DataLoader, _Dataset, AutoModel, _AutoTokenizer, _sched = import_torch_stack()

    class TransformerRewardModel(nn.Module):
        def __init__(self, backbone_name: str) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(backbone_name)
            if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            hidden = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(0.1)
            self.reward_head = nn.Linear(hidden, 1)

        def score(self, batch: Dict[str, Any], suffix: str) -> Any:
            outputs = self.backbone(
                input_ids=batch[f"input_ids_{suffix}"],
                attention_mask=batch[f"attention_mask_{suffix}"],
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            return self.reward_head(self.dropout(pooled)).squeeze(-1)

        def forward(self, batch: Dict[str, Any]) -> Any:
            return self.score(batch, "a") - self.score(batch, "b")

    return TransformerRewardModel(model_name)


def make_pair_dataset_class():
    _torch, _nn, _F, _DataLoader, Dataset, _AutoModel, _AutoTokenizer, _sched = import_torch_stack()

    class PairwisePreferenceDataset(Dataset):
        def __init__(self, obs: pd.DataFrame, docs_a: Sequence[str], docs_b: Sequence[str]) -> None:
            self.raw_idx = obs["raw_idx"].to_numpy(dtype=int)
            self.target = (obs["row_sign"].to_numpy(dtype=int) > 0).astype(np.float32)
            self.weight = obs["weight"].to_numpy(dtype=np.float32)
            self.docs_a = docs_a
            self.docs_b = docs_b

        def __len__(self) -> int:
            return int(len(self.raw_idx))

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            raw_idx = int(self.raw_idx[idx])
            return {
                "text_a": self.docs_a[raw_idx],
                "text_b": self.docs_b[raw_idx],
                "target": float(self.target[idx]),
                "weight": float(self.weight[idx]),
            }

    return PairwisePreferenceDataset


def make_collate(tokenizer: Any, max_length: int):
    torch, _nn, _F, _DataLoader, _Dataset, _AutoModel, _AutoTokenizer, _sched = import_torch_stack()

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        tok_a = tokenizer(
            [b["text_a"] for b in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok_b = tokenizer(
            [b["text_b"] for b in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids_a": tok_a["input_ids"],
            "attention_mask_a": tok_a["attention_mask"],
            "input_ids_b": tok_b["input_ids"],
            "attention_mask_b": tok_b["attention_mask"],
            "target": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
            "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        }

    return collate


def move_batch(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def autocast_context(torch: Any, device: Any, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_neural_reward_model(
    train_obs: pd.DataFrame,
    docs_a: Sequence[str],
    docs_b: Sequence[str],
    args: argparse.Namespace,
    split_index: int,
    group: str,
    aspect: str,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    torch, _nn, F, DataLoader, _Dataset, _AutoModel, AutoTokenizer, get_scheduler = import_torch_stack()

    if train_obs.empty:
        return None, None, {"fit_ok": False, "reason": "empty_train"}
    if train_obs["row_sign"].nunique() < 2:
        return None, None, {"fit_ok": False, "reason": "single_class"}
    if args.max_train_obs is not None and len(train_obs) > args.max_train_obs:
        train_obs = train_obs.sample(n=args.max_train_obs, random_state=args.seed + split_index).copy()

    torch.manual_seed(args.seed + split_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + split_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported on this GPU; using fp32.")
        args.precision = "fp32"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    PairwisePreferenceDataset = make_pair_dataset_class()
    dataset = PairwisePreferenceDataset(train_obs, docs_a, docs_b)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
    )

    model = make_reward_model(args.model_name, args.gradient_checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, args.grad_accum)))
    total_updates = max(1, args.epochs * updates_per_epoch)
    warmup_steps = int(round(args.warmup_ratio * total_updates))
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))

    model.train()
    start = time.time()
    epoch_losses: List[float] = []
    update_count = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.precision):
                logits = model(batch)
                losses = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
                weights = batch["weight"]
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()

            weighted_loss_sum += float((losses.detach() * weights).sum().cpu())
            weight_sum += float(weights.detach().sum().cpu())
            if step % max(1, args.grad_accum) == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1

        epoch_loss = weighted_loss_sum / max(weight_sum, 1e-8)
        epoch_losses.append(epoch_loss)
        print(
            f"split={split_index} group={group} aspect={aspect} "
            f"epoch={epoch + 1}/{args.epochs} train_loss={epoch_loss:.4f}"
        )

    fit_meta = {
        "fit_ok": True,
        "reason": "",
        "model_name": args.model_name,
        "device": str(device),
        "precision": args.precision,
        "n_train_obs": int(len(train_obs)),
        "train_weight": float(train_obs["weight"].sum()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "max_length": int(args.max_length),
        "updates": int(update_count),
        "train_seconds": float(time.time() - start),
        "final_train_loss": float(epoch_losses[-1]) if epoch_losses else np.nan,
    }
    return model, tokenizer, fit_meta


def predict_p_a(
    model: Any,
    tokenizer: Any,
    obs: pd.DataFrame,
    docs_a: Sequence[str],
    docs_b: Sequence[str],
    args: argparse.Namespace,
) -> np.ndarray:
    torch, _nn, _F, DataLoader, _Dataset, _AutoModel, _AutoTokenizer, _sched = import_torch_stack()
    device = next(model.parameters()).device
    PairwisePreferenceDataset = make_pair_dataset_class()
    dataset = PairwisePreferenceDataset(obs, docs_a, docs_b)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.precision):
                logits = model(batch)
            probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
    if not probs:
        return np.array([], dtype=float)
    return np.concatenate(probs).astype(float)


def analyze_triplets(
    split_index: int,
    group: str,
    aspect: str,
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    args: argparse.Namespace,
    fit_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    nodes = sorted(set(train_obs["i"]).union(set(train_obs["j"])))
    triplets = list(itertools.combinations(nodes, 3))
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
        evaluation = prediction_metrics(test_region, test_region["neural_p_a"].to_numpy(dtype=float))

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
            "reward_fit_final_train_loss": float(fit_meta.get("final_train_loss", np.nan)),
            "reward_fit_train_seconds": float(fit_meta.get("train_seconds", np.nan)),
        }
        row.update({f"train_{k}": v for k, v in residual.items()})
        row.update({f"train_{k}": v for k, v in cycle.items()})
        row.update(evaluation)
        rows.append(row)
    return rows


def save_model_artifact(model: Any, tokenizer: Any, outdir: Path, split_index: int, group: str, aspect: str) -> str:
    path = outdir / "models" / f"split_{split_index:04d}" / f"{group}_{aspect}"
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "backbone")
    tokenizer.save_pretrained(path / "tokenizer")
    state_path = path / "reward_head.pt"
    import torch

    torch.save(model.reward_head.state_dict(), state_path)
    return str(path)


def run_training(args: argparse.Namespace) -> None:
    torch, _nn, _F, _DataLoader, _Dataset, _AutoModel, _AutoTokenizer, _sched = import_torch_stack()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpus={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")

    if args.n_splits < 1:
        raise ValueError("--n-splits must be positive.")
    if args.split_index is None:
        split_indices = list(range(args.n_splits))
        output_split_index = None
    else:
        normalized = args.split_index - args.split_index_base
        if normalized < 0 or normalized >= args.n_splits:
            raise ValueError(
                f"Split index {args.split_index} with base {args.split_index_base} "
                f"maps to {normalized}, outside 0..{args.n_splits - 1}."
            )
        split_indices = [normalized]
        output_split_index = normalized

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args)
    flat = flatten_annotations(raw)
    docs_a, docs_b = response_texts(raw)
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()

    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {"empty_obs": 0, "fit_failed": 0}

    for split_index in split_indices:
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
                model, tokenizer, fit_meta = train_neural_reward_model(
                    train_obs=train_obs,
                    docs_a=docs_a,
                    docs_b=docs_b,
                    args=args,
                    split_index=split_index,
                    group=group,
                    aspect=aspect,
                )
                artifact_path = ""
                if model is not None and tokenizer is not None and args.save_model:
                    artifact_path = save_model_artifact(model, tokenizer, args.outdir, split_index, group, aspect)
                fit_rows.append(
                    {
                        "split_index": split_index,
                        "group": group,
                        "aspect": aspect,
                        "artifact_path": artifact_path,
                        **fit_meta,
                    }
                )
                if model is None or tokenizer is None:
                    skipped["fit_failed"] += 1
                    continue
                if not test_obs.empty:
                    test_obs = test_obs.copy()
                    test_obs["neural_p_a"] = predict_p_a(model, tokenizer, test_obs, docs_a, docs_b, args)
                all_rows.extend(
                    analyze_triplets(
                        split_index=split_index,
                        group=group,
                        aspect=aspect,
                        train_obs=train_obs,
                        test_obs=test_obs,
                        args=args,
                        fit_meta=fit_meta,
                    )
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results = pd.DataFrame(all_rows)
    fits = pd.DataFrame(fit_rows)
    summary = summarize_results(results) if not results.empty else pd.DataFrame()
    result_path, fit_path, summary_path, metadata_path = output_paths(args.outdir, output_split_index)
    results.to_csv(result_path, index=False)
    fits.to_csv(fit_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "run",
        "dataset": args.dataset,
        "split": args.split,
        "arrow": str(args.arrow) if args.arrow else "",
        "model_name": args.model_name,
        "n_raw_rows": int(len(raw)),
        "n_annotation_rows": int(len(flat)),
        "n_prompt_ids": int(len(prompt_ids)),
        "n_region_rows": int(len(results)),
        "n_splits_requested": int(args.n_splits),
        "split_indices_run": split_indices,
        "groups": args.groups,
        "aspects": args.aspects,
        "test_frac": args.test_frac,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
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


def combine_outputs(pattern: str, outdir: Path) -> None:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region-result files matched --combine-glob: {pattern}")
    frames = [pd.read_csv(path) for path in matches]
    rows = pd.concat(frames, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path, _fit_path, summary_path, metadata_path = output_paths(outdir, split_index=None)
    rows.to_csv(result_path, index=False)
    summary = summarize_results(rows)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "combine",
        "combine_glob": pattern,
        "n_input_files": len(matches),
        "n_region_rows": int(len(rows)),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


def main() -> None:
    args = parse_args()
    if args.combine_glob:
        combine_outputs(args.combine_glob, args.outdir)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
