#!/usr/bin/env python3
"""Matched scalar-versus-antisymmetric LoRA reward experiment on MultiPref.

The scalar arm is the rank-zero member of the same model family used by the
interaction arm. The latter adds a low-rank skew bilinear term and therefore
can represent cyclic pairwise structure while retaining swap antisymmetry.
Held-out prompt folds are non-overlapping by default.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from antisymmetric_reward_head import make_antisymmetric_reward_head
from reward_projection_diagnostics import COMPARISON_METRICS
from run_multipref_cycles_v4 import ASPECTS
from run_multipref_downstream_alignment import (
    region_mask,
    stable_prompt_kfold_split,
    summarize_results,
)
from run_multipref_lora_reward_downstream import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_MODULES,
    analyze_triplets,
    assert_finite_tensor,
    import_lora_stack,
    make_lora_reward_model,
)
from run_multipref_neural_reward_downstream import (
    autocast_context,
    load_raw,
    make_collate,
    make_pair_dataset_class,
    move_batch,
    response_texts,
    token_truncation_stats,
)
from run_multipref_routing_uplift import prediction_metrics
from run_multipref_text_reward_downstream import flatten_annotations, sorted_edge_observations


DEFAULT_OUTDIR = Path("src/results/multipref_lora_nested_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="allenai/multipref")
    parser.add_argument("--split", default="train")
    parser.add_argument("--arrow", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument("--split-index-base", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--modules-to-save", nargs="+", default=["score"])
    parser.add_argument("--zero-init-score", action="store_true", default=True)
    parser.add_argument("--trainable-fp32", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--interaction-rank", type=int, default=16)
    parser.add_argument("--interaction-dropout", type=float, default=0.1)
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None)
    parser.add_argument("--max-raw-rows", type=int, default=None)
    parser.add_argument("--max-train-obs", type=int, default=None)
    parser.add_argument("--include-same-model-train", action="store_true", default=True)
    parser.add_argument("--exclude-same-model-train", action="store_false", dest="include_same_model_train")
    parser.add_argument("--tie-training", choices=["drop", "half"], default="half")
    parser.add_argument("--tie-weight", type=float, default=1.0)
    parser.add_argument("--combine-glob", default=None)
    return parser.parse_args()


def output_paths(outdir: Path, split_index: Optional[int]) -> Tuple[Path, Path, Path, Path, Path]:
    stem = "multipref_lora_nested_reward"
    if split_index is not None:
        stem += f"_split_{split_index:04d}"
    return (
        outdir / f"{stem}_region_results.csv",
        outdir / f"{stem}_fit_summary.csv",
        outdir / f"{stem}_predictions.csv",
        outdir / f"{stem}_summary.csv",
        outdir / f"{stem}_metadata.json",
    )


def _last_token(hidden: Any, attention_mask: Any) -> Any:
    import torch

    positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, positions]


def make_nested_model(args: argparse.Namespace, device: Any, pad_token_id: int, rank: int) -> Any:
    torch, _F, _Loader, _Model, _Tokenizer, _LC, _TT, _gpm, _sched = import_lora_stack()
    import torch.nn as nn

    base = make_lora_reward_model(args, device, pad_token_id)
    hidden_size = int(base.config.hidden_size)
    head = make_antisymmetric_reward_head(hidden_size, rank, args.interaction_dropout)
    head = head.to(device=device)

    class NestedRewardModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_reward_model = base
            self.reward_head = head
            self.rank = int(rank)

        def encode(self, batch: Dict[str, Any], suffix: str) -> Any:
            mask = batch[f"attention_mask_{suffix}"]
            outputs = self.base_reward_model(
                input_ids=batch[f"input_ids_{suffix}"],
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = _last_token(outputs.hidden_states[-1], mask)
            return hidden.to(dtype=self.reward_head.scalar.weight.dtype)

        def forward(self, batch: Dict[str, Any]) -> Any:
            return self.reward_head(self.encode(batch, "a"), self.encode(batch, "b"))

    model = NestedRewardModel().to(device)
    if args.precision == "fp32":
        model = model.float()
    return model


def train_variant(
    train_obs: pd.DataFrame,
    docs_a: Sequence[str],
    docs_b: Sequence[str],
    args: argparse.Namespace,
    split_index: int,
    group: str,
    aspect: str,
    rank: int,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    torch, F, DataLoader, _Model, AutoTokenizer, _LC, _TT, _gpm, get_scheduler = import_lora_stack()
    variant = "scalar" if rank == 0 else "interaction"
    if train_obs.empty:
        return None, None, {"fit_ok": False, "reason": "empty_train", "variant": variant}
    if train_obs["target"].nunique() < 2:
        return None, None, {"fit_ok": False, "reason": "single_class", "variant": variant}
    if args.max_train_obs is not None and len(train_obs) > args.max_train_obs:
        train_obs = train_obs.sample(n=args.max_train_obs, random_state=args.seed + split_index).copy()

    training_seed = args.seed + split_index
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but is unsupported on this GPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    truncation = token_truncation_stats(tokenizer, train_obs, docs_a, docs_b, args.max_length)
    model = make_nested_model(args, device, tokenizer.pad_token_id, rank)
    dataset_class = make_pair_dataset_class()
    loader = DataLoader(
        dataset_class(train_obs, docs_a, docs_b),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(training_seed),
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, args.grad_accum)))
    total_updates = max(1, args.epochs * updates_per_epoch)
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=int(round(args.warmup_ratio * total_updates)),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))

    start = time.time()
    epoch_losses: List[float] = []
    updates = 0
    # Model construction consumes a rank-dependent number of random draws.
    # Reset here so the two arms use matched dropout streams during training.
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        loss_sum = 0.0
        weight_sum = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.precision):
                logits = model(batch)
                assert_finite_tensor(torch, logits, f"{variant} logits", split_index, group, aspect, step)
                losses = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
                assert_finite_tensor(torch, losses, f"{variant} losses", split_index, group, aspect, step)
                weights = batch["weight"]
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
                loss = loss / max(1, args.grad_accum)
                assert_finite_tensor(torch, loss, f"{variant} loss", split_index, group, aspect, step)
            scaler.scale(loss).backward()
            loss_sum += float((losses.detach() * weights).sum().cpu())
            weight_sum += float(weights.detach().sum().cpu())
            if step % max(1, args.grad_accum) == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        epoch_loss = loss_sum / max(weight_sum, 1e-8)
        epoch_losses.append(epoch_loss)
        print(
            f"split={split_index} group={group} aspect={aspect} variant={variant} "
            f"epoch={epoch + 1}/{args.epochs} train_loss={epoch_loss:.4f}",
            flush=True,
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return model, tokenizer, {
        "fit_ok": True,
        "reason": "",
        "variant": variant,
        "interaction_rank": int(rank),
        "model_name": args.model_name,
        "device": str(device),
        "precision": args.precision,
        "training_seed": int(training_seed),
        "n_train_obs": int(len(train_obs)),
        "train_weight": float(train_obs["weight"].sum()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "max_length": int(args.max_length),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": " ".join(args.lora_target_modules),
        "interaction_dropout": float(args.interaction_dropout),
        "trainable_parameters": int(trainable),
        "updates": int(updates),
        "final_train_loss": float(epoch_losses[-1]),
        "train_seconds": float(time.time() - start),
        **truncation,
    }


def predict_variant(
    model: Any,
    tokenizer: Any,
    obs: pd.DataFrame,
    docs_a: Sequence[str],
    docs_b: Sequence[str],
    args: argparse.Namespace,
) -> np.ndarray:
    torch, _F, DataLoader, _Model, _Tokenizer, _LC, _TT, _gpm, _sched = import_lora_stack()
    device = next(model.parameters()).device
    dataset_class = make_pair_dataset_class()
    loader = DataLoader(
        dataset_class(obs, docs_a, docs_b),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    output: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.precision):
                logits = model(batch)
            output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output).astype(float) if output else np.array([], dtype=float)


def add_interaction_metrics(rows: List[Dict[str, Any]], test_obs: pd.DataFrame) -> None:
    for row in rows:
        triplet = tuple(str(row["triplet"]).split(" | "))
        region = test_obs[region_mask(test_obs, triplet)]
        metrics = prediction_metrics(region, region["interaction_p_a"].to_numpy(dtype=float))
        for metric in COMPARISON_METRICS:
            value = float(metrics[metric])
            row[f"interaction_{metric}"] = value
            row[f"scalar_minus_interaction_{metric}"] = float(row[metric]) - value


def _split_indices(args: argparse.Namespace) -> Tuple[List[int], Optional[int]]:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2 for non-overlapping prompt folds.")
    if args.split_index is None:
        return list(range(args.n_splits)), None
    index = args.split_index - args.split_index_base
    if index < 0 or index >= args.n_splits:
        raise ValueError(f"split index maps to {index}, outside 0..{args.n_splits - 1}.")
    return [index], index


def run_training(args: argparse.Namespace) -> None:
    if args.interaction_rank < 1:
        raise ValueError("--interaction-rank must be positive.")
    torch, _F, _Loader, _Model, _Tokenizer, _LC, _TT, _gpm, _sched = import_lora_stack()
    split_indices, output_index = _split_indices(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args)
    flat = flatten_annotations(raw, preserve_ties=(args.tie_training == "half"))
    docs_a, docs_b = response_texts(raw)
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()
    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for split_index in split_indices:
        train_prompts, test_prompts = stable_prompt_kfold_split(
            prompt_ids,
            seed=args.seed,
            fold_index=split_index,
            n_folds=args.n_splits,
        )
        for group in args.groups:
            for aspect in args.aspects:
                reward_obs = sorted_edge_observations(
                    flat,
                    aspect=aspect,
                    group=group,
                    include_same_model=args.include_same_model_train,
                    include_ties=(args.tie_training == "half"),
                    tie_weight=args.tie_weight,
                )
                graph_obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                reward_train = reward_obs[reward_obs["prompt_id"].astype(str).isin(train_prompts)].copy()
                train_obs = graph_obs[graph_obs["prompt_id"].astype(str).isin(train_prompts)].copy()
                test_obs = graph_obs[graph_obs["prompt_id"].astype(str).isin(test_prompts)].copy()
                if reward_train.empty or train_obs.empty or test_obs.empty:
                    continue

                scalar, scalar_tokenizer, scalar_meta = train_variant(
                    reward_train, docs_a, docs_b, args, split_index, group, aspect, rank=0
                )
                scalar_meta.update({"split_index": split_index, "group": group, "aspect": aspect})
                fit_rows.append(scalar_meta)
                if scalar is None or scalar_tokenizer is None:
                    continue
                test_obs["neural_p_a"] = predict_variant(
                    scalar, scalar_tokenizer, test_obs, docs_a, docs_b, args
                )
                del scalar
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                interaction, interaction_tokenizer, interaction_meta = train_variant(
                    reward_train,
                    docs_a,
                    docs_b,
                    args,
                    split_index,
                    group,
                    aspect,
                    rank=args.interaction_rank,
                )
                interaction_meta.update({"split_index": split_index, "group": group, "aspect": aspect})
                fit_rows.append(interaction_meta)
                if interaction is None or interaction_tokenizer is None:
                    continue
                test_obs["interaction_p_a"] = predict_variant(
                    interaction, interaction_tokenizer, test_obs, docs_a, docs_b, args
                )
                rows = analyze_triplets(
                    split_index,
                    group,
                    aspect,
                    train_obs,
                    test_obs,
                    args,
                    scalar_meta,
                )
                add_interaction_metrics(rows, test_obs)
                all_rows.extend(rows)
                predictions = test_obs.copy()
                predictions.insert(0, "aspect", aspect)
                predictions.insert(0, "group", group)
                predictions.insert(0, "split_index", split_index)
                prediction_rows.extend(predictions.to_dict(orient="records"))
                del interaction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results = pd.DataFrame(all_rows)
    fits = pd.DataFrame(fit_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(results) if not results.empty else pd.DataFrame()
    result_path, fit_path, prediction_path, summary_path, metadata_path = output_paths(args.outdir, output_index)
    results.to_csv(result_path, index=False)
    fits.to_csv(fit_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "runner": "matched_nested_lora_reward",
        "model_name": args.model_name,
        "split_design": "deterministic_nonoverlapping_prompt_kfold",
        "n_splits": args.n_splits,
        "split_indices_run": split_indices,
        "seed": args.seed,
        "groups": args.groups,
        "aspects": args.aspects,
        "interaction_rank": args.interaction_rank,
        "n_raw_rows": int(len(raw)),
        "n_prompt_ids": int(len(prompt_ids)),
        "n_region_rows": int(len(results)),
        "n_prediction_rows": int(len(predictions)),
        "result_path": str(result_path),
        "fit_path": str(fit_path),
        "prediction_path": str(prediction_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for path in [result_path, fit_path, prediction_path, summary_path, metadata_path]:
        print(f"Wrote {path}", flush=True)


def combine_outputs(pattern: str, outdir: Path) -> None:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region result files matched: {pattern}")
    frames = [pd.read_csv(path) for path in matches]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if rows.empty:
        raise SystemExit("All matched region result files were empty.")
    outdir.mkdir(parents=True, exist_ok=True)
    result_path, _fit, _prediction, summary_path, metadata_path = output_paths(outdir, None)
    rows.to_csv(result_path, index=False)
    summarize_results(rows).to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps({"mode": "combine", "input_files": matches, "n_region_rows": int(len(rows))}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.combine_glob:
        combine_outputs(args.combine_glob, args.outdir)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
