#!/usr/bin/env python3
"""LoRA LLM reward-model downstream experiment for MultiPref.

This is the LLM-backbone companion to run_multipref_neural_reward_downstream.py.
It fine-tunes a causal/instruction model with a scalar sequence-classification
head and LoRA adapters:

    r_theta(prompt, completion)
    P(A preferred to B) = sigmoid(r_theta(prompt, A) - r_theta(prompt, B)).

The script keeps the downstream estimand identical to the encoder RM runs:
train on held-in prompt IDs, evaluate on held-out prompt IDs, then test whether
train-set rho_cyc predicts held-out reward-model failures by region.

For a paper-quality LoRA robustness check, prefer a small instruction LLM first
and run only group=all across all aspects before scaling the grid.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from run_multipref_cycles_v4 import ASPECTS
from reward_projection_diagnostics import (
    COMPARISON_METRICS,
    edge_lookup_predictions,
    projection_diagnostics,
)
from run_multipref_downstream_alignment import (
    aggregate_edges,
    hodge_residual,
    region_mask,
    stable_prompt_split,
    summarize_results,
    triplet_cycle_stats,
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


DEFAULT_OUTDIR = Path("src/results/multipref_lora_reward")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="allenai/multipref", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Hugging Face split.")
    parser.add_argument("--arrow", type=Path, default=None, help="Optional cached Arrow file.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=ASPECTS)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--split-index", type=int, default=None, help="Run one split only.")
    parser.add_argument("--split-index-base", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
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
    parser.add_argument("--no-zero-init-score", action="store_false", dest="zero_init_score")
    parser.add_argument("--trainable-fp32", action="store_true", default=True)
    parser.add_argument("--no-trainable-fp32", action="store_false", dest="trainable_fp32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-train-edge-weight", type=float, default=5.0)
    parser.add_argument("--min-test-weight", type=float, default=5.0)
    parser.add_argument("--min-test-edges", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-triplets", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-raw-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--max-train-obs", type=int, default=None, help="Optional training observation cap.")
    parser.add_argument("--include-same-model-train", action="store_true", default=True)
    parser.add_argument("--exclude-same-model-train", action="store_false", dest="include_same_model_train")
    parser.add_argument("--tie-training", choices=["drop", "half"], default="half")
    parser.add_argument("--tie-weight", type=float, default=1.0)
    parser.add_argument("--save-model", action="store_true", help="Save LoRA adapters and tokenizers.")
    parser.add_argument(
        "--combine-glob",
        type=str,
        default=None,
        help="Combine existing split CSVs matching this glob instead of training.",
    )
    return parser.parse_args()


def output_paths(outdir: Path, split_index: Optional[int]) -> Tuple[Path, Path, Path, Path]:
    if split_index is None:
        stem = "multipref_lora_reward"
    else:
        stem = f"multipref_lora_reward_split_{split_index:04d}"
    return (
        outdir / f"{stem}_region_results.csv",
        outdir / f"{stem}_fit_summary.csv",
        outdir / f"{stem}_summary.csv",
        outdir / f"{stem}_metadata.json",
    )


def prediction_output_path(outdir: Path, split_index: Optional[int]) -> Path:
    stem = "multipref_lora_reward" if split_index is None else f"multipref_lora_reward_split_{split_index:04d}"
    return outdir / f"{stem}_predictions.csv"


def import_lora_stack():
    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Missing LoRA RM dependencies. Install at least: "
            "pip install torch transformers datasets peft accelerate sentencepiece protobuf"
        ) from exc
    return (
        torch,
        F,
        DataLoader,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
        get_linear_schedule_with_warmup,
    )


def dtype_for_precision(torch: Any, precision: str, device: Any) -> Optional[Any]:
    if precision == "fp32" or device.type != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


def make_lora_reward_model(args: argparse.Namespace, device: Any, pad_token_id: Optional[int]) -> Any:
    (
        torch,
        _F,
        _DataLoader,
        AutoModelForSequenceClassification,
        _AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
        _sched,
    ) = import_lora_stack()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        torch_dtype=dtype_for_precision(torch, args.precision, device),
        trust_remote_code=args.trust_remote_code,
    )
    if args.precision == "fp32":
        model = model.float()
    if pad_token_id is not None:
        model.config.pad_token_id = int(pad_token_id)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.zero_init_score and hasattr(model, "score"):
        with torch.no_grad():
            model.score.weight.zero_()
            if getattr(model.score, "bias", None) is not None:
                model.score.bias.zero_()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        modules_to_save=args.modules_to_save,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if args.trainable_fp32 and args.precision == "fp32":
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    return model.to(device)


def score_batch(model: Any, batch: Dict[str, Any], suffix: str) -> Any:
    outputs = model(
        input_ids=batch[f"input_ids_{suffix}"],
        attention_mask=batch[f"attention_mask_{suffix}"],
    )
    return outputs.logits.squeeze(-1)


def assert_finite_tensor(torch: Any, tensor: Any, name: str, split_index: int, group: str, aspect: str, step: Optional[int]) -> None:
    if not torch.isfinite(tensor).all():
        where = f"split={split_index} group={group} aspect={aspect}"
        if step is not None:
            where += f" step={step}"
        raise FloatingPointError(f"Non-finite {name} during LoRA reward run at {where}.")


def train_lora_reward_model(
    train_obs: pd.DataFrame,
    docs_a: Sequence[str],
    docs_b: Sequence[str],
    args: argparse.Namespace,
    split_index: int,
    group: str,
    aspect: str,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    torch, F, DataLoader, _Model, AutoTokenizer, _LoraConfig, _TaskType, _get_peft_model, get_scheduler = import_lora_stack()

    if train_obs.empty:
        return None, None, {"fit_ok": False, "reason": "empty_train"}
    target_values = train_obs["target"] if "target" in train_obs else (train_obs["row_sign"] > 0).astype(float)
    if target_values.nunique() < 2:
        return None, None, {"fit_ok": False, "reason": "single_class"}
    if args.max_train_obs is not None and len(train_obs) > args.max_train_obs:
        train_obs = train_obs.sample(n=args.max_train_obs, random_state=args.seed + split_index).copy()

    torch.manual_seed(args.seed + split_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + split_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported on this GPU; using fp32.", flush=True)
        args.precision = "fp32"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    truncation_meta = token_truncation_stats(tokenizer, train_obs, docs_a, docs_b, args.max_length)

    model = make_lora_reward_model(args, device, tokenizer.pad_token_id)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

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

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
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
                logits = score_batch(model, batch, "a") - score_batch(model, batch, "b")
                assert_finite_tensor(torch, logits, "train logits", split_index, group, aspect, step)
                losses = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
                assert_finite_tensor(torch, losses, "train losses", split_index, group, aspect, step)
                weights = batch["weight"]
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
                loss = loss / max(1, args.grad_accum)
                assert_finite_tensor(torch, loss, "train loss", split_index, group, aspect, step)
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
        if not math.isfinite(epoch_loss):
            raise FloatingPointError(
                f"Non-finite epoch loss during LoRA reward run at split={split_index} "
                f"group={group} aspect={aspect}."
            )
        epoch_losses.append(epoch_loss)
        print(
            f"split={split_index} group={group} aspect={aspect} "
            f"epoch={epoch + 1}/{args.epochs} train_loss={epoch_loss:.4f}",
            flush=True,
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
        "eval_batch_size": int(args.eval_batch_size or max(1, args.batch_size * 2)),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_length": int(args.max_length),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": " ".join(args.lora_target_modules),
        "modules_to_save": " ".join(args.modules_to_save),
        "zero_init_score": bool(args.zero_init_score),
        "trainable_fp32": bool(args.trainable_fp32),
        "updates": int(update_count),
        "train_seconds": float(time.time() - start),
        "final_train_loss": float(epoch_losses[-1]) if epoch_losses else np.nan,
        **truncation_meta,
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
    torch, _F, DataLoader, _Model, _Tokenizer, _LoraConfig, _TaskType, _get_peft_model, _sched = import_lora_stack()
    device = next(model.parameters()).device
    PairwisePreferenceDataset = make_pair_dataset_class()
    dataset = PairwisePreferenceDataset(obs, docs_a, docs_b)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size or max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.precision):
                logits = score_batch(model, batch, "a") - score_batch(model, batch, "b")
            assert_finite_tensor(torch, logits, "eval logits", -1, "predict", "predict", step)
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
        projection = projection_diagnostics(
            train_edges,
            triplet,
            min_support=args.min_train_edge_weight,
        )
        edge_lookup = prediction_metrics(
            test_region,
            edge_lookup_predictions(train_edges, test_region, smoothing=1.0),
        )

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
        row.update({f"train_{k}": v for k, v in projection.items()})
        row.update(evaluation)
        row.update({f"edge_lookup_{k}": v for k, v in edge_lookup.items() if k in COMPARISON_METRICS})
        for metric in COMPARISON_METRICS:
            row[f"uplift_{metric}"] = float(evaluation[metric]) - float(edge_lookup[metric])
        rows.append(row)
    return rows


def save_model_artifact(model: Any, tokenizer: Any, outdir: Path, split_index: int, group: str, aspect: str) -> str:
    path = outdir / "models" / f"split_{split_index:04d}" / f"{group}_{aspect}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
    return str(path)


def run_training(args: argparse.Namespace) -> None:
    torch, _F, _DataLoader, _Model, _Tokenizer, _LoraConfig, _TaskType, _get_peft_model, _sched = import_lora_stack()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpus={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"gpu[{i}]={torch.cuda.get_device_name(i)}", flush=True)

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
    flat = flatten_annotations(raw, preserve_ties=(args.tie_training == "half"))
    docs_a, docs_b = response_texts(raw)
    prompt_ids = flat["prompt_id"].astype(str).unique().tolist()

    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
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
                reward_obs = sorted_edge_observations(
                    flat,
                    aspect=aspect,
                    group=group,
                    include_same_model=args.include_same_model_train,
                    include_ties=(args.tie_training == "half"),
                    tie_weight=args.tie_weight,
                )
                graph_obs = sorted_edge_observations(flat, aspect=aspect, group=group)
                if reward_obs.empty or graph_obs.empty:
                    skipped["empty_obs"] += 1
                    continue
                reward_train_obs = reward_obs[reward_obs["prompt_id"].isin(train_prompts)].copy()
                train_obs = graph_obs[graph_obs["prompt_id"].isin(train_prompts)].copy()
                test_obs = graph_obs[graph_obs["prompt_id"].isin(test_prompts)].copy()
                model, tokenizer, fit_meta = train_lora_reward_model(
                    train_obs=reward_train_obs,
                    docs_a=docs_a,
                    docs_b=docs_b,
                    args=args,
                    split_index=split_index,
                    group=group,
                    aspect=aspect,
                )
                fit_meta["n_input_train_ties"] = int((reward_train_obs["target"] == 0.5).sum())
                fit_meta["n_input_train_same_model"] = int(
                    (reward_train_obs["model_a"] == reward_train_obs["model_b"]).sum()
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
                    prediction_frame = test_obs.copy()
                    prediction_frame.insert(0, "aspect", aspect)
                    prediction_frame.insert(0, "group", group)
                    prediction_frame.insert(0, "split_index", split_index)
                    prediction_rows.extend(prediction_frame.to_dict(orient="records"))
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
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(results) if not results.empty else pd.DataFrame()
    result_path, fit_path, summary_path, metadata_path = output_paths(args.outdir, output_split_index)
    results.to_csv(result_path, index=False)
    fits.to_csv(fit_path, index=False)
    predictions_path = prediction_output_path(args.outdir, output_split_index)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "run",
        "runner": "lora_sequence_classification_reward",
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
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "modules_to_save": args.modules_to_save,
        "zero_init_score": args.zero_init_score,
        "trainable_fp32": args.trainable_fp32,
        "include_same_model_train": args.include_same_model_train,
        "tie_training": args.tie_training,
        "tie_weight": args.tie_weight,
        "min_train_edge_weight": args.min_train_edge_weight,
        "min_test_weight": args.min_test_weight,
        "min_test_edges": args.min_test_edges,
        "tau": args.tau,
        "skipped": skipped,
        "result_path": str(result_path),
        "fit_path": str(fit_path),
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {fit_path}", flush=True)
    print(f"Wrote {predictions_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)


def combine_outputs(pattern: str, outdir: Path) -> None:
    matches = sorted(path for path in glob.glob(pattern) if path.endswith("_region_results.csv"))
    if not matches:
        raise SystemExit(f"No region-result files matched --combine-glob: {pattern}")
    frames = []
    skipped_empty = []
    for path in matches:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            skipped_empty.append(path)
            continue
        if frame.empty:
            skipped_empty.append(path)
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit("All matched region-result files were empty. Check the split logs.")
    rows = pd.concat(frames, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path, _fit_path, summary_path, metadata_path = output_paths(outdir, split_index=None)
    rows.to_csv(result_path, index=False)
    summary = summarize_results(rows)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "mode": "combine",
        "runner": "lora_sequence_classification_reward",
        "combine_glob": pattern,
        "n_input_files": len(matches),
        "n_nonempty_input_files": len(frames),
        "skipped_empty_files": skipped_empty,
        "n_region_rows": int(len(rows)),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
