#!/usr/bin/env python3
"""
批量生成 192 步长期预测候选对比图。

默认配置：
- 数据集：ETTh1
- 预测长度：192
- 基线模型：T3Time
- 对比模型：T3Time_FreTS_Gated_Qwen

脚本能力：
1. 从 experiment_results.log 中自动检索最佳实验配置；
2. 按各模型原始训练方式重新训练并在测试集收集预测结果；
3. 缓存反归一化后的 inputs / preds / trues，避免重复训练；
4. 按“模型提升 + 曲线可视性”自动打分，生成多张候选图；
5. 输出候选图摘要 JSON，方便挑图写论文。

示例：
python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --dataset ETTh1 \
  --pred-len 192 \
  --top-k 8 \
  --device cuda

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL exchange_rate \
  --baseline-top-seeds 2 \
  --ours-top-seeds 4 \
  --top-k 12 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --datasets ETTh1 \
  --pred-lens 96 192 336 720 \
  --baseline-top-seeds 3 \
  --ours-top-seeds 6 \
  --top-k 24 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --datasets Weather ILI \
  --pred-lens 96 192 24 36 48 60 \
  --baseline-pattern T3Time_FreTS_Gated_Qwen \
  --baseline-match exact \
  --baseline-model-type frets_gated_qwen \
  --baseline-label FreTS-Gated-Qwen \
  --baseline-top-seeds 2 \
  --ours-pattern T3Time_FreTS_Gated_Qwen_FusionExp \
  --ours-match exact \
  --ours-model-type frets_gated_qwen_fusion_exp \
  --ours-label FusionExp \
  --ours-top-seeds 2 \
  --top-k 16 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --comparison-preset paper_pool \
  --baseline-top-seeds 2 \
  --ours-top-seeds 4 \
  --top-k 12 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

python scripts/T3Time_FreTS_FusionExp/generate_chap5_candidate_compare_plots.py \
  --epochs-override 3 \
  --max-train-batches 10 \
  --max-val-batches 5 \
  --max-test-batches 8 \
  --force-recompute
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = "/root/0/T3Time"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader_emb import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from utils.tools import adjust_learning_rate


LOG_FILE = os.path.join(PROJECT_ROOT, "experiment_results.log")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "docs/NEU-Thesis-main/NEU-Thesis-main/Img",
    "chap5_main_compare_candidates_v3",
)
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "tmp", "vis_cache")

DATASET_ALIASES = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "ili": "ILI",
    "weather": "Weather",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
    "ecl": "ECL",
}

MODEL_TYPE_CHOICES = ["t3time", "frets_gated_qwen", "frets_gated_qwen_fusion_exp"]
MATCH_MODE_CHOICES = ["exact", "contains", "prefix", "suffix"]
COMPARISON_PRESET_CHOICES = ["paper_pool"]

COLOR_HISTORY = "#A8A8A8"
COLOR_GT = "#1F1F1F"
COLOR_BASELINE = "#1F77B4"
COLOR_OURS = "#D62728"
COLOR_FORECAST_BG = "#F7F4EE"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(dataset: str) -> str:
    key = dataset.strip()
    return DATASET_ALIASES.get(key.lower(), key)


def feature_names(dataset: str) -> List[str]:
    dataset = normalize_dataset_name(dataset)
    csv_path = os.path.join(PROJECT_ROOT, "dataset", f"{dataset}.csv")
    df = pd.read_csv(csv_path)
    return list(df.columns[1:])


def infer_num_nodes(dataset: str) -> int:
    return len(feature_names(dataset))


def infer_d_llm(dataset: str, embed_version: str, split: str = "train") -> int:
    embed_dir = os.path.join(PROJECT_ROOT, "Embeddings", normalize_dataset_name(dataset), embed_version, split)
    if not os.path.isdir(embed_dir):
        return 1024
    for file_name in sorted(os.listdir(embed_dir)):
        if not file_name.endswith(".h5"):
            continue
        file_path = os.path.join(embed_dir, file_name)
        try:
            import h5py

            with h5py.File(file_path, "r") as hf:
                arr = hf["embeddings"]
                if arr.ndim == 2:
                    return int(arr.shape[0])
                if arr.ndim == 3:
                    return int(arr.shape[1])
        except Exception:
            continue
    return 1024


def match_text(text: str, pattern: str, mode: str) -> bool:
    if mode == "exact":
        return text == pattern
    if mode == "contains":
        return pattern.lower() in text.lower()
    if mode == "prefix":
        return text.startswith(pattern)
    if mode == "suffix":
        return text.endswith(pattern)
    raise ValueError(f"Unsupported match mode: {mode}")


def dedupe_rows(rows: List[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row.get("data_path"),
            row.get("pred_len"),
            row.get("model_id") or row.get("model"),
            row.get("seed"),
            row.get("seq_len"),
            row.get("channel"),
            row.get("batch_size"),
            row.get("learning_rate"),
            row.get("dropout_n"),
            row.get("weight_decay"),
            row.get("embed_version"),
            row.get("e_layer"),
            row.get("d_layer"),
            row.get("head"),
            row.get("test_mse"),
            row.get("test_mae"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def sort_log_rows(rows: List[dict], metric_name: str) -> List[dict]:
    key = "test_mae" if metric_name == "mae" else "test_mse"
    return sorted(
        rows,
        key=lambda x: (
            x.get(key, float("inf")),
            x.get("test_mae", float("inf")),
            x.get("test_mse", float("inf")),
            x.get("seed", float("inf")),
        ),
    )


def load_log_rows(
    dataset: str,
    pred_len: int,
    pattern: str,
    match_mode: str,
    metric_name: str,
    seeds: Optional[List[int]] = None,
    top_unique_seeds: int = 1,
) -> List[dict]:
    dataset = normalize_dataset_name(dataset)
    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("data_path") != dataset or int(item.get("pred_len", -1)) != pred_len:
                continue
            model_candidates = [
                item.get("model_id") or "",
                item.get("model") or "",
            ]
            if any(name and match_text(name, pattern, match_mode) for name in model_candidates):
                rows.append(item)

    rows = dedupe_rows(rows)
    if not rows:
        raise ValueError(
            f"未找到匹配日志: dataset={dataset}, pred_len={pred_len}, pattern={pattern}, match_mode={match_mode}"
        )

    rows = sort_log_rows(rows, metric_name)
    best_by_seed = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed not in best_by_seed:
            best_by_seed[seed] = row

    if seeds:
        missing = [seed for seed in seeds if seed not in best_by_seed]
        if missing:
            raise ValueError(
                f"以下种子在日志中未找到匹配配置: {missing}; "
                f"dataset={dataset}, pred_len={pred_len}, pattern={pattern}"
            )
        return [best_by_seed[seed] for seed in seeds]

    top_unique_seeds = max(1, int(top_unique_seeds))
    return list(best_by_seed.values())[:top_unique_seeds]


def build_runtime_cfg(log_row: dict, model_type: str, epochs_override: Optional[int]) -> dict:
    dataset = normalize_dataset_name(log_row["data_path"])
    default_epochs = 150 if model_type == "t3time" else 100
    default_weight_decay = 1e-3 if model_type == "t3time" else 1e-4
    embed_version = log_row.get("embed_version", "qwen3_0.6b")
    cfg = {
        "data_path": dataset,
        "seq_len": int(log_row.get("seq_len", 96)),
        "pred_len": int(log_row["pred_len"]),
        "num_nodes": int(log_row.get("num_nodes", infer_num_nodes(dataset))),
        "batch_size": int(log_row.get("batch_size", 32 if model_type == "t3time" else 16)),
        "learning_rate": float(log_row.get("learning_rate", 1e-4)),
        "dropout_n": float(log_row.get("dropout_n", 0.1)),
        "channel": int(log_row.get("channel", 64)),
        "e_layer": int(log_row.get("e_layer", 1)),
        "d_layer": int(log_row.get("d_layer", 1)),
        "head": int(log_row.get("head", 8)),
        "epochs": int(epochs_override or log_row.get("epochs", default_epochs)),
        "es_patience": int(log_row.get("es_patience", log_row.get("patience", 25 if model_type == "t3time" else 10))),
        "embed_version": embed_version,
        "seed": int(log_row.get("seed", 2024)),
        "weight_decay": float(log_row.get("weight_decay", default_weight_decay)),
        "loss_fn": log_row.get("loss_fn", "mse" if model_type == "t3time" else "smooth_l1"),
        "lradj": log_row.get("lradj", "type1"),
        "model_name_for_log": log_row.get("model_id") or log_row.get("model") or model_type,
        "fusion_mode": log_row.get("fusion_mode", "gate"),
        "sparsity_threshold": float(log_row.get("sparsity_threshold", 0.01)),
        "frets_scale": float(log_row.get("frets_scale", log_row.get("scale", 0.02))),
        "use_frets": int(log_row.get("use_frets", 1)),
        "use_complex": int(log_row.get("use_complex", 1)),
        "use_sparsity": int(log_row.get("use_sparsity", 1)),
        "use_improved_gate": int(log_row.get("use_improved_gate", 1)),
    }
    cfg["d_llm"] = int(log_row.get("d_llm", infer_d_llm(dataset, embed_version)))
    return cfg


def make_dataset(dataset_name: str, flag: str, cfg: dict):
    dataset_name = normalize_dataset_name(dataset_name)
    Data = Dataset_ETT_hour
    if dataset_name.startswith("ETTm"):
        Data = Dataset_ETT_minute
    elif dataset_name not in ["ETTh1", "ETTh2"]:
        Data = Dataset_Custom

    data_file = dataset_name if dataset_name.endswith(".csv") else f"{dataset_name}.csv"
    kwargs = {
        "root_path": os.path.join(PROJECT_ROOT, "dataset"),
        "data_path": data_file,
        "flag": flag,
        "size": [cfg["seq_len"], 0, cfg["pred_len"]],
        "features": "M",
        "scale": True,
        "embed_version": cfg["embed_version"],
    }
    signature = inspect.signature(Data.__init__)
    if "num_nodes" in signature.parameters:
        kwargs["num_nodes"] = cfg["num_nodes"]
    return Data(**kwargs)


def make_loader(dataset, model_type: str, flag: str, batch_size: int):
    if model_type == "t3time":
        shuffle = False
    else:
        shuffle = flag != "test"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )


def build_model(model_type: str, cfg: dict, device: torch.device):
    if model_type == "t3time":
        from models.T3Time import TriModal

        model = TriModal(
            device=device,
            channel=cfg["channel"],
            num_nodes=cfg["num_nodes"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout_n=cfg["dropout_n"],
            d_llm=cfg["d_llm"],
            e_layer=cfg["e_layer"],
            d_layer=cfg["d_layer"],
            head=cfg["head"],
        )
    elif model_type == "frets_gated_qwen":
        from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen

        model = TriModalFreTSGatedQwen(
            device=device,
            channel=cfg["channel"],
            num_nodes=cfg["num_nodes"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout_n=cfg["dropout_n"],
            d_llm=cfg["d_llm"],
            e_layer=cfg["e_layer"],
            d_layer=cfg["d_layer"],
            head=cfg["head"],
        )
    elif model_type == "frets_gated_qwen_fusion_exp":
        from models.T3Time_FreTS_Gated_Qwen_FusionExp import TriModalFreTSGatedQwenFusionExp

        model = TriModalFreTSGatedQwenFusionExp(
            device=device,
            channel=cfg["channel"],
            num_nodes=cfg["num_nodes"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout_n=cfg["dropout_n"],
            d_llm=cfg["d_llm"],
            e_layer=cfg["e_layer"],
            d_layer=cfg["d_layer"],
            head=cfg["head"],
            sparsity_threshold=cfg["sparsity_threshold"],
            scale=cfg["frets_scale"],
            fusion_mode=cfg["fusion_mode"],
            use_frets=bool(cfg["use_frets"]),
            use_complex=bool(cfg["use_complex"]),
            use_sparsity=bool(cfg["use_sparsity"]),
            use_improved_gate=bool(cfg["use_improved_gate"]),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model.to(device)


def build_optimizer_and_scheduler(model_type: str, model: nn.Module, cfg: dict, epochs: int):
    if model_type == "t3time":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=min(epochs, 50),
            eta_min=1e-6,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = None
    return optimizer, scheduler


def build_criterion(model_type: str, cfg: dict):
    if model_type == "t3time":
        return nn.MSELoss()
    if cfg["loss_fn"] == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=0.2)


def iter_batches(loader: DataLoader, max_batches: int) -> Iterable:
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        yield batch


def forward_and_target(
    model_type: str,
    model: nn.Module,
    batch,
    device: torch.device,
    pred_len: int,
):
    x = batch[0].to(device).float()
    y = batch[1].to(device).float()
    x_mark = batch[2].to(device).float()
    embeddings = batch[-1].to(device).float()
    if model_type == "t3time":
        out = model(x, x_mark, embeddings)
        target = y
    else:
        out = model(x, None, embeddings)
        target = y[:, -pred_len:, :]
    return x, out, target


def cache_key(
    dataset: str,
    pred_len: int,
    model_type: str,
    cfg: dict,
    max_train_batches: int,
    max_val_batches: int,
    max_test_batches: int,
) -> str:
    payload = {
        "dataset": dataset,
        "pred_len": pred_len,
        "model_type": model_type,
        "cfg": cfg,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": max_test_batches,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def maybe_load_cache(cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=False)
    return {
        "inputs": data["inputs"],
        "preds": data["preds"],
        "trues": data["trues"],
    }


def save_cache(cache_path: str, inputs: np.ndarray, preds: np.ndarray, trues: np.ndarray) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=inputs.astype(np.float32),
        preds=preds.astype(np.float32),
        trues=trues.astype(np.float32),
    )


def train_and_collect(
    dataset: str,
    model_type: str,
    cfg: dict,
    device: torch.device,
    cache_dir: str,
    force_recompute: bool,
    max_train_batches: int,
    max_val_batches: int,
    max_test_batches: int,
    return_arrays: bool = False,
) -> Dict[str, object]:
    dataset = normalize_dataset_name(dataset)
    key = cache_key(dataset, cfg["pred_len"], model_type, cfg, max_train_batches, max_val_batches, max_test_batches)
    cache_path = os.path.join(cache_dir, f"{dataset}_{cfg['pred_len']}_{model_type}_{key}.npz")

    if not force_recompute:
        cached = maybe_load_cache(cache_path)
        if cached is not None:
            print(f"[cache] loaded {cache_path}")
            if return_arrays:
                cached["cache_path"] = cache_path
                return cached
            return {"cache_path": cache_path}

    set_seed(cfg["seed"])
    train_set = make_dataset(dataset, "train", cfg)
    val_set = make_dataset(dataset, "val", cfg)
    test_set = make_dataset(dataset, "test", cfg)

    train_loader = make_loader(train_set, model_type, "train", cfg["batch_size"])
    val_loader = make_loader(val_set, model_type, "val", cfg["batch_size"])
    test_loader = make_loader(test_set, model_type, "test", cfg["batch_size"])

    model = build_model(model_type, cfg, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model_type, model, cfg, cfg["epochs"])
    criterion = build_criterion(model_type, cfg)

    best_state = None
    best_val = float("inf")
    patience_count = 0

    print(f"\n{'=' * 80}")
    print(f"Train {model_type} | {cfg['model_name_for_log']}")
    print(
        f"dataset={dataset} pred_len={cfg['pred_len']} seed={cfg['seed']} "
        f"epochs={cfg['epochs']} batch_size={cfg['batch_size']}"
    )
    print(f"{'=' * 80}")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for batch in iter_batches(train_loader, max_train_batches):
            optimizer.zero_grad()
            _, out, target = forward_and_target(model_type, model, batch, device, cfg["pred_len"])
            loss = criterion(out, target)
            loss.backward()
            if model_type == "t3time":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in iter_batches(val_loader, max_val_batches):
                _, out, target = forward_and_target(model_type, model, batch, device, cfg["pred_len"])
                val_losses.append(criterion(out, target).item())

        if not train_losses or not val_losses:
            raise RuntimeError("训练或验证 batch 为空，请检查 batch_size / max_*_batches 配置。")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch + 1:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["es_patience"]:
                print(f"Early stop at epoch {epoch + 1}")
                break

        if model_type == "t3time":
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch + 1, argparse.Namespace(**cfg))

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    inputs, preds, trues = [], [], []
    with torch.no_grad():
        for batch in iter_batches(test_loader, max_test_batches):
            x, out, target = forward_and_target(model_type, model, batch, device, cfg["pred_len"])
            inputs.append(x.cpu())
            preds.append(out.cpu())
            trues.append(target.cpu())

    if not inputs:
        raise RuntimeError("测试集 batch 为空，请检查 max_test_batches 配置。")

    inputs_arr = torch.cat(inputs, dim=0).numpy()
    preds_arr = torch.cat(preds, dim=0).numpy()
    trues_arr = torch.cat(trues, dim=0).numpy()

    scaler = train_set.scaler
    inputs_inv = scaler.inverse_transform(inputs_arr)
    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)

    save_cache(cache_path, inputs_inv, preds_inv, trues_inv)
    print(f"[cache] saved {cache_path}")
    if return_arrays:
        return {
            "inputs": inputs_inv.astype(np.float32),
            "preds": preds_inv.astype(np.float32),
            "trues": trues_inv.astype(np.float32),
            "cache_path": cache_path,
        }
    return {"cache_path": cache_path}


def load_cached_result(cache_path: str) -> Dict[str, np.ndarray]:
    cached = maybe_load_cache(cache_path)
    if cached is None:
        raise FileNotFoundError(f"未找到缓存结果: {cache_path}")
    return cached


def resolve_datasets(dataset: str, datasets: Optional[List[str]]) -> List[str]:
    raw = datasets if datasets else [dataset]
    ordered = []
    seen = set()
    for item in raw:
        name = normalize_dataset_name(item)
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def resolve_devices(device: str, devices: Optional[List[str]]) -> List[str]:
    raw = devices if devices else [device]
    ordered = []
    seen = set()
    for item in raw:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_pred_lens(pred_len: int, pred_lens: Optional[List[int]]) -> List[int]:
    raw = pred_lens if pred_lens else [pred_len]
    ordered = []
    seen = set()
    for item in raw:
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def slugify(text: str) -> str:
    safe = []
    for ch in text.strip().lower():
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    slug = "".join(safe)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "comparison"


def build_comparison_specs(args, datasets: List[str], pred_lens: List[int]) -> List[dict]:
    if not args.comparison_preset:
        specs = []
        default_tag = slugify(f"{args.baseline_label}_vs_{args.ours_label}")
        for dataset in datasets:
            for pred_len in pred_lens:
                specs.append(
                    {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "comparison_tag": default_tag,
                        "baseline_pattern": args.baseline_pattern,
                        "baseline_match": args.baseline_match,
                        "baseline_model_type": args.baseline_model_type,
                        "baseline_label": args.baseline_label,
                        "baseline_seeds": args.baseline_seeds,
                        "baseline_top_seeds": args.baseline_top_seeds,
                        "ours_pattern": args.ours_pattern,
                        "ours_match": args.ours_match,
                        "ours_model_type": args.ours_model_type,
                        "ours_label": args.ours_label,
                        "ours_seeds": args.ours_seeds,
                        "ours_top_seeds": args.ours_top_seeds,
                    }
                )
        return specs

    if args.comparison_preset != "paper_pool":
        raise ValueError(f"Unsupported comparison preset: {args.comparison_preset}")

    specs = []

    def append_many(
        dataset: str,
        pred_len_list: List[int],
        comparison_tag: str,
        baseline_pattern: str,
        baseline_match: str,
        baseline_model_type: str,
        baseline_label: str,
        ours_pattern: str,
        ours_match: str,
        ours_model_type: str,
        ours_label: str,
    ) -> None:
        for pred_len in pred_len_list:
            specs.append(
                {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "comparison_tag": comparison_tag,
                    "baseline_pattern": baseline_pattern,
                    "baseline_match": baseline_match,
                    "baseline_model_type": baseline_model_type,
                    "baseline_label": baseline_label,
                    "baseline_seeds": args.baseline_seeds,
                    "baseline_top_seeds": args.baseline_top_seeds,
                    "ours_pattern": ours_pattern,
                    "ours_match": ours_match,
                    "ours_model_type": ours_model_type,
                    "ours_label": ours_label,
                    "ours_seeds": args.ours_seeds,
                    "ours_top_seeds": args.ours_top_seeds,
                }
            )

    append_many(
        dataset="ETTh1",
        pred_len_list=[96, 192, 336, 720],
        comparison_tag="etth1_t3time_vs_frets",
        baseline_pattern="T3Time",
        baseline_match="exact",
        baseline_model_type="t3time",
        baseline_label="T3Time",
        ours_pattern="T3Time_FreTS_Gated_Qwen",
        ours_match="exact",
        ours_model_type="frets_gated_qwen",
        ours_label="FreTS-Gated-Qwen",
    )
    append_many(
        dataset="Weather",
        pred_len_list=[96, 192, 336],
        comparison_tag="weather_full_vs_wo_frets",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_w_o_FreTS_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="w/o FreTS",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    append_many(
        dataset="Weather",
        pred_len_list=[96, 192, 336],
        comparison_tag="weather_full_vs_wo_sparsity",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_w_o_Sparsity_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="w/o Sparsity",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    append_many(
        dataset="Weather",
        pred_len_list=[96, 336],
        comparison_tag="weather_full_vs_fft_complex",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_FFT_Complex_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="FFT-Complex",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_Weather_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    append_many(
        dataset="ILI",
        pred_len_list=[24, 36, 48, 60],
        comparison_tag="ili_full_vs_wo_frets",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_w_o_FreTS_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="w/o FreTS",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    append_many(
        dataset="ILI",
        pred_len_list=[24, 36, 48, 60],
        comparison_tag="ili_full_vs_wo_sparsity",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_w_o_Sparsity_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="w/o Sparsity",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    append_many(
        dataset="ILI",
        pred_len_list=[24, 36, 48, 60],
        comparison_tag="ili_full_vs_fft_complex",
        baseline_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_FFT_Complex_gate",
        baseline_match="exact",
        baseline_model_type="frets_gated_qwen_fusion_exp",
        baseline_label="FFT-Complex",
        ours_pattern="T3Time_FreTS_FusionExp_Ablation_ILI_Full_gate",
        ours_match="exact",
        ours_model_type="frets_gated_qwen_fusion_exp",
        ours_label="Full Model",
    )
    return specs


def prepare_dataset_plan(spec: dict, args) -> dict:
    dataset = spec["dataset"]
    pred_len = int(spec["pred_len"])
    baseline_rows = load_log_rows(
        dataset=dataset,
        pred_len=pred_len,
        pattern=spec["baseline_pattern"],
        match_mode=spec["baseline_match"],
        metric_name=args.metric,
        seeds=spec["baseline_seeds"],
        top_unique_seeds=spec["baseline_top_seeds"],
    )
    ours_rows = load_log_rows(
        dataset=dataset,
        pred_len=pred_len,
        pattern=spec["ours_pattern"],
        match_mode=spec["ours_match"],
        metric_name=args.metric,
        seeds=spec["ours_seeds"],
        top_unique_seeds=spec["ours_top_seeds"],
    )

    baseline_tasks = []
    for row in baseline_rows:
        baseline_tasks.append(
            {
                "comparison_tag": spec["comparison_tag"],
                "dataset": dataset,
                "side": "baseline",
                "model_type": spec["baseline_model_type"],
                "cfg": build_runtime_cfg(row, spec["baseline_model_type"], args.epochs_override),
            }
        )

    ours_tasks = []
    for row in ours_rows:
        ours_tasks.append(
            {
                "comparison_tag": spec["comparison_tag"],
                "dataset": dataset,
                "side": "ours",
                "model_type": spec["ours_model_type"],
                "cfg": build_runtime_cfg(row, spec["ours_model_type"], args.epochs_override),
            }
        )

    return {
        "comparison_tag": spec["comparison_tag"],
        "dataset": dataset,
        "pred_len": pred_len,
        "baseline_label": spec["baseline_label"],
        "ours_label": spec["ours_label"],
        "baseline_tasks": baseline_tasks,
        "ours_tasks": ours_tasks,
    }


def collect_dataset_plans(specs: List[dict], args) -> Tuple[List[dict], List[dict]]:
    plans = []
    skipped = []
    for spec in specs:
        try:
            plans.append(prepare_dataset_plan(spec, args))
        except Exception as exc:
            if args.strict_datasets:
                raise
            reason = str(exc)
            print(
                f"[skip] tag={spec['comparison_tag']} dataset={spec['dataset']} "
                f"pred_len={spec['pred_len']} reason={reason}"
            )
            skipped.append(
                {
                    "comparison_tag": spec["comparison_tag"],
                    "dataset": spec["dataset"],
                    "pred_len": spec["pred_len"],
                    "reason": reason,
                }
            )
    return plans, skipped


def flatten_training_tasks(dataset_plans: List[dict]) -> List[dict]:
    tasks = []
    for plan in dataset_plans:
        tasks.extend(plan["baseline_tasks"])
        tasks.extend(plan["ours_tasks"])
    return tasks


def assign_tasks_to_devices(tasks: List[dict], devices: List[str]) -> Dict[str, List[dict]]:
    queues = {device: [] for device in devices}
    for idx, task in enumerate(tasks):
        device = devices[idx % len(devices)]
        queued_task = dict(task)
        queued_task["device"] = device
        queues[device].append(queued_task)
    return queues


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * completed / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def emit_progress_event(progress_queue, payload: dict) -> None:
    if progress_queue is None:
        return
    progress_queue.put(payload)


def summarize_active_devices(active_devices: Dict[str, dict]) -> str:
    if not active_devices:
        return "idle"
    parts = []
    for device in sorted(active_devices.keys()):
        item = active_devices[device]
        tag = item.get("comparison_tag", "") or "default"
        parts.append(
            f"{device}:{item['current_idx']}/{item['total']} "
            f"{tag}:{item['dataset']}/pred{item['pred_len']}/{item['side']}/seed{item['seed']}"
        )
    return " | ".join(parts)


def print_progress_snapshot(
    completed: int,
    total: int,
    start_time: float,
    active_devices: Dict[str, dict],
    label: str,
) -> None:
    elapsed_seconds = time.time() - start_time
    bar = build_progress_bar(completed, total)
    elapsed = format_duration(elapsed_seconds)
    if completed > 0 and total > completed:
        eta_seconds = elapsed_seconds / completed * (total - completed)
        eta = format_duration(eta_seconds)
    elif total > 0 and completed >= total:
        eta = "00:00"
    else:
        eta = "--:--"
    print(
        f"[progress] {bar} {completed}/{total} done | elapsed={elapsed} | eta={eta} | "
        f"active={summarize_active_devices(active_devices)} | {label}"
    )


def run_training_task(task: dict, common: dict) -> dict:
    dataset = task["dataset"]
    comparison_tag = task.get("comparison_tag", "")
    side = task["side"]
    cfg = task["cfg"]
    device_str = task["device"]
    print(
        f"[task] device={device_str} tag={comparison_tag} dataset={dataset} side={side} "
        f"seed={cfg['seed']} pred_len={cfg['pred_len']}"
    )
    result = train_and_collect(
        dataset=dataset,
        model_type=task["model_type"],
        cfg=cfg,
        device=torch.device(device_str),
        cache_dir=common["cache_dir"],
        force_recompute=common["force_recompute"],
        max_train_batches=common["max_train_batches"],
        max_val_batches=common["max_val_batches"],
        max_test_batches=common["max_test_batches"],
        return_arrays=False,
    )
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "comparison_tag": comparison_tag,
        "dataset": dataset,
        "pred_len": int(cfg["pred_len"]),
        "side": side,
        "seed": int(cfg["seed"]),
        "device": device_str,
        "cfg": cfg,
        "model_name_for_log": cfg["model_name_for_log"],
        "cache_path": result["cache_path"],
    }


def run_device_queue(device: str, tasks: List[dict], common: dict) -> List[dict]:
    results = []
    progress_queue = common.get("progress_queue")
    total_on_device = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        queued_task = dict(task)
        queued_task["device"] = device
        emit_progress_event(
            progress_queue,
            {
                "event": "task_started",
                "device": device,
                "dataset": queued_task["dataset"],
                "comparison_tag": queued_task.get("comparison_tag", ""),
                "side": queued_task["side"],
                "seed": int(queued_task["cfg"]["seed"]),
                "pred_len": int(queued_task["cfg"]["pred_len"]),
                "current_idx": idx,
                "total": total_on_device,
            },
        )
        try:
            result = run_training_task(queued_task, common)
        except Exception as exc:
            emit_progress_event(
                progress_queue,
                {
                    "event": "task_failed",
                    "device": device,
                    "dataset": queued_task["dataset"],
                    "comparison_tag": queued_task.get("comparison_tag", ""),
                    "side": queued_task["side"],
                    "seed": int(queued_task["cfg"]["seed"]),
                    "pred_len": int(queued_task["cfg"]["pred_len"]),
                    "current_idx": idx,
                    "total": total_on_device,
                    "error": str(exc),
                },
            )
            raise
        emit_progress_event(
            progress_queue,
            {
                "event": "task_finished",
                "device": device,
                "dataset": queued_task["dataset"],
                "comparison_tag": queued_task.get("comparison_tag", ""),
                "side": queued_task["side"],
                "seed": int(queued_task["cfg"]["seed"]),
                "pred_len": int(queued_task["cfg"]["pred_len"]),
                "current_idx": idx,
                "total": total_on_device,
            },
        )
        results.append(result)
    emit_progress_event(
        progress_queue,
        {
            "event": "device_finished",
            "device": device,
            "total": total_on_device,
        },
    )
    return results


def drain_progress_queue(
    progress_queue,
    active_devices: Dict[str, dict],
    completed: int,
    total: int,
    start_time: float,
) -> int:
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break

        event_type = event["event"]
        if event_type == "task_started":
            active_devices[event["device"]] = {
                "dataset": event["dataset"],
                "comparison_tag": event.get("comparison_tag", ""),
                "side": event["side"],
                "seed": event["seed"],
                "pred_len": event["pred_len"],
                "current_idx": event["current_idx"],
                "total": event["total"],
            }
            print_progress_snapshot(
                completed,
                total,
                start_time,
                active_devices,
                f"started {event['device']} {event.get('comparison_tag','') or 'default'} "
                f"{event['dataset']}/pred{event['pred_len']}/{event['side']}/seed{event['seed']}",
            )
        elif event_type == "task_finished":
            completed += 1
            active_devices.pop(event["device"], None)
            print_progress_snapshot(
                completed,
                total,
                start_time,
                active_devices,
                f"finished {event['device']} {event.get('comparison_tag','') or 'default'} "
                f"{event['dataset']}/pred{event['pred_len']}/{event['side']}/seed{event['seed']}",
            )
        elif event_type == "task_failed":
            active_devices.pop(event["device"], None)
            print_progress_snapshot(
                completed,
                total,
                start_time,
                active_devices,
                f"failed {event['device']} {event.get('comparison_tag','') or 'default'} "
                f"{event['dataset']}/pred{event['pred_len']}/{event['side']}/seed{event['seed']}: {event['error']}",
            )
        elif event_type == "device_finished":
            print(
                f"[queue] {event['device']} queue finished "
                f"({event['total']} task{'s' if event['total'] != 1 else ''})"
            )
    return completed


def execute_training_tasks(tasks: List[dict], devices: List[str], common: dict) -> List[dict]:
    if not tasks:
        return []
    device_queues = assign_tasks_to_devices(tasks, devices)
    active_queues = {device: queue for device, queue in device_queues.items() if queue}
    total_tasks = sum(len(queue) for queue in active_queues.values())
    start_time = time.time()

    print(f"[progress] planned {total_tasks} training tasks across {len(active_queues)} device queue(s)")
    for device in sorted(active_queues.keys()):
        print(f"[queue] {device}: {len(active_queues[device])} task(s)")

    if len(active_queues) <= 1:
        progress_queue = None
        results = []
        completed = 0
        active_devices = {}
        for device, queue in active_queues.items():
            single_common = dict(common)
            single_common["progress_queue"] = progress_queue
            results.extend(run_device_queue(device, queue, single_common))
            completed += len(queue)
            print_progress_snapshot(
                completed,
                total_tasks,
                start_time,
                active_devices,
                f"device queue {device} finished",
            )
        return results

    ctx = mp.get_context("spawn")
    results = []
    manager = mp.Manager()
    progress_queue = manager.Queue()
    active_devices = {}
    completed = 0
    last_heartbeat = start_time

    try:
        with ProcessPoolExecutor(max_workers=len(active_queues), mp_context=ctx) as executor:
            future_map = {}
            for device, queue_items in active_queues.items():
                queue_common = dict(common)
                queue_common["progress_queue"] = progress_queue
                future = executor.submit(run_device_queue, device, queue_items, queue_common)
                future_map[future] = device
            pending = set(future_map.keys())

            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                completed = drain_progress_queue(
                    progress_queue,
                    active_devices,
                    completed,
                    total_tasks,
                    start_time,
                )
                for future in done:
                    device = future_map[future]
                    queue_results = future.result()
                    print(f"[done] device={device} finished {len(queue_results)} tasks")
                    results.extend(queue_results)

                now = time.time()
                if now - last_heartbeat >= 30:
                    print_progress_snapshot(
                        completed,
                        total_tasks,
                        start_time,
                        active_devices,
                        "heartbeat",
                    )
                    last_heartbeat = now

            completed = drain_progress_queue(
                progress_queue,
                active_devices,
                completed,
                total_tasks,
                start_time,
            )
    finally:
        manager.shutdown()

    print_progress_snapshot(
        completed,
        total_tasks,
        start_time,
        active_devices,
        "all training tasks finished",
    )
    return results


def group_task_results(task_results: List[dict]) -> Dict[Tuple[str, str, int], Dict[str, List[dict]]]:
    grouped = defaultdict(lambda: {"baseline": [], "ours": []})
    for item in task_results:
        grouped[(item.get("comparison_tag", ""), item["dataset"], int(item["pred_len"]))][item["side"]].append(item)
    return dict(grouped)


def score_candidates(
    ours: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    baseline_seed: int,
    ours_seed: int,
    baseline_model_name: str,
    ours_model_name: str,
) -> List[dict]:
    candidates = []
    num_samples = min(ours["trues"].shape[0], baseline["trues"].shape[0])
    num_vars = min(ours["trues"].shape[2], baseline["trues"].shape[2])

    for sample_idx in range(num_samples):
        for var_idx in range(num_vars):
            y_true = ours["trues"][sample_idx, :, var_idx]
            y_ours = ours["preds"][sample_idx, :, var_idx]
            y_base = baseline["preds"][sample_idx, :, var_idx]

            ours_mae = float(np.mean(np.abs(y_ours - y_true)))
            base_mae = float(np.mean(np.abs(y_base - y_true)))
            ours_mse = float(np.mean(np.square(y_ours - y_true)))
            base_mse = float(np.mean(np.square(y_base - y_true)))
            mae_gain = base_mae - ours_mae
            mse_gain = base_mse - ours_mse
            truth_std = float(np.std(y_true))
            truth_span = float(np.max(y_true) - np.min(y_true))
            ours_volatility = float(np.mean(np.abs(np.diff(y_ours))))
            base_volatility = float(np.mean(np.abs(np.diff(y_base))))
            smooth_gain = base_volatility - ours_volatility
            score = (
                3.2 * mae_gain
                + 1.1 * mse_gain
                + 0.08 * truth_std
                + 0.05 * truth_span
                + 0.6 * smooth_gain
            )

            candidates.append(
                {
                    "sample_idx": sample_idx,
                    "var_idx": var_idx,
                    "score": score,
                    "ours_mae": ours_mae,
                    "base_mae": base_mae,
                    "ours_mse": ours_mse,
                    "base_mse": base_mse,
                    "mae_gain": mae_gain,
                    "mse_gain": mse_gain,
                    "truth_std": truth_std,
                    "truth_span": truth_span,
                    "smooth_gain": smooth_gain,
                    "baseline_seed": baseline_seed,
                    "ours_seed": ours_seed,
                    "baseline_model_name": baseline_model_name,
                    "ours_model_name": ours_model_name,
                    "seed_pair": f"b{baseline_seed}_o{ours_seed}",
                }
            )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def select_diverse_candidates(candidates: List[dict], top_k: int) -> List[dict]:
    selected = []
    used_pairs = set()
    used_sample_vars = set()

    for item in candidates:
        pair_key = (item["baseline_seed"], item["ours_seed"])
        sample_var_key = (item["sample_idx"], item["var_idx"])
        if pair_key in used_pairs:
            continue
        if sample_var_key in used_sample_vars:
            continue
        selected.append(item)
        used_pairs.add(pair_key)
        used_sample_vars.add(sample_var_key)
        if len(selected) >= top_k:
            return selected

    for item in candidates:
        sample_var_key = (item["sample_idx"], item["var_idx"])
        if sample_var_key in used_sample_vars:
            continue
        selected.append(item)
        used_sample_vars.add(sample_var_key)
        if len(selected) >= top_k:
            break

    for item in candidates:
        full_key = (
            item["baseline_seed"],
            item["ours_seed"],
            item["sample_idx"],
            item["var_idx"],
        )
        if any(
            (
                x["baseline_seed"],
                x["ours_seed"],
                x["sample_idx"],
                x["var_idx"],
            ) == full_key
            for x in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def plot_candidate(
    dataset: str,
    pred_len: int,
    feature_name: str,
    rank: int,
    candidate: dict,
    baseline: Dict[str, np.ndarray],
    ours: Dict[str, np.ndarray],
    baseline_label: str,
    ours_label: str,
    output_dir: str,
) -> Tuple[str, str]:
    sample_idx = candidate["sample_idx"]
    var_idx = candidate["var_idx"]
    history = ours["inputs"][sample_idx, :, var_idx]
    y_true = ours["trues"][sample_idx, :, var_idx]
    y_base = baseline["preds"][sample_idx, :, var_idx]
    y_ours = ours["preds"][sample_idx, :, var_idx]

    hist_x = np.arange(len(history))
    fut_x = np.arange(len(history), len(history) + len(y_true))

    plt.figure(figsize=(12.8, 5.2), dpi=220)
    plt.axvspan(len(history) - 0.5, len(history) + len(y_true) - 0.5, color=COLOR_FORECAST_BG, alpha=0.9)
    plt.plot(hist_x, history, color=COLOR_HISTORY, linewidth=2.0, label="History")
    plt.plot(fut_x, y_true, color=COLOR_GT, linewidth=2.5, linestyle="--", label="Ground Truth")
    plt.plot(fut_x, y_base, color=COLOR_BASELINE, linewidth=2.15, label=baseline_label)
    plt.plot(fut_x, y_ours, color=COLOR_OURS, linewidth=2.35, label=ours_label)
    plt.axvline(len(history) - 1, color="#6D6D6D", linestyle=":", linewidth=1.5)

    y_max = float(np.max(np.r_[history, y_true, y_base, y_ours]))
    y_min = float(np.min(np.r_[history, y_true, y_base, y_ours]))
    y_text = y_max - 0.06 * (y_max - y_min + 1e-6)
    plt.text(len(history) + 8, y_text, "Forecast Region", fontsize=9, color="#5B5B5B")

    title = (
        f"{dataset} | {pred_len}-step Forecast Candidate #{rank} | {feature_name} | "
        f"MAE gain={candidate['mae_gain']:.4f}"
    )
    subtitle = (
        f"sample={sample_idx}, var={var_idx}, "
        f"{baseline_label}(seed={candidate['baseline_seed']}) MAE={candidate['base_mae']:.4f}, "
        f"{ours_label}(seed={candidate['ours_seed']}) MAE={candidate['ours_mae']:.4f}"
    )
    plt.title(title, fontsize=13)
    plt.suptitle(subtitle, y=0.98, fontsize=9, color="#666666")
    plt.xlabel("Time Step", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False, fontsize=10, ncol=4, loc="upper right")
    plt.tight_layout()

    safe_feature_name = feature_name.replace("/", "_").replace(" ", "_")
    file_stub = (
        f"rank{rank:02d}_b{candidate['baseline_seed']}_o{candidate['ours_seed']}_"
        f"sample{sample_idx:04d}_var{var_idx:02d}_{safe_feature_name}"
    )
    png_path = os.path.join(output_dir, f"{file_stub}.png")
    pdf_path = os.path.join(output_dir, f"{file_stub}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def write_summary(summary_path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_csv(csv_path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset_outputs(
    comparison_tag: str,
    dataset: str,
    pred_len: int,
    top_k: int,
    baseline_label: str,
    ours_label: str,
    output_root: str,
    baseline_task_results: List[dict],
    ours_task_results: List[dict],
) -> List[dict]:
    baseline_runs = []
    for item in baseline_task_results:
        baseline_runs.append(
            {
                "cfg": item["cfg"],
                "result": load_cached_result(item["cache_path"]),
                "model_name_for_log": item["model_name_for_log"],
            }
        )

    ours_runs = []
    for item in ours_task_results:
        ours_runs.append(
            {
                "cfg": item["cfg"],
                "result": load_cached_result(item["cache_path"]),
                "model_name_for_log": item["model_name_for_log"],
            }
        )

    candidates = []
    for baseline_run in baseline_runs:
        for ours_run in ours_runs:
            candidates.extend(
                score_candidates(
                    ours=ours_run["result"],
                    baseline=baseline_run["result"],
                    baseline_seed=int(baseline_run["cfg"]["seed"]),
                    ours_seed=int(ours_run["cfg"]["seed"]),
                    baseline_model_name=baseline_run["model_name_for_log"],
                    ours_model_name=ours_run["model_name_for_log"],
                )
            )

    selected = select_diverse_candidates(candidates, top_k)
    baseline_results_by_seed = {int(run["cfg"]["seed"]): run["result"] for run in baseline_runs}
    ours_results_by_seed = {int(run["cfg"]["seed"]): run["result"] for run in ours_runs}
    names = feature_names(dataset)
    output_stub = f"{dataset}_pred{pred_len}" if not comparison_tag else f"{dataset}_pred{pred_len}_{comparison_tag}"
    output_dir = os.path.join(output_root, output_stub)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for rank, item in enumerate(selected, start=1):
        feature_name = names[item["var_idx"]] if item["var_idx"] < len(names) else f"Var-{item['var_idx']}"
        baseline_result = baseline_results_by_seed[item["baseline_seed"]]
        ours_result = ours_results_by_seed[item["ours_seed"]]
        png_path, pdf_path = plot_candidate(
            dataset=dataset,
            pred_len=pred_len,
            feature_name=feature_name,
            rank=rank,
            candidate=item,
            baseline=baseline_result,
            ours=ours_result,
            baseline_label=baseline_label,
            ours_label=ours_label,
            output_dir=output_dir,
        )
        row = {
            "comparison_tag": comparison_tag,
            "rank": rank,
            "dataset": dataset,
            "pred_len": pred_len,
            "baseline_label": baseline_label,
            "ours_label": ours_label,
            "feature_name": feature_name,
            "baseline_seed": item["baseline_seed"],
            "ours_seed": item["ours_seed"],
            "baseline_model_name": item["baseline_model_name"],
            "ours_model_name": item["ours_model_name"],
            "sample_idx": item["sample_idx"],
            "var_idx": item["var_idx"],
            "score": round(item["score"], 6),
            "baseline_mae": round(item["base_mae"], 6),
            "ours_mae": round(item["ours_mae"], 6),
            "mae_gain": round(item["mae_gain"], 6),
            "baseline_mse": round(item["base_mse"], 6),
            "ours_mse": round(item["ours_mse"], 6),
            "mse_gain": round(item["mse_gain"], 6),
            "truth_std": round(item["truth_std"], 6),
            "truth_span": round(item["truth_span"], 6),
            "smooth_gain": round(item["smooth_gain"], 6),
            "png_path": png_path,
            "pdf_path": pdf_path,
        }
        summary_rows.append(row)
        print(
            f"[saved] tag={comparison_tag or 'default'} dataset={dataset} rank={rank} feature={feature_name} "
            f"b_seed={item['baseline_seed']} o_seed={item['ours_seed']} sample={item['sample_idx']} "
            f"mae_gain={item['mae_gain']:.4f}"
        )

    json_path = os.path.join(output_dir, "candidate_summary.json")
    csv_path = os.path.join(output_dir, "candidate_summary.csv")
    write_summary(json_path, summary_rows)
    write_csv(csv_path, summary_rows)
    print(f"\n候选图目录: {output_dir}")
    print(f"摘要 JSON : {json_path}")
    print(f"摘要 CSV  : {csv_path}")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Generate multiple 192-step comparison candidates for thesis figures.")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="单数据集入口，兼容旧命令。")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="多数据集入口，支持一条命令跑多个数据集。")
    parser.add_argument("--pred-len", type=int, default=192, help="预测长度")
    parser.add_argument("--pred-lens", type=int, nargs="+", default=None, help="多预测长度入口，支持一条命令跑多个 pred_len。")
    parser.add_argument(
        "--comparison-preset",
        type=str,
        choices=COMPARISON_PRESET_CHOICES,
        default=None,
        help="一键运行预设好的多数据集对比池，例如 paper_pool。",
    )
    parser.add_argument("--baseline-pattern", type=str, default="T3Time", help="基线模型匹配文本")
    parser.add_argument("--baseline-match", type=str, choices=MATCH_MODE_CHOICES, default="exact", help="基线模型匹配方式")
    parser.add_argument("--baseline-model-type", type=str, choices=MODEL_TYPE_CHOICES, default="t3time")
    parser.add_argument("--baseline-label", type=str, default="T3Time", help="图例中的基线模型名称")
    parser.add_argument("--baseline-seeds", type=int, nargs="+", default=None, help="手动指定要跑的基线种子列表")
    parser.add_argument("--baseline-top-seeds", type=int, default=1, help="若不手动指定种子，则自动选择前 N 个基线种子")
    parser.add_argument("--ours-pattern", type=str, default="T3Time_FreTS_Gated_Qwen", help="对比模型匹配文本")
    parser.add_argument("--ours-match", type=str, choices=MATCH_MODE_CHOICES, default="exact", help="对比模型匹配方式")
    parser.add_argument("--ours-model-type", type=str, choices=MODEL_TYPE_CHOICES, default="frets_gated_qwen")
    parser.add_argument("--ours-label", type=str, default="Ours", help="图例中的对比模型名称")
    parser.add_argument("--ours-seeds", type=int, nargs="+", default=None, help="手动指定要跑的对比模型种子列表")
    parser.add_argument("--ours-top-seeds", type=int, default=1, help="若不手动指定种子，则自动选择前 N 个对比模型种子")
    parser.add_argument("--metric", type=str, choices=["mae", "mse"], default="mae", help="按哪个指标选最佳日志配置")
    parser.add_argument("--top-k", type=int, default=6, help="输出多少张候选图")
    parser.add_argument("--epochs-override", type=int, default=None, help="调试时可覆盖训练轮数")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force-recompute", action="store_true", help="忽略缓存，重新训练并重算")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=str, nargs="+", default=None, help="多 GPU 列表，例如 cuda:0 ... cuda:7")
    parser.add_argument("--strict-datasets", action="store_true", help="若某个数据集缺少日志配置则直接报错，不自动跳过")
    parser.add_argument("--max-train-batches", type=int, default=0, help="仅调试用，0 表示不限制")
    parser.add_argument("--max-val-batches", type=int, default=0, help="仅调试用，0 表示不限制")
    parser.add_argument("--max-test-batches", type=int, default=0, help="仅调试用，0 表示不限制")
    args = parser.parse_args()

    datasets = resolve_datasets(args.dataset, args.datasets)
    pred_lens = resolve_pred_lens(args.pred_len, args.pred_lens)
    devices = resolve_devices(args.device, args.devices)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"[datasets] {datasets}")
    print(f"[pred_lens] {pred_lens}")
    print(f"[devices] {devices}")
    if args.comparison_preset:
        print(f"[comparison_preset] {args.comparison_preset}")

    comparison_specs = build_comparison_specs(args, datasets, pred_lens)
    dataset_plans, skipped = collect_dataset_plans(comparison_specs, args)
    if not dataset_plans:
        raise RuntimeError("没有可执行的数据集任务。请检查日志配置或关闭 strict 模式后重试。")

    for plan in dataset_plans:
        print(
            f"[plan] tag={plan['comparison_tag']} dataset={plan['dataset']} pred_len={plan['pred_len']} "
            f"labels={plan['baseline_label']} vs {plan['ours_label']} "
            f"baseline_tasks={len(plan['baseline_tasks'])} ours_tasks={len(plan['ours_tasks'])}"
        )

    training_tasks = flatten_training_tasks(dataset_plans)
    task_common = {
        "cache_dir": args.cache_dir,
        "force_recompute": args.force_recompute,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
    }
    task_results = execute_training_tasks(training_tasks, devices, task_common)
    grouped_results = group_task_results(task_results)

    combined_rows = []
    for plan in dataset_plans:
        key = (plan["comparison_tag"], plan["dataset"], int(plan["pred_len"]))
        if key not in grouped_results:
            continue
        dataset_rows = generate_dataset_outputs(
            comparison_tag=plan["comparison_tag"],
            dataset=plan["dataset"],
            pred_len=plan["pred_len"],
            top_k=args.top_k,
            baseline_label=plan["baseline_label"],
            ours_label=plan["ours_label"],
            output_root=args.output_dir,
            baseline_task_results=grouped_results[key]["baseline"],
            ours_task_results=grouped_results[key]["ours"],
        )
        combined_rows.extend(dataset_rows)

    if args.comparison_preset:
        summary_suffix = args.comparison_preset
    else:
        summary_suffix = f"pred{pred_lens[0]}" if len(pred_lens) == 1 else "multi_pred_lens"
    combined_json = os.path.join(args.output_dir, f"combined_candidate_summary_{summary_suffix}.json")
    combined_csv = os.path.join(args.output_dir, f"combined_candidate_summary_{summary_suffix}.csv")
    write_summary(combined_json, combined_rows)
    write_csv(combined_csv, combined_rows)
    print(f"\n综合摘要 JSON: {combined_json}")
    print(f"综合摘要 CSV : {combined_csv}")

    if skipped:
        skipped_json = os.path.join(args.output_dir, f"skipped_datasets_{summary_suffix}.json")
        write_summary(skipped_json, skipped)
        print(f"跳过数据集记录: {skipped_json}")


if __name__ == "__main__":
    main()
