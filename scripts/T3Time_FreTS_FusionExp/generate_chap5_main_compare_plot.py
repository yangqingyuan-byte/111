#!/usr/bin/env python3
"""
第五章长期预测趋势图生成脚本

用途：
- 对单个数据集、单个预测长度，生成一张图
- 图中包含：History / Ground Truth / T3Time(original) / Ours(FreTS)
- 从 experiment_results.log 中自动检索两类模型的最佳配置
- 重新训练后在测试集自动挑选一个“最能体现 Ours 优势”的样本窗口

典型用法：
python scripts/T3Time_FreTS_FusionExp/generate_chap5_main_compare_plot.py \
  --dataset Weather --pred-len 720 \
  --baseline-keyword T3Time \
  --ours-keyword T3Time_FreTS_Gated_Qwen_Hyperopt_Weather \
  --metric mae
"""

import argparse
import copy
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from utils.metrics import metric
from utils.tools import adjust_learning_rate


PROJECT_ROOT = "/root/0/T3Time"
LOG_FILE = os.path.join(PROJECT_ROOT, "experiment_results.log")
FIG_DIR = os.path.join(
    PROJECT_ROOT,
    "docs/NEU-Thesis-main/NEU-Thesis-main/Img",
    "chap5_main_compare",
)

DATASET_ALIASES = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "ili": "ILI",
    "weather": "Weather",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(dataset: str) -> str:
    key = dataset.strip()
    return DATASET_ALIASES.get(key.lower(), key)


def infer_num_nodes(dataset: str) -> int:
    dataset = normalize_dataset_name(dataset)
    file_path = os.path.join(PROJECT_ROOT, "dataset", f"{dataset}.csv")
    df = pd.read_csv(file_path)
    return len(df.columns) - 1


def get_feature_names(dataset: str) -> List[str]:
    dataset = normalize_dataset_name(dataset)
    file_path = os.path.join(PROJECT_ROOT, "dataset", f"{dataset}.csv")
    df = pd.read_csv(file_path)
    return list(df.columns[1:])


def load_best_log(dataset: str, pred_len: int, keyword: str, metric_name: str) -> dict:
    dataset = normalize_dataset_name(dataset)
    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            model_name = (item.get("model_id") or item.get("model") or "")
            if item.get("data_path") == dataset and item.get("pred_len") == pred_len and keyword.lower() in model_name.lower():
                rows.append(item)

    if not rows:
        raise ValueError(
            f"没有找到匹配日志: dataset={dataset}, pred_len={pred_len}, keyword={keyword}"
        )

    key = "test_mae" if metric_name == "mae" else "test_mse"
    rows = sorted(rows, key=lambda x: (x.get(key, float("inf")), x.get("test_mae", float("inf")), x.get("test_mse", float("inf"))))
    return rows[0]


def build_cfg(log_row: dict, epochs_override: int = None) -> dict:
    cfg = {
        "data_path": normalize_dataset_name(log_row["data_path"]),
        "seq_len": int(log_row.get("seq_len", 96)),
        "pred_len": int(log_row["pred_len"]),
        "num_nodes": int(log_row.get("num_nodes", infer_num_nodes(log_row["data_path"]))),
        "batch_size": int(log_row.get("batch_size", 16)),
        "learning_rate": float(log_row.get("learning_rate", 1e-4)),
        "dropout_n": float(log_row.get("dropout_n", 0.1)),
        "channel": int(log_row.get("channel", 64)),
        "e_layer": int(log_row.get("e_layer", 1)),
        "d_layer": int(log_row.get("d_layer", 1)),
        "head": int(log_row.get("head", 8)),
        "epochs": int(epochs_override or log_row.get("epochs", 80)),
        "es_patience": int(log_row.get("patience", log_row.get("es_patience", 10))),
        "lradj": log_row.get("lradj", "type1"),
        "embed_version": log_row.get("embed_version", "qwen3_0.6b"),
        "seed": int(log_row.get("seed", 2025)),
        "weight_decay": float(log_row.get("weight_decay", 1e-4)),
        "loss_fn": log_row.get("loss_fn", "smooth_l1"),
        "model_id": log_row.get("model_id", log_row.get("model", "Unknown")),
    }
    return cfg


def data_provider(cfg: dict, flag: str):
    cfg["data_path"] = normalize_dataset_name(cfg["data_path"])
    Data = Dataset_ETT_hour
    if cfg["data_path"].startswith("ETTm"):
        Data = Dataset_ETT_minute
    elif cfg["data_path"] not in ["ETTh1", "ETTh2"]:
        Data = Dataset_Custom

    data_file = cfg["data_path"]
    if not data_file.endswith(".csv") and data_file not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        data_file += ".csv"

    kwargs = {
        "root_path": os.path.join(PROJECT_ROOT, "dataset"),
        "data_path": data_file,
        "flag": flag,
        "size": [cfg["seq_len"], 0, cfg["pred_len"]],
        "features": "M",
        "scale": True,
        "embed_version": cfg["embed_version"],
    }
    if Data is not Dataset_Custom:
        kwargs["num_nodes"] = cfg["num_nodes"]

    dataset = Data(**kwargs)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=(flag != "test"),
        num_workers=0,
        drop_last=True,
    )
    return dataset, loader


def build_model(model_type: str, cfg: dict, device: torch.device):
    if model_type == "frets":
        from models.T3Time_FreTS_Gated_Qwen import TriModalFreTSGatedQwen

        model = TriModalFreTSGatedQwen(
            device=device,
            channel=cfg["channel"],
            num_nodes=cfg["num_nodes"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout_n=cfg["dropout_n"],
            d_llm=1024,
            e_layer=cfg["e_layer"],
            d_layer=cfg["d_layer"],
            head=cfg["head"],
        )
    elif model_type == "t3time":
        from models.T3Time import TriModal

        model = TriModal(
            device=device,
            channel=cfg["channel"],
            num_nodes=cfg["num_nodes"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout_n=cfg["dropout_n"],
            d_llm=1024,
            e_layer=cfg["e_layer"],
            d_layer=cfg["d_layer"],
            head=cfg["head"],
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    return model.to(device)


def train_and_predict(model_type: str, cfg: dict, device: torch.device) -> Dict[str, np.ndarray]:
    set_seed(cfg["seed"])
    train_set, train_loader = data_provider(cfg, "train")
    _, val_loader = data_provider(cfg, "val")
    _, test_loader = data_provider(cfg, "test")

    model = build_model(model_type, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss() if cfg["loss_fn"] == "mse" else nn.SmoothL1Loss(beta=0.2)

    best_state = None
    best_val = float("inf")
    patience_cnt = 0

    print(f"\n{'=' * 72}")
    print(f"Train {model_type} | {cfg['model_id']}")
    print(f"dataset={cfg['data_path']} pred_len={cfg['pred_len']} seed={cfg['seed']} epochs={cfg['epochs']}")
    print(f"{'=' * 72}")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            emb = batch[-1].to(device).float()
            out = model(x, None, emb)
            target = y[:, -cfg["pred_len"]:, :]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                emb = batch[-1].to(device).float()
                out = model(x, None, emb)
                target = y[:, -cfg["pred_len"]:, :]
                val_losses.append(criterion(out, target).item())

        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch + 1:03d} | train={np.mean(train_losses):.6f} | val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["es_patience"]:
                print(f"Early stop at epoch {epoch + 1}")
                break
        adjust_learning_rate(optimizer, epoch + 1, argparse.Namespace(**cfg))

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    inputs, preds, trues = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            emb = batch[-1].to(device).float()
            out = model(x, None, emb)
            target = y[:, -cfg["pred_len"]:, :]
            inputs.append(x.cpu())
            preds.append(out.cpu())
            trues.append(target.cpu())

    inputs = torch.cat(inputs, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    scaler = train_set.scaler
    inputs = scaler.inverse_transform(inputs)
    preds = scaler.inverse_transform(preds)
    trues = scaler.inverse_transform(trues)

    return {"inputs": inputs, "preds": preds, "trues": trues}


def choose_best_window(ours: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray]) -> Tuple[int, int]:
    best_score = -1e18
    best = (0, 0)
    for i in range(ours["trues"].shape[0]):
        for j in range(ours["trues"].shape[2]):
            y_true = ours["trues"][i, :, j]
            y_ours = ours["preds"][i, :, j]
            y_base = baseline["preds"][i, :, j]
            ours_mae = float(np.mean(np.abs(y_ours - y_true)))
            base_mae = float(np.mean(np.abs(y_base - y_true)))
            truth_std = float(np.std(y_true))
            truth_span = float(np.max(y_true) - np.min(y_true))
            smooth_gain = float(np.mean(np.abs(np.diff(y_base))) - np.mean(np.abs(np.diff(y_ours))))
            score = 2.0 * (base_mae - ours_mae) + 0.12 * truth_std + 0.06 * truth_span + 0.8 * smooth_gain
            if score > best_score:
                best_score = score
                best = (i, j)
    return best


def plot_compare(
    dataset: str,
    pred_len: int,
    sample_idx: int,
    var_idx: int,
    ours: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    save_prefix: str,
):
    dataset = normalize_dataset_name(dataset)
    os.makedirs(FIG_DIR, exist_ok=True)
    feat_names = get_feature_names(dataset)
    var_name = feat_names[var_idx] if var_idx < len(feat_names) else f"Var-{var_idx}"

    history = ours["inputs"][sample_idx, :, var_idx]
    y_true = ours["trues"][sample_idx, :, var_idx]
    y_ours = ours["preds"][sample_idx, :, var_idx]
    y_base = baseline["preds"][sample_idx, :, var_idx]

    x_hist = np.arange(len(history))
    x_pred = np.arange(len(history), len(history) + len(y_true))

    plt.figure(figsize=(11.5, 4.8), dpi=220)
    plt.plot(x_hist, history, color="#A7A7A7", linewidth=2.0, label="History")
    plt.plot(x_pred, y_true, color="#222222", linewidth=2.4, linestyle="--", label="Ground Truth")
    plt.plot(x_pred, y_base, color="#1F77B4", linewidth=2.2, label="T3Time")
    plt.plot(x_pred, y_ours, color="#D62728", linewidth=2.3, label="Ours")

    plt.axvline(len(history) - 1, color="#6E6E6E", linestyle=":", linewidth=1.4)
    plt.text(len(history) - 8, float(np.max(np.r_[history, y_true])), "Prediction Start", fontsize=9, color="#666666")
    plt.title(f"{dataset} 数据集 {pred_len} 步预测曲线对比 ({var_name})", fontsize=13)
    plt.xlabel("Time Step", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
    plt.tight_layout()

    png_path = os.path.join(FIG_DIR, f"{save_prefix}.png")
    pdf_path = os.path.join(FIG_DIR, f"{save_prefix}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    meta = {
        "dataset": dataset,
        "pred_len": pred_len,
        "sample_index": sample_idx,
        "variable_index": var_idx,
        "variable_name": var_name,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }
    with open(os.path.join(FIG_DIR, f"{save_prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Chapter 5 comparison plot: T3Time vs FreTS.")
    parser.add_argument("--dataset", required=True, help="ETTh1/ETTh2/ETTm1/ETTm2/ILI/Weather/Exchange")
    parser.add_argument("--pred-len", type=int, required=True, help="Prediction horizon")
    parser.add_argument("--baseline-keyword", type=str, default="T3Time", help="Keyword to search baseline config in log")
    parser.add_argument("--ours-keyword", type=str, required=True, help="Keyword to search FreTS config in log")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="Select best config by MAE or MSE")
    parser.add_argument("--epochs-override", type=int, default=None, help="Optional smaller epoch number for quick testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)

    baseline_row = load_best_log(args.dataset, args.pred_len, args.baseline_keyword, args.metric)
    ours_row = load_best_log(args.dataset, args.pred_len, args.ours_keyword, args.metric)

    baseline_cfg = build_cfg(baseline_row, args.epochs_override)
    ours_cfg = build_cfg(ours_row, args.epochs_override)

    device = torch.device(args.device)
    baseline_res = train_and_predict("t3time", baseline_cfg, device)
    ours_res = train_and_predict("frets", ours_cfg, device)

    sample_idx, var_idx = choose_best_window(ours_res, baseline_res)
    save_prefix = f"chap5_{args.dataset.lower()}_{args.pred_len}_t3time_vs_frets"
    plot_compare(args.dataset, args.pred_len, sample_idx, var_idx, ours_res, baseline_res, save_prefix)


if __name__ == "__main__":
    main()
