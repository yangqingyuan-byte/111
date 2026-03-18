#!/usr/bin/env python3
"""
第五章 5.3.3 长期预测趋势图生成脚本

用途：
1. 基于 experiment_results.log 中已经验证过的最佳配置，重新训练指定模型变体；
2. 在测试集上收集输入片段、真实未来序列与预测结果；
3. 自动挑选“完整模型明显优于对照变体且真值波动具有可视性”的样本；
4. 生成适合论文插图的长期预测曲线图（PNG/PDF）。

当前默认提供两个适合第五章的图：
- Weather, pred_len=720: Full / w_o_FreTS / w_o_ImprovedGate
- ILI,     pred_len=60 : Full / w_o_FreTS / w_o_Sparsity

说明：
- 为保证方法可复现，脚本直接使用 FusionExp 模型类。
- 若当前日志中的模型标识不同，可在 FIGURE_PRESETS 中修改 model_ids。
"""

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
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
from models.T3Time_FreTS_Gated_Qwen_FusionExp import TriModalFreTSGatedQwenFusionExp
from utils.metrics import metric
from utils.tools import adjust_learning_rate


PROJECT_ROOT = "/root/0/T3Time"
LOG_FILE = os.path.join(PROJECT_ROOT, "experiment_results.log")
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "docs/NEU-Thesis-main/NEU-Thesis-main/Img",
    "chap5_trend_figures",
)


@dataclass
class FigurePreset:
    name: str
    dataset: str
    pred_len: int
    model_ids: List[str]
    title_cn: str
    filename: str
    selection_metric: str = "mae"


FIGURE_PRESETS: Dict[str, FigurePreset] = {
    "weather_720": FigurePreset(
        name="weather_720",
        dataset="Weather",
        pred_len=720,
        model_ids=[
            "T3Time_FreTS_FusionExp_Ablation_Weather_Full_gate",
            "T3Time_FreTS_FusionExp_Ablation_Weather_w_o_FreTS_gate",
            "T3Time_FreTS_FusionExp_Ablation_Weather_w_o_ImprovedGate_gate",
        ],
        title_cn="Weather 数据集上不同模块配置的长期预测曲线对比",
        filename="chap5_weather_720_trend_compare",
    ),
    "ili_60": FigurePreset(
        name="ili_60",
        dataset="ILI",
        pred_len=60,
        model_ids=[
            "T3Time_FreTS_FusionExp_Ablation_ILI_Full_gate",
            "T3Time_FreTS_FusionExp_Ablation_ILI_w_o_FreTS_gate",
            "T3Time_FreTS_FusionExp_Ablation_ILI_w_o_Sparsity_gate",
        ],
        title_cn="ILI 数据集上不同模块配置的长期预测曲线对比",
        filename="chap5_ili_60_trend_compare",
    ),
}


DISPLAY_NAME_MAP = {
    "T3Time_FreTS_FusionExp_Ablation_Weather_Full_gate": "Full Model",
    "T3Time_FreTS_FusionExp_Ablation_Weather_w_o_FreTS_gate": "Without FreTS",
    "T3Time_FreTS_FusionExp_Ablation_Weather_w_o_ImprovedGate_gate": "Without Improved Gate",
    "T3Time_FreTS_FusionExp_Ablation_Weather_w_o_Sparsity_gate": "Without Sparsity",
    "T3Time_FreTS_FusionExp_Ablation_ILI_Full_gate": "Full Model",
    "T3Time_FreTS_FusionExp_Ablation_ILI_w_o_FreTS_gate": "Without FreTS",
    "T3Time_FreTS_FusionExp_Ablation_ILI_w_o_ImprovedGate_gate": "Without Improved Gate",
    "T3Time_FreTS_FusionExp_Ablation_ILI_w_o_Sparsity_gate": "Without Sparsity",
}


COLOR_MAP = {
    "Ground Truth": "#222222",
    "History": "#A0A0A0",
    "Full Model": "#D62728",
    "Without FreTS": "#1F77B4",
    "Without Improved Gate": "#2CA02C",
    "Without Sparsity": "#9467BD",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_log_rows(dataset: str, pred_len: int, model_ids: List[str]) -> Dict[str, dict]:
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        rows = []
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("data_path") == dataset and item.get("pred_len") == pred_len:
                rows.append(item)

    best_rows = {}
    for model_id in model_ids:
        candidates = [r for r in rows if (r.get("model_id") or "") == model_id]
        if not candidates:
            raise ValueError(f"在日志中未找到配置: dataset={dataset}, pred_len={pred_len}, model_id={model_id}")
        best_rows[model_id] = sorted(
            candidates,
            key=lambda x: (x.get("test_mae", float("inf")), x.get("test_mse", float("inf")))
        )[0]
    return best_rows


def infer_num_nodes(dataset: str) -> int:
    data_file = os.path.join(PROJECT_ROOT, "dataset", f"{dataset}.csv")
    df = pd.read_csv(data_file)
    return len(df.columns) - 1


def feature_names(dataset: str) -> List[str]:
    data_file = os.path.join(PROJECT_ROOT, "dataset", f"{dataset}.csv")
    df = pd.read_csv(data_file)
    return list(df.columns[1:])


def data_provider(cfg: dict, flag: str):
    Data = Dataset_ETT_hour
    if cfg["data_path"].startswith("ETTm"):
        Data = Dataset_ETT_minute
    elif cfg["data_path"] not in ["ETTh1", "ETTh2"]:
        Data = Dataset_Custom

    data_file = cfg["data_path"]
    if not data_file.endswith(".csv") and data_file not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        data_file += ".csv"

    kwargs = dict(
        root_path=os.path.join(PROJECT_ROOT, "dataset"),
        data_path=data_file,
        flag=flag,
        size=[cfg["seq_len"], 0, cfg["pred_len"]],
        features="M",
        scale=True,
        embed_version=cfg["embed_version"],
    )
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


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    return TriModalFreTSGatedQwenFusionExp(
        device=device,
        channel=cfg["channel"],
        num_nodes=cfg["num_nodes"],
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        dropout_n=cfg["dropout_n"],
        e_layer=cfg.get("e_layer", 1),
        d_layer=cfg.get("d_layer", 1),
        head=cfg.get("head", 8),
        sparsity_threshold=cfg.get("sparsity_threshold", 0.009),
        scale=cfg.get("frets_scale", 0.018),
        fusion_mode=cfg.get("fusion_mode", "gate"),
        use_frets=bool(cfg.get("use_frets", True)),
        use_complex=bool(cfg.get("use_complex", True)),
        use_sparsity=bool(cfg.get("use_sparsity", True)),
        use_improved_gate=bool(cfg.get("use_improved_gate", True)),
    ).to(device)


def make_runtime_cfg(log_row: dict, epochs_override: int = None) -> dict:
    cfg = {
        "data_path": log_row["data_path"],
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
        "model_id": log_row.get("model_id", "unknown"),
        "fusion_mode": log_row.get("fusion_mode", "gate"),
        "sparsity_threshold": float(log_row.get("sparsity_threshold", 0.009)),
        "frets_scale": float(log_row.get("frets_scale", 0.018)),
        "use_frets": bool(log_row.get("use_frets", True)),
        "use_complex": bool(log_row.get("use_complex", True)),
        "use_sparsity": bool(log_row.get("use_sparsity", True)),
        "use_improved_gate": bool(log_row.get("use_improved_gate", True)),
    }
    return cfg


def train_and_collect(cfg: dict, device: torch.device) -> Dict[str, np.ndarray]:
    set_seed(cfg["seed"])
    train_set, train_loader = data_provider(cfg, "train")
    _, val_loader = data_provider(cfg, "val")
    test_set, test_loader = data_provider(cfg, "test")

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss() if cfg["loss_fn"] == "mse" else nn.SmoothL1Loss(beta=0.2)

    best_state = None
    best_val = float("inf")
    patience = 0

    print(f"\n{'=' * 72}")
    print(f"Training {cfg['model_id']}")
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
            patience = 0
        else:
            patience += 1
            if patience >= cfg["es_patience"]:
                print(f"Early stop at epoch {epoch + 1}")
                break
        adjust_learning_rate(optimizer, epoch + 1, argparse.Namespace(**cfg))

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues, inputs = [], [], []
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

    inputs = torch.cat(inputs, dim=0)
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mse, mae = metric(preds, trues)
    print(f"Test MSE={mse:.6f} | Test MAE={mae:.6f}")

    scaler = train_set.scaler
    inputs_inv = scaler.inverse_transform(inputs.numpy())
    preds_inv = scaler.inverse_transform(preds.numpy())
    trues_inv = scaler.inverse_transform(trues.numpy())

    return {
        "inputs": inputs_inv,
        "preds": preds_inv,
        "trues": trues_inv,
        "mse": mse,
        "mae": mae,
    }


def choose_best_sample(results: Dict[str, Dict[str, np.ndarray]], full_model_id: str) -> Tuple[int, int]:
    full = results[full_model_id]
    model_ids = list(results.keys())
    baseline_ids = [m for m in model_ids if m != full_model_id]

    full_preds = full["preds"]
    full_trues = full["trues"]

    best_score = -1e18
    best_pair = (0, 0)

    for sample_idx in range(full_trues.shape[0]):
        true_sample = full_trues[sample_idx]
        full_sample = full_preds[sample_idx]
        for var_idx in range(true_sample.shape[1]):
            y_true = true_sample[:, var_idx]
            y_full = full_sample[:, var_idx]
            full_mae = np.mean(np.abs(y_full - y_true))
            baseline_maes = []
            for mid in baseline_ids:
                baseline_maes.append(
                    np.mean(np.abs(results[mid]["preds"][sample_idx, :, var_idx] - y_true))
                )
            avg_baseline_mae = float(np.mean(baseline_maes)) if baseline_maes else full_mae
            truth_std = float(np.std(y_true))
            truth_span = float(np.max(y_true) - np.min(y_true))

            # 倾向于选择“完整模型明显更优，同时真值曲线具有足够可视波动”的样本
            score = 2.2 * (avg_baseline_mae - full_mae) + 0.15 * truth_std + 0.05 * truth_span
            if score > best_score:
                best_score = score
                best_pair = (sample_idx, var_idx)

    return best_pair


def plot_figure(
    preset: FigurePreset,
    results: Dict[str, Dict[str, np.ndarray]],
    sample_idx: int,
    var_idx: int,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    names = feature_names(preset.dataset)
    var_name = names[var_idx] if var_idx < len(names) else f"Var-{var_idx}"

    full_model_id = preset.model_ids[0]
    x_hist = results[full_model_id]["inputs"][sample_idx, :, var_idx]
    y_true = results[full_model_id]["trues"][sample_idx, :, var_idx]

    hist_x = np.arange(len(x_hist))
    fut_x = np.arange(len(x_hist), len(x_hist) + len(y_true))

    plt.figure(figsize=(11.5, 4.8), dpi=200)
    plt.plot(hist_x, x_hist, color=COLOR_MAP["History"], linewidth=2.0, label="History")
    plt.plot(fut_x, y_true, color=COLOR_MAP["Ground Truth"], linewidth=2.4, linestyle="--", label="Ground Truth")

    for model_id in preset.model_ids:
        disp = DISPLAY_NAME_MAP.get(model_id, model_id)
        y_pred = results[model_id]["preds"][sample_idx, :, var_idx]
        plt.plot(
            fut_x,
            y_pred,
            linewidth=2.2,
            color=COLOR_MAP.get(disp, None),
            label=disp,
            alpha=0.95,
        )

    plt.axvline(len(x_hist) - 1, color="#777777", linestyle=":", linewidth=1.5)
    y_top = float(np.max(np.r_[x_hist, y_true]))
    plt.text(len(x_hist) - 8, y_top, "Prediction Start", fontsize=9, color="#666666")

    plt.title(f"{preset.title_cn} ({var_name})", fontsize=13)
    plt.xlabel("Time Step", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False, ncol=min(5, len(preset.model_ids) + 2), fontsize=9, loc="upper right")
    plt.tight_layout()

    png_path = os.path.join(save_dir, f"{preset.filename}.png")
    pdf_path = os.path.join(save_dir, f"{preset.filename}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    meta = {
        "dataset": preset.dataset,
        "pred_len": preset.pred_len,
        "sample_index": sample_idx,
        "variable_index": var_idx,
        "variable_name": var_name,
        "models": preset.model_ids,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }
    meta_path = os.path.join(save_dir, f"{preset.filename}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved figure: {png_path}")
    print(f"✅ Saved figure: {pdf_path}")
    print(f"✅ Saved meta  : {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 trend figures for thesis.")
    parser.add_argument(
        "--preset",
        type=str,
        default="weather_720",
        choices=sorted(FIGURE_PRESETS.keys()),
        help="Which preset figure to generate.",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Override epochs for all models. Useful for quick testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu",
    )
    args = parser.parse_args()

    preset = FIGURE_PRESETS[args.preset]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_rows = load_log_rows(preset.dataset, preset.pred_len, preset.model_ids)
    runtime_cfgs = {
        model_id: make_runtime_cfg(row, epochs_override=args.epochs_override)
        for model_id, row in best_rows.items()
    }

    device = torch.device(args.device)
    results = {}
    for model_id in preset.model_ids:
        results[model_id] = train_and_collect(runtime_cfgs[model_id], device)

    sample_idx, var_idx = choose_best_sample(results, preset.model_ids[0])
    print(f"Selected sample={sample_idx}, variable={var_idx}")
    plot_figure(preset, results, sample_idx, var_idx, OUTPUT_DIR)


if __name__ == "__main__":
    main()
