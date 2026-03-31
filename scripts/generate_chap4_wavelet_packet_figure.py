from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = PROJECT_ROOT / "docs" / "NEU-Thesis-main" / "NEU-Thesis-main"
IMG_DIR = THESIS_DIR / "Img"
OUTPUT_PATH = IMG_DIR / "chap4_wavelet_packet_decomposition_python.png"


def load_fonts() -> tuple[FontProperties, FontProperties]:
    sans_candidates = [
        THESIS_DIR / "simhei.ttf",
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    serif_candidates = [
        THESIS_DIR / "simsun.ttc",
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]

    def first_existing(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    sans_path = first_existing(sans_candidates)
    serif_path = first_existing(serif_candidates)

    sans = FontProperties(fname=str(sans_path)) if sans_path else FontProperties(family="DejaVu Sans")
    serif = FontProperties(fname=str(serif_path)) if serif_path else sans
    return sans, serif


def interp_signal(values: np.ndarray, points: int = 240) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, points)
    return np.interp(x_new, x_old, values)


def normalize_signal(values: np.ndarray) -> np.ndarray:
    values = values - np.mean(values)
    scale = np.max(np.abs(values))
    return values / scale if scale > 1e-8 else values


def haar_packet_level2(signal: np.ndarray) -> dict[str, np.ndarray]:
    def split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        even = arr[::2]
        odd = arr[1::2]
        approx = (even + odd) / np.sqrt(2.0)
        detail = (even - odd) / np.sqrt(2.0)
        return approx, detail

    a1, d1 = split(signal)
    aa, ad = split(a1)
    da, dd = split(d1)
    return {"A1": a1, "D1": d1, "AA": aa, "AD": ad, "DA": da, "DD": dd}


def make_demo_signals(n: int = 128) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    trend = 0.9 * np.sin(2 * np.pi * 1.1 * t - 0.25)
    low_osc = 0.45 * np.sin(2 * np.pi * 3.0 * t + 0.55)
    mid_osc = 0.24 * np.sin(2 * np.pi * 7.5 * t - 0.15)
    high_osc = 0.10 * np.sin(2 * np.pi * 15.0 * t + 0.2)
    burst = 0.25 * np.exp(-((t - 0.70) / 0.045) ** 2)
    base = trend + low_osc + mid_osc + high_osc + burst

    signals: list[np.ndarray] = []
    offsets = [0.52, 0.20, -0.12, -0.45]
    phases = [0.0, 0.35, 0.6, 0.9]
    for idx, (offset, phase) in enumerate(zip(offsets, phases)):
        variant = (
            0.82 * base
            + 0.10 * np.sin(2 * np.pi * (idx + 1.8) * t + phase)
            + 0.06 * np.cos(2 * np.pi * 10.0 * t + 0.4 * idx)
        )
        signals.append(variant + offset)

    packet = haar_packet_level2(base)
    return t, signals, packet


def add_text(ax, x: float, y: float, text: str, font: FontProperties, size: float, **kwargs) -> None:
    ax.text(x, y, text, fontproperties=font, fontsize=size, **kwargs)


def add_wave_ax(fig, rect, values: np.ndarray, line_color: str, bg_color: str = "#ffffff") -> None:
    wave_ax = fig.add_axes(rect)
    wave_ax.set_facecolor(bg_color)
    wave_ax.plot(np.linspace(0, 1, len(values)), values, color=line_color, linewidth=2.0)
    wave_ax.set_xlim(0, 1)
    wave_ax.set_ylim(-1.12, 1.12)
    wave_ax.set_xticks([])
    wave_ax.set_yticks([])
    for spine in wave_ax.spines.values():
        spine.set_visible(False)


def build_display_subband_waves(packet: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, 240)

    aa = 0.88 * normalize_signal(interp_signal(packet["AA"]))

    # AD: gentle low-frequency variation with mild local undulation.
    ad = (
        0.62 * np.sin(2 * np.pi * 6.0 * x + 0.22) * (0.92 + 0.08 * np.cos(2 * np.pi * 1.2 * x))
        + 0.12 * np.sin(2 * np.pi * 2.6 * x - 0.40)
        + 0.06 * normalize_signal(interp_signal(packet["AD"]))
    )
    ad = normalize_signal(ad)

    # DA: denser and more energetic oscillation, clearly separated from AD.
    da = (
        (0.46 + 0.34 * x) * np.sin(2 * np.pi * 10.8 * x - 0.12)
        + 0.24 * np.sin(2 * np.pi * 15.8 * x + 0.48)
        + 0.10 * np.cos(2 * np.pi * 7.2 * x - 0.35)
        + 0.05 * normalize_signal(interp_signal(packet["DA"]))
    )
    da = normalize_signal(da)

    # DD: sharp burst-like high-frequency details with pronounced spikes.
    carrier = np.sin(2 * np.pi * 18.0 * x + 0.18)
    envelope = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * x - 0.8))
    spikes = (
        0.95 * np.exp(-((x - 0.18) / 0.020) ** 2)
        - 0.85 * np.exp(-((x - 0.34) / 0.016) ** 2)
        + 1.05 * np.exp(-((x - 0.58) / 0.018) ** 2)
        - 0.90 * np.exp(-((x - 0.74) / 0.015) ** 2)
        + 0.85 * np.exp(-((x - 0.89) / 0.020) ** 2)
    )
    dd = envelope * carrier + 0.55 * spikes + 0.03 * normalize_signal(interp_signal(packet["DD"]))
    dd = normalize_signal(dd)

    return {"AA": aa, "AD": ad, "DA": da, "DD": dd}


def main() -> None:
    sans_font, serif_font = load_fonts()
    _, multi_signals, packet = make_demo_signals()
    display_waves = build_display_subband_waves(packet)

    fig = plt.figure(figsize=(14, 5.9), dpi=300, facecolor="white")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stage_style = dict(fill=False, edgecolor="#8a8a8a", linewidth=1.15, linestyle=(0, (1.5, 1.5)))
    ax.add_patch(Rectangle((0.31, 0.20), 0.22, 0.58, **stage_style))
    ax.add_patch(Rectangle((0.55, 0.12), 0.41, 0.74, **stage_style))

    input_ax = fig.add_axes([0.03, 0.37, 0.10, 0.24])
    input_ax.set_facecolor("white")
    palette = ["#2f2f2f", "#5f7ea8", "#b88a4a", "#7b9d61"]
    for signal, color in zip(multi_signals, palette):
        input_ax.plot(np.linspace(0, 1, len(signal)), normalize_signal(signal), color=color, linewidth=1.6)
    input_ax.set_xticks([])
    input_ax.set_yticks([])
    for spine in input_ax.spines.values():
        spine.set_visible(False)
    add_text(ax, 0.08, 0.30, "多变量时序输入", sans_font, 14.5, ha="center", va="center", color="#333333")

    pre_box = FancyBboxPatch(
        (0.14, 0.43),
        0.13,
        0.14,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="white",
        edgecolor="#444444",
        linewidth=1.5,
    )
    ax.add_patch(pre_box)
    add_text(ax, 0.205, 0.50, "数据预处理", sans_font, 18, ha="center", va="center", color="#1d1d1d")

    level1_box_specs = [
        (0.35, 0.57, 0.13, 0.10, "#dde4ef", "近似分量 A1\n低频主趋势"),
        (0.35, 0.37, 0.13, 0.10, "#efe4da", "细节分量 D1\n高频细节"),
    ]
    for x, y, w, h, color, label in level1_box_specs:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=color,
            edgecolor="#666666",
            linewidth=1.4,
        )
        ax.add_patch(box)
        add_text(ax, x + w / 2, y + h / 2, label, sans_font, 15, ha="center", va="center", color="#1d1d1d")

    leaf_specs = [
        ("AA", "叶子子带 AA\n最低频子带", "#f1e5c8", "#2f2f2f"),
        ("AD", "叶子子带 AD\n低频变化子带", "#dde7d3", "#2f2f2f"),
        ("DA", "叶子子带 DA\n中高频子带", "#d8e5ea", "#2f2f2f"),
        ("DD", "叶子子带 DD\n最高频子带", "#ead8de", "#2f2f2f"),
    ]
    leaf_y = [0.69, 0.53, 0.37, 0.21]
    wave_bg = ["#fcfaf5", "#f7faf4", "#f4f8fa", "#faf6f8"]
    for (key, label, box_color, line_color), y, bg in zip(leaf_specs, leaf_y, wave_bg):
        box = FancyBboxPatch(
            (0.59, y),
            0.20,
            0.12,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=box_color,
            edgecolor="#666666",
            linewidth=1.4,
        )
        ax.add_patch(box)
        add_text(ax, 0.69, y + 0.06, label, sans_font, 15, ha="center", va="center", color="#1d1d1d")
        values = display_waves[key]
        add_wave_ax(fig, [0.81, y + 0.02, 0.11, 0.08], values, line_color, bg)

    arrow_style = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#222222")
    arrows = [
        ((0.11, 0.50), (0.14, 0.50)),
        ((0.27, 0.50), (0.35, 0.62)),
        ((0.27, 0.50), (0.35, 0.42)),
        ((0.48, 0.62), (0.59, 0.75)),
        ((0.48, 0.62), (0.59, 0.59)),
        ((0.48, 0.42), (0.59, 0.43)),
        ((0.48, 0.42), (0.59, 0.27)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, **arrow_style))

    add_text(ax, 0.42, 0.22, "第 1 层小波包分解", sans_font, 14, ha="center", va="center", color="#333333")
    add_text(ax, 0.755, 0.14, "第 2 层小波包分解与叶子子带", sans_font, 14, ha="center", va="center", color="#333333")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
