"""
Efficiency and Quantization Figures for EmberVLM.

Section 5 of the visualization spec:

  5.1  Quantization impact – multi-panel (accuracy / size / latency vs bits)

These plots are designed for the "Efficiency" section of a conference
paper or supplementary material.

Every function follows the convention::

    def plot_*(…, save_path_base: str | None) -> Tuple[plt.Figure, Image.Image]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── non-interactive backend ───────────────────────────────────────────
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

from embervlm.monitoring.style import (
    COLORS,
    FIGSIZE_2COL,
    FIGSIZE_2COL_TALL,
    PALETTE,
    create_figure,
    fig_to_pil,
    save_figure,
    set_paper_style,
    tight_layout_with_padding,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 5.1  QUANTIZATION IMPACT
# =====================================================================


def plot_quantization_impact(
    results: List[Dict[str, Any]],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_2COL_TALL,
) -> Tuple[plt.Figure, Image.Image]:
    """Multi-panel figure: accuracy / model size / latency vs bit-width.

    **Paper purpose**: Demonstrates that EmberVLM retains strong
    accuracy under aggressive quantization while achieving significant
    size and latency reductions – important for edge deployment.

    Parameters
    ----------
    results : list of dict
        Each dict must contain:

        * ``"bits"`` (int) – bit-width (e.g. 16, 8, 4).
        * ``"accuracy"`` (float) – task accuracy.
        * ``"size_mb"`` (float) – model size in MB.
        * ``"latency_ms"`` (float) – inference latency in ms.
    save_path_base : str, optional
        Path stem for PDF + PNG export.

    Returns
    -------
    (fig, pil_image)
    """
    set_paper_style()

    # Sort by bits descending (16 → 8 → 4)
    results = sorted(results, key=lambda r: r["bits"], reverse=True)

    bits = [r["bits"] for r in results]
    accs = [r["accuracy"] for r in results]
    sizes = [r["size_mb"] for r in results]
    lats = [r["latency_ms"] for r in results]

    bit_labels = [f"{b}-bit" for b in bits]
    x = np.arange(len(bits))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: accuracy vs bits (line) ──────────────────────────
    ax = axes[0]
    ax.plot(x, accs, marker="o", color=COLORS["primary"], linewidth=1.5,
            markersize=6, zorder=3)
    for i, (xi, acc) in enumerate(zip(x, accs)):
        ax.annotate(f"{acc:.1%}", (xi, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=COLORS["primary"])
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")

    # ── Panel 2: size vs bits (bar) ───────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, sizes, color=COLORS["secondary"], edgecolor="white",
                  linewidth=0.5, width=0.55)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.02,
                f"{s:.0f}", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size")
    ax.grid(axis="y")

    # ── Panel 3: latency vs bits (bar) ────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, lats, color=COLORS["success"], edgecolor="white",
                  linewidth=0.5, width=0.55)
    for bar, l in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(lats) * 0.02,
                f"{l:.0f} ms", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency")
    ax.grid(axis="y")

    fig.suptitle("Quantization Impact on EmberVLM", fontsize=11, fontweight="bold", y=1.02)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 5.2  DEVICE LATENCY COMPARISON (bonus utility)
# =====================================================================


def plot_device_latency_comparison(
    device_results: List[Dict[str, Any]],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_2COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Grouped bar chart comparing latency across devices / backends.

    **Paper purpose**: Supplementary figure showing real-world deployment
    latency on edge hardware (Jetson, RPi, phone, etc.).

    Parameters
    ----------
    device_results : list of dict
        Each dict:

        * ``"device"`` (str) – e.g. "Jetson Orin Nano", "RPi 5".
        * ``"fp16_ms"`` (float) – latency at FP16.
        * ``"int8_ms"`` (float, optional) – latency at INT8.
        * ``"int4_ms"`` (float, optional) – latency at INT4.
    """
    set_paper_style()

    devices = [d["device"] for d in device_results]
    y = np.arange(len(devices))
    bar_h = 0.25
    precisions = []
    for key, label, color in [
        ("fp16_ms", "FP16", COLORS["primary"]),
        ("int8_ms", "INT8", COLORS["secondary"]),
        ("int4_ms", "INT4", COLORS["success"]),
    ]:
        if any(key in d for d in device_results):
            precisions.append((key, label, color))

    fig, ax = plt.subplots(figsize=figsize)

    n = len(precisions)
    offsets = np.linspace(-bar_h * (n - 1) / 2, bar_h * (n - 1) / 2, n)

    for (key, label, color), off in zip(precisions, offsets):
        vals = [d.get(key, 0) for d in device_results]
        ax.barh(y + off, vals, bar_h * 0.9, label=label, color=color,
                edgecolor="white", linewidth=0.4)
        for yi, v in zip(y, vals):
            if v > 0:
                ax.text(v + max(vals) * 0.01, yi + off, f"{v:.0f}",
                        va="center", fontsize=7, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(devices)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Inference Latency by Device")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    ax.invert_yaxis()
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil
