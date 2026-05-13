"""
Publication-Quality Paper Figures for EmberVLM.

Implements the core figures needed for an ML conference submission:

  3.1  Parameter efficiency comparison (horizontal grouped bar)
  3.2  Accuracy–efficiency Pareto frontier (scatter + front)
  3.3  Multi-stage training overview (concatenated loss + stage shading)
  3.4  t-SNE / UMAP embedding alignment (scatter + match lines)
  3.5  Robot precision–recall curves (one per class)
  3.7  Calibration / reliability diagram (ECE + confidence histogram)
  4.1  Hallucination score histograms (before vs after VA)
  4.2  VA burst timeline (per-token p(VA))

Every function follows the convention::

    def plot_*(…, save_path_base: str | None) -> Tuple[plt.Figure, Image.Image]

All functions are importable and reusable in notebooks / scripts.

Dependencies: numpy, matplotlib, PIL, sklearn (for t-SNE).
Optional: umap-learn (falls back to t-SNE).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── ensure non-interactive backend ────────────────────────────────────
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from PIL import Image

from embervlm.monitoring.style import (
    COLORS,
    FIGSIZE_1COL,
    FIGSIZE_2COL,
    FIGSIZE_2COL_TALL,
    MARKERS,
    PALETTE,
    ROBOT_COLORS,
    STAGE_COLORS,
    STAGE_LABELS,
    create_figure,
    fig_to_pil,
    robot_color,
    save_figure,
    set_paper_style,
    tight_layout_with_padding,
)

logger = logging.getLogger(__name__)

# =====================================================================
# 3.1  PARAMETER EFFICIENCY COMPARISON
# =====================================================================


def plot_parameter_efficiency(
    models_info: List[Dict[str, Any]],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_2COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Horizontal grouped bar chart: total vs trainable parameters.

    **Paper purpose**: Shows that EmberVLM achieves competitive
    performance with a fraction of total / trainable parameters
    compared to baselines.

    Parameters
    ----------
    models_info : list of dict
        Each dict must contain ``"name"`` (str), ``"total_params"`` (float),
        ``"trainable_params"`` (float).  Values are in *raw* counts
        (e.g. ``138e6``).
    save_path_base : str, optional
        Path stem for PDF + PNG export.

    Returns
    -------
    (fig, pil_image)
    """
    set_paper_style()

    names = [m["name"] for m in models_info]
    total = np.array([m["total_params"] for m in models_info])
    train = np.array([m["trainable_params"] for m in models_info])

    y = np.arange(len(names))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars_total = ax.barh(
        y + bar_h / 2, total / 1e6, bar_h,
        label="Total", color=COLORS["primary"], edgecolor="white", linewidth=0.5,
    )
    bars_train = ax.barh(
        y - bar_h / 2, train / 1e6, bar_h,
        label="Trainable", color=COLORS["secondary"], edgecolor="white", linewidth=0.5,
    )

    # Annotate bars
    for bar, val in zip(bars_total, total):
        _label = _human_params(val)
        ax.text(
            bar.get_width() + max(total / 1e6) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            _label, va="center", fontsize=7, color="#333333",
        )
    for bar, val in zip(bars_train, train):
        _label = _human_params(val)
        ax.text(
            bar.get_width() + max(total / 1e6) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            _label, va="center", fontsize=7, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Parameters (millions)")
    ax.set_title("Parameter Efficiency Comparison")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.invert_yaxis()

    # Use log scale if range is > 100×
    if total.max() / total.min() > 100:
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (millions, log scale)")

    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 3.2  ACCURACY–EFFICIENCY PARETO FRONTIER
# =====================================================================


def plot_accuracy_efficiency_pareto(
    models_info: List[Dict[str, Any]],
    save_path_base: Optional[str] = None,
    *,
    x_key: str = "params_m",
    x_label: str = "Parameters (M)",
    y_key: str = "accuracy",
    y_label: str = "Accuracy",
    figsize: Tuple[float, float] = FIGSIZE_1COL,
) -> Tuple[plt.Figure, Image.Image]:
    """2-D scatter with optional Pareto front line.

    **Paper purpose**: Demonstrates EmberVLM sits on or near the
    Pareto frontier of accuracy vs. computational cost.

    Parameters
    ----------
    models_info : list of dict
        Keys: ``"name"``, *x_key* (float), *y_key* (float),
        optional ``"highlight"`` (bool).
    x_key, y_key : str
        Dict keys used for x / y axes.
    """
    set_paper_style()

    fig, ax = plt.subplots(figsize=figsize)

    xs = np.array([m[x_key] for m in models_info])
    ys = np.array([m[y_key] for m in models_info])
    highlights = [m.get("highlight", False) for m in models_info]

    # Draw non-highlighted first
    for i, m in enumerate(models_info):
        marker = "*" if highlights[i] else "o"
        ms = 10 if highlights[i] else 5
        color = COLORS["accent"] if highlights[i] else COLORS["neutral"]
        zorder = 5 if highlights[i] else 3
        ax.scatter(
            xs[i], ys[i], marker=marker, s=ms ** 2, color=color,
            edgecolors="white", linewidths=0.5, zorder=zorder,
        )
        # Label highlighted points directly
        if highlights[i]:
            ax.annotate(
                m["name"], (xs[i], ys[i]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, fontweight="bold", color=color,
            )

    # Pareto front (maximize y, minimize x)
    _draw_pareto_front(ax, xs, ys, minimize_x=True, maximize_y=True)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Accuracy–Efficiency Trade-off")

    # Custom legend
    legend_els = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS["accent"],
                    markersize=8, label="EmberVLM (ours)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["neutral"],
                    markersize=6, label="Baseline"),
        plt.Line2D([0], [0], color=COLORS["primary"], linestyle="--",
                    linewidth=1, label="Pareto front"),
    ]
    ax.legend(handles=legend_els, loc="lower right", framealpha=0.8)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 3.3  MULTI-STAGE TRAINING OVERVIEW
# =====================================================================


def plot_multistage_training_overview(
    stage_histories: Dict[str, Dict[str, List[float]]],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_2COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Single figure with concatenated loss across all training stages.

    **Paper purpose**: Gives the reader a single-glance overview of the
    full training trajectory, with coloured background bands denoting
    stage boundaries.

    Parameters
    ----------
    stage_histories : dict
        Keyed by ``"stage1"``, ``"stage2"``, etc.  Each value is a dict
        with at least ``"loss"`` (list of floats) and optionally
        ``"val_loss"`` and ``"steps"``.
    """
    set_paper_style()

    ordered = sorted(stage_histories.keys())
    if not ordered:
        logger.warning("Empty stage_histories – nothing to plot")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, fig_to_pil(fig)

    fig, ax = plt.subplots(figsize=figsize)

    global_step = 0
    boundaries: list[int] = [0]

    for stage_key in ordered:
        hist = stage_histories[stage_key]
        loss = np.array(hist.get("loss", []))
        val_loss = np.array(hist.get("val_loss", []))
        steps = np.array(hist.get("steps", np.arange(len(loss))))
        if len(loss) == 0:
            continue

        local_steps = np.arange(len(loss)) + global_step
        colour = STAGE_COLORS.get(stage_key, PALETTE[len(boundaries) % len(PALETTE)])

        ax.plot(local_steps, loss, color=colour, linewidth=1.2, alpha=0.85,
                label=f"{STAGE_LABELS.get(stage_key, stage_key)} – train")

        if len(val_loss) > 0:
            # val points are usually fewer – spread evenly
            val_steps = np.linspace(local_steps[0], local_steps[-1], len(val_loss))
            ax.plot(val_steps, val_loss, color=colour, linewidth=1.2,
                    linestyle="--", marker="o", markersize=3, alpha=0.65,
                    label=f"{STAGE_LABELS.get(stage_key, stage_key)} – val")

        global_step += len(loss)
        boundaries.append(global_step)

    # Shade stage regions
    for i, stage_key in enumerate(ordered):
        if i >= len(boundaries) - 1:
            break
        colour = STAGE_COLORS.get(stage_key, PALETTE[i % len(PALETTE)])
        ax.axvspan(boundaries[i], boundaries[i + 1], alpha=0.07, color=colour)
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(
            mid, ax.get_ylim()[1] * 0.97,
            STAGE_LABELS.get(stage_key, stage_key),
            ha="center", va="top", fontsize=7, color=colour, fontweight="bold",
        )

    ax.set_xlabel("Global Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Multi-Stage Training Overview")
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.8)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 3.4  t-SNE / UMAP EMBEDDING ALIGNMENT
# =====================================================================


def plot_embedding_alignment(
    embeddings_image: np.ndarray,
    embeddings_text: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path_base: Optional[str] = None,
    *,
    method: str = "tsne",
    n_match_lines: int = 50,
    figsize: Tuple[float, float] = FIGSIZE_1COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Scatter plot of image and text embeddings in 2-D.

    **Paper purpose**: Visualises how well image–text pairs are
    aligned in the shared embedding space after Stage 1.

    Parameters
    ----------
    embeddings_image : ndarray, shape [N, D]
    embeddings_text : ndarray, shape [N, D]
        Paired – row *i* of each array corresponds to the same sample.
    labels : ndarray, shape [N], optional
        Integer or string labels for colouring by class / source.
    method : ``"tsne"`` | ``"umap"``
        Dimensionality-reduction algorithm (falls back to t-SNE if
        UMAP is unavailable).
    n_match_lines : int
        Number of random image–text pairs connected by thin lines.
    """
    set_paper_style()

    N = len(embeddings_image)
    combined = np.concatenate([embeddings_image, embeddings_text], axis=0)  # [2N, D]

    # Dimensionality reduction
    coords = _reduce_2d(combined, method=method)
    img_coords = coords[:N]
    txt_coords = coords[N:]

    fig, ax = plt.subplots(figsize=figsize)

    # Match lines (thin, muted)
    rng = np.random.default_rng(42)
    subset = rng.choice(N, size=min(n_match_lines, N), replace=False)
    for idx in subset:
        ax.plot(
            [img_coords[idx, 0], txt_coords[idx, 0]],
            [img_coords[idx, 1], txt_coords[idx, 1]],
            color=COLORS["neutral"], alpha=0.15, linewidth=0.5, zorder=1,
        )

    # Scatter
    if labels is not None:
        unique_labels = np.unique(labels)
        for li, lab in enumerate(unique_labels):
            mask = labels == lab
            c = PALETTE[li % len(PALETTE)]
            ax.scatter(img_coords[mask, 0], img_coords[mask, 1],
                       marker="o", s=12, color=c, alpha=0.6, label=f"Img – {lab}", zorder=2)
            ax.scatter(txt_coords[mask, 0], txt_coords[mask, 1],
                       marker="^", s=12, color=c, alpha=0.6, label=f"Txt – {lab}", zorder=2)
    else:
        ax.scatter(img_coords[:, 0], img_coords[:, 1],
                   marker="o", s=12, color=COLORS["primary"], alpha=0.5,
                   label="Image", zorder=2)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
                   marker="^", s=12, color=COLORS["secondary"], alpha=0.5,
                   label="Text", zorder=2)

    method_name = method.upper()
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.set_title(f"Image–Text Alignment ({method_name})")
    ax.legend(loc="best", fontsize=6, framealpha=0.8, markerscale=1.5)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 3.5  ROBOT PRECISION–RECALL CURVES
# =====================================================================


def plot_robot_precision_recall(
    pr_curves: Dict[str, Dict[str, np.ndarray]],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_1COL,
) -> Tuple[plt.Figure, Image.Image]:
    """One PR curve per robot class with per-class AP in the legend.

    **Paper purpose**: Demonstrates per-class detection quality for the
    robot selection task.

    Parameters
    ----------
    pr_curves : dict
        ``{robot_name: {"precision": ndarray, "recall": ndarray, "ap": float}}``
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in pr_curves.items():
        prec = np.asarray(data["precision"])
        rec = np.asarray(data["recall"])
        ap = data.get("ap", np.trapz(prec, rec) if len(prec) > 1 else 0.0)
        colour = robot_color(name)
        ax.plot(rec, prec, color=colour, linewidth=1.4,
                label=f"{name} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision–Recall by Robot Class")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 3.7  CALIBRATION / RELIABILITY DIAGRAM
# =====================================================================


def plot_calibration_curve(
    calibration_stats: Dict[str, Any],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_1COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Reliability diagram + confidence histogram.

    **Paper purpose**: Shows whether model confidence is well-calibrated.

    Parameters
    ----------
    calibration_stats : dict
        Output of ``compute_confidence_calibration`` – must contain
        ``bin_accuracies``, ``bin_confidences``, ``bin_counts``,
        ``ece``, ``mce``.
    """
    set_paper_style()

    bin_acc = np.asarray(calibration_stats["bin_accuracies"])
    bin_conf = np.asarray(calibration_stats["bin_confidences"])
    bin_cnt = np.asarray(calibration_stats["bin_counts"])
    ece = calibration_stats.get("ece", np.nan)
    mce = calibration_stats.get("mce", np.nan)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(figsize[0], figsize[1] * 1.5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # -- top: reliability diagram --
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax1.bar(
        bin_conf, bin_acc, width=0.08, color=COLORS["primary"],
        edgecolor="white", linewidth=0.4, alpha=0.85,
        label=f"Observed (ECE={ece:.3f})",
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # Annotate MCE
    if not np.isnan(mce):
        ax1.text(
            0.97, 0.05, f"MCE = {mce:.3f}",
            transform=ax1.transAxes, ha="right", va="bottom",
            fontsize=7, color=COLORS["accent"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["accent"], alpha=0.8),
        )

    # -- bottom: histogram of bin counts --
    ax2.bar(
        bin_conf, bin_cnt, width=0.08, color=COLORS["neutral"],
        edgecolor="white", linewidth=0.4,
    )
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 4.1  HALLUCINATION SCORE HISTOGRAMS
# =====================================================================


def plot_hallucination_histograms(
    halluc_scores_before: List[float],
    halluc_scores_after: List[float],
    save_path_base: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = FIGSIZE_1COL,
    bins: int = 30,
) -> Tuple[plt.Figure, Image.Image]:
    """Overlapping histograms (or KDEs) for hallucination scores
    before vs. after the VA Refiner.

    **Paper purpose**: Visual evidence that the VA Refiner reduces
    hallucination scores at inference time.

    Parameters
    ----------
    halluc_scores_before, halluc_scores_after : list of float
        Per-sample hallucination rates / scores.
    bins : int
        Histogram bins.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    before = np.asarray(halluc_scores_before)
    after = np.asarray(halluc_scores_after)

    ax.hist(
        before, bins=bins, density=True, alpha=0.55,
        color=COLORS["accent"], edgecolor="white", linewidth=0.4,
        label=f"Before VA (mean={before.mean():.3f})",
    )
    ax.hist(
        after, bins=bins, density=True, alpha=0.55,
        color=COLORS["success"], edgecolor="white", linewidth=0.4,
        label=f"After VA (mean={after.mean():.3f})",
    )

    # Median lines
    ax.axvline(np.median(before), color=COLORS["accent"], linestyle="--",
               linewidth=1.0, alpha=0.8)
    ax.axvline(np.median(after), color=COLORS["success"], linestyle="--",
               linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Hallucination Score")
    ax.set_ylabel("Density")
    ax.set_title("Hallucination Score Distribution")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
# 4.2  VA BURST TIMELINE
# =====================================================================


def plot_va_burst_timeline(
    token_indices: np.ndarray,
    va_probs: np.ndarray,
    flags_burst: np.ndarray,
    save_path_base: Optional[str] = None,
    *,
    threshold: float = 0.7,
    figsize: Tuple[float, float] = FIGSIZE_2COL,
) -> Tuple[plt.Figure, Image.Image]:
    """Per-token p(VA) with burst highlights.

    **Paper purpose**: Illustrates how the VA Refiner detects
    hallucination bursts during auto-regressive generation.

    Parameters
    ----------
    token_indices : ndarray, shape [T]
    va_probs : ndarray, shape [T]
        Probability of visual-absence at each generated token.
    flags_burst : ndarray, shape [T]
        Boolean – ``True`` where a burst was detected.
    threshold : float
        p(VA) threshold drawn as horizontal line.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    token_indices = np.asarray(token_indices)
    va_probs = np.asarray(va_probs)
    flags_burst = np.asarray(flags_burst, dtype=bool)

    # Line plot of p(VA)
    ax.plot(token_indices, va_probs, color=COLORS["primary"], linewidth=1.0,
            label="p(VA)", zorder=2)

    # Above-threshold markers
    above = va_probs >= threshold
    if above.any():
        ax.scatter(
            token_indices[above], va_probs[above],
            color=COLORS["accent"], s=18, zorder=3,
            marker="v", label=f"p(VA) >= {threshold}",
        )

    # Burst spans
    if flags_burst.any():
        _shade_bursts(ax, token_indices, flags_burst,
                      color=COLORS["accent"], alpha=0.12, label="Burst window")

    # Threshold line
    ax.axhline(threshold, color=COLORS["neutral"], linestyle=":",
               linewidth=0.8, label=f"Threshold ({threshold})")

    ax.set_xlabel("Token Index")
    ax.set_ylabel("p(Visual-Absence)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("VA Refiner – Per-Token Hallucination Probability")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    tight_layout_with_padding(fig)

    if save_path_base:
        save_figure(fig, save_path_base)

    pil = fig_to_pil(fig)
    plt.close(fig)
    return fig, pil


# =====================================================================
#  PRIVATE HELPERS
# =====================================================================


def _human_params(n: float) -> str:
    """Convert raw parameter count to human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(int(n))


def _draw_pareto_front(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> None:
    """Draw a dashed line connecting non-dominated points."""
    n = len(xs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (xs[j] <= xs[i]) if minimize_x else (xs[j] >= xs[i])
            y_better = (ys[j] >= ys[i]) if maximize_y else (ys[j] <= ys[i])
            x_strict = (xs[j] < xs[i]) if minimize_x else (xs[j] > xs[i])
            y_strict = (ys[j] > ys[i]) if maximize_y else (ys[j] < ys[i])
            if x_better and y_better and (x_strict or y_strict):
                dominated[i] = True
                break

    front_idx = np.where(~dominated)[0]
    if len(front_idx) < 2:
        return
    order = np.argsort(xs[front_idx])
    front_idx = front_idx[order]
    ax.plot(xs[front_idx], ys[front_idx], color=COLORS["primary"],
            linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)


def _reduce_2d(X: np.ndarray, method: str = "tsne") -> np.ndarray:
    """Reduce *X* to 2-D via UMAP (if available) or t-SNE."""
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            return reducer.fit_transform(X)
        except ImportError:
            logger.info("umap-learn not installed – falling back to t-SNE")

    from sklearn.manifold import TSNE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perp = min(30, max(5, len(X) // 5))
        return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)


def _shade_bursts(
    ax: plt.Axes,
    token_indices: np.ndarray,
    flags: np.ndarray,
    *,
    color: str,
    alpha: float = 0.15,
    label: Optional[str] = None,
) -> None:
    """Shade contiguous burst regions on a timeline axis."""
    in_burst = False
    start = 0
    labelled = False
    for i, f in enumerate(flags):
        if f and not in_burst:
            start = i
            in_burst = True
        elif not f and in_burst:
            lbl = label if not labelled else None
            ax.axvspan(token_indices[start], token_indices[i - 1],
                       color=color, alpha=alpha, label=lbl)
            labelled = True
            in_burst = False
    if in_burst:
        lbl = label if not labelled else None
        ax.axvspan(token_indices[start], token_indices[-1],
                   color=color, alpha=alpha, label=lbl)
