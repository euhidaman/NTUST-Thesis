"""
Central Styling Utility for EmberVLM Publication-Quality Figures.

All plotting modules should import from here to ensure consistent
typography, color palette, and export settings across the entire
visualization suite. Targets NeurIPS / ICLR / ICML figure quality.

Usage
-----
    from embervlm.monitoring.style import (
        set_paper_style, save_figure, COLORS, ROBOT_COLORS,
        STAGE_COLORS, PALETTE, tight_layout_with_padding,
    )
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# ── non-interactive backend (must precede pyplot import) ──────────────
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)

# =====================================================================
# 1.  COLOUR PALETTE  (colorblind-safe, CB-friendly)
# =====================================================================
# Wong (2011) "Points of significance: Color blindness" – Nature Methods
# Extended with a few carefully chosen extras.

PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # rose / pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # grey
]

COLORS: dict[str, str] = {
    "primary":    "#0072B2",
    "secondary":  "#E69F00",
    "accent":     "#D55E00",
    "success":    "#009E73",
    "warning":    "#F0E442",
    "neutral":    "#999999",
    "rose":       "#CC79A7",
    "sky":        "#56B4E9",
    "train":      "#0072B2",
    "val":        "#E69F00",
    "i2t":        "#009E73",
    "t2i":        "#CC79A7",
    "baseline":   "#999999",
    "target":     "#D55E00",
}

# Consistent per-robot colours (used in Stage 3 + PR curves + radar)
ROBOT_COLORS: dict[str, str] = {
    "Drone":             "#0072B2",
    "Humanoid":          "#E69F00",
    "Wheeled":           "#009E73",
    "Wheeled Robot":     "#009E73",
    "Wheels":            "#009E73",
    "Legged":            "#CC79A7",
    "Legged Robot":      "#CC79A7",
    "Legs":              "#CC79A7",
    "Underwater":        "#56B4E9",
    "Underwater Robot":  "#56B4E9",
}

# Consistent per-stage colours
STAGE_COLORS: dict[str, str] = {
    "stage1": "#0072B2",
    "stage2": "#E69F00",
    "stage3": "#009E73",
    "stage4": "#CC79A7",
}

STAGE_LABELS: dict[str, str] = {
    "stage1": "Stage 1: Alignment",
    "stage2": "Stage 2: Instruction Tuning",
    "stage3": "Stage 3: Robot Selection",
    "stage4": "Stage 4: CoT Reasoning",
}

# Markers for scatter plots
MARKERS: list[str] = ["o", "s", "D", "^", "v", "P", "X", "*"]

# =====================================================================
# 2.  FIGURE SIZE PRESETS  (inches)
# =====================================================================
# For ML conference papers (NeurIPS / ICLR / ICML columns are ~3.25 in)
FIGSIZE_1COL: Tuple[float, float] = (3.5, 2.6)     # single-column
FIGSIZE_2COL: Tuple[float, float] = (7.0, 3.0)     # full-width
FIGSIZE_2COL_TALL: Tuple[float, float] = (7.0, 5.0)
FIGSIZE_SQUARE: Tuple[float, float] = (3.5, 3.5)

# =====================================================================
# 3.  DPI SETTINGS
# =====================================================================
DPI_SAVE: int = 300       # for PDF / PNG export
DPI_PREVIEW: int = 150    # for W&B or quick inspection

# =====================================================================
# 4.  set_paper_style()
# =====================================================================

_STYLE_APPLIED = False


def set_paper_style(font_family: str = "serif") -> None:
    """Configure matplotlib + seaborn for publication-quality output.

    Idempotent – calling multiple times is safe.

    Parameters
    ----------
    font_family : str
        ``"serif"`` (Computer Modern / Times) or ``"sans-serif"`` (Helvetica).
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    # -- base style -------------------------------------------------
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        try:
            plt.style.use("seaborn-paper")
        except Exception:
            logger.debug("Seaborn paper style not available; using defaults")

    # -- rcParams ---------------------------------------------------
    rc = {
        # figure
        "figure.dpi":          DPI_PREVIEW,
        "savefig.dpi":         DPI_SAVE,
        "figure.figsize":      FIGSIZE_2COL,
        "figure.facecolor":    "white",
        "figure.edgecolor":    "white",
        # font
        "font.family":         font_family,
        "font.size":           9,
        "axes.labelsize":      10,
        "axes.titlesize":      11,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "legend.title_fontsize": 9,
        # axes
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "axes.grid.which":     "major",
        "grid.alpha":          0.30,
        "grid.linewidth":      0.5,
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#333333",
        "axes.linewidth":      0.8,
        # ticks
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        # legend
        "legend.framealpha":   0.80,
        "legend.edgecolor":    "#cccccc",
        "legend.fancybox":     True,
        # lines
        "lines.linewidth":     1.5,
        "lines.markersize":    4,
        # savefig
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "savefig.transparent": False,
    }

    # LaTeX-like serif if available
    if font_family == "serif":
        rc["font.serif"] = [
            "CMU Serif", "Computer Modern", "Times New Roman",
            "DejaVu Serif", "serif",
        ]
    else:
        rc["font.sans-serif"] = [
            "Helvetica", "Arial", "DejaVu Sans", "sans-serif",
        ]

    matplotlib.rcParams.update(rc)

    # seaborn palette
    try:
        import seaborn as sns
        sns.set_palette(PALETTE)
    except Exception:
        pass

    _STYLE_APPLIED = True
    logger.debug("Paper style applied")


# =====================================================================
# 5.  save_figure()
# =====================================================================

def save_figure(
    fig: plt.Figure,
    path_base: str,
    *,
    dpi: int = DPI_SAVE,
    close: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
) -> None:
    """Save *fig* as vector (PDF) and raster (PNG) side-by-side.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_base : str
        Path **without** extension – ``.pdf`` and ``.png`` are appended.
    dpi : int
        Resolution for raster export.
    close : bool
        If ``True``, call ``plt.close(fig)`` after saving.
    formats : sequence of str
        File extensions to save (default: ``("pdf", "png")``).
    """
    path = Path(path_base)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        try:
            fig.savefig(str(out), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
            logger.debug("Saved %s", out)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", out, exc)

    if close:
        plt.close(fig)


# =====================================================================
# 6.  fig_to_pil()  &  tight_layout_with_padding()
# =====================================================================

def fig_to_pil(fig: plt.Figure, dpi: int = DPI_PREVIEW) -> Image.Image:
    """Render a matplotlib Figure to a PIL Image (for W&B / inline display)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def tight_layout_with_padding(
    fig: plt.Figure,
    *,
    pad: float = 1.08,
    h_pad: Optional[float] = None,
    w_pad: Optional[float] = None,
    rect: Optional[Sequence[float]] = None,
) -> None:
    """Call ``fig.tight_layout`` with safe padding to avoid clipped labels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    pad, h_pad, w_pad, rect :
        Forwarded to ``fig.tight_layout()``.
    """
    try:
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
    except Exception as exc:
        logger.debug("tight_layout fallback: %s", exc)
        try:
            fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        except Exception:
            pass


# =====================================================================
# 7.  Convenience: create_figure()
# =====================================================================

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    squeeze: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Thin wrapper around ``plt.subplots`` with paper-style defaults.

    Ensures ``set_paper_style`` is called and returns consistent types.
    """
    set_paper_style()
    if figsize is None:
        if ncols == 1 and nrows == 1:
            figsize = FIGSIZE_1COL
        elif ncols >= 2:
            figsize = FIGSIZE_2COL if nrows == 1 else FIGSIZE_2COL_TALL
        else:
            figsize = FIGSIZE_2COL_TALL
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=squeeze, **kwargs)
    return fig, axes


# =====================================================================
# 8.  robot_color() helper
# =====================================================================

def robot_color(name: str) -> str:
    """Return the canonical colour for a robot class name (case-insensitive)."""
    for key, val in ROBOT_COLORS.items():
        if key.lower() == name.lower():
            return val
    # fallback
    idx = hash(name) % len(PALETTE)
    return PALETTE[idx]
