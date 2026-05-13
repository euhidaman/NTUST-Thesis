"""
Training Visualization Script for EmberVLM

Generates real-time plots and analysis from training logs and WandB data.
Uses the shared paper style from ``embervlm.monitoring.style``.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so the embervlm package is found
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Non-interactive backend (must come before pyplot)
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
except ImportError:
    sns = None

# ── shared paper style ────────────────────────────────────────────────
from embervlm.monitoring.style import (
    COLORS, PALETTE, ROBOT_COLORS, STAGE_COLORS, STAGE_LABELS,
    set_paper_style, save_figure as _save_fig, fig_to_pil,
    tight_layout_with_padding, DPI_SAVE, robot_color,
)


class TrainingVisualizer:
    """Real-time training visualization from log files or WandB.

    All figures use the shared paper style and are exported as PDF + PNG
    via :func:`save_figure`.
    """

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        set_paper_style()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_stage1_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 1 (alignment) training curves.

        Improvements over original:
        - Consistent colour coding (train/val, i2t/t2i) from the palette.
        - Shaded region for similarity separation.
        - Light grid, white background, legible legends.
        - Fixed y-limits where sensible.
        - Saves PDF + PNG via ``save_figure``.
        """
        if not metrics_history:
            logger.warning("No metrics to plot for Stage 1")
            return

        df = pd.DataFrame(metrics_history)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Stage 1: Visual–Language Alignment", fontsize=12, fontweight="bold")

        # 1. Contrastive Loss ──────────────────────────────────────
        ax = axes[0, 0]
        if "contrastive_loss" in df.columns:
            ax.plot(df.index, df["contrastive_loss"], color=COLORS["train"],
                    linewidth=1.2, alpha=0.4, label="Train")
        if "val_contrastive_loss" in df.columns:
            ax.plot(df.index, df["val_contrastive_loss"], color=COLORS["val"],
                    linewidth=1.5, label="Validation")
        ax.set_title("Contrastive Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

        # 2. Top-1 Accuracy ────────────────────────────────────────
        ax = axes[0, 1]
        if "val_acc_i2t" in df.columns:
            ax.plot(df.index, df["val_acc_i2t"] * 100, color=COLORS["i2t"],
                    linewidth=1.5, label="Image→Text")
        if "val_acc_t2i" in df.columns:
            ax.plot(df.index, df["val_acc_t2i"] * 100, color=COLORS["t2i"],
                    linewidth=1.5, label="Text→Image")
        ax.axhline(y=5, color=COLORS["baseline"], linestyle="--", linewidth=0.8,
                   alpha=0.7, label="Random (5%)")
        ax.set_title("Retrieval Accuracy (Top-1)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

        # 3. Top-5 Accuracy ────────────────────────────────────────
        ax = axes[0, 2]
        if "val_acc_i2t_top5" in df.columns:
            ax.plot(df.index, df["val_acc_i2t_top5"] * 100, color=COLORS["i2t"],
                    linewidth=1.5, label="I→T (Top-5)")
        if "val_acc_t2i_top5" in df.columns:
            ax.plot(df.index, df["val_acc_t2i_top5"] * 100, color=COLORS["t2i"],
                    linewidth=1.5, label="T→I (Top-5)")
        ax.set_title("Retrieval Accuracy (Top-5)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

        # 4. Mean Reciprocal Rank ──────────────────────────────────
        ax = axes[1, 0]
        if "val_i2t_mrr" in df.columns:
            ax.plot(df.index, df["val_i2t_mrr"], color=COLORS["i2t"],
                    linewidth=1.5, label="I→T MRR")
        if "val_t2i_mrr" in df.columns:
            ax.plot(df.index, df["val_t2i_mrr"], color=COLORS["t2i"],
                    linewidth=1.5, label="T→I MRR")
        ax.set_title("Mean Reciprocal Rank")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MRR")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

        # 5. Similarity Separation ─────────────────────────────────
        ax = axes[1, 1]
        if "val_mean_pos_similarity" in df.columns and "val_mean_neg_similarity" in df.columns:
            pos = df["val_mean_pos_similarity"]
            neg = df["val_mean_neg_similarity"]
            ax.plot(df.index, pos, color=COLORS["success"], linewidth=1.5, label="Positive")
            ax.plot(df.index, neg, color=COLORS["accent"], linewidth=1.5, label="Negative")
            ax.fill_between(df.index, pos, neg, alpha=0.12, color=COLORS["primary"],
                           label="Separation")
        ax.set_title("Similarity Separation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(loc="best", fontsize=7, framealpha=0.8)

        # 6. Learning Rate ─────────────────────────────────────────
        ax = axes[1, 2]
        if "lr" in df.columns:
            ax.plot(df.index, df["lr"], color=COLORS["rose"], linewidth=1.5)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")

        tight_layout_with_padding(fig)

        _path = save_path or str(self.output_dir / "stage1_training")
        _save_fig(fig, _path.replace(".png", "").replace(".pdf", ""))
        logger.info("Saved Stage 1 visualization to %s", _path)
        plt.close(fig)
        
    def plot_stage2_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 2 (instruction tuning) training curves.

        Improvements:
        - Log-scale perplexity with dashed target/baseline lines in
          muted colours.
        - Relative change annotation (ΔPPL) in a text box.
        - Consistent palette.
        """
        if not metrics_history:
            logger.warning("No metrics to plot for Stage 2")
            return

        df = pd.DataFrame(metrics_history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Stage 2: Instruction Tuning", fontsize=12, fontweight="bold")

        # 1. Loss ──────────────────────────────────────────────────
        ax = axes[0, 0]
        if "loss" in df.columns:
            ax.plot(df.index, df["loss"], color=COLORS["train"], linewidth=1.2,
                    alpha=0.4, label="Train")
        if "val_loss" in df.columns:
            ax.plot(df.index, df["val_loss"], color=COLORS["val"], linewidth=1.5,
                    label="Validation")
        ax.set_title("Instruction Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

        # 2. Perplexity ────────────────────────────────────────────
        ax = axes[0, 1]
        if "val_perplexity" in df.columns:
            vals = df["val_perplexity"]
            ax.plot(df.index, vals, color=COLORS["primary"], linewidth=1.5)
            ax.axhline(y=20, color=COLORS["success"], linestyle="--", linewidth=0.8,
                       alpha=0.6, label="Target (PPL=20)")
            ax.axhline(y=50, color=COLORS["baseline"], linestyle="--", linewidth=0.8,
                       alpha=0.6, label="Baseline (PPL=50)")

            # Relative change annotation
            if len(vals) >= 2:
                first, last = vals.iloc[0], vals.iloc[-1]
                delta_pct = (last - first) / first * 100
                ax.text(
                    0.97, 0.95,
                    f"ΔPPL: {delta_pct:+.1f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLORS["primary"], alpha=0.85),
                )
        ax.set_title("Perplexity")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

        # 3. Token-level Accuracy ──────────────────────────────────
        ax = axes[1, 0]
        if "val_accuracy" in df.columns:
            ax.plot(df.index, df["val_accuracy"] * 100, color=COLORS["success"],
                    linewidth=1.5)
            ax.axhline(y=10, color=COLORS["baseline"], linestyle="--",
                       linewidth=0.8, alpha=0.6, label="Target (10%)")
        ax.set_title("Token-Level Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

        # 4. Learning Rate ─────────────────────────────────────────
        ax = axes[1, 1]
        if "lr" in df.columns:
            ax.plot(df.index, df["lr"], color=COLORS["rose"], linewidth=1.5)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")

        tight_layout_with_padding(fig)

        _path = save_path or str(self.output_dir / "stage2_training")
        _save_fig(fig, _path.replace(".png", "").replace(".pdf", ""))
        logger.info("Saved Stage 2 visualization to %s", _path)
        plt.close(fig)
    
    def plot_stage3_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 3 (robot selection) training curves.

        Improvements:
        - Consistent robot-class colours from ``ROBOT_COLORS``.
        - Small but non-cluttered markers on per-class F1.
        - Horizontal target line on Macro F1.
        - Fixed y-limits (0–1 for F1, 0–100 for accuracy).
        """
        if not metrics_history:
            logger.warning("No metrics to plot for Stage 3")
            return

        df = pd.DataFrame(metrics_history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Stage 3: Robot Fleet Selection", fontsize=12, fontweight="bold")

        # 1. Loss ──────────────────────────────────────────────────
        ax = axes[0, 0]
        if "loss" in df.columns:
            ax.plot(df.index, df["loss"], color=COLORS["train"], linewidth=1.2,
                    alpha=0.4, label="Train")
        if "val_loss" in df.columns:
            ax.plot(df.index, df["val_loss"], color=COLORS["val"], linewidth=1.5,
                    label="Validation")
        ax.set_title("Classification Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

        # 2. Overall Accuracy ─────────────────────────────────────
        ax = axes[0, 1]
        if "val_accuracy" in df.columns:
            ax.plot(df.index, df["val_accuracy"] * 100, color=COLORS["success"],
                    linewidth=1.5)
            ax.axhline(y=50, color=COLORS["baseline"], linestyle="--",
                       linewidth=0.8, alpha=0.6, label="Target (50%)")
        ax.set_title("Classification Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

        # 3. Per-Class F1 ─────────────────────────────────────────
        ax = axes[1, 0]
        robot_classes = ["Drone", "Humanoid", "Underwater", "Wheels", "Legs"]
        for cls in robot_classes:
            col = f"val_f1_{cls.lower()}"
            if col in df.columns:
                colour = robot_color(cls)
                ax.plot(df.index, df[col], linewidth=1.3, label=cls,
                        color=colour, marker="o", markersize=3)
        ax.set_title("Per-Class F1 Scores")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

        # 4. Macro F1 ─────────────────────────────────────────────
        ax = axes[1, 1]
        if "val_macro_f1" in df.columns:
            ax.plot(df.index, df["val_macro_f1"], color=COLORS["primary"],
                    linewidth=1.5, marker="o", markersize=3)
            ax.axhline(y=0.4, color=COLORS["target"], linestyle="--",
                       linewidth=0.8, alpha=0.6, label="Target (0.4)")
        ax.set_title("Macro F1 Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

        tight_layout_with_padding(fig)

        _path = save_path or str(self.output_dir / "stage3_training")
        _save_fig(fig, _path.replace(".png", "").replace(".pdf", ""))
        logger.info("Saved Stage 3 visualization to %s", _path)
        plt.close(fig)


def parse_train_log(log_path: str) -> Dict[str, List[Dict]]:
    """Parse training log file and extract metrics."""
    stage_metrics = {
        'stage1': [],
        'stage2': [],
        'stage3': [],
        'stage4': []
    }
    
    current_stage = None
    current_epoch_metrics = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect stage transitions
            if 'Stage 1 complete' in line or 'Starting Stage 1' in line:
                current_stage = 'stage1'
            elif 'Stage 2 complete' in line or 'Starting Stage 2' in line:
                current_stage = 'stage2'
            elif 'Stage 3 complete' in line or 'Starting Stage 3' in line:
                current_stage = 'stage3'
            elif 'Stage 4 complete' in line or 'Starting Stage 4' in line:
                current_stage = 'stage4'
            
            # Extract validation metrics
            if 'Validation metrics:' in line and current_stage:
                # Parse metrics dict from log line
                try:
                    metrics_str = line.split('Validation metrics:')[1].strip()
                    # Simple parsing (can be improved with json.loads if formatted correctly)
                    stage_metrics[current_stage].append(current_epoch_metrics.copy())
                    current_epoch_metrics = {}
                except Exception as e:
                    continue
    
    return stage_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize EmberVLM training progress")
    parser.add_argument('--log_path', type=str, default='train.log', help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='./outputs/visualizations', help='Output directory for plots')
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'stage1', 'stage2', 'stage3', 'stage4'], help='Which stage to visualize')
    
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(output_dir=args.output_dir)
    
    print(f"📊 Generating training visualizations from {args.log_path}...")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Note: For real-time monitoring, check your Weights & Biases dashboard!")
    print()


