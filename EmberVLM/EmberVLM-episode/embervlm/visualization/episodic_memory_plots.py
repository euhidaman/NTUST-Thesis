"""
Episodic Memory Visualisation — Real-Time Training Tracker + Plots

Provides ``EpisodicMemoryTracker``, a lightweight object that sits inside the
Stage 1.5 and Stage 5 training loops, collects per-step metrics, and
generates six publication-quality figures at the end of each stage.

Figures produced (each shows genuinely different information):
  1. memory_fill_curve       — cumulative slots used + write rate over steps
  2. novelty_landscape       — 3-timepoint novelty score distributions (histograms)
  3. scope_detector_learning — scope BCE loss + mean prediction confidence
  4. slot_utilization        — sorted slot usage distribution (log-scale)
  5. memory_embedding_map    — PCA/t-SNE of written slots coloured by access recency
  6. consolidation_dynamics  — Stage 5 losses + cosine similarity convergence
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — skipping episodic memory figures")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Style ──
_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_PAL = {
    "primary": "#FF5722",     # deep orange — writes / episode
    "secondary": "#2196F3",   # blue — reference / hallucinate
    "accent": "#4CAF50",      # green — positive signal
    "warn": "#FFC107",        # amber — threshold lines
    "neutral": "#78909C",     # blue-grey — grid / secondary
    "bg": "#FAFAFA",
}


def _apply_style():
    if HAS_MPL:
        plt.rcParams.update(_STYLE)


def _savefig(fig, output_dir: str, name: str) -> str:
    """Save as PNG (for quick viewing) and return path."""
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path}")
    return path


def _rolling_mean(arr, window: int = 20):
    """Simple rolling mean for smoothing."""
    if len(arr) < window:
        return np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# =========================================================================
#  Tracker — collects metrics inside training loops
# =========================================================================

class EpisodicMemoryTracker:
    """
    Lightweight metric accumulator for episodic memory training stages.

    Usage::

        tracker = EpisodicMemoryTracker()
        for step, batch in enumerate(loader):
            ...
            tracker.log_init_step(step, write_stats, scope_loss, scope_preds, memory)
        tracker.generate_plots(output_dir)
    """

    def __init__(self):
        # Stage 1.5 — memory init
        self.init_steps: List[int] = []
        self.num_written: List[int] = []
        self.sigma_mean: List[float] = []
        self.sigma_max: List[float] = []
        self.scope_loss: List[float] = []
        self.scope_pred_mean: List[float] = []
        self.scope_pred_std: List[float] = []
        self.slots_used: List[int] = []

        # Novelty snapshots (step → array of per-sample scores)
        self.novelty_snapshots: Dict[int, np.ndarray] = {}

        # Stage 5 — consolidation
        self.consol_steps: List[int] = []
        self.sup_loss: List[float] = []
        self.align_loss: List[float] = []
        self.cosine_sim: List[float] = []

        # Final memory state snapshots (numpy)
        self._final_M: Optional[np.ndarray] = None
        self._final_usage: Optional[np.ndarray] = None
        self._final_recency: Optional[np.ndarray] = None

    # ----- Stage 1.5 logging -----

    def log_init_step(
        self,
        step: int,
        write_stats: Dict[str, Any],
        scope_loss_val: Optional[float] = None,
        scope_preds: Optional["torch.Tensor"] = None,
        memory_controller: Optional[Any] = None,
    ):
        """Record one Stage 1.5 step."""
        self.init_steps.append(step)
        self.num_written.append(write_stats.get("num_written", 0))
        self.sigma_mean.append(write_stats.get("sigma_mean", 0.0))
        self.sigma_max.append(write_stats.get("sigma_max", 0.0))

        if scope_loss_val is not None:
            self.scope_loss.append(scope_loss_val)
        if scope_preds is not None:
            self.scope_pred_mean.append(scope_preds.mean().item())
            self.scope_pred_std.append(scope_preds.std().item())

        if memory_controller is not None:
            slots_active = int((memory_controller.usage_counts > 0).sum().item())
            self.slots_used.append(slots_active)

    def capture_novelty_snapshot(self, step: int, sigma: "torch.Tensor"):
        """Store the full batch novelty scores at a checkpoint step."""
        self.novelty_snapshots[step] = sigma.detach().cpu().numpy()

    def capture_final_memory(self, memory_controller: Any):
        """Snapshot the memory matrix and metadata at end of training."""
        import torch
        with torch.no_grad():
            self._final_M = memory_controller.M.detach().cpu().numpy()
            self._final_usage = memory_controller.usage_counts.detach().cpu().numpy()
            self._final_recency = memory_controller.last_access_step.detach().cpu().float().numpy()

    # ----- Stage 5 logging -----

    def log_consolidation_step(
        self,
        step: int,
        sup_loss_val: float,
        align_loss_val: float,
        cosine_sim_val: Optional[float] = None,
    ):
        """Record one Stage 5 step."""
        self.consol_steps.append(step)
        self.sup_loss.append(sup_loss_val)
        self.align_loss.append(align_loss_val)
        if cosine_sim_val is not None:
            self.cosine_sim.append(cosine_sim_val)

    # ----- Serialise -----

    def save_metrics(self, output_dir: str):
        """Dump all scalar metrics to JSON for later analysis."""
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "init_steps": self.init_steps,
            "num_written": self.num_written,
            "sigma_mean": self.sigma_mean,
            "sigma_max": self.sigma_max,
            "scope_loss": self.scope_loss,
            "scope_pred_mean": self.scope_pred_mean,
            "scope_pred_std": self.scope_pred_std,
            "slots_used": self.slots_used,
            "consol_steps": self.consol_steps,
            "sup_loss": self.sup_loss,
            "align_loss": self.align_loss,
            "cosine_sim": self.cosine_sim,
        }
        path = os.path.join(output_dir, "episodic_memory_metrics.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Saved metrics: {path}")

    # ----- Plot generation -----

    def generate_plots(self, output_dir: str) -> Dict[str, str]:
        """
        Generate all episodic memory plots from collected data.

        Returns dict mapping plot name → file path.
        """
        if not HAS_MPL:
            logger.warning("matplotlib unavailable — cannot generate plots")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        _apply_style()
        paths: Dict[str, str] = {}

        logger.info(f"Generating episodic memory visualisations in {output_dir} ...")

        # Save raw metrics alongside plots
        self.save_metrics(output_dir)

        # 1. Memory Fill Curve
        if self.slots_used:
            paths["memory_fill_curve"] = self._plot_fill_curve(output_dir)

        # 2. Novelty Landscape
        if self.novelty_snapshots or self.sigma_mean:
            paths["novelty_landscape"] = self._plot_novelty_landscape(output_dir)

        # 3. Scope Detector Learning
        if self.scope_loss:
            paths["scope_detector_learning"] = self._plot_scope_learning(output_dir)

        # 4. Slot Utilization
        if self._final_usage is not None:
            paths["slot_utilization"] = self._plot_slot_utilization(output_dir)

        # 5. Memory Embedding Map
        if self._final_M is not None:
            paths["memory_embedding_map"] = self._plot_embedding_map(output_dir)

        # 6. Consolidation Dynamics
        if self.consol_steps:
            paths["consolidation_dynamics"] = self._plot_consolidation(output_dir)

        logger.info(f"Generated {len(paths)} episodic memory figures")
        return paths

    # =====================================================================
    #  1. Memory Fill Curve — how fast does memory fill up?
    # =====================================================================

    def _plot_fill_curve(self, output_dir: str) -> str:
        """
        Left Y: cumulative unique slots used (shows capacity filling).
        Right Y: per-step write rate (shows when new knowledge is accepted).
        """
        fig, ax1 = plt.subplots(figsize=(9, 4.5))

        steps = np.array(self.init_steps[:len(self.slots_used)])
        slots = np.array(self.slots_used)

        # Slots used (cumulative fill)
        ax1.fill_between(steps, 0, slots, alpha=0.20, color=_PAL["primary"])
        ax1.plot(steps, slots, color=_PAL["primary"], linewidth=1.8, label="Slots Used")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Active Memory Slots", color=_PAL["primary"])
        ax1.tick_params(axis="y", labelcolor=_PAL["primary"])

        # Per-step write count (right axis)
        ax2 = ax1.twinx()
        written = np.array(self.num_written[:len(steps)])
        if len(written) > 10:
            smoothed = _rolling_mean(written, window=max(5, len(written) // 20))
            x_smooth = steps[:len(smoothed)]
            ax2.plot(x_smooth, smoothed, color=_PAL["accent"], linewidth=1.2,
                     alpha=0.8, label="Write Rate (smoothed)")
        else:
            ax2.bar(steps, written, width=max(1, steps[-1] / len(steps)),
                    alpha=0.3, color=_PAL["accent"])
        ax2.set_ylabel("Episodes Written per Step", color=_PAL["accent"])
        ax2.tick_params(axis="y", labelcolor=_PAL["accent"])

        ax1.set_title("Memory Population Over Training", fontweight="bold")
        ax1.grid(True, alpha=0.15, color=_PAL["neutral"])

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)

        fig.tight_layout()
        return _savefig(fig, output_dir, "memory_fill_curve")

    # =====================================================================
    #  2. Novelty Landscape — does memory learn to cover the data?
    # =====================================================================

    def _plot_novelty_landscape(self, output_dir: str) -> str:
        """
        If snapshots exist: overlaid histograms at early / mid / late steps
        showing the novelty distribution shifting left (memory covers more).
        Otherwise: rolling mean of sigma_mean over time.
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # (a) Time-series of sigma_mean
        ax = axes[0]
        if self.sigma_mean:
            steps = np.array(self.init_steps[:len(self.sigma_mean)])
            raw = np.array(self.sigma_mean)
            ax.plot(steps, raw, alpha=0.25, color=_PAL["neutral"], linewidth=0.5)
            if len(raw) > 10:
                smoothed = _rolling_mean(raw, window=max(5, len(raw) // 15))
                ax.plot(steps[:len(smoothed)], smoothed, color=_PAL["primary"],
                        linewidth=2.0, label="Mean Novelty (smoothed)")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Mean Novelty Score (σ)")
            ax.set_title("Novelty Score Over Time", fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.15)

        # (b) Distribution snapshots
        ax = axes[1]
        if self.novelty_snapshots:
            sorted_keys = sorted(self.novelty_snapshots.keys())
            # Pick up to 3 well-spaced snapshots
            if len(sorted_keys) >= 3:
                indices = [0, len(sorted_keys) // 2, -1]
            elif len(sorted_keys) == 2:
                indices = [0, -1]
            else:
                indices = [0]

            colors_hist = ["#90CAF9", "#FF9800", "#4CAF50"]
            for i, idx in enumerate(indices):
                key = sorted_keys[idx]
                scores = self.novelty_snapshots[key]
                col = colors_hist[i % len(colors_hist)]
                ax.hist(scores, bins=40, alpha=0.5, color=col,
                        label=f"Step {key}", edgecolor="white", linewidth=0.3)

            ax.set_xlabel("Novelty Score (σ)")
            ax.set_ylabel("Count")
            ax.set_title("Novelty Distribution Shift", fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.9)
        else:
            # Fallback: show sigma_max if no snapshots
            if self.sigma_max:
                steps = np.array(self.init_steps[:len(self.sigma_max)])
                ax.fill_between(steps, np.array(self.sigma_mean[:len(steps)]),
                                np.array(self.sigma_max[:len(steps)]),
                                alpha=0.3, color=_PAL["secondary"], label="σ range (mean→max)")
                ax.plot(steps, self.sigma_mean[:len(steps)], color=_PAL["primary"],
                        linewidth=1.0, label="σ mean")
                ax.set_xlabel("Training Step")
                ax.set_ylabel("Novelty Score")
                ax.set_title("Novelty Range Over Time", fontweight="bold")
                ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.15)

        fig.tight_layout()
        return _savefig(fig, output_dir, "novelty_landscape")

    # =====================================================================
    #  3. Scope Detector Learning — is the gating MLP learning?
    # =====================================================================

    def _plot_scope_learning(self, output_dir: str) -> str:
        """
        Left Y: scope detector BCE loss (should decrease).
        Right Y: mean prediction + confidence band (should become bimodal).
        """
        fig, ax1 = plt.subplots(figsize=(9, 4.5))

        steps = list(range(len(self.scope_loss)))

        # Loss
        raw_loss = np.array(self.scope_loss)
        ax1.plot(steps, raw_loss, alpha=0.2, color=_PAL["neutral"], linewidth=0.5)
        if len(raw_loss) > 10:
            smoothed = _rolling_mean(raw_loss, max(5, len(raw_loss) // 15))
            ax1.plot(range(len(smoothed)), smoothed, color=_PAL["secondary"],
                     linewidth=2.0, label="Scope BCE Loss")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("BCE Loss", color=_PAL["secondary"])
        ax1.tick_params(axis="y", labelcolor=_PAL["secondary"])

        # Mean prediction + std band
        if self.scope_pred_mean:
            ax2 = ax1.twinx()
            pred_mean = np.array(self.scope_pred_mean)
            ax2.plot(range(len(pred_mean)), pred_mean, color=_PAL["primary"],
                     linewidth=1.5, label="Mean Scope Pred")
            if self.scope_pred_std:
                pred_std = np.array(self.scope_pred_std[:len(pred_mean)])
                ax2.fill_between(range(len(pred_mean)),
                                 pred_mean - pred_std, pred_mean + pred_std,
                                 alpha=0.15, color=_PAL["primary"])
            ax2.set_ylabel("Scope Prediction (mean ± std)", color=_PAL["primary"])
            ax2.tick_params(axis="y", labelcolor=_PAL["primary"])
            ax2.set_ylim(-0.05, 1.05)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc="upper right", framealpha=0.9)

        ax1.set_title("Scope Detector Training Progress", fontweight="bold")
        ax1.grid(True, alpha=0.15)
        fig.tight_layout()
        return _savefig(fig, output_dir, "scope_detector_learning")

    # =====================================================================
    #  4. Slot Utilization — are all memory slots being used?
    # =====================================================================

    def _plot_slot_utilization(self, output_dir: str) -> str:
        """
        Sorted bar chart of per-slot usage counts (log scale).
        Shows memory balance vs. hotspots or dead slots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        usage = self._final_usage.copy()
        K = len(usage)

        # (a) Sorted usage (descending)
        ax = axes[0]
        sorted_usage = np.sort(usage)[::-1]
        active_count = int((usage > 0).sum())
        x = np.arange(K)
        ax.bar(x, sorted_usage, width=1.0, color=_PAL["primary"], alpha=0.7, edgecolor="none")
        if sorted_usage.max() > 0:
            ax.set_yscale("symlog", linthresh=1)
        ax.axvline(active_count, color=_PAL["warn"], linestyle="--", linewidth=1.2,
                   label=f"Active boundary ({active_count}/{K})")
        ax.set_xlabel("Slot Rank (sorted by usage)")
        ax.set_ylabel("Access Count (symlog)")
        ax.set_title("Slot Usage Distribution", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.15)

        # (b) Usage histogram (log bins)
        ax = axes[1]
        nonzero = usage[usage > 0]
        if len(nonzero) > 0:
            max_val = nonzero.max()
            bins = np.logspace(0, np.log10(max(max_val, 2)), 30)
            ax.hist(nonzero, bins=bins, color=_PAL["secondary"], alpha=0.7,
                    edgecolor="white", linewidth=0.3)
            ax.set_xscale("log")
        dead = K - len(nonzero)
        ax.set_xlabel("Usage Count")
        ax.set_ylabel("Number of Slots")
        ax.set_title(f"Usage Histogram ({dead} dead slots)", fontweight="bold")
        ax.grid(True, alpha=0.15)

        fig.tight_layout()
        return _savefig(fig, output_dir, "slot_utilization")

    # =====================================================================
    #  5. Memory Embedding Map — what did memory learn?
    # =====================================================================

    def _plot_embedding_map(self, output_dir: str) -> str:
        """
        PCA (or t-SNE) of the K memory slot embeddings.
        Coloured by access recency — bright = recently used, dark = stale.
        Only plots slots that have been written (usage > 0).
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        M = self._final_M
        usage = self._final_usage
        recency = self._final_recency

        # Filter to active slots
        active_mask = usage > 0
        if active_mask.sum() < 3:
            ax.text(0.5, 0.5, f"Only {active_mask.sum()} active slots — too few to visualise",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Memory Slot Embeddings", fontweight="bold")
            fig.tight_layout()
            return _savefig(fig, output_dir, "memory_embedding_map")

        M_active = M[active_mask]
        recency_active = recency[active_mask]
        usage_active = usage[active_mask]

        # Dimensionality reduction
        if HAS_SKLEARN and M_active.shape[0] >= 10:
            try:
                if M_active.shape[0] > 50:
                    perp = min(30, M_active.shape[0] // 2)
                    coords = TSNE(n_components=2, perplexity=perp,
                                  random_state=42, n_iter=500).fit_transform(M_active)
                    method = "t-SNE"
                else:
                    coords = PCA(n_components=2).fit_transform(M_active)
                    method = "PCA"
            except Exception:
                coords = PCA(n_components=2).fit_transform(M_active)
                method = "PCA"
        elif M_active.shape[1] >= 2:
            coords = M_active[:, :2]
            method = "First 2 dims"
        else:
            ax.text(0.5, 0.5, "Cannot project to 2D", ha="center", va="center",
                    transform=ax.transAxes)
            fig.tight_layout()
            return _savefig(fig, output_dir, "memory_embedding_map")

        # Normalise recency to [0, 1] for colour mapping
        if recency_active.max() > recency_active.min():
            norm_rec = (recency_active - recency_active.min()) / (recency_active.max() - recency_active.min())
        else:
            norm_rec = np.ones_like(recency_active) * 0.5

        # Size proportional to usage (clamped)
        sizes = np.clip(usage_active / max(usage_active.max(), 1) * 100, 8, 120)

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=norm_rec, cmap="YlOrRd", s=sizes,
                             alpha=0.75, edgecolors="white", linewidths=0.3)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Access Recency (recent → bright)")

        ax.set_xlabel(f"{method} Dim 1")
        ax.set_ylabel(f"{method} Dim 2")
        n_active = int(active_mask.sum())
        ax.set_title(f"Memory Slot Embeddings — {n_active} Active Slots ({method})",
                     fontweight="bold")
        ax.grid(True, alpha=0.1)

        fig.tight_layout()
        return _savefig(fig, output_dir, "memory_embedding_map")

    # =====================================================================
    #  6. Consolidation Dynamics — Stage 5 convergence
    # =====================================================================

    def _plot_consolidation(self, output_dir: str) -> str:
        """
        Dual-axis: supervised loss + alignment loss (left Y),
        cosine similarity (right Y) over consolidation steps.
        """
        fig, ax1 = plt.subplots(figsize=(9, 4.5))

        steps = np.array(self.consol_steps)

        # Losses
        ax1.plot(steps, self.sup_loss, color=_PAL["secondary"], linewidth=1.5,
                 alpha=0.7, label="Supervised Loss")
        ax1.plot(steps, self.align_loss, color=_PAL["primary"], linewidth=1.5,
                 alpha=0.7, label="Alignment Loss (MSE)")
        ax1.set_xlabel("Consolidation Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.15)

        # Cosine similarity (right axis)
        if self.cosine_sim:
            ax2 = ax1.twinx()
            ax2.plot(steps[:len(self.cosine_sim)], self.cosine_sim,
                     color=_PAL["accent"], linewidth=1.8,
                     label="Cosine Sim (model ↔ memory)")
            ax2.set_ylabel("Cosine Similarity", color=_PAL["accent"])
            ax2.tick_params(axis="y", labelcolor=_PAL["accent"])
            ax2.set_ylim(-0.1, 1.1)
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []

        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

        ax1.set_title("Stage 5 — Memory Consolidation Convergence", fontweight="bold")
        fig.tight_layout()
        return _savefig(fig, output_dir, "consolidation_dynamics")
