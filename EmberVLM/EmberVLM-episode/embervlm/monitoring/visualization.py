"""
Publication-Quality Visualization Module for EmberVLM

Generates conference-ready plots and attention visualizations logged to W&B.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Configure matplotlib for non-interactive use BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg', force=True)  # Force non-interactive backend

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Try to import and configure seaborn
try:
    import seaborn as sns
    sns.set_palette("husl")
    logger.info("Seaborn configured successfully")
except Exception as e:
    logger.warning(f"Failed to configure seaborn: {e}")
    sns = None

# Try to set matplotlib style
try:
    plt.style.use('seaborn-v0_8-paper')
except Exception as e:
    logger.warning(f"Failed to set matplotlib style: {e}, using default")

from typing import Dict, Any, Optional, List, Tuple
import io
from PIL import Image
from pathlib import Path

# Import shared paper style
try:
    from embervlm.monitoring.style import set_paper_style, COLORS, tight_layout_with_padding, save_figure, DPI_SAVE
except ImportError:
    try:
        from .style import set_paper_style, COLORS, tight_layout_with_padding, save_figure, DPI_SAVE
    except ImportError:
        def set_paper_style():
            pass  # Fallback: no-op if style module unavailable
        COLORS = {
            "success": "#4CAF50",
            "accent": "#FF6B6B",
        }
        def tight_layout_with_padding(fig, **kwargs):
            fig.tight_layout()
        def save_figure(fig, path_base, **kwargs):
            fig.savefig(f"{path_base}.png", bbox_inches='tight', dpi=200)
        DPI_SAVE = 300


class TrainingVisualizer:
    """Generates publication-quality visualizations for W&B logging."""

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        logger.info(f"Initializing TrainingVisualizer with output_dir={output_dir}")
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create output directory: {e}")

        # Apply shared paper style from style.py
        set_paper_style()

    def visualize_attention_on_image(
        self,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        text_tokens: List[str],
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Overlay attention map on image for visualization.

        Args:
            image: Image tensor [3, H, W]
            attention_map: Attention weights [num_heads, num_text_tokens, num_visual_tokens]
            text_tokens: List of text tokens
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        # Average across heads and text tokens to get spatial attention
        # attention_map: [num_heads, num_text_tokens, num_visual_tokens]
        spatial_attention = attention_map.mean(dim=(0, 1))  # [num_visual_tokens]

        # Reshape to 2D (assuming square grid of visual tokens)
        grid_size = int(np.sqrt(spatial_attention.shape[0]))
        attention_2d = spatial_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Prepare image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention heatmap
        im = axes[1].imshow(attention_2d, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(img_np)
        attention_overlay = F.interpolate(
            torch.from_numpy(attention_2d).unsqueeze(0).unsqueeze(0),
            size=(img_np.shape[0], img_np.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        axes[2].imshow(attention_overlay, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image for W&B
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return fig, pil_image

    def plot_loss_decomposition(
        self,
        metrics_history: Dict[str, List[float]],
        stage_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot stacked loss components over time.

        Args:
            metrics_history: Dictionary of metric names to values
            stage_name: Name of training stage
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Filter loss components (all keys containing 'loss')
        all_loss_keys = [k for k in metrics_history.keys() if 'loss' in k.lower()]

        # Find primary loss key
        primary_loss_key = None
        for candidate in ['loss', 'robot_loss', 'sft_loss', 'total_loss']:
            if candidate in metrics_history and len(metrics_history[candidate]) > 0:
                primary_loss_key = candidate
                break
        if primary_loss_key is None and all_loss_keys:
            primary_loss_key = all_loss_keys[0]

        # Component loss keys (excluding primary)
        loss_keys = [k for k in all_loss_keys if k != primary_loss_key]

        # Plot 1: Stacked area chart of loss components
        if loss_keys:
            # Use minimum length across all arrays
            min_len = min(len(metrics_history[k]) for k in loss_keys)
            steps = np.arange(min_len)
            loss_data = np.array([metrics_history[k][:min_len] for k in loss_keys])

            axes[0].stackplot(steps, loss_data, labels=loss_keys, alpha=0.7)
            axes[0].set_ylabel('Loss Components')
            axes[0].set_title(f'{stage_name}: Loss Decomposition')
            axes[0].legend(loc='upper right', ncol=2)
            axes[0].grid(True, alpha=0.3)
        elif primary_loss_key:
            # If no components, just show the primary loss
            total_loss = metrics_history[primary_loss_key]
            steps = np.arange(len(total_loss))
            axes[0].fill_between(steps, total_loss, alpha=0.7, label=primary_loss_key)
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'{stage_name}: Loss Progression')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Primary loss with trend
        if primary_loss_key and primary_loss_key in metrics_history:
            total_loss = metrics_history[primary_loss_key]
            steps = np.arange(len(total_loss))
            axes[1].plot(steps, total_loss, linewidth=2, label=primary_loss_key.replace('_', ' ').title(), color='#2E86AB')

            # Add moving average trend
            window = min(50, len(total_loss) // 10)
            if window > 1:
                moving_avg = np.convolve(total_loss, np.ones(window)/window, mode='valid')
                axes[1].plot(steps[window-1:], moving_avg, '--', linewidth=2,
                           label=f'Trend (MA-{window})', color='#A23B72', alpha=0.8)

            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Loss Value')
            axes[1].set_title(f'{stage_name}: Training Loss Progression')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return fig, pil_image

    def plot_gradient_distribution(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot distribution of gradients per module.

        Args:
            gradients: Dictionary of module names to gradient tensors
            step: Current training step
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Collect statistics
        modules = []
        means = []
        stds = []
        norms = []

        for name, grad in gradients.items():
            if grad is not None:
                grad_flat = grad.flatten()
                modules.append(name)
                means.append(grad_flat.mean().item())
                stds.append(grad_flat.std().item())
                norms.append(grad.norm().item())

        x = np.arange(len(modules))

        # Plot 1: Gradient statistics
        ax1 = axes[0]
        width = 0.35
        ax1.bar(x - width/2, means, width, label='Mean', alpha=0.8)
        ax1.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8)
        ax1.set_ylabel('Gradient Value')
        ax1.set_title(f'Gradient Statistics (Step {step})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modules, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Gradient norms (log scale)
        ax2 = axes[1]
        ax2.bar(x, norms, alpha=0.8, color='#F18F01')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norms by Module')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modules, rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return fig, pil_image

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        stage_name: str,
        save_path: Optional[str] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        *,
        highlight_diagonal: bool = True,
        annotate_errors: bool = True,
        top_k_errors: int = 3,
    ) -> Tuple[plt.Figure, Image.Image]:
        """Plot confusion matrix for classification tasks.

        Args:
            predictions: Predicted class indices (or None if confusion_matrix provided)
            labels: True class indices (or None if confusion_matrix provided)
            class_names: List of class names
            stage_name: Name of training stage
            save_path: Optional path to save figure
            confusion_matrix: Pre-computed confusion matrix (optional)
            highlight_diagonal: If ``True``, draw thicker borders on the
                diagonal cells and use a distinct colour to emphasise
                correct predictions.
            annotate_errors: If ``True``, annotate the *top_k_errors* most
                frequent misclassifications directly on the normalised
                matrix (e.g. "Drone->Wheels: 12%").
            top_k_errors: Number of top misclassifications to annotate.

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        from sklearn.metrics import confusion_matrix as sklearn_cm

        # Compute confusion matrix if not provided
        if confusion_matrix is not None:
            cm = confusion_matrix
        else:
            cm = sklearn_cm(labels, predictions)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Raw counts
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[0], cbar_kws={'label': 'Count'})
        else:
            axes[0].imshow(cm, cmap='Blues')
        axes[0].set_title(f'{stage_name}: Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Plot 2: Normalized
        if sns is not None:
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        else:
            axes[1].imshow(cm_normalized, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title(f'{stage_name}: Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        # ── highlight diagonal ────────────────────────────────────
        if highlight_diagonal:
            n_cls = len(class_names)
            for ax_i in axes:
                for k in range(n_cls):
                    ax_i.add_patch(plt.Rectangle(
                        (k, k), 1, 1, fill=False,
                        edgecolor=COLORS["success"], linewidth=2.5,
                    ))

        # ── annotate top-K errors ─────────────────────────────────
        if annotate_errors:
            n_cls = len(class_names)
            # Collect off-diagonal entries
            errors = []
            for i in range(n_cls):
                for j in range(n_cls):
                    if i != j and cm_normalized[i, j] > 0:
                        errors.append((i, j, cm_normalized[i, j]))
            errors.sort(key=lambda t: t[2], reverse=True)
            for i_true, j_pred, val in errors[:top_k_errors]:
                label = f"{class_names[i_true]}\u2192{class_names[j_pred]}: {val:.0%}"
                axes[1].text(
                    j_pred + 0.5, i_true + 0.5, f"\n\n{label}",
                    ha="center", va="center", fontsize=6,
                    color=COLORS["accent"], fontweight="bold",
                )

        tight_layout_with_padding(fig)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
            # Also save legacy single PNG for backward compat
            fig.savefig(save_path, bbox_inches='tight', dpi=DPI_SAVE)

        # Convert to PIL Image for W&B
        pil_image = fig_to_pil(fig)
        plt.close(fig)

        return fig, pil_image

    def plot_convergence_analysis(
        self,
        metrics_history: Dict[str, List[float]],
        stage_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot convergence indicators and stability metrics.

        Args:
            metrics_history: Dictionary of metric names to values
            stage_name: Name of training stage
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Find primary loss key
        primary_loss_key = None
        for candidate in ['loss', 'robot_loss', 'sft_loss', 'total_loss']:
            if candidate in metrics_history and len(metrics_history[candidate]) > 0:
                primary_loss_key = candidate
                break
        if primary_loss_key is None:
            loss_keys = [k for k in metrics_history.keys() if 'loss' in k.lower()]
            if loss_keys:
                primary_loss_key = loss_keys[0]

        # Plot 1: Loss and validation loss
        ax1 = fig.add_subplot(gs[0, :])
        steps = None
        if primary_loss_key and primary_loss_key in metrics_history:
            steps = np.arange(len(metrics_history[primary_loss_key]))
            ax1.plot(steps, metrics_history[primary_loss_key],
                    label=f'Training {primary_loss_key.replace("_", " ").title()}', linewidth=2)
        if 'val_loss' in metrics_history and steps is not None:
            val_steps = np.linspace(0, len(steps)-1, len(metrics_history['val_loss']))
            ax1.plot(val_steps, metrics_history['val_loss'],
                    label='Validation Loss', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{stage_name}: Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate schedule
        ax2 = fig.add_subplot(gs[1, 0])
        if 'lr' in metrics_history:
            ax2.plot(metrics_history['lr'], linewidth=2, color='#E63946')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Gradient norm or other metrics
        ax3 = fig.add_subplot(gs[1, 1])
        if 'grad_norm' in metrics_history:
            ax3.plot(metrics_history['grad_norm'], linewidth=2, color='#2A9D8F')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Norm Evolution')
            ax3.grid(True, alpha=0.3)
        elif 'robot_accuracy' in metrics_history:
            ax3.plot(metrics_history['robot_accuracy'], linewidth=2, color='#2A9D8F')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Robot Accuracy')
            ax3.set_title('Robot Selection Accuracy')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Accuracy metrics
        ax4 = fig.add_subplot(gs[2, 0])
        acc_keys = [k for k in metrics_history.keys() if 'acc' in k.lower()]
        for key in acc_keys[:3]:  # Limit to 3 for clarity
            ax4.plot(metrics_history[key], label=key.replace('_', ' ').title(), linewidth=2, alpha=0.8)
        if acc_keys:
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy Metrics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Loss variance (stability indicator)
        ax5 = fig.add_subplot(gs[2, 1])
        if primary_loss_key and primary_loss_key in metrics_history and len(metrics_history[primary_loss_key]) > 50:
            window = 50
            loss = np.array(metrics_history[primary_loss_key])
            rolling_var = np.array([
                np.var(loss[max(0, i-window):i+1])
                for i in range(len(loss))
            ])
            ax5.plot(rolling_var, linewidth=2, color='#F4A261')
            ax5.set_xlabel('Step')
            ax5.set_ylabel(f'Loss Variance (window={window})')
            ax5.set_title('Training Stability')
            ax5.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return fig, pil_image

    def close(self):
        """Clean up matplotlib resources."""
        plt.close('all')

