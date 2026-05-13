"""
Advanced Visualization Module for EmberVLM

Provides comprehensive research-quality visualizations including:
- 3D surface plots for layer-wise training dynamics
- Frozen vs trained comparisons
- Layer budget rank distribution
- Pareto frontier analysis
- Ablation study tornado charts

All visualizations are designed for W&B logging and publication quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import io
import logging

# Configure matplotlib for non-interactive use
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

# Import seaborn if available
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from PIL import Image


# ========================= COLOR SCHEMES =========================

# Publication-quality color schemes
ROBOT_COLORS = {
    'Drone': '#4E79A7',
    'Underwater Robot': '#76B7B2',
    'Underwater': '#76B7B2',
    'Humanoid': '#F28E2B',
    'Robot with Wheels': '#59A14F',
    'Wheeled': '#59A14F',
    'Robot with Legs': '#E15759',
    'Legged': '#E15759',
}

STAGE_COLORS = {
    1: '#4E79A7',  # Blue - Visual Alignment
    2: '#F28E2B',  # Orange - Instruction Tuning
    3: '#59A14F',  # Green - Robot Selection
    4: '#E15759',  # Red - Reasoning
}

COMPARISON_COLORS = {
    'frozen': '#FFB5B5',
    'trained': '#90EE90',
    'baseline': '#CCCCCC',
}


def _fig_to_pil(fig: plt.Figure, dpi: int = 150) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_image


def _save_and_return(fig: plt.Figure, save_path: Optional[Path], dpi: int = 200) -> Tuple[plt.Figure, Image.Image]:
    """Save figure and return both figure and PIL image."""
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved visualization to {save_path}")
    return fig, _fig_to_pil(fig, dpi)


class AdvancedVisualizer:
    """
    Advanced visualizations for EmberVLM training analysis.

    Provides publication-quality plots for:
    - Layer-wise training dynamics (3D surfaces)
    - Frozen vs trained comparisons
    - Layer importance ranking
    - Performance Pareto frontiers
    - Ablation studies
    """

    def __init__(self, output_dir: str = "./outputs/visualizations/advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History tracking for time-series visualizations
        self.layer_accuracy_history = {}  # {layer_idx: {step: accuracy}}
        self.robot_accuracy_history = {}  # {robot: {step: accuracy}}

        logger.info(f"AdvancedVisualizer initialized with output_dir={self.output_dir}")

    # ======================= 3D SURFACE PLOTS =======================

    def plot_layer_training_dynamics_3d(
        self,
        accuracy_data: np.ndarray,
        layer_names: List[str] = None,
        step_labels: List[int] = None,
        title: str = "Layer-wise Training Dynamics",
        robot_name: str = None,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create 3D surface plot showing accuracy evolution across layers and training steps.

        Inspired by research paper Figure 4 showing layer-wise metric evolution.

        Args:
            accuracy_data: 2D array [n_layers, n_steps] of accuracy values
            layer_names: Names for each layer
            step_labels: Step numbers for x-axis
            title: Plot title
            robot_name: Optional robot name for coloring
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        n_layers, n_steps = accuracy_data.shape

        if layer_names is None:
            layer_names = [f'Layer {i}' for i in range(n_layers)]
        if step_labels is None:
            step_labels = list(range(n_steps))

        # Create meshgrid
        layers = np.arange(n_layers)
        steps = np.arange(n_steps)
        layers_mesh, steps_mesh = np.meshgrid(layers, steps)

        # Get colormap based on robot
        if robot_name and robot_name in ROBOT_COLORS:
            base_color = ROBOT_COLORS[robot_name]
            cmap = self._create_gradient_cmap(base_color)
        else:
            cmap = 'viridis'

        # Plot surface
        surf = ax.plot_surface(
            layers_mesh, steps_mesh, accuracy_data.T,
            cmap=cmap, alpha=0.85,
            edgecolor='gray', linewidth=0.3,
            antialiased=True
        )

        # Labels and styling
        ax.set_xlabel('Layer Index', fontsize=11, labelpad=10)
        ax.set_ylabel('Training Steps', fontsize=11, labelpad=10)
        ax.set_zlabel('Accuracy', fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

        # Set layer ticks
        ax.set_xticks(layers[::max(1, n_layers//5)])
        if n_steps > 10:
            step_ticks = np.linspace(0, n_steps-1, 6).astype(int)
            ax.set_yticks(step_ticks)
            ax.set_yticklabels([str(step_labels[i]) for i in step_ticks])

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label='Accuracy', pad=0.1)

        # Adjust view angle for best visibility
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        save_path = self.output_dir / "layer_dynamics_3d" / f"{robot_name or 'all'}.png" if save else None
        return _save_and_return(fig, save_path)

    def plot_multi_robot_3d_surfaces(
        self,
        robot_accuracy_data: Dict[str, np.ndarray],
        layer_names: List[str] = None,
        step_labels: List[int] = None,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create multi-panel 3D surface plots for each robot type.

        Args:
            robot_accuracy_data: {robot_name: [n_layers, n_steps] array}
            layer_names: Names for each layer
            step_labels: Step numbers
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        robots = list(robot_accuracy_data.keys())
        n_robots = len(robots)

        # Create subplot grid
        cols = min(3, n_robots)
        rows = (n_robots + cols - 1) // cols

        fig = plt.figure(figsize=(6*cols, 5*rows))

        for idx, robot in enumerate(robots):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

            data = robot_accuracy_data[robot]
            n_layers, n_steps = data.shape

            layers = np.arange(n_layers)
            steps = np.arange(n_steps)
            layers_mesh, steps_mesh = np.meshgrid(layers, steps)

            color = ROBOT_COLORS.get(robot, '#4E79A7')
            cmap = self._create_gradient_cmap(color)

            surf = ax.plot_surface(
                layers_mesh, steps_mesh, data.T,
                cmap=cmap, alpha=0.8,
                edgecolor='gray', linewidth=0.2
            )

            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel('Step', fontsize=9)
            ax.set_zlabel('Acc', fontsize=9)
            ax.set_title(robot, fontsize=11, fontweight='bold')
            ax.view_init(elev=20, azim=45)

        fig.suptitle('Layer-wise Training Dynamics by Robot Type', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / "multi_robot_3d" / "multi_robot_3d_surfaces.png" if save else None
        return _save_and_return(fig, save_path)

    def plot_image_text_similarity_evolution_3d(
        self,
        similarity_history: List[np.ndarray],
        step_labels: List[int] = None,
        title: str = "Image-Text Similarity Evolution",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        3D surface showing image-text similarity across visual tokens and training.

        Args:
            similarity_history: List of [n_visual_tokens] similarity arrays per step
            step_labels: Training step labels
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Stack similarities
        similarity_matrix = np.array(similarity_history)  # [n_steps, n_tokens]
        n_steps, n_tokens = similarity_matrix.shape

        if step_labels is None:
            step_labels = list(range(n_steps))

        # Create meshgrid
        tokens = np.arange(n_tokens)
        steps = np.arange(n_steps)
        tokens_mesh, steps_mesh = np.meshgrid(tokens, steps)

        # Plot surface with cool-to-warm colormap
        surf = ax.plot_surface(
            tokens_mesh, steps_mesh, similarity_matrix,
            cmap='RdYlBu_r', alpha=0.85,
            edgecolor='gray', linewidth=0.2
        )

        ax.set_xlabel('Visual Token Position', fontsize=11, labelpad=10)
        ax.set_ylabel('Training Step', fontsize=11, labelpad=10)
        ax.set_zlabel('Cosine Similarity', fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label='Similarity', pad=0.1)
        ax.view_init(elev=25, azim=-45)

        plt.tight_layout()

        save_path = self.output_dir / "similarity_evolution" / "similarity_evolution_3d.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= COMPARISON PLOTS =======================

    def plot_frozen_trained_comparison(
        self,
        frozen_metrics: Dict[str, float],
        trained_metrics: Dict[str, float],
        categories: List[str] = None,
        title: str = "Frozen vs Trained Comparison",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create bar chart comparing frozen vs trained model performance.

        Inspired by research paper Figure 5 style.

        Args:
            frozen_metrics: {category: score} for frozen model
            trained_metrics: {category: score} for trained model
            categories: List of categories to compare
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if categories is None:
            categories = list(frozen_metrics.keys())

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(categories))
        width = 0.35

        frozen_vals = [frozen_metrics.get(c, 0) for c in categories]
        trained_vals = [trained_metrics.get(c, 0) for c in categories]

        # Create bars with patterns
        bars1 = ax.bar(x - width/2, frozen_vals, width,
                       label='Frozen Backbone',
                       color=COMPARISON_COLORS['frozen'],
                       edgecolor='gray', linewidth=1.5,
                       hatch='//')
        bars2 = ax.bar(x + width/2, trained_vals, width,
                       label='Trained (Fine-tuned)',
                       color=COMPARISON_COLORS['trained'],
                       edgecolor='gray', linewidth=1.5)

        # Add reference lines
        frozen_avg = np.mean(frozen_vals)
        trained_avg = np.mean(trained_vals)
        ax.axhline(y=frozen_avg, color='#FF6B6B', linestyle='--',
                   alpha=0.7, label=f'Frozen Avg: {frozen_avg:.3f}')
        ax.axhline(y=trained_avg, color='#4CAF50', linestyle='--',
                   alpha=0.7, label=f'Trained Avg: {trained_avg:.3f}')

        # Add improvement annotations
        for i, (f, t) in enumerate(zip(frozen_vals, trained_vals)):
            improvement = ((t - f) / (f + 1e-8)) * 100
            color = '#4CAF50' if improvement > 0 else '#FF6B6B'
            sign = '+' if improvement > 0 else ''
            ax.annotate(f'{sign}{improvement:.1f}%',
                       xy=(i + width/2, t),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', fontsize=8, color=color,
                       fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right')
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(max(frozen_vals), max(trained_vals)) * 1.15)

        plt.tight_layout()

        save_path = self.output_dir / "frozen_trained" / "frozen_trained_comparison.png" if save else None
        return _save_and_return(fig, save_path)

    def plot_layers_trained_accuracy_curves(
        self,
        accuracy_by_layers: Dict[int, Dict[str, float]],
        robot_names: List[str] = None,
        title: str = "Accuracy vs Layers Trained",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot accuracy curves showing performance vs number of layers trained.

        Similar to Figure 6 in research papers.

        Args:
            accuracy_by_layers: {n_layers_trained: {robot: accuracy}}
            robot_names: List of robot types to plot
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        layers_trained = sorted(accuracy_by_layers.keys())

        if robot_names is None:
            robot_names = list(accuracy_by_layers[layers_trained[0]].keys())

        for robot in robot_names:
            accuracies = [accuracy_by_layers[l].get(robot, 0) for l in layers_trained]
            color = ROBOT_COLORS.get(robot, '#4E79A7')

            ax.plot(layers_trained, accuracies, 'o-',
                    linewidth=2.5, markersize=8,
                    label=robot, color=color)

            # Add shaded region for confidence
            ax.fill_between(layers_trained,
                           [a - 0.02 for a in accuracies],
                           [a + 0.02 for a in accuracies],
                           alpha=0.15, color=color)

        ax.set_xlabel('Number of Layers Trained', fontsize=11)
        ax.set_ylabel('Robot Selection Accuracy', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        save_path = self.output_dir / "layers_trained" / "layers_trained_curves.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= LAYER IMPORTANCE PLOTS =======================

    def plot_layer_budget_rank_distribution(
        self,
        importance_scores: Dict[str, np.ndarray],
        layer_names: List[str] = None,
        title: str = "Layer Budget Rank Distribution",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create stacked bar chart showing layer importance rankings.

        Inspired by Figure 7 showing symmetric/FIM/LSN rank distributions.

        Args:
            importance_scores: {method_name: [layer_importance_scores]}
            layer_names: Names for each layer
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_methods = len(importance_scores)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3*n_methods))
        if n_methods == 1:
            axes = [axes]

        cmap = plt.cm.viridis

        for ax, (method, scores) in zip(axes, importance_scores.items()):
            n_layers = len(scores)
            if layer_names is None:
                layer_labels = [f'L{i}' for i in range(n_layers)]
            else:
                layer_labels = layer_names[:n_layers]

            # Normalize scores for coloring
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            colors = [cmap(s) for s in norm_scores]

            # Compute ranks
            ranks = n_layers - np.argsort(np.argsort(scores))

            bars = ax.bar(range(n_layers), scores, color=colors, edgecolor='white', linewidth=1.5)

            # Add rank labels on bars
            for i, (bar, rank) in enumerate(zip(bars, ranks)):
                height = bar.get_height()
                ax.annotate(f'R{rank}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, fontweight='bold')

            ax.set_xticks(range(n_layers))
            ax.set_xticklabels(layer_labels, rotation=45, ha='right')
            ax.set_ylabel('Importance', fontsize=10)
            ax.set_title(f'{method} Layer Importance', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / "layer_budget" / "layer_budget_ranks.png" if save else None
        return _save_and_return(fig, save_path)

    def plot_attention_ffn_importance(
        self,
        attention_importance: np.ndarray,
        ffn_importance: np.ndarray,
        layer_names: List[str] = None,
        title: str = "Attention vs FFN Layer Importance",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare attention and FFN layer importance scores.

        Args:
            attention_importance: Importance scores for attention layers
            ffn_importance: Importance scores for FFN layers
            layer_names: Names for each layer
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        n_layers = len(attention_importance)
        if layer_names is None:
            layer_names = [f'Layer {i}' for i in range(n_layers)]

        x = np.arange(n_layers)
        width = 0.35

        ax.bar(x - width/2, attention_importance, width,
               label='Attention', color='#4E79A7', edgecolor='white')
        ax.bar(x + width/2, ffn_importance, width,
               label='FFN', color='#F28E2B', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_ylabel('Importance Score', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.output_dir / "attention_ffn" / "attention_ffn_importance.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= PARETO FRONTIER =======================

    def plot_pareto_frontier(
        self,
        models: Dict[str, Dict[str, float]],
        x_metric: str = "params",
        y_metric: str = "accuracy",
        size_metric: str = "memory",
        color_metric: str = "deployable",
        highlight_model: str = "EmberVLM",
        title: str = "Model Efficiency Pareto Frontier",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot Pareto frontier for model efficiency analysis.

        Args:
            models: {model_name: {metric: value}}
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            size_metric: Metric for point size
            color_metric: Metric for color (categorical)
            highlight_model: Model to highlight
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        model_names = list(models.keys())
        x_vals = [models[m].get(x_metric, 0) for m in model_names]
        y_vals = [models[m].get(y_metric, 0) for m in model_names]
        sizes = [models[m].get(size_metric, 50) for m in model_names]
        colors = [models[m].get(color_metric, 1) for m in model_names]

        # Normalize sizes
        max_size = max(sizes) if sizes else 1
        sizes_norm = [100 + 400 * (s / max_size) for s in sizes]

        # Color mapping
        cmap = plt.cm.RdYlGn
        color_vals = [cmap(c) if isinstance(c, (int, float)) else '#CCCCCC' for c in colors]

        # Plot all models
        for i, name in enumerate(model_names):
            is_highlight = name == highlight_model
            scatter = ax.scatter(
                x_vals[i], y_vals[i],
                s=sizes_norm[i],
                c=[color_vals[i]],
                alpha=0.9 if is_highlight else 0.6,
                edgecolors='black' if is_highlight else 'gray',
                linewidths=3 if is_highlight else 1,
                zorder=10 if is_highlight else 5
            )

            # Label
            ax.annotate(
                name,
                xy=(x_vals[i], y_vals[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9 if is_highlight else 7,
                fontweight='bold' if is_highlight else 'normal',
                color='black' if is_highlight else 'gray'
            )

        # Find and plot Pareto frontier
        pareto_idx = self._find_pareto_frontier(x_vals, y_vals, minimize_x=True, maximize_y=True)
        pareto_x = [x_vals[i] for i in pareto_idx]
        pareto_y = [y_vals[i] for i in pareto_idx]

        # Sort by x for line
        sorted_idx = np.argsort(pareto_x)
        ax.plot([pareto_x[i] for i in sorted_idx],
                [pareto_y[i] for i in sorted_idx],
                'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')

        ax.set_xlabel(f'{x_metric.replace("_", " ").title()} (log scale)', fontsize=11)
        ax.set_ylabel(f'{y_metric.replace("_", " ").title()}', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / "pareto_frontier" / "pareto_frontier.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= ABLATION STUDY =======================

    def plot_ablation_tornado(
        self,
        ablation_results: Dict[str, float],
        baseline: float,
        title: str = "Ablation Study: Component Impact",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create tornado chart for ablation study results.

        Args:
            ablation_results: {component_removed: accuracy}
            baseline: Baseline accuracy with all components
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        components = list(ablation_results.keys())
        impacts = [baseline - ablation_results[c] for c in components]

        # Sort by absolute impact
        sorted_idx = np.argsort(np.abs(impacts))[::-1]
        components = [components[i] for i in sorted_idx]
        impacts = [impacts[i] for i in sorted_idx]

        # Create horizontal bars
        y_pos = np.arange(len(components))
        colors = ['#FF6B6B' if i > 0 else '#4CAF50' for i in impacts]

        bars = ax.barh(y_pos, impacts, color=colors, edgecolor='white', linewidth=1.5)

        # Add baseline reference
        ax.axvline(x=0, color='black', linewidth=2, linestyle='-')

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.set_xlabel('Impact on Accuracy (Baseline - Ablated)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Add annotations
        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            sign = '+' if impact < 0 else '-'
            ax.annotate(f'{sign}{abs(impact):.3f}',
                       xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(5 if width >= 0 else -5, 0),
                       textcoords='offset points',
                       ha='left' if width >= 0 else 'right',
                       va='center', fontsize=9, fontweight='bold')

        # Add baseline annotation
        ax.annotate(f'Baseline: {baseline:.3f}',
                   xy=(0, len(components) - 0.5),
                   fontsize=10, fontweight='bold',
                   ha='center', va='bottom')

        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        save_path = self.output_dir / "ablation_tornado" / "ablation_tornado.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= CONFUSION EVOLUTION =======================

    def plot_confusion_evolution(
        self,
        confusion_matrices: List[np.ndarray],
        step_labels: List[int],
        class_names: List[str],
        title: str = "Confusion Matrix Evolution",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create multi-panel showing confusion matrix evolution over training.

        Args:
            confusion_matrices: List of confusion matrices at different steps
            step_labels: Training step for each matrix
            class_names: Names of classes
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_matrices = len(confusion_matrices)
        cols = min(4, n_matrices)
        rows = (n_matrices + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.array(axes).flatten() if n_matrices > 1 else [axes]

        for idx, (cm, step) in enumerate(zip(confusion_matrices, step_labels)):
            ax = axes[idx]

            # Normalize
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

            if HAS_SEABORN:
                sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names,
                           ax=ax, vmin=0, vmax=1, cbar=False)
            else:
                im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(class_names, fontsize=7)

            accuracy = np.trace(cm) / (np.sum(cm) + 1e-8)
            ax.set_title(f'Step {step}\nAcc: {accuracy:.1%}', fontsize=10)

        # Hide unused axes
        for idx in range(n_matrices, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / "confusion_evolution" / "confusion_evolution.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= CROSS-STAGE ANALYSIS =======================

    def plot_benchmark_heatmap(
        self,
        results: Dict[int, Dict[str, float]],
        title: str = "Benchmark Performance Across Stages",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create heatmap of benchmark scores across training stages.

        Args:
            results: {stage: {benchmark: score}}
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        stages = sorted(results.keys())
        benchmarks = list(results[stages[0]].keys()) if stages else []

        # Create matrix
        matrix = np.array([[results[s].get(b, 0) for b in benchmarks] for s in stages])

        fig, ax = plt.subplots(figsize=(max(10, len(benchmarks)), max(6, len(stages))))

        if HAS_SEABORN:
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=benchmarks,
                       yticklabels=[f'Stage {s}' for s in stages],
                       ax=ax, vmin=0, vmax=1,
                       cbar_kws={'label': 'Score'})
        else:
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(benchmarks)))
            ax.set_yticks(range(len(stages)))
            ax.set_xticklabels(benchmarks, rotation=45, ha='right')
            ax.set_yticklabels([f'Stage {s}' for s in stages])

            for i in range(len(stages)):
                for j in range(len(benchmarks)):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)

            fig.colorbar(im, ax=ax, label='Score')

        ax.set_title(title, fontsize=13, fontweight='bold')

        plt.tight_layout()

        save_path = self.output_dir / "benchmark_heatmap" / "benchmark_heatmap.png" if save else None
        return _save_and_return(fig, save_path)

    def plot_carbon_treemap(
        self,
        emissions_data: Dict[str, float],
        title: str = "Carbon Footprint by Component",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create treemap visualization of carbon emissions.

        Args:
            emissions_data: {component: kg_co2}
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        try:
            import squarify
            has_squarify = True
        except ImportError:
            has_squarify = False

        fig, ax = plt.subplots(figsize=(12, 8))

        if not has_squarify:
            # Fallback to bar chart
            components = list(emissions_data.keys())
            values = list(emissions_data.values())
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(components)))

            ax.bar(components, values, color=colors, edgecolor='white')
            ax.set_ylabel('CO₂ Emissions (kg)')
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
        else:
            components = list(emissions_data.keys())
            values = list(emissions_data.values())

            # Normalize for colors
            norm_vals = np.array(values)
            norm_vals = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min() + 1e-8)
            colors = plt.cm.RdYlGn_r(norm_vals)

            # Labels with values
            labels = [f'{c}\n{v:.4f} kg' for c, v in zip(components, values)]

            squarify.plot(sizes=values, label=labels, color=colors, alpha=0.8, ax=ax,
                         edgecolor='white', linewidth=2)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')

        total = sum(emissions_data.values())
        ax.annotate(f'Total: {total:.4f} kg CO₂',
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        save_path = self.output_dir / "carbon_treemap" / "carbon_treemap.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= TASK-ROBOT PERFORMANCE =======================

    def plot_task_robot_performance_heatmap(
        self,
        performance: Dict[str, Dict[str, float]],
        title: str = "Task Type vs Robot Performance",
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create heatmap of robot selection accuracy by task type.

        Args:
            performance: {task_type: {robot: accuracy}}
            title: Plot title
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        tasks = list(performance.keys())
        robots = list(performance[tasks[0]].keys()) if tasks else []

        matrix = np.array([[performance[t].get(r, 0) for r in robots] for t in tasks])

        fig, ax = plt.subplots(figsize=(max(10, len(robots)*1.5), max(6, len(tasks)*0.8)))

        if HAS_SEABORN:
            sns.heatmap(matrix, annot=True, fmt='.2f',
                       cmap='YlOrRd',
                       xticklabels=robots, yticklabels=tasks,
                       ax=ax, vmin=0, vmax=1,
                       cbar_kws={'label': 'Selection Accuracy'})
        else:
            im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(robots)))
            ax.set_yticks(range(len(tasks)))
            ax.set_xticklabels(robots, rotation=45, ha='right')
            ax.set_yticklabels(tasks)
            fig.colorbar(im, ax=ax, label='Selection Accuracy')

        ax.set_xlabel('Robot Type', fontsize=11)
        ax.set_ylabel('Task Type', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        plt.tight_layout()

        save_path = self.output_dir / "task_robot_heatmap" / "task_robot_heatmap.png" if save else None
        return _save_and_return(fig, save_path)

    # ======================= HELPER METHODS =======================

    def _create_gradient_cmap(self, base_color: str):
        """Create a custom colormap from white to the base color."""
        import matplotlib.colors as mcolors

        # Parse hex color
        rgb = mcolors.to_rgb(base_color)

        # Create colormap from white to base color
        colors = ['white', base_color]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=256)
        return cmap

    def _find_pareto_frontier(self, x_vals, y_vals, minimize_x=True, maximize_y=True):
        """Find indices of points on Pareto frontier."""
        n = len(x_vals)
        pareto_idx = []

        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i == j:
                    continue

                x_better = (x_vals[j] < x_vals[i] if minimize_x else x_vals[j] > x_vals[i])
                y_better = (y_vals[j] > y_vals[i] if maximize_y else y_vals[j] < y_vals[i])
                x_equal = (x_vals[j] == x_vals[i])
                y_equal = (y_vals[j] == y_vals[i])

                if ((x_better or x_equal) and (y_better or y_equal) and
                    (x_better or y_better)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_idx.append(i)

        return pareto_idx

    def update_history(self, step: int, layer_accuracies: Dict[int, float] = None,
                       robot_accuracies: Dict[str, float] = None):
        """Update history for time-series visualizations."""
        if layer_accuracies:
            for layer_idx, acc in layer_accuracies.items():
                if layer_idx not in self.layer_accuracy_history:
                    self.layer_accuracy_history[layer_idx] = {}
                self.layer_accuracy_history[layer_idx][step] = acc

        if robot_accuracies:
            for robot, acc in robot_accuracies.items():
                if robot not in self.robot_accuracy_history:
                    self.robot_accuracy_history[robot] = {}
                self.robot_accuracy_history[robot][step] = acc

    def get_layer_accuracy_matrix(self) -> np.ndarray:
        """Get layer accuracy history as matrix [n_layers, n_steps]."""
        if not self.layer_accuracy_history:
            return np.array([])

        layers = sorted(self.layer_accuracy_history.keys())
        steps = sorted(set(s for l in layers for s in self.layer_accuracy_history[l].keys()))

        matrix = np.zeros((len(layers), len(steps)))
        for i, layer in enumerate(layers):
            for j, step in enumerate(steps):
                matrix[i, j] = self.layer_accuracy_history[layer].get(step, 0)

        return matrix

    def close(self):
        """Clean up matplotlib resources."""
        plt.close('all')

