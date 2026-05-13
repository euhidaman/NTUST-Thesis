"""
Comprehensive Metrics Tracker for EmberVLM

Tracks all quantitative metrics required for publication-quality evaluation:
- Loss decomposition
- Optimization health (gradients, learning rate, parameter updates)
- Representation quality (entropy, cosine similarity, feature collapse)
- Stability (variance, oscillation)
- Module-specific metrics

All metrics are logged to WandB and saved locally with publication-quality
visualizations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Comprehensive metrics tracking for all training stages.

    Tracks:
    - Core optimization metrics (loss, accuracy, LR)
    - Training health (gradient norms, parameter updates)
    - Stability (variance, oscillation)
    - Representation quality (attention entropy, feature collapse)
    - Module-specific metrics (vision, fusion, language, reasoning)
    """

    def __init__(
        self,
        output_dir: str,
        wandb_logger: Optional[Any] = None,
        window_size: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_logger = wandb_logger
        self.window_size = window_size

        # Metric histories
        self.metrics_history = defaultdict(list)
        self.step_history = []

        # Sliding windows for variance computation
        self.loss_window = deque(maxlen=window_size)
        self.grad_window = deque(maxlen=window_size)
        self.lr_window = deque(maxlen=window_size)

        # Module-specific trackers
        self.module_metrics = {
            'vision_encoder': defaultdict(list),
            'fusion_module': defaultdict(list),
            'language_model': defaultdict(list),
            'reasoning_module': defaultdict(list),
        }

        logger.info(f"Initialized MetricsTracker with output_dir={output_dir}")

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        commit: bool = True,
    ):
        """
        Log metrics for a single training step.

        Args:
            step: Current training step
            metrics: Dictionary of metric names and values
            commit: Whether to commit to WandB
        """
        self.step_history.append(step)

        # Store all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.metrics_history[key].append(float(value))

        # Update sliding windows
        if 'loss' in metrics:
            self.loss_window.append(metrics['loss'])
        if 'grad_norm' in metrics:
            self.grad_window.append(metrics['grad_norm'])
        if 'lr' in metrics:
            self.lr_window.append(metrics['lr'])

        # Compute derived metrics
        derived_metrics = self._compute_derived_metrics()
        metrics.update(derived_metrics)

        # Log to WandB
        if self.wandb_logger and commit:
            self.wandb_logger.log(metrics, step=step)

        # Save metrics periodically
        if step % 100 == 0:
            self._save_metrics()

    def log_loss_decomposition(
        self,
        step: int,
        loss_components: Dict[str, float],
    ):
        """
        Log individual loss components for analysis.

        Args:
            step: Current step
            loss_components: Dict of loss component names and values
        """
        # Log each component
        for name, value in loss_components.items():
            metric_name = f'loss/{name}'
            self.metrics_history[metric_name].append(value)

            if self.wandb_logger:
                self.wandb_logger.log({metric_name: value}, step=step)

    def log_gradient_statistics(
        self,
        step: int,
        model: torch.nn.Module,
        detailed: bool = True,
    ):
        """
        Log comprehensive gradient statistics.

        Args:
            step: Current step
            model: The model to analyze
            detailed: Whether to log per-module statistics
        """
        total_norm = 0.0
        module_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2

                # Track by module
                module_name = name.split('.')[0]
                if module_name not in module_norms:
                    module_norms[module_name] = 0.0
                module_norms[module_name] += param_norm ** 2

        total_norm = np.sqrt(total_norm)

        # Log global norm
        metrics = {'grad_norm_global': total_norm}
        self.metrics_history['grad_norm_global'].append(total_norm)

        # Log per-module norms
        if detailed:
            for module_name, norm_sq in module_norms.items():
                norm = np.sqrt(norm_sq)
                metric_name = f'grad_norm/{module_name}'
                metrics[metric_name] = norm
                self.module_metrics[module_name]['grad_norm'].append(norm)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def log_parameter_update_magnitude(
        self,
        step: int,
        model: torch.nn.Module,
        previous_params: Dict[str, torch.Tensor],
    ):
        """
        Log magnitude of parameter updates (useful for detecting learning saturation).

        Args:
            step: Current step
            model: The model
            previous_params: Previous parameter values
        """
        update_magnitudes = {}

        for name, param in model.named_parameters():
            if name in previous_params and param.requires_grad:
                update = (param.data - previous_params[name]).norm().item()
                param_norm = param.data.norm().item()
                relative_update = update / (param_norm + 1e-8)

                module_name = name.split('.')[0]
                if module_name not in update_magnitudes:
                    update_magnitudes[module_name] = []
                update_magnitudes[module_name].append(relative_update)

        # Log average per module
        metrics = {}
        for module_name, updates in update_magnitudes.items():
            avg_update = np.mean(updates)
            metric_name = f'param_update/{module_name}'
            metrics[metric_name] = avg_update
            self.module_metrics[module_name]['param_update'].append(avg_update)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def log_attention_statistics(
        self,
        step: int,
        attention_weights: torch.Tensor,
        prefix: str = 'attention',
    ):
        """
        Log attention distribution statistics.

        Args:
            step: Current step
            attention_weights: Attention weights [B, num_heads, seq_len, seq_len]
            prefix: Prefix for metric names
        """
        if attention_weights is None or attention_weights.numel() == 0:
            return

        # Compute entropy of attention distribution
        # Average over batch and heads
        attn = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        # Entropy per query position
        entropies = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)

        metrics = {
            f'{prefix}/entropy_mean': entropies.mean().item(),
            f'{prefix}/entropy_std': entropies.std().item(),
            f'{prefix}/entropy_min': entropies.min().item(),
            f'{prefix}/entropy_max': entropies.max().item(),
        }

        # Check for attention collapse (all attention on one position)
        max_attention_per_query = attn.max(dim=-1)[0]
        metrics[f'{prefix}/max_attention_mean'] = max_attention_per_query.mean().item()

        # Store and log
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def log_feature_statistics(
        self,
        step: int,
        features: torch.Tensor,
        name: str = 'features',
    ):
        """
        Log feature distribution statistics (for detecting collapse).

        Args:
            step: Current step
            features: Feature tensor [B, dim] or [B, seq_len, dim]
            name: Name for this feature set
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # Average over sequence

        # Basic statistics
        metrics = {
            f'{name}/mean': features.mean().item(),
            f'{name}/std': features.std().item(),
            f'{name}/min': features.min().item(),
            f'{name}/max': features.max().item(),
        }

        # Compute pairwise cosine similarity distribution
        features_norm = F.normalize(features, p=2, dim=-1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        similarities = similarity_matrix[mask]

        metrics[f'{name}/cosine_similarity_mean'] = similarities.mean().item()
        metrics[f'{name}/cosine_similarity_std'] = similarities.std().item()

        # Check for collapse (high similarity indicates collapse)
        high_similarity_ratio = (similarities > 0.9).float().mean().item()
        metrics[f'{name}/high_similarity_ratio'] = high_similarity_ratio

        # Store and log
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def log_qk_norm_statistics(
        self,
        step: int,
        query: torch.Tensor,
        key: torch.Tensor,
        prefix: str = 'qk_norm',
    ):
        """
        Log Query-Key normalization statistics.

        Args:
            step: Current step
            query: Query tensor
            key: Key tensor
            prefix: Prefix for metric names
        """
        q_norm = query.norm(dim=-1)
        k_norm = key.norm(dim=-1)

        metrics = {
            f'{prefix}/query_mean': q_norm.mean().item(),
            f'{prefix}/query_std': q_norm.std().item(),
            f'{prefix}/key_mean': k_norm.mean().item(),
            f'{prefix}/key_std': k_norm.std().item(),
        }

        # Store and log
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def log_vision_text_similarity(
        self,
        step: int,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """
        Log cosine similarity between vision and text representations.

        Args:
            step: Current step
            vision_features: Vision features [B, dim] or [B, num_visual_tokens, dim]
            text_features: Text features [B, dim] or [B, seq_len, dim]
        """
        # Pool if needed
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)

        # Normalize
        vision_norm = F.normalize(vision_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity
        similarity = (vision_norm * text_norm).sum(dim=-1)

        metrics = {
            'vision_text_similarity/mean': similarity.mean().item(),
            'vision_text_similarity/std': similarity.std().item(),
            'vision_text_similarity/min': similarity.min().item(),
            'vision_text_similarity/max': similarity.max().item(),
        }

        # Store and log
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def _compute_derived_metrics(self) -> Dict[str, float]:
        """Compute derived metrics from windowed history."""
        metrics = {}

        # Loss variance
        if len(self.loss_window) >= 20:
            metrics['loss_variance'] = np.var(list(self.loss_window))
            metrics['loss_std'] = np.std(list(self.loss_window))

        # Gradient stability
        if len(self.grad_window) >= 20:
            metrics['grad_variance'] = np.var(list(self.grad_window))
            metrics['grad_std'] = np.std(list(self.grad_window))

        # Learning rate tracking
        if len(self.lr_window) >= 1:
            metrics['lr_current'] = list(self.lr_window)[-1]

        return metrics

    def _save_metrics(self):
        """Save metrics history to disk."""
        metrics_file = self.output_dir / 'metrics_history.json'

        # Convert to serializable format
        serializable_metrics = {}
        for key, values in self.metrics_history.items():
            serializable_metrics[key] = values

        with open(metrics_file, 'w') as f:
            json.dump({
                'steps': self.step_history,
                'metrics': serializable_metrics,
            }, f, indent=2)

        logger.debug(f"Saved metrics to {metrics_file}")

    def generate_loss_decomposition_plot(
        self,
        save_path: Optional[Path] = None,
        loss_components: Optional[List[str]] = None,
    ):
        """
        Generate publication-quality loss decomposition plot.

        Args:
            save_path: Where to save the plot
            loss_components: List of loss component names to plot
        """
        if save_path is None:
            save_path = self.output_dir / 'loss_decomposition.png'

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Stacked area chart
        if loss_components:
            data = []
            labels = []
            for component in loss_components:
                key = f'loss/{component}'
                if key in self.metrics_history:
                    data.append(self.metrics_history[key])
                    labels.append(component)

            if data:
                ax1.stackplot(self.step_history[:len(data[0])], *data, labels=labels, alpha=0.7)
                ax1.set_ylabel('Loss (Stacked)')
                ax1.set_title('Loss Component Decomposition (Stacked)')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)

        # Plot 2: Individual traces
        colors = sns.color_palette("husl", len(loss_components) if loss_components else 5)
        for i, component in enumerate(loss_components or []):
            key = f'loss/{component}'
            if key in self.metrics_history:
                ax2.plot(self.step_history[:len(self.metrics_history[key])],
                        self.metrics_history[key],
                        label=component, color=colors[i], linewidth=2, alpha=0.8)

        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components (Individual)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved loss decomposition plot to {save_path}")

        # Log to wandb
        if self.wandb_logger:
            import wandb
            self.wandb_logger.log({'plots/loss_decomposition': wandb.Image(str(save_path))})

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked metrics."""
        summary = {}

        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1],
                }

        return summary


def create_metrics_tracker(
    output_dir: str,
    wandb_logger: Optional[Any] = None,
    window_size: int = 100,
) -> MetricsTracker:
    """Factory function to create a MetricsTracker."""
    return MetricsTracker(
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        window_size=window_size,
    )

