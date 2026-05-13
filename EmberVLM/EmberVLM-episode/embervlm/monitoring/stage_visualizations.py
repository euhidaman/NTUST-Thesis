"""
Stage-Specific Visualization Module for EmberVLM

Provides comprehensive visualizations for each training stage:
- Stage 1: Visual-Language Alignment
- Stage 2: Instruction Tuning
- Stage 2.5: Hallucination Assessment (VA Refiner)
- Stage 3: Robot Selection
- Stage 4: Chain-of-Thought Reasoning
- Collated: Cross-stage training summary

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

logger = logging.getLogger(__name__)

# Import seaborn if available
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not available, using matplotlib only")

# Import sklearn if available
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix as sklearn_cm
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available, some visualizations disabled")

from PIL import Image


def _fig_to_pil(fig: plt.Figure, dpi: int = 150) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_image


def _viz_save_path(base_dir: Path, viz_type: str, filename: str) -> Path:
    """Route a visualization file into a type-specific sub-folder.

    Creates ``base_dir / viz_type /`` if it doesn't exist and returns the
    full path ``base_dir / viz_type / filename``.
    """
    subdir = base_dir / viz_type
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / filename


class Stage1Visualizer:
    """Visualizations for Stage 1: Visual-Language Alignment."""

    def __init__(self, output_dir: str = "./outputs/stage1/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_similarity_matrix(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot image-text similarity matrix (batch-level CLIP-style).

        Args:
            image_embeds: Image embeddings [B, D]
            text_embeds: Text embeddings [B, D]
            step: Training step
            save: Whether to save to disk

        Returns:
            Tuple of (figure, PIL image)
        """
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(image_embeds, text_embeds.T).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Raw similarity
        im1 = axes[0].imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('Image-Text Similarity Matrix')
        axes[0].set_xlabel('Text Index')
        axes[0].set_ylabel('Image Index')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Highlight diagonal (correct matches)
        batch_size = similarity.shape[0]
        for i in range(batch_size):
            axes[0].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                            fill=False, edgecolor='blue', linewidth=2))

        # Softmax scores (retrieval probabilities)
        i2t_probs = F.softmax(torch.from_numpy(similarity), dim=1).numpy()
        im2 = axes[1].imshow(i2t_probs, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Image→Text Retrieval Probabilities')
        axes[1].set_xlabel('Text Index')
        axes[1].set_ylabel('Image Index')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Add diagonal accuracy annotation
        diagonal_acc = np.mean(np.argmax(similarity, axis=1) == np.arange(batch_size))
        fig.suptitle(f'Step {step} | Batch Retrieval Accuracy: {diagonal_acc:.1%}', fontsize=12)

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "similarity_matrix", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_embedding_tsne(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        n_samples: int = 100,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot t-SNE visualization of image and text embeddings.

        Args:
            image_embeds: Image embeddings [N, D]
            text_embeds: Text embeddings [N, D]
            step: Training step
            n_samples: Max samples to visualize
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for t-SNE")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 't-SNE requires sklearn', ha='center', va='center')
            return fig, _fig_to_pil(fig)

        # Subsample if needed
        n = min(n_samples, image_embeds.size(0))
        img_emb = image_embeds[:n].cpu().numpy()
        txt_emb = text_embeds[:n].cpu().numpy()

        # Combine embeddings
        combined = np.vstack([img_emb, txt_emb])
        labels = ['Image'] * n + ['Text'] * n

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, n-1), random_state=42)
        coords = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot with different colors
        img_coords = coords[:n]
        txt_coords = coords[n:]

        ax.scatter(img_coords[:, 0], img_coords[:, 1], c='#2E86AB',
                   label='Image Embeddings', alpha=0.7, s=50)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1], c='#A23B72',
                   label='Text Embeddings', alpha=0.7, s=50, marker='^')

        # Draw lines between matching pairs
        for i in range(n):
            ax.plot([img_coords[i, 0], txt_coords[i, 0]],
                   [img_coords[i, 1], txt_coords[i, 1]],
                   'gray', alpha=0.2, linewidth=0.5)

        ax.set_title(f't-SNE Visualization of Embeddings (Step {step})')
        ax.legend()
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "tsne", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_retrieval_examples(
        self,
        images: List[Image.Image],
        captions: List[str],
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        top_k: int = 3,
        n_queries: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Show top-K retrieval examples (image→text and text→image).

        Args:
            images: List of PIL images
            captions: List of caption strings
            image_embeds: Image embeddings [N, D]
            text_embeds: Text embeddings [N, D]
            step: Training step
            top_k: Number of top matches to show
            n_queries: Number of query examples
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Normalize and compute similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        similarity = torch.matmul(image_embeds, text_embeds.T).cpu()

        n_queries = min(n_queries, len(images))

        fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(4*(top_k+1), 4*n_queries))

        for i in range(n_queries):
            # Show query image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Query Image {i}', fontsize=10)
            axes[i, 0].axis('off')

            # Get top-K text matches
            scores, indices = similarity[i].topk(top_k)

            for j, (score, idx) in enumerate(zip(scores, indices)):
                is_correct = (idx == i)
                color = 'green' if is_correct else 'red'

                axes[i, j+1].text(0.5, 0.5,
                                  f"Rank {j+1}\nScore: {score:.3f}\n\n{captions[idx][:100]}...",
                                  ha='center', va='center', wrap=True, fontsize=8,
                                  color=color)
                axes[i, j+1].set_xlim(0, 1)
                axes[i, j+1].set_ylim(0, 1)
                axes[i, j+1].axis('off')
                axes[i, j+1].set_title('✓ Correct' if is_correct else '✗ Wrong',
                                       color=color, fontsize=9)

        fig.suptitle(f'Image→Text Retrieval Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "retrieval_examples", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_cross_attention(
        self,
        attention_weights: torch.Tensor,
        image: Image.Image,
        text_tokens: List[str],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Visualize cross-attention between image regions and text tokens.

        Args:
            attention_weights: Attention [num_heads, text_len, visual_tokens]
            image: Original PIL image
            text_tokens: List of text tokens
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Average across heads
        attn = attention_weights.mean(dim=0).cpu().numpy()  # [text_len, visual_tokens]

        # Create figure
        n_tokens = min(10, len(text_tokens))  # Limit tokens shown
        fig, axes = plt.subplots(2, n_tokens // 2 + 1, figsize=(20, 8))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Get grid size for visual tokens
        n_visual = attn.shape[1]
        grid_size = int(np.sqrt(n_visual))

        # Show attention for each token
        for i, token in enumerate(text_tokens[:n_tokens]):
            ax = axes[i + 1]

            token_attn = attn[i].reshape(grid_size, grid_size)
            token_attn_resized = np.array(
                Image.fromarray(token_attn).resize(image.size, Image.BILINEAR)
            )

            ax.imshow(image)
            ax.imshow(token_attn_resized, cmap='hot', alpha=0.5)
            ax.set_title(f'"{token}"', fontsize=9)
            ax.axis('off')

        # Hide unused axes
        for i in range(n_tokens + 1, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'Cross-Attention Visualization (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "cross_attention", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage2Visualizer:
    """Visualizations for Stage 2: Instruction Tuning."""

    def __init__(self, output_dir: str = "./outputs/stage2/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_generation_examples(
        self,
        images: List[Image.Image],
        instructions: List[str],
        generated: List[str],
        ground_truth: List[str],
        step: int,
        n_examples: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Display generation examples with images.

        Args:
            images: Input images
            instructions: Instruction prompts
            generated: Generated responses
            ground_truth: Ground truth responses
            step: Training step
            n_examples: Number of examples to show
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(images))

        fig, axes = plt.subplots(n_examples, 3, figsize=(18, 5*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            # Image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')

            # Instruction + Generated
            text = f"Instruction:\n{instructions[i][:200]}\n\n---\nGenerated:\n{generated[i][:300]}"
            axes[i, 1].text(0.05, 0.95, text, ha='left', va='top', wrap=True,
                           fontsize=9, transform=axes[i, 1].transAxes)
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].set_title('Instruction & Generation', fontsize=10)
            axes[i, 1].axis('off')

            # Ground truth
            axes[i, 2].text(0.05, 0.95, f"Ground Truth:\n{ground_truth[i][:400]}",
                           ha='left', va='top', wrap=True, fontsize=9,
                           transform=axes[i, 2].transAxes, color='green')
            axes[i, 2].set_xlim(0, 1)
            axes[i, 2].set_ylim(0, 1)
            axes[i, 2].set_title('Ground Truth', fontsize=10, color='green')
            axes[i, 2].axis('off')

        fig.suptitle(f'Generation Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "generation_examples", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_token_probability_distribution(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot distribution of predicted token probabilities.

        Args:
            logits: Model logits [B, T, V]
            labels: Target labels [B, T]
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Get probabilities for correct tokens
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        if labels.dtype != torch.long:
            labels = labels.long()

        probs = F.softmax(logits, dim=-1)  # [B, T, V]

        # Gather probabilities of correct tokens
        batch_size, seq_len, vocab_size = probs.shape

        # Mask out padding (-100)
        valid_mask = labels != -100

        # Flatten and gather
        flat_probs = probs.view(-1, vocab_size)
        flat_labels = labels.view(-1).clamp(min=0)

        correct_probs = flat_probs.gather(1, flat_labels.unsqueeze(1)).squeeze()
        correct_probs = correct_probs[valid_mask.view(-1)].float().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of correct token probabilities
        axes[0].hist(correct_probs, bins=50, alpha=0.7, color='#2E86AB', edgecolor='white')
        axes[0].axvline(np.mean(correct_probs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(correct_probs):.3f}')
        axes[0].set_xlabel('Probability of Correct Token')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Correct Token Probabilities')
        axes[0].legend()
        axes[0].set_xlim(0, 1)

        # Confidence categories
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels_cat = ['Very Low\n(0-0.1)', 'Low\n(0.1-0.3)', 'Medium\n(0.3-0.5)',
                      'High\n(0.5-0.7)', 'Very High\n(0.7-0.9)', 'Confident\n(0.9-1.0)']
        counts, _ = np.histogram(correct_probs, bins=bins)

        colors = ['#FF6B6B', '#FFA06B', '#FFD93D', '#6BCB77', '#4D96FF', '#6A5ACD']
        axes[1].bar(labels_cat, counts, color=colors, edgecolor='white')
        axes[1].set_xlabel('Confidence Category')
        axes[1].set_ylabel('Token Count')
        axes[1].set_title('Token Confidence Distribution')

        # Add percentages
        total = sum(counts)
        for i, (count, label) in enumerate(zip(counts, labels_cat)):
            axes[1].text(i, count + total*0.01, f'{count/total*100:.1f}%',
                        ha='center', fontsize=9)

        fig.suptitle(f'Token Probability Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "token_probs", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_response_length_distribution(
        self,
        generated_lengths: List[int],
        target_lengths: List[int],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare generated vs target response lengths.

        Args:
            generated_lengths: List of generated response lengths
            target_lengths: List of target response lengths
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overlapping histograms
        bins = np.linspace(0, max(max(generated_lengths), max(target_lengths)), 30)
        axes[0].hist(target_lengths, bins=bins, alpha=0.6, label='Target', color='#2E86AB')
        axes[0].hist(generated_lengths, bins=bins, alpha=0.6, label='Generated', color='#A23B72')
        axes[0].axvline(np.mean(target_lengths), color='#2E86AB', linestyle='--',
                       label=f'Target Mean: {np.mean(target_lengths):.1f}')
        axes[0].axvline(np.mean(generated_lengths), color='#A23B72', linestyle='--',
                       label=f'Gen Mean: {np.mean(generated_lengths):.1f}')
        axes[0].set_xlabel('Response Length (tokens)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Response Length Distribution')
        axes[0].legend()

        # Scatter plot
        axes[1].scatter(target_lengths, generated_lengths, alpha=0.5, s=20)
        max_len = max(max(target_lengths), max(generated_lengths))
        axes[1].plot([0, max_len], [0, max_len], 'r--', label='Perfect Match')
        axes[1].set_xlabel('Target Length')
        axes[1].set_ylabel('Generated Length')
        axes[1].set_title('Target vs Generated Length')
        axes[1].legend()

        fig.suptitle(f'Response Length Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "response_lengths", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage3Visualizer:
    """Visualizations for Stage 3: Robot Selection."""

    ROBOT_NAMES = ["Drone", "Underwater", "Humanoid", "Wheeled", "Legged"]
    ROBOT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    def __init__(self, output_dir: str = "./outputs/stage3/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confusion_history = []

    def plot_confusion_matrix(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot confusion matrix for robot selection.

        Args:
            predictions: Predicted robot indices
            labels: True robot indices
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for confusion matrix")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Confusion matrix requires sklearn', ha='center', va='center')
            return fig, _fig_to_pil(fig)

        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        cm = sklearn_cm(labels, predictions, labels=list(range(5)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # Store for history tracking
        self.confusion_history.append(cm)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Raw counts
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.ROBOT_NAMES, yticklabels=self.ROBOT_NAMES,
                       ax=axes[0], cbar_kws={'label': 'Count'})
        else:
            im = axes[0].imshow(cm, cmap='Blues')
            axes[0].set_xticks(range(5))
            axes[0].set_yticks(range(5))
            axes[0].set_xticklabels(self.ROBOT_NAMES, rotation=45, ha='right')
            axes[0].set_yticklabels(self.ROBOT_NAMES)
            for i in range(5):
                for j in range(5):
                    axes[0].text(j, i, str(cm[i, j]), ha='center', va='center')
            plt.colorbar(im, ax=axes[0])

        axes[0].set_title(f'Robot Selection Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Robot')
        axes[0].set_xlabel('Predicted Robot')

        # Normalized
        if HAS_SEABORN:
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=self.ROBOT_NAMES, yticklabels=self.ROBOT_NAMES,
                       ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        else:
            im = axes[1].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1].set_xticks(range(5))
            axes[1].set_yticks(range(5))
            axes[1].set_xticklabels(self.ROBOT_NAMES, rotation=45, ha='right')
            axes[1].set_yticklabels(self.ROBOT_NAMES)
            for i in range(5):
                for j in range(5):
                    axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center')
            plt.colorbar(im, ax=axes[1])

        axes[1].set_title(f'Robot Selection Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Robot')
        axes[1].set_xlabel('Predicted Robot')

        # Add overall accuracy
        acc = np.trace(cm) / (np.sum(cm) + 1e-8)
        fig.suptitle(f'Step {step} | Overall Accuracy: {acc:.1%}', fontsize=14)

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "confusion_matrix", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_per_robot_radar(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create radar chart for per-robot performance.

        Args:
            metrics: Dict with keys 'precision', 'recall', 'f1' each containing
                     per-robot values
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        categories = self.ROBOT_NAMES
        n_cats = len(categories)

        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        metric_names = ['precision', 'recall', 'f1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for metric_name, color in zip(metric_names, colors):
            if metric_name in metrics:
                values = [metrics[metric_name].get(robot, 0) for robot in categories]
                values += values[:1]  # Close the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=metric_name.upper(), color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'Per-Robot Performance Metrics (Step {step})', size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "robot_radar", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_confidence_calibration(
        self,
        confidences: Union[torch.Tensor, np.ndarray],
        correct: Union[torch.Tensor, np.ndarray],
        step: int,
        n_bins: int = 10,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot confidence calibration diagram and ECE.

        Args:
            confidences: Model confidence scores
            correct: Boolean array of correct predictions
            step: Training step
            n_bins: Number of calibration bins
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if torch.is_tensor(confidences):
            confidences = confidences.cpu().numpy()
        if torch.is_tensor(correct):
            correct = correct.cpu().numpy()

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(correct[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((lower + upper) / 2)
                bin_counts.append(0)

        # Calculate ECE
        ece = sum(count * abs(acc - conf) for count, acc, conf in
                  zip(bin_counts, bin_accuracies, bin_confidences)) / (sum(bin_counts) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Reliability diagram
        ax1 = axes[0]
        bin_centers = (bin_lowers + bin_uppers) / 2
        ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7,
               label='Accuracy', color='#2E86AB', edgecolor='white')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram | ECE: {ece:.4f}')
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Confidence histogram
        ax2 = axes[1]
        ax2.hist(confidences, bins=20, alpha=0.7, color='#A23B72', edgecolor='white')
        ax2.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()

        fig.suptitle(f'Calibration Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "calibration", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_reasoning_examples(
        self,
        tasks: List[str],
        reasoning_chains: List[str],
        predictions: List[str],
        ground_truth: List[str],
        correct: List[bool],
        step: int,
        n_examples: int = 6,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Display reasoning chain examples with correctness.

        Args:
            tasks: Task descriptions
            reasoning_chains: Generated reasoning
            predictions: Predicted robots
            ground_truth: True robots
            correct: Correctness flags
            step: Training step
            n_examples: Number to show
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(tasks))

        # Show mix of correct and incorrect
        correct_idx = [i for i, c in enumerate(correct) if c]
        incorrect_idx = [i for i, c in enumerate(correct) if not c]

        # Balance examples
        n_correct = min(n_examples // 2, len(correct_idx))
        n_incorrect = min(n_examples - n_correct, len(incorrect_idx))

        selected = correct_idx[:n_correct] + incorrect_idx[:n_incorrect]

        fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4*len(selected)))
        if len(selected) == 1:
            axes = [axes]

        for ax_idx, i in enumerate(selected):
            ax = axes[ax_idx]

            is_correct = correct[i]
            color = 'green' if is_correct else 'red'
            symbol = '✓' if is_correct else '✗'

            text = f"{symbol} Task: {tasks[i][:150]}...\n\n"
            text += f"Reasoning: {reasoning_chains[i][:300]}...\n\n"
            text += f"Predicted: {predictions[i]} | Ground Truth: {ground_truth[i]}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor=color, linewidth=2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        fig.suptitle(f'Reasoning Chain Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "reasoning_examples", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage4Visualizer:
    """Visualizations for Stage 4: Chain-of-Thought Reasoning."""

    def __init__(self, output_dir: str = "./outputs/stage4/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase1_metrics = []
        self.phase2_metrics = []

    def plot_reasoning_quality_metrics(
        self,
        coherence_scores: List[float],
        consistency_scores: List[float],
        step_counts: List[int],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot reasoning quality metrics dashboard.

        Args:
            coherence_scores: List of coherence scores
            consistency_scores: List of consistency scores
            step_counts: List of reasoning step counts
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Coherence distribution
        axes[0, 0].hist(coherence_scores, bins=20, alpha=0.7, color='#2E86AB', edgecolor='white')
        axes[0, 0].axvline(np.mean(coherence_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(coherence_scores):.3f}')
        axes[0, 0].set_xlabel('Coherence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reasoning Coherence Distribution')
        axes[0, 0].legend()

        # Consistency distribution
        axes[0, 1].hist(consistency_scores, bins=20, alpha=0.7, color='#A23B72', edgecolor='white')
        axes[0, 1].axvline(np.mean(consistency_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(consistency_scores):.3f}')
        axes[0, 1].set_xlabel('Consistency Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Logical Consistency Distribution')
        axes[0, 1].legend()

        # Step count distribution
        unique_steps, counts = np.unique(step_counts, return_counts=True)
        axes[1, 0].bar(unique_steps, counts, alpha=0.7, color='#4ECDC4', edgecolor='white')
        axes[1, 0].set_xlabel('Number of Reasoning Steps')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reasoning Step Count Distribution')

        # Coherence vs Steps scatter
        axes[1, 1].scatter(step_counts, coherence_scores, alpha=0.5, s=30, c='#45B7D1')

        # Add trend line
        z = np.polyfit(step_counts, coherence_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(step_counts), max(step_counts), 100)
        axes[1, 1].plot(x_line, p(x_line), 'r--', label='Trend')

        axes[1, 1].set_xlabel('Number of Reasoning Steps')
        axes[1, 1].set_ylabel('Coherence Score')
        axes[1, 1].set_title('Reasoning Steps vs Coherence')
        axes[1, 1].legend()

        fig.suptitle(f'Reasoning Quality Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "reasoning_quality", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_phase_comparison(
        self,
        phase1_history: Dict[str, List[float]],
        phase2_history: Dict[str, List[float]],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare Phase 1 and Phase 2 training metrics.

        Args:
            phase1_history: Phase 1 metrics history
            phase2_history: Phase 2 metrics history
            step: Current step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss comparison
        ax = axes[0, 0]
        if 'loss' in phase1_history:
            ax.plot(phase1_history['loss'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'loss' in phase2_history:
            offset = len(phase1_history.get('loss', []))
            x = np.arange(offset, offset + len(phase2_history['loss']))
            ax.plot(x, phase2_history['loss'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5, label='Phase Transition')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy comparison
        ax = axes[0, 1]
        if 'robot_accuracy' in phase1_history:
            ax.plot(phase1_history['robot_accuracy'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'robot_accuracy' in phase2_history:
            offset = len(phase1_history.get('robot_accuracy', []))
            x = np.arange(offset, offset + len(phase2_history['robot_accuracy']))
            ax.plot(x, phase2_history['robot_accuracy'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Robot Selection Accuracy by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reasoning loss
        ax = axes[1, 0]
        if 'consistency_loss' in phase1_history:
            ax.plot(phase1_history['consistency_loss'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'consistency_loss' in phase2_history:
            offset = len(phase1_history.get('consistency_loss', []))
            x = np.arange(offset, offset + len(phase2_history['consistency_loss']))
            ax.plot(x, phase2_history['consistency_loss'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Consistency Loss')
        ax.set_title('Reasoning Consistency Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary bar chart
        ax = axes[1, 1]
        metrics = ['loss', 'robot_accuracy']
        phase1_final = [phase1_history.get(m, [0])[-1] if phase1_history.get(m) else 0 for m in metrics]
        phase2_final = [phase2_history.get(m, [0])[-1] if phase2_history.get(m) else 0 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, phase1_final, width, label='Phase 1 Final', color='#2E86AB')
        ax.bar(x + width/2, phase2_final, width, label='Phase 2 Final', color='#A23B72')
        ax.set_xticks(x)
        ax.set_xticklabels(['Loss', 'Accuracy'])
        ax.set_ylabel('Value')
        ax.set_title('Final Metrics Comparison')
        ax.legend()

        fig.suptitle(f'Phase 1 vs Phase 2 Training Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "phase_comparison", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_cot_examples(
        self,
        tasks: List[str],
        without_cot: List[Tuple[str, bool]],  # (prediction, correct)
        with_cot: List[Tuple[str, str, bool]],  # (reasoning, prediction, correct)
        step: int,
        n_examples: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare predictions with and without Chain-of-Thought.

        Args:
            tasks: Task descriptions
            without_cot: (prediction, correct) without CoT
            with_cot: (reasoning, prediction, correct) with CoT
            step: Training step
            n_examples: Number of examples
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(tasks))

        fig, axes = plt.subplots(n_examples, 2, figsize=(18, 4*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            # Without CoT
            ax = axes[i, 0]
            pred, correct = without_cot[i]
            color = 'green' if correct else 'red'
            symbol = '✓' if correct else '✗'

            text = f"Task: {tasks[i][:150]}...\n\n"
            text += f"{symbol} Direct Prediction: {pred}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='#FFF3E0',
                            edgecolor=color, linewidth=2))
            ax.set_title('Without Chain-of-Thought', fontsize=11)
            ax.axis('off')

            # With CoT
            ax = axes[i, 1]
            reasoning, pred, correct = with_cot[i]
            color = 'green' if correct else 'red'
            symbol = '✓' if correct else '✗'

            text = f"Reasoning:\n{reasoning[:250]}...\n\n"
            text += f"{symbol} Prediction: {pred}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='#E8F5E9',
                            edgecolor=color, linewidth=2))
            ax.set_title('With Chain-of-Thought', fontsize=11)
            ax.axis('off')

        fig.suptitle(f'Chain-of-Thought Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "cot_comparison", f"step{step}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class CrossStageVisualizer:
    """Visualizations for cross-stage analysis."""

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_metrics = {1: {}, 2: {}, 3: {}, 4: {}}

    def update_stage_metrics(self, stage: int, metrics: Dict[str, float]):
        """Update metrics for a stage."""
        for key, value in metrics.items():
            if key not in self.stage_metrics[stage]:
                self.stage_metrics[stage][key] = []
            self.stage_metrics[stage][key].append(value)

    def plot_stage_progression(
        self,
        eval_results: Dict[int, Dict[str, float]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot performance progression across stages.

        Args:
            eval_results: {stage_num: {benchmark: score}}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        stages = sorted(eval_results.keys())
        benchmarks = list(eval_results[stages[0]].keys()) if stages else []

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(benchmarks))
        width = 0.2

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for i, stage in enumerate(stages):
            scores = [eval_results[stage].get(b, 0) for b in benchmarks]
            ax.bar(x + i*width, scores, width, label=f'Stage {stage}',
                  color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Score')
        ax.set_title('Benchmark Performance Across Training Stages')
        ax.set_xticks(x + width * (len(stages) - 1) / 2)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "stage_progression", "stage_progression.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_training_summary(
        self,
        all_metrics: Dict[int, Dict[str, List[float]]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot training summary across all stages.

        Args:
            all_metrics: {stage: {metric: [values]}}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}

        # Loss progression
        ax = axes[0, 0]
        offset = 0
        for stage in sorted(all_metrics.keys()):
            if 'loss' in all_metrics[stage]:
                values = all_metrics[stage]['loss']
                x = np.arange(offset, offset + len(values))
                ax.plot(x, values, label=f'Stage {stage}', color=colors[stage], linewidth=2)
                offset += len(values)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Across All Stages')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[0, 1]
        offset = 0
        for stage in sorted(all_metrics.keys()):
            if 'lr' in all_metrics[stage]:
                values = all_metrics[stage]['lr']
                x = np.arange(offset, offset + len(values))
                ax.plot(x, values, label=f'Stage {stage}', color=colors[stage], linewidth=2)
                offset += len(values)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Per-stage final metrics
        ax = axes[1, 0]
        stage_names = [f'Stage {s}' for s in sorted(all_metrics.keys())]
        final_losses = [all_metrics[s].get('loss', [0])[-1] if all_metrics[s].get('loss') else 0
                       for s in sorted(all_metrics.keys())]

        ax.bar(stage_names, final_losses, color=[colors[s] for s in sorted(all_metrics.keys())])
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Per Stage')
        ax.grid(True, alpha=0.3, axis='y')

        # Training time (if available)
        ax = axes[1, 1]
        if any('training_time' in all_metrics[s] for s in all_metrics):
            times = [sum(all_metrics[s].get('training_time', [0])) for s in sorted(all_metrics.keys())]
            ax.pie(times, labels=stage_names, colors=[colors[s] for s in sorted(all_metrics.keys())],
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Training Time Distribution')
        else:
            ax.text(0.5, 0.5, 'Training time not tracked', ha='center', va='center')
            ax.axis('off')

        fig.suptitle('EmberVLM Training Summary', fontsize=16)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "training_summary", "training_summary.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_carbon_footprint(
        self,
        emissions_per_stage: Dict[int, float],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot carbon emissions by stage.

        Args:
            emissions_per_stage: {stage: kg_co2}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        stages = [f'Stage {s}' for s in sorted(emissions_per_stage.keys())]
        emissions = [emissions_per_stage[s] for s in sorted(emissions_per_stage.keys())]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # Bar chart
        axes[0].bar(stages, emissions, color=colors[:len(stages)], edgecolor='white')
        axes[0].set_ylabel('CO₂ Emissions (kg)')
        axes[0].set_title('Carbon Footprint by Stage')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Cumulative line
        cumulative = np.cumsum(emissions)
        axes[1].plot(stages, cumulative, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        axes[1].fill_between(stages, 0, cumulative, alpha=0.3, color='#2E86AB')
        axes[1].set_ylabel('Cumulative CO₂ (kg)')
        axes[1].set_title(f'Cumulative Emissions: {cumulative[-1]:.3f} kg CO₂')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('EmberVLM Carbon Footprint Analysis', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "carbon_footprint", "carbon_footprint.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_benchmark_progression_heatmap(
        self,
        benchmark_scores: Dict[str, Dict[str, float]],  # {benchmark: {stage: score}}
        save: bool = True
    ) -> Tuple[plt.Figure, Image.Image]:
        """Create heatmap showing benchmark score progression across stages."""
        fig, ax = plt.subplots(figsize=(12, 8))

        benchmarks = sorted(benchmark_scores.keys())
        stages = sorted(list(benchmark_scores[benchmarks[0]].keys())) if benchmarks else []

        if not benchmarks or not stages:
            return fig, _fig_to_pil(fig)

        matrix = np.zeros((len(benchmarks), len(stages)))
        for i, bench in enumerate(benchmarks):
            for j, stage in enumerate(stages):
                matrix[i, j] = benchmark_scores[bench].get(stage, 0)

        matrix_norm = matrix / matrix.max(axis=1, keepdims=True).clip(min=1e-10)
        im = ax.imshow(matrix_norm, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(stages)))
        ax.set_yticks(np.arange(len(benchmarks)))
        ax.set_xticklabels(stages)
        ax.set_yticklabels(benchmarks)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Score', rotation=270, labelpad=20)

        for i in range(len(benchmarks)):
            for j in range(len(stages)):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha='center', va='center',
                             color='white' if matrix_norm[i, j] > 0.5 else 'black',
                             fontsize=9)

        ax.set_title('Benchmark Score Progression Across Training Stages',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Training Stage', fontsize=12)
        ax.set_ylabel('Benchmark', fontsize=12)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "benchmark_progression", "benchmark_progression_heatmap.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_ablation_tornado(
        self,
        ablation_results: Dict[str, float],
        baseline_score: float,
        save: bool = True
    ) -> Tuple[plt.Figure, Image.Image]:
        """Create tornado chart showing ablation study results."""
        fig, ax = plt.subplots(figsize=(12, 8))

        components = list(ablation_results.keys())
        impacts = [ablation_results[c] - baseline_score for c in components]
        sorted_idx = np.argsort(np.abs(impacts))[::-1]

        components_sorted = [components[i] for i in sorted_idx]
        impacts_sorted = [impacts[i] for i in sorted_idx]

        colors = ['#E15759' if imp < 0 else '#59A14F' for imp in impacts_sorted]
        y_pos = np.arange(len(components_sorted))

        bars = ax.barh(y_pos, impacts_sorted, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Baseline')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(components_sorted, fontsize=11)
        ax.set_xlabel('Impact on Performance (Δ Accuracy)', fontsize=12)
        ax.set_title('Ablation Study: Component Impact Analysis',
                    fontsize=14, fontweight='bold', pad=20)

        for bar, impact in zip(bars, impacts_sorted):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                   f'{impact:+.3f}',
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.legend()
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "ablation_tornado", "ablation_tornado.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_parameter_efficiency_pareto(
        self,
        models_data: List[Dict[str, Union[str, float]]],
        save: bool = True
    ) -> Tuple[plt.Figure, Image.Image]:
        """Create Pareto frontier plot for parameter efficiency analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))

        names = [m['name'] for m in models_data]
        params = np.array([m['params'] for m in models_data])
        accuracies = np.array([m['accuracy'] for m in models_data])
        memories = np.array([m.get('memory', 100) for m in models_data])

        bubble_sizes = (memories / memories.max()) * 1000
        efficiencies = accuracies / (params / 1e6)

        scatter = ax.scatter(params / 1e6, accuracies, s=bubble_sizes,
                           c=efficiencies, cmap='RdYlGn', alpha=0.6,
                           edgecolors='black', linewidth=2)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Efficiency (Acc/M Params)', rotation=270, labelpad=20)

        # Find Pareto frontier
        pareto_idx = []
        for i in range(len(params)):
            is_pareto = True
            for j in range(len(params)):
                if i != j:
                    if params[j] <= params[i] and accuracies[j] >= accuracies[i]:
                        if params[j] < params[i] or accuracies[j] > accuracies[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_idx.append(i)

        if pareto_idx:
            pareto_idx_sorted = sorted(pareto_idx, key=lambda i: params[i])
            pareto_params = params[pareto_idx_sorted] / 1e6
            pareto_accs = accuracies[pareto_idx_sorted]
            ax.plot(pareto_params, pareto_accs, 'r--', linewidth=2,
                   alpha=0.7, label='Pareto Frontier')

        for i, name in enumerate(names):
            ax.annotate(name, (params[i] / 1e6, accuracies[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        ember_idx = [i for i, n in enumerate(names) if 'EmberVLM' in n or 'Ember' in n]
        if ember_idx:
            ax.scatter(params[ember_idx] / 1e6, accuracies[ember_idx],
                      s=1500, facecolors='none', edgecolors='red',
                      linewidths=3, label='EmberVLM')

        ax.set_xlabel('Parameters (Millions)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Parameter Efficiency Pareto Frontier',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if params.max() / params.min() > 100:
            ax.set_xscale('log')

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "parameter_efficiency", "parameter_efficiency_pareto.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)


# =============================================================================
# Hallucination Visualizer — for Stage 2.5 VA analysis
# =============================================================================

class HallucinationVisualizer:
    """Visualizations for hallucination analysis and VA refiner effectiveness.

    Produces plots showing:
    - Per-category hallucination rate (text vs vision tasks)
    - Response quality distribution with hallucination flags
    - Hallucination type breakdown (repetition, gibberish, ungrounded, etc.)
    - VA refiner impact: before/after comparison
    - Hallucination heatmap across prompt categories
    """

    # Warm color palette for hallucination severity
    COLORS = {
        'pass': '#4CAF50',       # Green
        'fail_halluc': '#F44336', # Red
        'fail_other': '#FF9800',  # Orange
        'text': '#42A5F5',       # Blue
        'vision': '#AB47BC',     # Purple
        'bg_light': '#FAFAFA',
        'grid': '#E0E0E0',
    }

    def __init__(self, output_dir: str = "./outputs/stage2_5/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_hallucination_dashboard(
        self,
        coherence_results: Dict[str, Any],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Comprehensive hallucination analysis dashboard from Stage 2.5 evaluation.

        Produces a 2x3 figure:
          [0,0] Per-category pass/fail bar chart
          [0,1] Text vs Vision hallucination comparison
          [0,2] Hallucination type breakdown (pie)
          [1,0] Response length vs pass/fail violin
          [1,1] Per-category hallucination rate heatmap
          [1,2] Overall quality gauge

        Args:
            coherence_results: Full dict from CoherenceChecker.run_coherence_tests()
            save: Whether to save to disk
        """
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor('#FAFAFA')
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        categories = coherence_results.get('categories', {})
        tests = coherence_results.get('tests', [])

        # ── [0,0] Per-category pass/fail stacked bar ──
        ax = fig.add_subplot(gs[0, 0])
        self._plot_category_pass_fail(ax, categories)

        # ── [0,1] Text vs Vision hallucination comparison ──
        ax = fig.add_subplot(gs[0, 1])
        self._plot_text_vs_vision(ax, coherence_results)

        # ── [0,2] Failure type breakdown ──
        ax = fig.add_subplot(gs[0, 2])
        self._plot_failure_type_pie(ax, tests)

        # ── [1,0] Response length analysis ──
        ax = fig.add_subplot(gs[1, 0])
        self._plot_response_length_analysis(ax, tests)

        # ── [1,1] Category hallucination rate heatmap ──
        ax = fig.add_subplot(gs[1, 1])
        self._plot_category_heatmap(ax, categories)

        # ── [1,2] Overall quality gauge ──
        ax = fig.add_subplot(gs[1, 2])
        self._plot_quality_gauge(ax, coherence_results)

        fig.suptitle(
            'Hallucination Analysis Dashboard — Stage 2.5 Evaluation',
            fontsize=16, fontweight='bold', y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            save_path = _viz_save_path(self.output_dir, "hallucination_dashboard", "hallucination_dashboard.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
            logger.info(f"  ✓ Saved hallucination dashboard: {save_path}")

        return fig, _fig_to_pil(fig)

    def _plot_category_pass_fail(self, ax: plt.Axes, categories: Dict):
        """Stacked horizontal bar: passed vs failed per category."""
        if not categories:
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Category Results')
            return

        cat_names = list(categories.keys())
        passed = [categories[c].get('passed', 0) for c in cat_names]
        total = [categories[c].get('total', 1) for c in cat_names]
        failed = [t - p for t, p in zip(total, passed)]

        y_pos = np.arange(len(cat_names))
        ax.barh(y_pos, passed, color=self.COLORS['pass'], label='Passed', edgecolor='white', height=0.6)
        ax.barh(y_pos, failed, left=passed, color=self.COLORS['fail_halluc'], label='Failed', edgecolor='white', height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=10)
        ax.set_xlabel('Number of Tests', fontsize=10)
        ax.set_title('Per-Category Pass/Fail', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor(self.COLORS['bg_light'])

        # Add count labels
        for i, (p, f) in enumerate(zip(passed, failed)):
            if p > 0:
                ax.text(p / 2, i, str(p), ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            if f > 0:
                ax.text(p + f / 2, i, str(f), ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    def _plot_text_vs_vision(self, ax: plt.Axes, results: Dict):
        """Grouped bar comparing text-only vs vision-language hallucination rates."""
        text_passed = results.get('text_passed', 0)
        text_total = results.get('text_total', 1)
        vision_passed = results.get('vision_passed', 0)
        vision_total = results.get('vision_total', 1)

        text_rate = (1 - text_passed / max(text_total, 1)) * 100
        vision_rate = (1 - vision_passed / max(vision_total, 1)) * 100
        text_pass_rate = text_passed / max(text_total, 1) * 100
        vision_pass_rate = vision_passed / max(vision_total, 1) * 100

        x = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x - width / 2, [text_pass_rate, vision_pass_rate], width,
                       label='Pass Rate %', color=[self.COLORS['text'], self.COLORS['vision']],
                       edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, [text_rate, vision_rate], width,
                       label='Hallucination Rate %', color=[self.COLORS['fail_halluc']] * 2,
                       alpha=0.7, edgecolor='white', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(['Text-Only', 'Vision-Language'], fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title('Text vs Vision Hallucination', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor(self.COLORS['bg_light'])

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f'{h:.0f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _plot_failure_type_pie(self, ax: plt.Axes, tests: List[Dict]):
        """Pie chart of failure reasons."""
        failure_types = {
            'Hallucination': 0,
            'Repetition': 0,
            'Too Short': 0,
            'Gibberish': 0,
            'Code Output': 0,
            'Other Failure': 0,
        }
        n_passed = 0

        for test in tests:
            if test.get('passed', False):
                n_passed += 1
                continue
            reason = test.get('reason', '').lower()
            if 'hallucination' in reason or 'negative pattern' in reason:
                failure_types['Hallucination'] += 1
            elif 'repetition' in reason or 'unique' in reason:
                failure_types['Repetition'] += 1
            elif 'too short' in reason or 'length' in reason or 'empty' in reason:
                failure_types['Too Short'] += 1
            elif 'gibberish' in reason or 'special char' in reason:
                failure_types['Gibberish'] += 1
            elif 'code' in reason:
                failure_types['Code Output'] += 1
            else:
                failure_types['Other Failure'] += 1

        # Include passed in pie
        labels = ['Passed'] + [k for k, v in failure_types.items() if v > 0]
        sizes = [n_passed] + [v for v in failure_types.values() if v > 0]
        colors_list = [self.COLORS['pass']] + [
            {'Hallucination': '#F44336', 'Repetition': '#FF5722', 'Too Short': '#FF9800',
             'Gibberish': '#795548', 'Code Output': '#607D8B', 'Other Failure': '#9E9E9E'}[k]
            for k, v in failure_types.items() if v > 0
        ]

        if sum(sizes) == 0:
            ax.text(0.5, 0.5, 'No test data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Failure Type Breakdown')
            return

        explode = [0.05] * len(sizes)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_list, autopct='%1.0f%%',
            explode=explode, startangle=90, textprops={'fontsize': 9},
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('Response Quality Breakdown', fontsize=12, fontweight='bold')

    def _plot_response_length_analysis(self, ax: plt.Axes, tests: List[Dict]):
        """Box plots of response lengths for passed vs failed tests."""
        passed_lens = []
        failed_lens = []
        for test in tests:
            resp = test.get('response', '')
            length = len(resp.split())
            if test.get('passed', False):
                passed_lens.append(length)
            else:
                failed_lens.append(length)

        data_to_plot = []
        labels_to_plot = []
        colors_bp = []
        if passed_lens:
            data_to_plot.append(passed_lens)
            labels_to_plot.append(f'Passed\n(n={len(passed_lens)})')
            colors_bp.append(self.COLORS['pass'])
        if failed_lens:
            data_to_plot.append(failed_lens)
            labels_to_plot.append(f'Failed\n(n={len(failed_lens)})')
            colors_bp.append(self.COLORS['fail_halluc'])

        if not data_to_plot:
            ax.text(0.5, 0.5, 'No response data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Response Length Distribution')
            return

        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, widths=0.5,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='gold', markersize=8))

        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Response Length (words)', fontsize=10)
        ax.set_title('Response Length: Passed vs Failed', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor(self.COLORS['bg_light'])

    def _plot_category_heatmap(self, ax: plt.Axes, categories: Dict):
        """Heatmap showing hallucination rate per category."""
        if not categories:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Category Hallucination Rate')
            return

        cat_names = list(categories.keys())
        rates = []
        for c in cat_names:
            total = categories[c].get('total', 1)
            passed = categories[c].get('passed', 0)
            rates.append((1 - passed / max(total, 1)) * 100)

        # Reshape for imshow (single row heatmap - or make it a grid)
        n = len(cat_names)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        matrix = np.full((nrows, ncols), np.nan)
        names_grid = [[''] * ncols for _ in range(nrows)]

        for i, (name, rate) in enumerate(zip(cat_names, rates)):
            r, c_idx = divmod(i, ncols)
            matrix[r, c_idx] = rate
            names_grid[r][c_idx] = name.replace('_', '\n').title()

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(nrows):
            for j in range(ncols):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 50 else 'black'
                    ax.text(j, i, f'{names_grid[i][j]}\n{val:.0f}%',
                           ha='center', va='center', fontsize=9, fontweight='bold', color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hallucination Rate %', fontsize=9)
        ax.set_title('Hallucination Rate by Category', fontsize=12, fontweight='bold')

    def _plot_quality_gauge(self, ax: plt.Axes, results: Dict):
        """Semi-circular gauge showing overall quality score."""
        score = results.get('overall_score', 0.0)

        # Draw arc
        theta = np.linspace(np.pi, 0, 100)
        r_outer = 1.0
        r_inner = 0.7

        # Background arc (grey)
        ax.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_inner,
                        np.sin(theta) * r_outer, alpha=0.1, color='grey')

        # Score arc
        n_filled = int(score)
        theta_filled = np.linspace(np.pi, np.pi - (np.pi * score / 100), max(2, n_filled))

        if score >= 70:
            gauge_color = self.COLORS['pass']
        elif score >= 40:
            gauge_color = self.COLORS['fail_other']
        else:
            gauge_color = self.COLORS['fail_halluc']

        for t in theta_filled:
            ax.plot([np.cos(t) * r_inner, np.cos(t) * r_outer],
                   [np.sin(t) * r_inner, np.sin(t) * r_outer],
                   color=gauge_color, linewidth=2, alpha=0.8)

        ax.text(0, 0.35, f'{score:.1f}%', ha='center', va='center',
               fontsize=28, fontweight='bold', color=gauge_color)
        ax.text(0, 0.05, 'Overall Quality', ha='center', va='center',
               fontsize=11, color='#333333')

        passed = results.get('passed_count', 0)
        total = results.get('total_count', 0)
        ax.text(0, -0.15, f'{passed}/{total} tests passed', ha='center', va='center',
               fontsize=10, color='#666666')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Overall Coherence Score', fontsize=12, fontweight='bold')

    def plot_hallucination_over_training(
        self,
        halluc_rates_per_stage: Dict[str, float],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Bar chart showing hallucination rate reduction across training stages.

        Args:
            halluc_rates_per_stage: {'Stage 1': 0.45, 'Stage 2': 0.30, 'Stage 2.5': 0.15, ...}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#FAFAFA')

        stages = list(halluc_rates_per_stage.keys())
        rates = [halluc_rates_per_stage[s] * 100 for s in stages]

        colors = []
        for r in rates:
            if r >= 50:
                colors.append('#F44336')
            elif r >= 30:
                colors.append('#FF9800')
            elif r >= 15:
                colors.append('#FFC107')
            else:
                colors.append('#4CAF50')

        bars = ax.bar(stages, rates, color=colors, edgecolor='white', linewidth=2, width=0.6)

        # Add trend line
        x_numeric = np.arange(len(stages))
        if len(x_numeric) >= 2:
            z = np.polyfit(x_numeric, rates, 1)
            p = np.poly1d(z)
            ax.plot(x_numeric, p(x_numeric), '--', color='#333333', linewidth=2, alpha=0.6, label='Trend')
            ax.legend(fontsize=10)

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Hallucination Rate (%)', fontsize=12)
        ax.set_title('Hallucination Rate Reduction Across Training',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        ax.set_ylim(0, max(rates) * 1.2 if rates else 100)

        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "hallucination_over_training", "hallucination_over_training.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

    def plot_va_refiner_impact(
        self,
        without_va_results: Dict[str, Any],
        with_va_results: Dict[str, Any],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Side-by-side comparison of model outputs with and without VA refiner.

        Shows how the VA hallucination suppression module improves output quality.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#FAFAFA')

        # [0] Score comparison
        ax = axes[0]
        metrics = ['overall_score', 'text_score', 'vision_score']
        labels = ['Overall', 'Text', 'Vision']
        without_vals = [without_va_results.get(m, 0) for m in metrics]
        with_vals = [with_va_results.get(m, 0) for m in metrics]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, without_vals, width, label='Without VA', color='#EF5350', alpha=0.8, edgecolor='white')
        ax.bar(x + width / 2, with_vals, width, label='With VA', color='#66BB6A', alpha=0.8, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('Quality Score Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # [1] Pass rate comparison per category
        ax = axes[1]
        without_cats = without_va_results.get('categories', {})
        with_cats = with_va_results.get('categories', {})
        all_cats = sorted(set(list(without_cats.keys()) + list(with_cats.keys())))

        if all_cats:
            y = np.arange(len(all_cats))
            without_rates = [(without_cats.get(c, {}).get('passed', 0) / max(without_cats.get(c, {}).get('total', 1), 1)) * 100 for c in all_cats]
            with_rates = [(with_cats.get(c, {}).get('passed', 0) / max(with_cats.get(c, {}).get('total', 1), 1)) * 100 for c in all_cats]
            h = 0.35
            ax.barh(y - h / 2, without_rates, h, label='Without VA', color='#EF5350', alpha=0.8)
            ax.barh(y + h / 2, with_rates, h, label='With VA', color='#66BB6A', alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([c.replace('_', ' ').title() for c in all_cats], fontsize=9)
            ax.set_xlabel('Pass Rate (%)', fontsize=11)
            ax.legend(fontsize=9)
        ax.set_title('Per-Category Pass Rate', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # [2] Improvement delta
        ax = axes[2]
        if all_cats:
            deltas = [w - wo for w, wo in zip(with_rates, without_rates)]
            colors_delta = ['#4CAF50' if d >= 0 else '#F44336' for d in deltas]
            ax.barh(y, deltas, color=colors_delta, edgecolor='white', height=0.5)
            ax.set_yticks(y)
            ax.set_yticklabels([c.replace('_', ' ').title() for c in all_cats], fontsize=9)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Δ Pass Rate (%)', fontsize=11)

            for i, d in enumerate(deltas):
                ax.text(d + (1 if d >= 0 else -1), i, f'{d:+.0f}%', va='center',
                       ha='left' if d >= 0 else 'right', fontsize=9, fontweight='bold')
        ax.set_title('VA Refiner Improvement', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        fig.suptitle('VA Hallucination Suppression — Impact Analysis',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            save_path = _viz_save_path(self.output_dir, "va_refiner_impact", "va_refiner_impact.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)


# =============================================================================
# Collated Cross-Stage Visualizer — end-of-training summary
# =============================================================================

class CollatedVisualizer:
    """
    Generates beautiful cross-stage training summary visualizations at the end
    of the complete training pipeline.

    Reads saved metrics from each stage and produces comprehensive,
    publication-quality figures showing:
    - How the model learned across all 4 stages
    - Loss curves, accuracy trajectories, hallucination reduction
    - Robot selection improvement over stages
    - How different training objectives contribute to final capability
    """

    STAGE_COLORS = {
        'Stage 1': '#2196F3',   # Blue
        'Stage 2': '#9C27B0',   # Purple
        'Stage 2.5': '#FF9800', # Orange
        'Stage 3': '#4CAF50',   # Green
        'Stage 4': '#F44336',   # Red
    }

    STAGE_LABELS = {
        'Stage 1': 'Visual-Language\nAlignment',
        'Stage 2': 'Instruction\nTuning',
        'Stage 2.5': 'Hallucination\nAssessment',
        'Stage 3': 'Robot Fleet\nSelection',
        'Stage 4': 'Chain-of-Thought\nReasoning',
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        stage2_5_results: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Generate all collated visualizations.

        Args:
            stage_metrics: {
                'stage1': {'loss': [...], 'acc_i2t': [...], 'acc_t2i': [...], 'lr': [...]},
                'stage2': {'loss': [...], 'val_accuracy': [...], 'val_perplexity': [...], ...},
                'stage3': {'robot_loss': [...], 'robot_accuracy': [...], 'macro_f1': [...]},
                'stage4': {'loss': [...], 'robot_accuracy': [...], 'phase1_loss': [...], 'phase2_loss': [...]},
            }
            stage2_5_results: Stage 2.5 evaluation_summary.json contents

        Returns:
            Dict of {plot_name: saved_path}
        """
        saved = {}
        logger.info("📊 Generating collated cross-stage visualizations...")

        # 1. Grand loss + accuracy timeline
        try:
            fig, _ = self.plot_training_timeline(stage_metrics)
            path = str(_viz_save_path(self.output_dir, "training_timeline", "training_timeline.png"))
            saved['training_timeline'] = path
            logger.info(f"  ✓ training_timeline.png")
        except Exception as e:
            logger.warning(f"  ✗ training_timeline: {e}")

        # 2. Stage-by-stage capability radar
        try:
            fig, _ = self.plot_capability_radar(stage_metrics, stage2_5_results)
            path = str(_viz_save_path(self.output_dir, "capability_radar", "capability_radar.png"))
            saved['capability_radar'] = path
            logger.info(f"  ✓ capability_radar.png")
        except Exception as e:
            logger.warning(f"  ✗ capability_radar: {e}")

        # 3. Hallucination journey
        try:
            fig, _ = self.plot_hallucination_journey(stage_metrics, stage2_5_results)
            path = str(_viz_save_path(self.output_dir, "hallucination_journey", "hallucination_journey.png"))
            saved['hallucination_journey'] = path
            logger.info(f"  ✓ hallucination_journey.png")
        except Exception as e:
            logger.warning(f"  ✗ hallucination_journey: {e}")

        # 4. Learning dynamics heatmap
        try:
            fig, _ = self.plot_learning_dynamics_heatmap(stage_metrics)
            path = str(_viz_save_path(self.output_dir, "learning_dynamics_heatmap", "learning_dynamics_heatmap.png"))
            saved['learning_dynamics_heatmap'] = path
            logger.info(f"  ✓ learning_dynamics_heatmap.png")
        except Exception as e:
            logger.warning(f"  ✗ learning_dynamics_heatmap: {e}")

        # 5. Final achievement summary
        try:
            fig, _ = self.plot_achievement_summary(stage_metrics, stage2_5_results)
            path = str(_viz_save_path(self.output_dir, "achievement_summary", "achievement_summary.png"))
            saved['achievement_summary'] = path
            logger.info(f"  ✓ achievement_summary.png")
        except Exception as e:
            logger.warning(f"  ✗ achievement_summary: {e}")

        logger.info(f"📊 Collated visualizations complete: {len(saved)} figures saved to {self.output_dir}")
        return saved

    def plot_training_timeline(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Grand timeline showing loss curves and key metrics across all stages.

        Layout: 3 rows
          Row 0: Continuous loss curve across all stages (shaded by stage)
          Row 1: Stage-specific key metric (accuracy, F1, perplexity)
          Row 2: Learning rate schedule across all stages
        """
        fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 2, 1]})
        fig.patch.set_facecolor('#FAFAFA')

        stage_order = ['stage1', 'stage2', 'stage3', 'stage4']
        stage_display = {'stage1': 'Stage 1', 'stage2': 'Stage 2', 'stage3': 'Stage 3', 'stage4': 'Stage 4'}
        offset = 0
        boundaries = []

        # ── Row 0: Continuous loss ──
        ax = axes[0]
        for stage_key in stage_order:
            if stage_key not in stage_metrics:
                continue
            sm = stage_metrics[stage_key]
            losses = sm.get('loss', [])
            if not losses:
                continue

            color = self.STAGE_COLORS.get(stage_display.get(stage_key, ''), '#888888')
            x = np.arange(offset, offset + len(losses))
            ax.plot(x, losses, color=color, linewidth=1.8, label=stage_display.get(stage_key, stage_key))
            ax.fill_between(x, losses, alpha=0.08, color=color)

            # Stage boundary
            if boundaries:
                ax.axvline(offset, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
            boundaries.append((offset, offset + len(losses), stage_display.get(stage_key, stage_key)))
            offset += len(losses)

        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss Across All Stages', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#FAFAFA')

        # Add stage labels at top
        for start, end, label in boundaries:
            mid = (start + end) / 2
            ax.annotate(label, xy=(mid, ax.get_ylim()[1]), fontsize=9,
                       ha='center', va='bottom', fontweight='bold', alpha=0.7)

        # ── Row 1: Stage-specific accuracy / key metric ──
        ax = axes[1]
        offset = 0
        for stage_key in stage_order:
            if stage_key not in stage_metrics:
                continue
            sm = stage_metrics[stage_key]
            color = self.STAGE_COLORS.get(stage_display.get(stage_key, ''), '#888888')

            # Pick the best available accuracy metric
            if stage_key == 'stage1':
                vals = sm.get('acc_i2t', sm.get('val_i2t_acc', []))
                metric_name = 'I2T Accuracy'
            elif stage_key == 'stage2':
                vals = sm.get('val_accuracy', [])
                metric_name = 'Token Accuracy'
            elif stage_key == 'stage3':
                vals = sm.get('robot_accuracy', sm.get('accuracy', []))
                metric_name = 'Robot Accuracy'
            elif stage_key == 'stage4':
                vals = sm.get('robot_accuracy', [])
                metric_name = 'Robot Accuracy'
            else:
                vals = []
                metric_name = ''

            if not vals:
                # Use loss reduction as fallback
                losses = sm.get('loss', [])
                offset += len(losses) if losses else 0
                continue

            x = np.arange(offset, offset + len(vals))
            ax.plot(x, vals, color=color, linewidth=1.8, marker='o', markersize=2, label=f'{stage_display.get(stage_key)}: {metric_name}')
            offset += len(sm.get('loss', vals))

        ax.set_ylabel('Accuracy / Key Metric', fontsize=12, fontweight='bold')
        ax.set_title('Key Performance Metrics by Stage', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#FAFAFA')

        # ── Row 2: Learning rate ──
        ax = axes[2]
        offset = 0
        for stage_key in stage_order:
            if stage_key not in stage_metrics:
                continue
            sm = stage_metrics[stage_key]
            lr_vals = sm.get('lr', [])
            if not lr_vals:
                offset += len(sm.get('loss', []))
                continue

            color = self.STAGE_COLORS.get(stage_display.get(stage_key, ''), '#888888')
            x = np.arange(offset, offset + len(lr_vals))
            ax.plot(x, lr_vals, color=color, linewidth=1.5)
            offset += len(sm.get('loss', lr_vals))

        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_xlabel('Global Training Step', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=12)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#FAFAFA')

        plt.tight_layout()

        if save:
            path = _viz_save_path(self.output_dir, "training_timeline", "training_timeline.png")
            fig.savefig(path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

    def plot_capability_radar(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        stage2_5_results: Optional[Dict] = None,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Radar chart showing model capabilities after each stage.

        Dimensions: Vision Alignment, Language Quality, Hallucination Control,
                    Robot Selection, Reasoning.
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#FAFAFA')

        dimensions = ['Vision\nAlignment', 'Language\nQuality', 'Hallucination\nControl',
                      'Robot\nSelection', 'Reasoning\nQuality']
        n_dims = len(dimensions)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        # Compute normalized scores per stage (0-100)
        stage_scores = {}

        # After Stage 1
        s1 = stage_metrics.get('stage1', {})
        if s1:
            i2t = s1.get('acc_i2t', s1.get('val_i2t_acc', [0]))
            final_i2t = i2t[-1] if i2t else 0
            stage_scores['After Stage 1'] = [
                min(final_i2t * 100, 100),  # Vision Alignment
                10,                          # Language Quality (not yet trained)
                50,                          # Hallucination Control (baseline)
                0,                           # Robot Selection (not yet)
                0,                           # Reasoning (not yet)
            ]

        # After Stage 2
        s2 = stage_metrics.get('stage2', {})
        if s2:
            val_acc = s2.get('val_accuracy', [0])
            final_acc = val_acc[-1] if val_acc else 0
            val_ppl = s2.get('val_perplexity', [100])
            final_ppl = val_ppl[-1] if val_ppl else 100
            lang_score = max(0, 100 - final_ppl * 2) if final_ppl < 50 else max(0, 50 - (final_ppl - 50))
            stage_scores['After Stage 2'] = [
                stage_scores.get('After Stage 1', [50])[0],  # Preserve S1
                min(final_acc * 100, 100),                     # Language Quality
                45,                                            # Hallucination (pre-VA)
                0,
                0,
            ]

        # After Stage 2.5 (hallucination assessment)
        if stage2_5_results:
            overall = stage2_5_results.get('scores', {}).get('overall', 0)
            vision = stage2_5_results.get('scores', {}).get('vision_language', 0)
            text = stage2_5_results.get('scores', {}).get('text_only', 0)
            prev = stage_scores.get('After Stage 2', [50, 50, 45, 0, 0])
            halluc_control = max(overall, 50)  # Higher coherence = better halluc control
            stage_scores['After Stage 2.5'] = [
                prev[0],
                prev[1],
                min(halluc_control, 100),
                0,
                0,
            ]

        # After Stage 3
        s3 = stage_metrics.get('stage3', {})
        if s3:
            robot_acc = s3.get('robot_accuracy', s3.get('accuracy', [0]))
            final_robot = robot_acc[-1] if robot_acc else 0
            macro_f1 = s3.get('macro_f1', [0])
            final_f1 = macro_f1[-1] if macro_f1 else 0
            prev = list(stage_scores.values())[-1] if stage_scores else [50, 50, 50, 0, 0]
            stage_scores['After Stage 3'] = [
                prev[0], prev[1], prev[2],
                min(final_robot * 100, 100) if final_robot <= 1 else min(final_robot, 100),
                0,
            ]

        # After Stage 4
        s4 = stage_metrics.get('stage4', {})
        if s4:
            robot_acc = s4.get('robot_accuracy', [0])
            final_robot4 = robot_acc[-1] if robot_acc else 0
            prev = list(stage_scores.values())[-1] if stage_scores else [50, 50, 50, 50, 0]
            # Reasoning quality from coherence scores
            coherence = s4.get('plan_coherence', [0])
            final_coh = coherence[-1] if coherence else 0.3
            stage_scores['After Stage 4'] = [
                prev[0], prev[1], prev[2],
                max(prev[3], min(final_robot4 * 100, 100) if final_robot4 <= 1 else min(final_robot4, 100)),
                min(final_coh * 100, 100) if final_coh <= 1 else min(final_coh, 100),
            ]

        # Plot each stage as a polygon
        alpha_levels = np.linspace(0.15, 0.5, len(stage_scores))
        line_widths = np.linspace(1.0, 2.5, len(stage_scores))
        stage_color_map = list(self.STAGE_COLORS.values())

        for idx, (stage_label, scores) in enumerate(stage_scores.items()):
            values = scores + scores[:1]  # Close polygon
            color = stage_color_map[idx % len(stage_color_map)]
            ax.plot(angles, values, 'o-', linewidth=line_widths[idx], label=stage_label, color=color)
            ax.fill(angles, values, alpha=alpha_levels[idx], color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
        ax.set_title('Model Capability Evolution Across Training Stages',
                    fontsize=14, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = _viz_save_path(self.output_dir, "capability_radar", "capability_radar.png")
            fig.savefig(path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

    def plot_hallucination_journey(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        stage2_5_results: Optional[Dict] = None,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Detailed visualization of hallucination control throughout training.

        Shows:
          - Left: Hallucination proxy metrics per stage (vision grounding loss,
            coherence failure rate, response quality)
          - Right: Before/After VA assessment comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor('#FAFAFA')

        # ── Left: Hallucination indicators across stages ──
        ax = axes[0]
        stages = []
        halluc_indicators = []
        indicator_labels = []

        # Stage 1: no explicit hallucination metric, use loss as proxy
        s1 = stage_metrics.get('stage1', {})
        if s1:
            losses = s1.get('loss', [])
            if losses:
                # Higher initial loss = more random/hallucinated outputs
                initial_loss = losses[0] if losses else 1.0
                final_loss = losses[-1] if losses else 1.0
                stages.append('Stage 1\n(Alignment)')
                halluc_indicators.append(min(final_loss / max(initial_loss, 0.01), 1.0) * 100)
                indicator_labels.append(f'Loss ratio: {final_loss:.2f}/{initial_loss:.2f}')

        # Stage 2: use vision_grounding_loss as hallucination proxy
        s2 = stage_metrics.get('stage2', {})
        if s2:
            vgl = s2.get('vision_grounding_loss', s2.get('rep_penalty', []))
            val_ppl = s2.get('val_perplexity', [])
            if val_ppl:
                # Perplexity > threshold indicates hallucination tendency
                final_ppl = val_ppl[-1] if val_ppl else 50
                halluc_rate = min(final_ppl / 100, 1.0) * 100
            elif vgl:
                halluc_rate = min(vgl[-1] * 100, 100) if vgl[-1] <= 1 else min(vgl[-1], 100)
            else:
                halluc_rate = 60  # Default estimate
            stages.append('Stage 2\n(Instruction)')
            halluc_indicators.append(halluc_rate)
            indicator_labels.append(f'Perplexity proxy: {halluc_rate:.0f}')

        # Stage 2.5: actual hallucination measurement
        if stage2_5_results:
            cats = stage2_5_results.get('coherence_results', {}).get('categories', {})
            vision_cats = ['image_description', 'image_understanding', 'object_recognition', 'color_recognition']
            vision_passed = sum(cats.get(c, {}).get('passed', 0) for c in vision_cats)
            vision_total = sum(cats.get(c, {}).get('total', 1) for c in vision_cats)
            halluc_rate = (1 - vision_passed / max(vision_total, 1)) * 100
            stages.append('Stage 2.5\n(VA Assessment)')
            halluc_indicators.append(halluc_rate)
            indicator_labels.append(f'Vision fail: {vision_total - vision_passed}/{vision_total}')

        # Stage 3: robot accuracy improvement = less hallucination in selection
        s3 = stage_metrics.get('stage3', {})
        if s3:
            robot_acc = s3.get('robot_accuracy', s3.get('accuracy', []))
            if robot_acc:
                # Inverse of accuracy = "wrong selection" proxy
                final_acc = robot_acc[-1] if robot_acc else 0
                if final_acc <= 1:
                    final_acc *= 100
                halluc_rate = 100 - final_acc
                stages.append('Stage 3\n(Robot Select)')
                halluc_indicators.append(halluc_rate)
                indicator_labels.append(f'Selection error: {halluc_rate:.0f}%')

        # Stage 4: reasoning quality as hallucination proxy
        s4 = stage_metrics.get('stage4', {})
        if s4:
            coh = s4.get('plan_coherence', s4.get('val_plan_coherence', []))
            if coh:
                final_coh = coh[-1] if coh else 0.5
                halluc_rate = (1 - final_coh) * 100 if final_coh <= 1 else max(0, 100 - final_coh)
            else:
                robot_acc = s4.get('robot_accuracy', [])
                final_acc = robot_acc[-1] if robot_acc else 0
                if final_acc <= 1:
                    final_acc *= 100
                halluc_rate = max(0, 100 - final_acc)
            stages.append('Stage 4\n(Reasoning)')
            halluc_indicators.append(halluc_rate)
            indicator_labels.append(f'Incoherence: {halluc_rate:.0f}%')

        if stages:
            colors = []
            for r in halluc_indicators:
                if r >= 60:
                    colors.append('#F44336')
                elif r >= 35:
                    colors.append('#FF9800')
                elif r >= 15:
                    colors.append('#FFC107')
                else:
                    colors.append('#4CAF50')

            bars = ax.bar(stages, halluc_indicators, color=colors, edgecolor='white', linewidth=2, width=0.6)

            # Trend line
            x_num = np.arange(len(stages))
            if len(x_num) >= 2:
                z = np.polyfit(x_num, halluc_indicators, 1)
                p = np.poly1d(z)
                ax.plot(x_num, p(x_num), '--', color='#333333', linewidth=2, alpha=0.6, label=f'Trend (slope={z[0]:.1f})')
                ax.legend(fontsize=10, loc='upper right')

            for bar, rate, lbl in zip(bars, halluc_indicators, indicator_labels):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_ylabel('Hallucination Indicator (%)', fontsize=12, fontweight='bold')
            ax.set_title('Hallucination Reduction Journey', fontsize=13, fontweight='bold')
            ax.set_ylim(0, max(halluc_indicators) * 1.25 if halluc_indicators else 100)
            ax.grid(axis='y', alpha=0.3)
            ax.set_facecolor('#FAFAFA')

        # ── Right: Stage 2.5 detailed category breakdown ──
        ax = axes[1]
        if stage2_5_results:
            cats = stage2_5_results.get('coherence_results', {}).get('categories', {})
            if cats:
                cat_names = list(cats.keys())
                pass_rates = [(cats[c].get('passed', 0) / max(cats[c].get('total', 1), 1)) * 100 for c in cat_names]
                fail_rates = [100 - r for r in pass_rates]

                y_pos = np.arange(len(cat_names))
                ax.barh(y_pos, pass_rates, color='#4CAF50', label='Coherent', height=0.5, edgecolor='white')
                ax.barh(y_pos, fail_rates, left=pass_rates, color='#F44336', label='Hallucinated', height=0.5, edgecolor='white')

                ax.set_yticks(y_pos)
                ax.set_yticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=10)
                ax.set_xlabel('Percentage (%)', fontsize=11)
                ax.legend(fontsize=10, loc='lower right')
                ax.set_xlim(0, 100)

                # Text-only vs Vision split annotation
                text_cats = ['math_basic', 'common_knowledge', 'completion', 'instruction_following', 'listing', 'reasoning']
                for i, c in enumerate(cat_names):
                    tag = '📝' if c in text_cats else '👁️'
                    ax.text(-3, i, tag, ha='right', va='center', fontsize=12)

                ax.set_title('Stage 2.5: Per-Category Coherence', fontsize=13, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Stage 2.5 evaluation\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#999999')
            ax.set_title('Stage 2.5 Category Breakdown')

        ax.set_facecolor('#FAFAFA')

        fig.suptitle('EmberVLM — Hallucination Control Throughout Training',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            path = _viz_save_path(self.output_dir, "hallucination_journey", "hallucination_journey.png")
            fig.savefig(path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

    def plot_learning_dynamics_heatmap(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Heatmap showing normalized metric values across training, giving an
        at-a-glance view of when the model improves on different objectives.

        Y-axis: Metric categories (loss, alignment accuracy, language quality,
                robot accuracy, reasoning coherence)
        X-axis: Training progress (binned into ~20 columns)
        Color: Normalized metric value (0=worst, 1=best)
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('#FAFAFA')

        # Collect all metric time-series, normalize to [0, 1]
        metric_rows = []
        row_labels = []
        stage_boundaries = []

        col_offset = 0
        n_bins = 40  # Total columns in the heatmap

        stage_order = ['stage1', 'stage2', 'stage3', 'stage4']
        stage_display = {'stage1': 'S1', 'stage2': 'S2', 'stage3': 'S3', 'stage4': 'S4'}
        metric_configs = {
            'stage1': [('loss', 'S1: Alignment Loss', True), ('acc_i2t', 'S1: I2T Accuracy', False)],
            'stage2': [('loss', 'S2: Instruction Loss', True), ('val_accuracy', 'S2: Token Accuracy', False)],
            'stage3': [('robot_loss', 'S3: Robot Loss', True), ('robot_accuracy', 'S3: Robot Accuracy', False)],
            'stage4': [('loss', 'S4: Reasoning Loss', True), ('robot_accuracy', 'S4: Robot Accuracy', False)],
        }

        # Calculate how many bins each stage gets
        total_steps = sum(len(stage_metrics.get(s, {}).get('loss', [])) for s in stage_order)
        if total_steps == 0:
            ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig, _fig_to_pil(fig)

        # Bin each metric into the heatmap
        all_rows = {}
        for stage_key in stage_order:
            sm = stage_metrics.get(stage_key, {})
            if not sm:
                continue
            stage_steps = len(sm.get('loss', []))
            stage_bins = max(1, int(n_bins * stage_steps / total_steps))

            for metric_key, label, invert in metric_configs.get(stage_key, []):
                vals = sm.get(metric_key, [])
                if not vals:
                    continue

                # Bin by averaging
                binned = []
                bin_size = max(1, len(vals) // stage_bins)
                for i in range(0, len(vals), bin_size):
                    chunk = vals[i:i + bin_size]
                    binned.append(np.mean(chunk))

                # Normalize to [0, 1] — for loss, invert so lower = better = 1.0
                arr = np.array(binned)
                if len(arr) > 1 and arr.max() != arr.min():
                    normalized = (arr - arr.min()) / (arr.max() - arr.min())
                    if invert:
                        normalized = 1.0 - normalized
                else:
                    normalized = np.ones_like(arr) * 0.5

                if label not in all_rows:
                    all_rows[label] = np.full(n_bins, np.nan)

                # Place in the right columns
                start_col = col_offset if label.startswith(stage_display.get(stage_key, '')) or label not in all_rows else 0
                # Reset: always use proper offset
                s_offset = sum(
                    max(1, int(n_bins * len(stage_metrics.get(sk, {}).get('loss', [])) / total_steps))
                    for sk in stage_order[:stage_order.index(stage_key)]
                    if sk in stage_metrics
                )
                for j, v in enumerate(normalized):
                    col = s_offset + j
                    if col < n_bins:
                        all_rows[label][col] = v

        if not all_rows:
            ax.text(0.5, 0.5, 'No metrics to display', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig, _fig_to_pil(fig)

        labels = list(all_rows.keys())
        matrix = np.array([all_rows[l] for l in labels])

        # Use masked array for NaNs
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color='#E0E0E0')

        im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Training Progress →', fontsize=12, fontweight='bold')
        ax.set_title('Learning Dynamics Heatmap (Green = Better)', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Normalized Performance (1.0 = Best)', fontsize=10)

        # Add stage boundary lines
        s_offset = 0
        for sk in stage_order:
            sm = stage_metrics.get(sk, {})
            if not sm:
                continue
            stage_steps = len(sm.get('loss', []))
            stage_bins = max(1, int(n_bins * stage_steps / total_steps))
            if s_offset > 0:
                ax.axvline(s_offset - 0.5, color='white', linewidth=2)
            mid = s_offset + stage_bins / 2
            ax.text(mid, -0.8, stage_display.get(sk, sk), ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=self.STAGE_COLORS.get(f'Stage {sk[-1]}', '#333'))
            s_offset += stage_bins

        ax.set_facecolor('#E0E0E0')
        plt.tight_layout()

        if save:
            path = _viz_save_path(self.output_dir, "learning_dynamics_heatmap", "learning_dynamics_heatmap.png")
            fig.savefig(path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

    def plot_achievement_summary(
        self,
        stage_metrics: Dict[str, Dict[str, Any]],
        stage2_5_results: Optional[Dict] = None,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Final achievement card — a visually striking summary of what the model
        learned, showing initial vs final performance for each stage.
        """
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#1A1A2E')

        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

        stage_configs = [
            ('stage1', 'Stage 1: Vision Alignment', 'loss', 'Alignment Loss', True, gs[0, 0]),
            ('stage2', 'Stage 2: Instruction Tuning', 'val_accuracy', 'Token Accuracy', False, gs[0, 1]),
            ('stage3', 'Stage 3: Robot Selection', 'robot_accuracy', 'Robot Accuracy', False, gs[0, 2]),
            ('stage4', 'Stage 4: CoT Reasoning', 'robot_accuracy', 'Robot Accuracy', False, gs[1, 0]),
        ]

        for stage_key, title, metric_key, metric_label, invert, gs_pos in stage_configs:
            ax = fig.add_subplot(gs_pos)
            ax.set_facecolor('#16213E')
            sm = stage_metrics.get(stage_key, {})
            vals = sm.get(metric_key, sm.get('loss', []))

            if not vals:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='#888888')
                ax.set_title(title, fontsize=11, color='white', fontweight='bold')
                ax.tick_params(colors='#888888')
                continue

            initial = vals[0]
            final = vals[-1]
            improvement = ((initial - final) / max(abs(initial), 1e-10)) * 100 if invert else ((final - initial) / max(abs(initial), 1e-10)) * 100

            color = self.STAGE_COLORS.get(title.split(':')[0].strip(), '#42A5F5')
            x = np.arange(len(vals))
            ax.plot(x, vals, color=color, linewidth=2)
            ax.fill_between(x, vals, alpha=0.2, color=color)
            ax.scatter([0, len(vals) - 1], [initial, final], color='white', s=60, zorder=5, edgecolor=color, linewidth=2)

            ax.set_title(title, fontsize=11, color='white', fontweight='bold')
            ax.tick_params(colors='#888888', labelsize=8)
            ax.grid(True, alpha=0.1, color='white')

            # Improvement badge
            badge_color = '#4CAF50' if improvement > 0 else '#F44336'
            ax.text(0.97, 0.95, f'{improvement:+.1f}%', transform=ax.transAxes,
                   ha='right', va='top', fontsize=14, fontweight='bold', color=badge_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0F3460', edgecolor=badge_color, linewidth=2))

            # Initial → Final
            ax.text(0.03, 0.05, f'{initial:.4f} → {final:.4f}', transform=ax.transAxes,
                   ha='left', va='bottom', fontsize=9, color='#BBBBBB')

        # Stage 2.5 summary card
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor('#16213E')
        if stage2_5_results:
            scores = stage2_5_results.get('scores', {})
            overall = scores.get('overall', 0)
            text = scores.get('text_only', 0)
            vision = scores.get('vision_language', 0)
            passed = stage2_5_results.get('quality_check_passed', False)

            ax.text(0.5, 0.85, 'Stage 2.5: Hallucination Assessment',
                   ha='center', va='top', transform=ax.transAxes, fontsize=11, fontweight='bold', color='white')

            # Big score
            badge_color = '#4CAF50' if passed else '#F44336'
            ax.text(0.5, 0.55, f'{overall:.0f}%', ha='center', va='center', transform=ax.transAxes,
                   fontsize=36, fontweight='bold', color=badge_color)
            ax.text(0.5, 0.35, '✅ PASSED' if passed else '❌ FAILED',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14, color=badge_color)
            ax.text(0.5, 0.15, f'Text: {text:.0f}%  |  Vision: {vision:.0f}%',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#BBBBBB')
        else:
            ax.text(0.5, 0.5, 'No Stage 2.5 Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#888888')
        ax.axis('off')

        # Model info card
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor('#16213E')
        ax.text(0.5, 0.9, 'EmberVLM', ha='center', va='top', transform=ax.transAxes,
               fontsize=18, fontweight='bold', color='#FF9800')
        ax.text(0.5, 0.7, 'Training Complete', ha='center', va='center', transform=ax.transAxes,
               fontsize=13, color='#4CAF50', fontweight='bold')

        # Count total steps
        total_steps = sum(len(stage_metrics.get(s, {}).get('loss', [])) for s in ['stage1', 'stage2', 'stage3', 'stage4'])
        info_text = f'Total training steps: {total_steps}\nStages completed: {sum(1 for s in ["stage1","stage2","stage3","stage4"] if s in stage_metrics)}/4'
        ax.text(0.5, 0.4, info_text, ha='center', va='center', transform=ax.transAxes,
               fontsize=10, color='#BBBBBB', linespacing=1.8)
        ax.axis('off')

        fig.suptitle('EmberVLM — Training Achievement Summary',
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            path = _viz_save_path(self.output_dir, "achievement_summary", "achievement_summary.png")
            fig.savefig(path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())

        return fig, _fig_to_pil(fig)

