"""
Attention Visualization for EmberVLM

LeJePA-inspired visualization tools for understanding
vision-language attention patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention patterns in EmberVLM.

    Provides:
    - Cross-modal attention heatmaps (image-text)
    - Self-attention patterns
    - Feature space projections
    - Attention rollout for deep networks
    """

    def __init__(self, model, image_size: int = 224, num_visual_tokens: int = 8):
        self.model = model
        self.image_size = image_size
        self.num_visual_tokens = num_visual_tokens

        self._attention_hooks = []
        self._attention_maps = {}

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def make_hook(name):
            def hook(module, input, output):
                # Capture attention weights if available
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1] if len(output) > 1 else None
                    if attn_weights is not None:
                        self._attention_maps[name] = attn_weights.detach().cpu()
            return hook

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower():
                hook = module.register_forward_hook(make_hook(name))
                self._attention_hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []
        self._attention_maps = {}

    @torch.no_grad()
    def extract_attention(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from model.

        Args:
            pixel_values: Input images [B, C, H, W]
            input_ids: Token IDs [B, seq_len]
            attention_mask: Optional attention mask

        Returns:
            Dictionary of attention maps
        """
        self._attention_maps = {}
        self._register_attention_hooks()

        try:
            # Forward pass with attention output
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            # Also capture from output if available
            if 'attentions' in outputs and outputs['attentions'] is not None:
                for i, attn in enumerate(outputs['attentions']):
                    if attn is not None:
                        self._attention_maps[f'layer_{i}'] = attn.detach().cpu()

        finally:
            self._remove_hooks()

        return self._attention_maps

    def visualize_cross_attention(
        self,
        attention_map: torch.Tensor,
        tokens: List[str],
        image: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Visualize cross-attention between image and text.

        Args:
            attention_map: Attention weights [num_heads, query_len, key_len]
            tokens: Text tokens
            image: Original image (optional)
            output_path: Path to save visualization

        Returns:
            Matplotlib figure if available
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return None

        # Average over heads
        if attention_map.dim() == 4:
            attention_map = attention_map[0]  # Remove batch
        attn_avg = attention_map.mean(dim=0).numpy()  # Average over heads

        # Separate visual and text attention
        num_visual = self.num_visual_tokens

        if attn_avg.shape[1] > num_visual:
            # Attention from text to visual tokens
            text_to_visual = attn_avg[num_visual:, :num_visual]
            # Attention from visual to text tokens
            visual_to_text = attn_avg[:num_visual, num_visual:]
        else:
            text_to_visual = attn_avg
            visual_to_text = attn_avg.T

        # Create visualization
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 2])

        # Original image
        if image is not None:
            ax0 = fig.add_subplot(gs[0])
            ax0.imshow(image)
            ax0.set_title('Input Image')
            ax0.axis('off')

        # Text-to-visual attention
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(text_to_visual, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Visual Tokens')
        ax1.set_ylabel('Text Tokens')
        ax1.set_title('Text → Visual Attention')

        if tokens and len(tokens) <= text_to_visual.shape[0]:
            ax1.set_yticks(range(min(len(tokens), text_to_visual.shape[0])))
            ax1.set_yticklabels(tokens[:text_to_visual.shape[0]], fontsize=8)

        plt.colorbar(im1, ax=ax1)

        # Visual-to-text attention
        ax2 = fig.add_subplot(gs[2])
        im2 = ax2.imshow(visual_to_text, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Text Tokens')
        ax2.set_ylabel('Visual Tokens')
        ax2.set_title('Visual → Text Attention')

        if tokens and len(tokens) <= visual_to_text.shape[1]:
            ax2.set_xticks(range(min(len(tokens), visual_to_text.shape[1])))
            ax2.set_xticklabels(tokens[:visual_to_text.shape[1]], rotation=45, ha='right', fontsize=8)

        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {output_path}")

        return fig

    def visualize_attention_on_image(
        self,
        attention_map: torch.Tensor,
        image: np.ndarray,
        token_index: int = 0,
        output_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Overlay attention weights on image.

        Args:
            attention_map: Attention weights
            image: Original image [H, W, C]
            token_index: Which text token to visualize
            output_path: Path to save visualization

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.ndimage import zoom
        except ImportError:
            logger.warning("matplotlib/scipy not available")
            return None

        # Get attention for specific token
        if attention_map.dim() == 4:
            attn = attention_map[0].mean(dim=0)  # [query, key]
        else:
            attn = attention_map.mean(dim=0) if attention_map.dim() == 3 else attention_map

        # Extract visual attention
        num_visual = self.num_visual_tokens
        visual_attn = attn[num_visual + token_index, :num_visual]

        # Reshape to spatial grid
        grid_size = int(np.sqrt(num_visual))
        if grid_size ** 2 != num_visual:
            grid_size = (2, 4)  # 8 tokens
        else:
            grid_size = (grid_size, grid_size)

        attn_grid = visual_attn.numpy().reshape(grid_size)

        # Upsample to image size
        scale_h = image.shape[0] / grid_size[0]
        scale_w = image.shape[1] / grid_size[1]
        attn_upsampled = zoom(attn_grid, (scale_h, scale_w), order=1)

        # Normalize
        attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention heatmap
        axes[1].imshow(attn_upsampled, cmap='jet')
        axes[1].set_title(f'Attention (Token {token_index})')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attn_upsampled, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig

    def attention_rollout(
        self,
        attention_maps: List[torch.Tensor],
        head_fusion: str = 'mean',
        discard_ratio: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute attention rollout across layers.

        Args:
            attention_maps: List of attention maps per layer
            head_fusion: How to combine heads ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attention to discard

        Returns:
            Rolled-out attention map
        """
        result = None

        for attention in attention_maps:
            if attention is None:
                continue

            # Fuse heads
            if head_fusion == 'mean':
                attention_fused = attention.mean(dim=1)
            elif head_fusion == 'max':
                attention_fused = attention.max(dim=1)[0]
            elif head_fusion == 'min':
                attention_fused = attention.min(dim=1)[0]
            else:
                attention_fused = attention.mean(dim=1)

            # Add identity (residual)
            batch_size, seq_len, _ = attention_fused.shape
            flat = attention_fused.view(batch_size, -1)

            # Discard low attention
            if discard_ratio > 0:
                threshold = flat.quantile(discard_ratio, dim=-1, keepdim=True)
                flat = flat.masked_fill(flat < threshold, 0)

            attention_fused = flat.view(batch_size, seq_len, seq_len)

            # Add identity
            I = torch.eye(seq_len, device=attention_fused.device)
            attention_fused = 0.5 * attention_fused + 0.5 * I

            # Normalize
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Multiply with previous result
            if result is None:
                result = attention_fused
            else:
                result = torch.bmm(attention_fused, result)

        return result

    def visualize_feature_space(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = 'tsne',
        output_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Visualize feature space using dimensionality reduction.

        Args:
            visual_features: Visual features [N_v, D]
            text_features: Text features [N_t, D]
            labels: Optional labels for points
            method: Reduction method ('tsne', 'pca', 'umap')
            output_path: Path to save visualization

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("sklearn/matplotlib not available")
            return None

        # Combine features
        visual_np = visual_features.cpu().numpy().reshape(-1, visual_features.shape[-1])
        text_np = text_features.cpu().numpy().reshape(-1, text_features.shape[-1])

        all_features = np.concatenate([visual_np, text_np], axis=0)

        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
        else:
            reducer = PCA(n_components=2)

        reduced = reducer.fit_transform(all_features)

        # Split back
        visual_reduced = reduced[:len(visual_np)]
        text_reduced = reduced[len(visual_np):]

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(visual_reduced[:, 0], visual_reduced[:, 1],
                  c='blue', alpha=0.6, label='Visual', s=100, marker='o')
        ax.scatter(text_reduced[:, 0], text_reduced[:, 1],
                  c='red', alpha=0.6, label='Text', s=100, marker='^')

        if labels:
            for i, label in enumerate(labels[:len(text_reduced)]):
                ax.annotate(label, (text_reduced[i, 0], text_reduced[i, 1]),
                           fontsize=8, alpha=0.7)

        ax.legend()
        ax.set_title(f'Feature Space ({method.upper()})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


def create_attention_video(
    visualizer: AttentionVisualizer,
    model,
    pixel_values: torch.Tensor,
    tokens: List[str],
    output_path: str,
    fps: int = 2,
):
    """
    Create video showing attention evolution across tokens.

    Args:
        visualizer: AttentionVisualizer instance
        model: EmberVLM model
        pixel_values: Input image
        tokens: Text tokens
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        logger.warning("matplotlib not available for video creation")
        return

    # This is a placeholder for video creation
    # Full implementation would create animated attention maps
    logger.info(f"Video creation would save to {output_path}")


def analyze_model_attention(
    model,
    dataloader,
    output_dir: str,
    num_samples: int = 10,
):
    """
    Analyze attention patterns on dataset samples.

    Args:
        model: EmberVLM model
        dataloader: Data loader
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = AttentionVisualizer(model)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']

        # Extract attention
        attention_maps = visualizer.extract_attention(pixel_values, input_ids)

        # Save visualizations
        for layer_name, attn_map in attention_maps.items():
            output_path = output_dir / f'sample_{i}_{layer_name}.png'
            visualizer.visualize_cross_attention(
                attn_map,
                tokens=[f'tok_{j}' for j in range(input_ids.shape[1])],
                output_path=str(output_path),
            )

    logger.info(f"Saved {num_samples} attention visualizations to {output_dir}")

