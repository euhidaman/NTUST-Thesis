"""
Multi-Scale Vision Feature Extraction

Extracts and fuses features from multiple layers of the vision encoder
to capture both low-level details and high-level semantics.

Inspired by:
- DeepStack (Qwen3-VL): Multi-level ViT feature fusion
- FPN (Feature Pyramid Networks): Multi-scale feature extraction
- LLaVA-Next: Anyres with multi-scale processing

Key benefits:
- Richer visual representations without extra parameters
- Early layers: edges, textures, fine details
- Late layers: objects, scenes, high-level concepts
- Lightweight fusion: Simple attention-based combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MultiScaleFusion(nn.Module):
    """
    Lightweight fusion module for combining multi-scale vision features.
    
    Uses learnable attention weights to combine features from different layers.
    Minimal parameters (~1K params for 4 scales).
    
    Args:
        num_scales: Number of feature scales to combine (default: 4)
        feature_dim: Dimension of features after projection (default: 384)
    """
    
    def __init__(self, num_scales: int = 4, feature_dim: int = 384):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # Learnable attention weights for each scale
        # Start with equal weights [0.25, 0.25, 0.25, 0.25]
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )
        
        # Optional: Layer-specific projections if features have different dims
        # (Usually not needed if we extract after same projection layer)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_scales)
        ])
        
        logger.info(f"✓ MultiScaleFusion initialized (scales={num_scales}, dim={feature_dim})")
        logger.info(f"   Added {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count number of parameters in fusion module."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features using learned attention weights.
        
        Args:
            features_list: List of features from different layers
                Each tensor shape: [B, num_tokens, feature_dim]
        
        Returns:
            Fused features: [B, num_tokens, feature_dim]
        """
        assert len(features_list) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(features_list)}"
        
        # Normalize each feature map
        normalized_features = [
            self.layer_norms[i](feat) for i, feat in enumerate(features_list)
        ]
        
        # Apply softmax to scale weights for smooth combination
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Weighted sum of features
        fused = sum(w * feat for w, feat in zip(weights, normalized_features))
        
        return fused


class MultiScaleVisionExtractor:
    """
    Helper class to extract features from multiple layers of vision encoder.
    
    Modifies the forward pass to capture intermediate representations.
    Works with both RepViT and MobileViT backbones.
    
    Args:
        vision_encoder: The vision encoder module
        layer_indices: Which layers to extract (default: [3, 6, 9, -1])
                      -1 means the final layer
        enable_fusion: Whether to fuse multi-scale features (default: True)
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        layer_indices: List[int] = None,
        enable_fusion: bool = True,
    ):
        self.vision_encoder = vision_encoder
        self.layer_indices = layer_indices or [3, 6, 9, -1]  # Early, mid, late, final
        self.enable_fusion = enable_fusion
        self.feature_dim = vision_encoder.output_dim
        
        # Create fusion module if enabled
        if self.enable_fusion:
            self.fusion = MultiScaleFusion(
                num_scales=len(self.layer_indices),
                feature_dim=self.feature_dim
            )
        else:
            self.fusion = None
        
        logger.info(f"✓ MultiScaleVisionExtractor initialized")
        logger.info(f"   Extracting from layers: {self.layer_indices}")
        logger.info(f"   Fusion enabled: {enable_fusion}")
    
    def extract_multi_scale_features(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers of vision encoder.
        
        Args:
            pixel_values: Input images [B, C, H, W]
        
        Returns:
            Dictionary containing:
                - visual_tokens: Final visual embeddings (fused if enabled)
                - pooled_output: Global pooled features
                - multi_scale_features: List of features from each layer (if not fused)
        """
        # Check if backbone supports output_hidden_states
        has_hidden_states = hasattr(self.vision_encoder.backbone, 'output_hidden_states')
        
        if has_hidden_states:
            # Use native output_hidden_states if available (MobileViT)
            with torch.set_grad_enabled(not self.vision_encoder.backbone.training):
                outputs = self.vision_encoder.backbone(
                    pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get all hidden states
            all_hidden_states = outputs.hidden_states
            
            # Select specific layers
            selected_features = []
            for idx in self.layer_indices:
                if idx == -1:
                    feat = all_hidden_states[-1]
                else:
                    feat = all_hidden_states[idx]
                selected_features.append(feat)
        
        else:
            # Fallback: Extract final features only (RepViT doesn't expose intermediate)
            # For RepViT, we'll just use the final output multiple times as approximation
            with torch.set_grad_enabled(not self.vision_encoder.backbone.training):
                final_features = self.vision_encoder.backbone.forward_features(pixel_values)
            
            # Approximate multi-scale by using same features
            # (Better than nothing, still gets fusion parameters)
            selected_features = [final_features] * len(self.layer_indices)
        
        # Process each feature map through vision encoder pipeline
        processed_features = []
        for features in selected_features:
            # features shape: [B, C, H, W]
            # Adaptive pooling
            pooled = self.vision_encoder.adaptive_pool(features)
            
            # Reshape to tokens
            B, C, h, w = pooled.shape
            tokens = pooled.permute(0, 2, 3, 1).reshape(B, h * w, C)
            
            # Layer norm + projection
            tokens = self.vision_encoder.ln_vision(tokens)
            tokens = self.vision_encoder.projection(tokens)
            
            processed_features.append(tokens)
        
        # Fuse if enabled
        if self.enable_fusion and self.fusion is not None:
            visual_tokens = self.fusion(processed_features)
        else:
            # Use final layer only
            visual_tokens = processed_features[-1]
        
        # Global pooled output (from final features)
        pooled_output = F.adaptive_avg_pool2d(selected_features[-1], 1).flatten(1)
        pooled_output = self.vision_encoder.projection[0](pooled_output)
        
        return {
            'visual_tokens': visual_tokens,
            'pooled_output': pooled_output,
            'multi_scale_features': processed_features if not self.enable_fusion else None,
        }
