"""
Fusion Module for EmberVLM

Bridges vision encoder features to language model space using
adapter blocks with bottleneck design for parameter efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class AdapterBlock(nn.Module):
    """
    Adapter block with bottleneck design.

    Inspired by TinyGPT-V but scaled down 8x for efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.

        Args:
            x: Input tensor [B, seq_len, input_dim]

        Returns:
            Output tensor [B, seq_len, output_dim]
        """
        residual = x if x.size(-1) == self.up_proj.out_features else None

        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.layer_norm(x)

        if residual is not None:
            x = x + residual

        return x


class QKNormFusion(nn.Module):
    """
    Query-Key normalization layer for stable fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=eps)
        self.norm_k = nn.LayerNorm(dim, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.norm_q(query), self.norm_k(key)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer for vision-language fusion.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        # QK normalization
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_norm = QKNormFusion(self.head_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.

        Args:
            query: Query tensor [B, query_len, query_dim]
            key: Key tensor [B, key_len, key_dim]
            value: Value tensor (default: same as key)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        if value is None:
            value = key

        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        output = self.out_proj(output)
        output = self.proj_dropout(output)

        return output, attn_weights


class FusionModule(nn.Module):
    """
    Fusion Module for EmberVLM.

    Maps vision encoder features to language model space using adapter blocks
    with QK-normalization for training stability.

    Includes a learnable gate (α) that controls how much vision information
    flows into the language model. This prevents the LM from being overwhelmed
    by vision features early in training.

    Architecture:
        RepViT_Features(8×384) → Linear(384→384) → LayerNorm →
        AdapterBlock(bottleneck=48) → Gated Output → TinyLLM_Input(8×384)
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        bottleneck_dim: int = 48,
        num_visual_tokens: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_qk_norm: bool = True,
        use_cross_attention: bool = False,
        num_cross_attention_heads: int = 6,  # Match tinyllm/30M-0.4 heads
        use_vision_gate: bool = True,  # Enable gated fusion
        initial_gate_value: float = 2.0,  # Start with gate OPEN (sigmoid(2)≈0.88) so vision contributes
        use_visual_self_attention: bool = True,
        num_visual_attention_heads: int = 6,
        use_image_summary: bool = True,
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_visual_tokens = num_visual_tokens
        self.use_cross_attention = use_cross_attention
        self.use_vision_gate = use_vision_gate
        self.use_visual_self_attention = use_visual_self_attention
        self.use_image_summary = use_image_summary

        # Initial projection
        self.vision_proj = nn.Linear(vision_dim, language_dim)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.pre_norm = nn.LayerNorm(language_dim)

        # Adapter block with larger bottleneck for DINOv2 alignment
        # Use language_dim // 4 (e.g. 144 for 576, 96 for 384) for better
        # capacity while staying compact.  The constructor arg is still
        # accepted but overridden here so that every config gets the
        # improved bottleneck.
        effective_bottleneck = int(language_dim // 4)
        self.adapter = AdapterBlock(
            input_dim=language_dim,
            bottleneck_dim=effective_bottleneck,
            output_dim=language_dim,
            dropout=dropout,
        )

        if self.use_visual_self_attention:
            self.visual_attn_norm = nn.LayerNorm(language_dim)
            self.visual_attn = nn.MultiheadAttention(
                language_dim,
                num_visual_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.visual_mlp_norm = nn.LayerNorm(language_dim)
            self.visual_mlp = nn.Sequential(
                nn.Linear(language_dim, 4 * language_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * language_dim, language_dim),
                nn.Dropout(dropout),
            )

        if self.use_image_summary:
            self.summary_token = nn.Parameter(torch.zeros(1, 1, language_dim))
            self.summary_norm = nn.LayerNorm(language_dim)
            self.summary_attn = nn.MultiheadAttention(
                language_dim,
                num_visual_attention_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Optional cross-attention
        if use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                query_dim=language_dim,
                key_dim=language_dim,
                hidden_dim=language_dim,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
            )

        # Output layer norm
        self.output_norm = nn.LayerNorm(language_dim)

        # Learnable vision gate (α)
        # Gate controls: output = α * vision_features
        # Initialize to 0 (or small value) so vision influence starts minimal
        # and model learns to use it gradually
        if use_vision_gate:
            # sigmoid(2.5) ≈ 0.924 — slightly stronger initial visual influence
            # than the previous default of 2.0 (sigmoid(2) ≈ 0.88).
            initial_gate_value = 2.5
            self.vision_gate = nn.Parameter(torch.tensor(initial_gate_value))
        else:
            self.vision_gate = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers."""
        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.zeros_(self.vision_proj.bias)
        if self.use_image_summary:
            nn.init.normal_(self.summary_token, mean=0.0, std=0.02)

    def get_gate_value(self) -> float:
        """Get current gate value (for logging/monitoring)."""
        if self.vision_gate is not None:
            # Apply sigmoid to get value in [0, 1]
            return torch.sigmoid(self.vision_gate).item()
        return 1.0

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion module.

        Args:
            visual_features: Vision encoder output [B, num_visual_tokens, vision_dim]
            text_features: Optional text features for cross-attention [B, text_len, language_dim]
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing:
                - fused_features: Fused visual features [B, num_visual_tokens, language_dim]
                - attention_weights: Optional attention weights
                - gate_value: Current gate value (for monitoring)
        """
        import logging
        logger = logging.getLogger(__name__)

        # DIAGNOSTIC: Log input visual features variance (should be different for different images)
        if not hasattr(self, '_fusion_logged'):
            input_mean = visual_features.mean().item()
            input_std = visual_features.std().item()
            input_var = visual_features.var().item()
            logger.info(f"[Fusion] INPUT visual_features: shape={visual_features.shape}, "
                       f"mean={input_mean:.4f}, std={input_std:.4f}, var={input_var:.4f}")

        # Project vision features to language space
        fused = self.vision_proj(visual_features)

        if not hasattr(self, '_fusion_logged'):
            proj_mean = fused.mean().item()
            proj_std = fused.std().item()
            logger.info(f"[Fusion] AFTER vision_proj: mean={proj_mean:.4f}, std={proj_std:.4f}")

        # Apply pre-normalization
        if self.use_layer_norm:
            fused = self.pre_norm(fused)
            if not hasattr(self, '_fusion_logged'):
                norm_mean = fused.mean().item()
                norm_std = fused.std().item()
                logger.info(f"[Fusion] AFTER pre_norm: mean={norm_mean:.4f}, std={norm_std:.4f}")

        # Apply adapter
        fused = self.adapter(fused)

        if not hasattr(self, '_fusion_logged'):
            adapt_mean = fused.mean().item()
            adapt_std = fused.std().item()
            logger.info(f"[Fusion] AFTER adapter: mean={adapt_mean:.4f}, std={adapt_std:.4f}")

        if self.use_visual_self_attention:
            attn_input = self.visual_attn_norm(fused)
            attn_out, _ = self.visual_attn(attn_input, attn_input, attn_input)
            fused = fused + attn_out
            mlp_input = self.visual_mlp_norm(fused)
            fused = fused + self.visual_mlp(mlp_input)

        if self.use_image_summary:
            summary_query = self.summary_norm(self.summary_token).expand(fused.size(0), -1, -1)
            summary_out, _ = self.summary_attn(summary_query, fused, fused)
            fused = fused + summary_out.expand(-1, fused.size(1), -1)

        # Optional cross-attention with text features
        attention_weights = None
        if self.use_cross_attention and text_features is not None:
            cross_out, attention_weights = self.cross_attention(
                query=fused,
                key=text_features,
                value=text_features,
                attention_mask=attention_mask,
            )
            fused = fused + cross_out

        # Output normalization
        fused = self.output_norm(fused)

        # Apply vision gate
        # Use sigmoid to bound gate in [0, 1]
        gate_value = 1.0
        if self.vision_gate is not None:
            gate_value = torch.sigmoid(self.vision_gate)
            fused = gate_value * fused

            if not hasattr(self, '_fusion_logged'):
                logger.info(f"[Fusion] Gate value: {gate_value.item():.4f} (raw param={self.vision_gate.item():.4f})")
                final_mean = fused.mean().item()
                final_std = fused.std().item()
                logger.info(f"[Fusion] FINAL output: mean={final_mean:.4f}, std={final_std:.4f}")
                self._fusion_logged = True

        return {
            'fused_features': fused,
            'attention_weights': attention_weights,
            'gate_value': gate_value if isinstance(gate_value, float) else gate_value.item(),
        }

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.language_dim

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
        }


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion for handling different visual feature resolutions.
    """

    def __init__(
        self,
        vision_dims: list = [128, 256, 384],
        language_dim: int = 768,
        bottleneck_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, language_dim),
                nn.LayerNorm(language_dim),
                nn.GELU(),
            )
            for dim in vision_dims
        ])

        self.fusion = nn.Sequential(
            nn.Linear(language_dim * len(vision_dims), language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.adapter = AdapterBlock(
            input_dim=language_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=language_dim,
            dropout=dropout,
        )

    def forward(
        self,
        multi_scale_features: list,
    ) -> torch.Tensor:
        """
        Fuse multi-scale visual features.

        Args:
            multi_scale_features: List of features at different scales

        Returns:
            Fused features [B, seq_len, language_dim]
        """
        projected = []
        for feat, proj in zip(multi_scale_features, self.projections):
            projected.append(proj(feat))

        # Concatenate along feature dimension
        concat = torch.cat(projected, dim=-1)
        fused = self.fusion(concat)
        fused = self.adapter(fused)

        return fused


class VisualProjector(nn.Module):
    """
    Simple visual projector for baseline comparison.
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim, language_dim),
            nn.LayerNorm(language_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_features)


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer with cross-attention to vision features.

    Architecture (inspired by BLIP-2):
    1. Self-attention on queries
    2. Cross-attention from queries to vision tokens
    3. FFN
    """

    def __init__(
        self,
        hidden_dim: int = 576,
        num_heads: int = 8,
        ffn_dim: int = 2304,  # 4x hidden_dim
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention on queries
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to vision tokens
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vision_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        vision_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through Q-Former layer.

        Args:
            queries: [B, num_queries, hidden_dim]
            vision_tokens: [B, num_vision_tokens, hidden_dim]

        Returns:
            Updated queries: [B, num_queries, hidden_dim]
        """
        # Self-attention
        q_norm = self.self_attn_norm(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self.dropout(self_attn_out)

        # Cross-attention to vision tokens
        # CRITICAL: This is where we extract visual information
        q_norm = self.cross_attn_norm(queries)
        v_norm = self.vision_norm(vision_tokens)
        cross_attn_out, cross_attn_weights = self.cross_attn(q_norm, v_norm, v_norm, need_weights=True)
        queries = queries + self.dropout(cross_attn_out)

        # Store attention weights for monitoring (optional)
        # High attention entropy = queries attending broadly
        # Low attention entropy = queries collapsing to same positions
        self._last_cross_attn_weights = cross_attn_weights

        # FFN
        q_norm = self.ffn_norm(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out

        return queries


class QFormerFusion(nn.Module):
    """
    Q-Former Fusion Module for EmberVLM (BLIP-2 style).

    Uses learnable query tokens that cross-attend to vision features,
    compressing variable-length vision tokens into fixed-length queries
    suitable for the language model.

    Key advantages over simple projection:
    1. Learns WHAT to extract from vision (via cross-attention)
    2. Compresses 256 vision tokens → 32 queries (8x compression)
    3. Preserves semantic relationships through multi-layer processing
    4. Each query can specialize (objects, colors, spatial, etc.)

    Architecture:
        Vision tokens [B, 256, 384]
              ↓
        Project to hidden_dim [B, 256, 576]
              ↓
        Learnable queries [B, 32, 576] cross-attend to vision
              ↓
        6 Q-Former layers
              ↓
        Output queries [B, 32, 576] → Language model
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 576,
        num_query_tokens: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers

        # Project vision tokens to language dimension if needed
        if vision_dim != language_dim:
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.LayerNorm(language_dim),
            )
        else:
            self.vision_proj = nn.LayerNorm(language_dim)

        # Learnable query tokens (the key innovation of Q-Former)
        # These learn to extract relevant information from vision tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, language_dim)
        )

        # Q-Former layers
        ffn_dim = language_dim * 4
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_dim=language_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(language_dim)

        # Initialize
        self._init_weights()

        # Log architecture
        import logging
        logger = logging.getLogger(__name__)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"QFormerFusion: {num_query_tokens} queries, {num_layers} layers, "
            f"{total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)"
        )

    def _init_weights(self):
        """Initialize weights."""
        # Initialize query tokens from normal distribution
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        # Initialize projection
        for module in self.vision_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Q-Former.

        Args:
            visual_features: Vision encoder output [B, num_vision_tokens, vision_dim]
            text_features: Optional (not used in Q-Former, kept for compatibility)
            attention_mask: Optional (not used)

        Returns:
            Dictionary containing:
                - fused_features: [B, num_query_tokens, language_dim]
                - attention_weights: None (could add for visualization)
                - gate_value: 1.0 (no gating in Q-Former)
        """
        batch_size = visual_features.size(0)

        # Project vision tokens to language dimension
        vision_tokens = self.vision_proj(visual_features)

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, vision_tokens)

        # Output normalization
        output = self.output_norm(queries)

        return {
            'fused_features': output,
            'attention_weights': None,
            'gate_value': 1.0,
        }

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.language_dim

    def get_num_query_tokens(self) -> int:
        """Return number of query tokens."""
        return self.num_query_tokens

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
        }

    def get_attention_entropy(self) -> float:
        """
        Compute average attention entropy across all Q-Former layers.

        High entropy (close to log(num_vision_tokens)) = queries attending broadly
        Low entropy (close to 0) = attention collapsed to few tokens

        This is useful for diagnosing degenerate Q-Former training.
        """
        entropies = []
        for layer in self.layers:
            if hasattr(layer, '_last_cross_attn_weights') and layer._last_cross_attn_weights is not None:
                # weights shape: [B, num_queries, num_vision_tokens]
                weights = layer._last_cross_attn_weights
                # Compute entropy: -sum(p * log(p))
                # Add small epsilon to avoid log(0)
                eps = 1e-8
                entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
                entropies.append(entropy.item())

        if entropies:
            return sum(entropies) / len(entropies)
        return 0.0

    def get_attention_sparsity(self) -> float:
        """
        Compute attention sparsity (how concentrated attention is).

        Returns ratio of max attention weight to uniform attention.
        High values (>>1) = attention very concentrated
        Values near 1 = attention is uniform/spread out
        """
        sparsities = []
        for layer in self.layers:
            if hasattr(layer, '_last_cross_attn_weights') and layer._last_cross_attn_weights is not None:
                weights = layer._last_cross_attn_weights
                # Max attention weight per query
                max_weights = weights.max(dim=-1)[0]  # [B, num_queries]
                # Uniform would be 1/num_vision_tokens
                uniform = 1.0 / weights.size(-1)
                sparsity = (max_weights / uniform).mean()
                sparsities.append(sparsity.item())

        if sparsities:
            return sum(sparsities) / len(sparsities)
        return 0.0


class CrossModalFusion(nn.Module):
    """
    Cross-Modal Fusion Module for bidirectional vision-text interaction.

    Instead of simple concatenation [vision | text], this module allows:
    1. Text tokens to attend to vision queries (text sees what's in image)
    2. Vision queries to attend to text tokens (vision sees the question/context)
    3. Gated residual connections to preserve original representations

    This creates richer representations BEFORE they enter the frozen LM.

    Architecture:
        Text Embeds [B, T, D]     Vision Queries [B, 32, D]
              │                          │
              ▼                          ▼
        ┌─────────────────────────────────────┐
        │     Bidirectional Cross-Attention    │
        │  text→vision: What's in the image?  │
        │  vision→text: What's the question?  │
        └─────────────────────────────────────┘
              │                          │
              ▼                          ▼
        ┌─────────┐                ┌─────────┐
        │  Gate   │                │  Gate   │
        │ (α·new) │                │ (β·new) │
        │+(1-α)old│                │+(1-β)old│
        └─────────┘                └─────────┘
              │                          │
              ▼                          ▼
        Enhanced Text            Enhanced Vision
              │                          │
              └──────────┬───────────────┘
                         ▼
                  Concatenate & Return
    """

    def __init__(
        self,
        hidden_dim: int = 576,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_cross_layers: int = 2,
        use_gating: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_cross_layers = num_cross_layers
        self.use_gating = use_gating

        # Cross-attention layers: text attends to vision
        self.text_to_vision_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_cross_layers)
        ])

        # Cross-attention layers: vision attends to text
        self.vision_to_text_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_cross_layers)
        ])

        # Learnable gates for residual blending
        if use_gating:
            # CRITICAL FIX: Initialize gates to ~0.6 so vision contributes from the start
            # Previous value (~0.12) was too small, causing vision to be ignored
            # sigmoid(0.5) ≈ 0.62 allows vision to influence outputs while still being learnable
            self.text_gate = nn.Parameter(torch.tensor(0.5))  # sigmoid(0.5) ≈ 0.62
            self.vision_gate = nn.Parameter(torch.tensor(0.5))  # sigmoid(0.5) ≈ 0.62

        self._init_weights()

        # Log architecture
        import logging
        logger = logging.getLogger(__name__)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CrossModalFusion: {num_cross_layers} layers, {num_heads} heads, "
            f"{total_params/1e6:.2f}M params"
        )

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        vision_queries: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply bidirectional cross-modal fusion.

        Args:
            vision_queries: Q-Former output [B, num_queries, hidden_dim]
            text_embeds: Text embeddings [B, seq_len, hidden_dim]
            text_attention_mask: Optional mask for text [B, seq_len]

        Returns:
            Dictionary with:
                - enhanced_vision: [B, num_queries, hidden_dim]
                - enhanced_text: [B, seq_len, hidden_dim]
                - text_gate_value: current text gate value
                - vision_gate_value: current vision gate value
        """
        # Store originals for residual
        original_text = text_embeds
        original_vision = vision_queries

        # Create key_padding_mask for attention if provided
        # MultiheadAttention expects: True = ignore, False = attend
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = (text_attention_mask == 0)  # Invert: 0 means padding

        # Apply cross-attention layers
        for t2v_layer, v2t_layer in zip(self.text_to_vision_layers, self.vision_to_text_layers):
            # Text attends to vision: "What objects/colors/relations are in the image?"
            text_normed = t2v_layer['norm'](text_embeds)
            cross_out, _ = t2v_layer['cross_attn'](
                query=text_normed,
                key=vision_queries,
                value=vision_queries,
            )
            text_embeds = text_embeds + cross_out
            text_embeds = text_embeds + t2v_layer['ffn'](t2v_layer['ffn_norm'](text_embeds))

            # Vision attends to text: "What is the question asking about?"
            vision_normed = v2t_layer['norm'](vision_queries)
            cross_out, _ = v2t_layer['cross_attn'](
                query=vision_normed,
                key=text_embeds,
                value=text_embeds,
                key_padding_mask=key_padding_mask,
            )
            vision_queries = vision_queries + cross_out
            vision_queries = vision_queries + v2t_layer['ffn'](v2t_layer['ffn_norm'](vision_queries))

        # Apply gated residual: blend enhanced with original
        text_gate_value = 1.0
        vision_gate_value = 1.0

        if self.use_gating:
            text_gate_value = torch.sigmoid(self.text_gate)
            vision_gate_value = torch.sigmoid(self.vision_gate)

            # Blend: gate * enhanced + (1 - gate) * original
            text_embeds = text_gate_value * text_embeds + (1 - text_gate_value) * original_text
            vision_queries = vision_gate_value * vision_queries + (1 - vision_gate_value) * original_vision

        return {
            'enhanced_vision': vision_queries,
            'enhanced_text': text_embeds,
            'text_gate_value': text_gate_value.item() if isinstance(text_gate_value, torch.Tensor) else text_gate_value,
            'vision_gate_value': vision_gate_value.item() if isinstance(vision_gate_value, torch.Tensor) else vision_gate_value,
        }
