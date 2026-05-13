"""
EmberVLM - Complete Model Implementation

A lightweight multimodal Vision-Language Model combining RepViT vision encoder
and TinyLLM language backbone for robot fleet selection with incident reasoning.

Uses pretrained models:
- Vision: RepViT-XXS from THU-MIG/RepViT (HuggingFace) [default]
- Vision Alternative: MobileViT-XS from timm (~2.3M params)
- Language: tinyllm/30M-0.4 from HuggingFace (GPT-2 style) [default]
- Language Alternatives: SmolLM-135M, SmolLM-360M from HuggingFace
"""

import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from embervlm.models.vision_encoder import (
    RepViTEncoder,
    MobileViTEncoder,
    ImagePreprocessor,
    create_vision_encoder,
    VISION_BACKBONE_REPVIT,
    VISION_BACKBONE_MOBILEVIT_XS,
)
from embervlm.models.language_model import (
    TinyLLMBackbone,
    TinyLLMConfig,
    PretrainedTinyLLMBackbone,
    SmolLMBackbone,
    create_language_backbone,
    PRETRAINED_TINYLLM_MODEL,
    PRETRAINED_SMOLLM_135M,
    BACKBONE_TINYLLM,
    BACKBONE_SMOLLM_135M,
)
from embervlm.models.fusion_module import FusionModule
from embervlm.models.reasoning_heads import ReasoningModule, ReasoningLoss
from embervlm.models.episodic_memory import EpisodicMemoryController, ScopeDetector


@dataclass
class EmberVLMConfig:
    """Configuration for EmberVLM model."""

    # Backbone selection
    vision_backbone: str = "repvit"  # Options: 'repvit', 'mobilevit_xs'
    language_backbone: str = "tinyllm"  # Options: 'tinyllm', 'smollm_135m'

    # Vision encoder
    vision_model: str = "repvit_m0_9"  # Using timm model: repvit_m0_9.dist_450e_in1k
    vision_pretrained: bool = True
    freeze_vision: bool = True
    num_visual_tokens: int = 8
    vision_output_dim: int = 384
    image_size: int = 224

    # Language model (tinyllm/30M-0.4 defaults)
    language_hidden_size: int = 384  # tinyllm/30M-0.4 uses 384
    language_num_layers: int = 6
    language_num_heads: int = 6  # tinyllm/30M-0.4 uses 6 heads
    language_vocab_size: int = 50257
    language_max_length: int = 1024
    freeze_language_base: bool = True
    unfreeze_last_layer: bool = True

    # Pretrained model settings
    use_pretrained_language: bool = True
    pretrained_language_model: str = "tinyllm/30M-0.4"

    def __post_init__(self):
        """Auto-configure parameters based on backbone selection."""
        # Configure vision parameters based on backbone
        if self.vision_backbone == VISION_BACKBONE_MOBILEVIT_XS:
            # MobileViT-XS configuration
            self.vision_model = "apple/mobilevit-x-small"
            self.vision_output_dim = 384  # MobileViT-XS outputs 384-dim features
            self.num_visual_tokens = 8  # MobileViT-XS uses 8 tokens
        elif self.vision_backbone == VISION_BACKBONE_REPVIT:
            # RepViT configuration (default)
            self.vision_model = "repvit_m0_9"
            self.vision_output_dim = 384
            self.num_visual_tokens = 8  # RepViT uses 8 tokens
        elif self.vision_backbone == 'dinov2_small':  # DINOv2-Small
            # DINOv2-Small configuration
            self.vision_model = "facebook/dinov2-small"
            self.vision_output_dim = 384
            self.num_visual_tokens = 258  # CLS(1) + patch_avg(1) + 16x16 patches(256)

        # Configure language parameters based on backbone
        if self.language_backbone == BACKBONE_SMOLLM_135M:
            # SmolLM-135M configuration
            self.pretrained_language_model = PRETRAINED_SMOLLM_135M
            self.language_hidden_size = 576
            self.language_num_layers = 30
            self.language_num_heads = 9
            self.language_vocab_size = 49152
        elif self.language_backbone == BACKBONE_TINYLLM:
            # TinyLLM configuration (default)
            self.pretrained_language_model = PRETRAINED_TINYLLM_MODEL
            self.language_hidden_size = 384
            self.language_num_layers = 6
            self.language_num_heads = 6
            self.language_vocab_size = 50257

    # Fusion module
    fusion_bottleneck_dim: int = 48
    fusion_dropout: float = 0.1
    use_qk_norm: bool = True

    # Reasoning module
    reasoning_enabled: bool = True
    reasoning_hidden_dim: int = 192  # Reduced to match smaller language model
    reasoning_num_layers: int = 2
    reasoning_num_heads: int = 4
    num_reasoning_steps: int = 4
    max_plan_steps: int = 5

    # Robot fleet
    num_robots: int = 5
    robot_names: List[str] = field(default_factory=lambda: [
                                   "Drone", "Humanoid", "Wheeled", "Legged", "Underwater"])

    # Special tokens
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "reasoning_start": "<|reasoning_start|>",
        "reasoning_end": "<|reasoning_end|>",
        "robot_selection": "<|robot_selection|>",
        "action_plan": "<|action_plan|>",
        "image_start": "<image>",
        "image_end": "</image>",
        "question_start": "<question>",
        "question_end": "</question>",
        "answer_start": "<answer>",
    })

    # VA hallucination mitigation (off by default)
    use_va_refiner: bool = False
    va_p_threshold: float = 0.7
    va_layer_indices: List[int] = field(default_factory=lambda: [10, 15, 20, 25, 28])

    # Episodic memory (episode branch only)
    use_episodic_memory: bool = False
    memory_slots: int = 512
    memory_addressing: str = "gaussian"  # "gaussian" or "pseudoinverse"
    memory_alpha: float = 1.0
    memory_temperature: float = 0.1
    memory_variance: float = 1.0
    scope_detection_method: str = "internal"
    novelty_threshold_novel: float = 0.7
    novelty_threshold_similar: float = 0.2
    enable_memory_consolidation: bool = False

    # Training
    dropout: float = 0.1
    initializer_range: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        config_dict = {
            'vision_backbone': self.vision_backbone,
            'language_backbone': self.language_backbone,
            'vision_model': self.vision_model,
            'vision_pretrained': self.vision_pretrained,
            'freeze_vision': self.freeze_vision,
            'num_visual_tokens': self.num_visual_tokens,
            'vision_output_dim': self.vision_output_dim,
            'image_size': self.image_size,
            'language_hidden_size': self.language_hidden_size,
            'language_num_layers': self.language_num_layers,
            'language_num_heads': self.language_num_heads,
            'language_vocab_size': self.language_vocab_size,
            'language_max_length': self.language_max_length,
            'freeze_language_base': self.freeze_language_base,
            'unfreeze_last_layer': self.unfreeze_last_layer,
            'use_pretrained_language': self.use_pretrained_language,
            'pretrained_language_model': self.pretrained_language_model,
            'fusion_bottleneck_dim': self.fusion_bottleneck_dim,
            'fusion_dropout': self.fusion_dropout,
            'use_qk_norm': self.use_qk_norm,
            'reasoning_enabled': self.reasoning_enabled,
            'reasoning_hidden_dim': self.reasoning_hidden_dim,
            'reasoning_num_layers': self.reasoning_num_layers,
            'reasoning_num_heads': self.reasoning_num_heads,
            'num_reasoning_steps': self.num_reasoning_steps,
            'max_plan_steps': self.max_plan_steps,
            'num_robots': self.num_robots,
            'robot_names': self.robot_names,
            'special_tokens': self.special_tokens,
            'dropout': self.dropout,
            'initializer_range': self.initializer_range,
            'use_va_refiner': self.use_va_refiner,
            'va_p_threshold': self.va_p_threshold,
            'va_layer_indices': self.va_layer_indices,
            'use_episodic_memory': self.use_episodic_memory,
            'memory_slots': self.memory_slots,
            'memory_addressing': self.memory_addressing,
            'memory_alpha': self.memory_alpha,
            'memory_temperature': self.memory_temperature,
            'memory_variance': self.memory_variance,
            'scope_detection_method': self.scope_detection_method,
            'novelty_threshold_novel': self.novelty_threshold_novel,
            'novelty_threshold_similar': self.novelty_threshold_similar,
            'enable_memory_consolidation': self.enable_memory_consolidation,
        }

        # Add vocab_size alias for easier access (same as language_vocab_size)
        config_dict['vocab_size'] = self.language_vocab_size

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmberVLMConfig':
        """Create config from dictionary.

        IMPORTANT: The backbone selection fields (vision_backbone, language_backbone)
        drive __post_init__ to set the derived fields (vision_model, language_hidden_size, etc).

        We trust the backbone fields and let __post_init__ derive the rest.
        This ensures that even if config.json was saved with wrong derived values
        (before the __post_init__ fix), loading will work correctly.

        This method is designed to be EXTRA ROBUST and will never fail due to
        logging errors or missing fields - it will always return a valid config.
        """
        # Create local logger VERY CAREFULLY to avoid any import/logging failures
        try:
            import logging
            _log = logging.getLogger(__name__)
        except:
            _log = None

        def _safe_log(msg, level='info'):
            """Safe logging that never raises exceptions"""
            try:
                if _log is not None:
                    getattr(_log, level)(msg)
            except:
                pass  # Silently ignore logging failures

        try:
            # Filter to only valid fields
            valid_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}

            # Extract backbone selection fields - these drive __post_init__
            backbone_fields = {
                'vision_backbone': valid_fields.get('vision_backbone', 'repvit'),
                'language_backbone': valid_fields.get('language_backbone', 'tinyllm'),
            }

            _safe_log(f"[from_dict] Creating config with vision={backbone_fields['vision_backbone']}, language={backbone_fields['language_backbone']}")

            # Create instance with backbone fields only
            # __post_init__ will set all derived fields correctly
            config = cls(**backbone_fields)

            # Get the default vocab size for this backbone (set by __post_init__)
            default_vocab_size = config.language_vocab_size

            # Now override with saved values, but SKIP fields that __post_init__ derives
            # EXCEPT language_vocab_size if it was explicitly resized (differs from default)
            derived_fields = {
                'vision_model', 'vision_output_dim', 'num_visual_tokens',
                'language_hidden_size', 'language_num_layers', 'language_num_heads',
                'pretrained_language_model',
            }

            for key, value in valid_fields.items():
                # Skip backbone fields (already set) and derived fields (set by __post_init__)
                if key in backbone_fields or key in derived_fields:
                    continue

                # Special case: language_vocab_size (or vocab_size alias) can be overridden
                # This handles cases where embeddings were resized (e.g., 49152 -> 49157)
                if key == 'language_vocab_size' or key == 'vocab_size':
                    saved_vocab_size = config_dict.get('language_vocab_size') or config_dict.get('vocab_size')
                    if saved_vocab_size and saved_vocab_size != default_vocab_size:
                        _safe_log(f"[from_dict] Overriding vocab_size: {default_vocab_size} -> {saved_vocab_size} (embeddings were resized)")
                        setattr(config, 'language_vocab_size', saved_vocab_size)
                    continue
                setattr(config, key, value)

            return config

        except Exception as e:
            # CRITICAL: If anything fails, at least return a config with the right backbones
            _safe_log(f"[from_dict] WARNING: Exception during config parsing: {e}. Using minimal config.", 'warning')

            # Extract backbones safely
            vision_backbone = config_dict.get('vision_backbone', 'repvit')
            language_backbone = config_dict.get('language_backbone', 'tinyllm')

            # Create minimal valid config
            return cls(vision_backbone=vision_backbone, language_backbone=language_backbone)


class EmberVLM(nn.Module):
    """
    EmberVLM - Tiny Multimodal VLM for Robot Fleet Selection.

    Architecture:
    - Vision Encoder: RepViT-XXS (frozen, ~5M params)
    - Language Model: TinyLLM-30M (last layer trainable)
    - Fusion Module: Adapter-based fusion (~1M params)
    - Reasoning Module: CoT generation heads (~3M params)

    Total: ~35M parameters (~5M trainable)
    """

    def __init__(self, config: Optional[EmberVLMConfig] = None):
        super().__init__()

        if config is None:
            config = EmberVLMConfig()

        self.config = config

        # Initialize components
        self._build_vision_encoder()
        self._build_language_model()
        self._build_fusion_module()

        if config.reasoning_enabled:
            self._build_reasoning_module()

        # Image preprocessor
        self.image_preprocessor = ImagePreprocessor(
            image_size=config.image_size)

        # Loss function
        self.reasoning_loss = ReasoningLoss(
            num_robots=config.num_robots,
        )

        # Special token IDs (will be set when tokenizer is loaded)
        self.special_token_ids = {}

        # VA hallucination refiner (lazy-initialised on first use)
        self._va_refiner: Optional["VARefiner"] = None  # noqa: F821

        # Episodic memory (episode branch)
        self.episodic_memory: Optional[EpisodicMemoryController] = None
        self.scope_detector: Optional[ScopeDetector] = None
        self._edge_runtime = None  # EdgeMemoryRuntime (set via prepare_for_edge)
        if config.use_episodic_memory:
            self.episodic_memory = EpisodicMemoryController(
                memory_slots=config.memory_slots,
                hidden_dim=config.language_hidden_size,
                addressing=config.memory_addressing,
                alpha=config.memory_alpha,
                variance=config.memory_variance,
                temperature=config.memory_temperature,
                novelty_threshold_novel=config.novelty_threshold_novel,
                novelty_threshold_similar=config.novelty_threshold_similar,
            )
            self.scope_detector = ScopeDetector(
                input_dim=config.language_hidden_size,
                method=config.scope_detection_method,
            )

    # ------------------------------------------------------------------
    # Edge deployment helpers
    # ------------------------------------------------------------------

    def prepare_for_inference(self):
        """Pre-warm episodic memory addressing tables for fast inference.

        Call once after loading a checkpoint and before running generate().
        This precomputes Gaussian norms / pseudoinverse caches so the first
        inference call doesn't pay the setup cost.
        """
        if self.episodic_memory is not None:
            self.episodic_memory.prepare_for_inference()

    def prepare_for_edge(
        self,
        quantize: bool = False,
        cache_addressing: bool = True,
        scope_fast_path: bool = True,
    ):
        """Convert episodic memory to edge-optimised runtime.

        After calling this, the ``forward()`` method will use
        ``EdgeMemoryRuntime`` instead of the full controller — stripping
        training-only state and using precomputed addressing tables with
        optional int8 quantisation.

        Args:
            quantize: Use int8 memory matrix (saves ~4× VRAM).
            cache_addressing: Precompute addressing tables.
            scope_fast_path: Enable running-mean scope detector gate.
        """
        if self.episodic_memory is None:
            logger.warning("prepare_for_edge: no episodic memory configured")
            return

        from embervlm.models.edge_memory import EdgeMemoryRuntime, EdgeMemoryConfig

        cfg = EdgeMemoryConfig(
            quantize=quantize,
            cache_addressing=cache_addressing,
            scope_fast_path=scope_fast_path,
        )
        self._edge_runtime = EdgeMemoryRuntime.from_trained(
            controller=self.episodic_memory,
            scope_detector=self.scope_detector,
            config=cfg,
        )
        self._edge_runtime = self._edge_runtime.to(self.device)
        logger.info("EmberVLM: edge memory runtime active")

    def benchmark_memory_latency(self, **kwargs):
        """Run latency benchmark on the episodic memory subsystem.

        See ``edge_memory.benchmark_edge_latency`` for full parameter docs.
        Returns a dict of latency measurements.
        """
        from embervlm.models.edge_memory import benchmark_edge_latency, format_benchmark_report

        runtime = self._edge_runtime
        if runtime is None:
            # Build a temporary edge runtime for benchmarking
            if self.episodic_memory is None:
                raise RuntimeError("No episodic memory configured")
            self.prepare_for_edge(quantize=kwargs.pop("quantize", False))
            runtime = self._edge_runtime

        lm_head = None
        if hasattr(self.language_model, 'get_output_embeddings'):
            lm_head = self.language_model.get_output_embeddings()

        results = benchmark_edge_latency(
            runtime,
            device=str(self.device),
            include_lm_head=lm_head is not None,
            lm_head=lm_head,
            **kwargs,
        )
        logger.info("\n" + format_benchmark_report(results))
        return results

    # ------------------------------------------------------------------
    # VA Refiner  (lazy property – avoids overhead when disabled)
    # ------------------------------------------------------------------
    @property
    def va_refiner(self):
        """Return a ready VARefiner, creating it on first access."""
        if self._va_refiner is None:
            from embervlm.models.hallucination import VARefiner, VAConfig
            va_cfg = VAConfig(
                hidden_dim=self.config.language_hidden_size,
                layer_indices=self.config.va_layer_indices,
                p_threshold=self.config.va_p_threshold,
            )
            self._va_refiner = VARefiner(self.language_model, va_cfg)
            # Move to same device as the model
            self._va_refiner.feature_extractor.to(self.device)
            self._va_refiner.classifier.to(self.device)
        return self._va_refiner

    def sync_tokenizer_and_embeddings(
        self,
        tokenizer,
        add_special_tokens: bool = True,
        force_resize: bool = False,
        logger: Optional[Any] = None,
    ) -> int:
        """
        Ensure tokenizer special tokens and model embeddings are consistent.

        Returns the final vocab size used by the model.
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)

        if tokenizer is None:
            return self.config.language_vocab_size

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Ensure special tokens are present in tokenizer
        if add_special_tokens and hasattr(self.config, 'special_tokens'):
            desired_tokens = list(self.config.special_tokens.values())
            existing = tokenizer.additional_special_tokens or []
            new_tokens = [t for t in desired_tokens if t not in existing]
            if new_tokens:
                tokenizer.add_special_tokens({'additional_special_tokens': existing + new_tokens})

        # Compute required vocab size (must cover special IDs too)
        special_ids = [
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            getattr(tokenizer, 'bos_token_id', None),
        ]
        max_special_id = max([i for i in special_ids if i is not None] + [-1])
        required_vocab_size = max(len(tokenizer), max_special_id + 1)

        # Determine current embedding size
        current_vocab_size = None
        if hasattr(self.language_model, 'get_input_embeddings'):
            try:
                current_vocab_size = self.language_model.get_input_embeddings().weight.shape[0]
            except Exception:
                current_vocab_size = None
        if current_vocab_size is None and hasattr(self.language_model, 'model'):
            if hasattr(self.language_model.model, 'get_input_embeddings'):
                try:
                    current_vocab_size = self.language_model.model.get_input_embeddings().weight.shape[0]
                except Exception:
                    current_vocab_size = None
        if current_vocab_size is None and hasattr(self.language_model, 'config'):
            current_vocab_size = getattr(self.language_model.config, 'vocab_size', None)

        # Resize embeddings if needed
        if current_vocab_size is None or force_resize or current_vocab_size != required_vocab_size:
            logger.warning(
                f"Token embedding size mismatch. Tokenizer: {required_vocab_size}, Model: {current_vocab_size}. Resizing."
            )
            try:
                if hasattr(self.language_model, 'resize_token_embeddings'):
                    self.language_model.resize_token_embeddings(required_vocab_size)
                elif hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'resize_token_embeddings'):
                    self.language_model.model.resize_token_embeddings(required_vocab_size)
            except Exception as e:
                logger.error(f"Failed to resize embeddings: {e}")
                raise

        # Update configs to be consistent
        self.config.language_vocab_size = required_vocab_size
        if hasattr(self.language_model, 'config'):
            self.language_model.config.vocab_size = required_vocab_size
        if hasattr(self.language_model, 'hf_config'):
            self.language_model.hf_config.vocab_size = required_vocab_size
        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'config'):
            self.language_model.model.config.vocab_size = required_vocab_size

        # Ensure special token IDs are within bounds
        safe_id = required_vocab_size - 1 if required_vocab_size > 0 else 0
        if tokenizer.eos_token_id is None or tokenizer.eos_token_id >= required_vocab_size:
            tokenizer.eos_token_id = safe_id
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= required_vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(tokenizer, 'bos_token_id', None) is not None and tokenizer.bos_token_id >= required_vocab_size:
            tokenizer.bos_token_id = safe_id

        # Cache special token IDs
        self.special_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v)
            for k, v in getattr(self.config, 'special_tokens', {}).items()
            if v is not None
        }

        return required_vocab_size

    def _build_vision_encoder(self):
        """Initialize vision encoder based on backbone selection."""
        # Use factory function for backbone selection
        self.vision_encoder = create_vision_encoder(
            backbone_type=self.config.vision_backbone,
            model_name=self.config.vision_model if self.config.vision_backbone == VISION_BACKBONE_REPVIT else None,
            pretrained=self.config.vision_pretrained,
            freeze=self.config.freeze_vision,
            num_visual_tokens=self.config.num_visual_tokens,
            output_dim=self.config.vision_output_dim,
            image_size=self.config.image_size,
        )

    def _build_language_model(self):
        """Initialize language model based on backbone selection."""
        if self.config.use_pretrained_language:
            # Determine model name from backbone type
            model_name = self.config.pretrained_language_model
            backbone_type = self.config.language_backbone

            # Override model_name if using SmolLM backbone
            if backbone_type == BACKBONE_SMOLLM_135M:
                model_name = PRETRAINED_SMOLLM_135M

            # Use factory function for backbone selection
            self.language_model = create_language_backbone(
                use_pretrained=True,
                model_name=model_name,
                backbone_type=backbone_type,
                freeze_base=self.config.freeze_language_base,
                unfreeze_last_layer=self.config.unfreeze_last_layer,
            )
            # Update config with actual model dimensions
            self.config.language_hidden_size = self.language_model.config.hidden_size
            self.config.language_num_layers = self.language_model.config.num_hidden_layers
            self.config.language_num_heads = self.language_model.config.num_attention_heads
            self.config.language_vocab_size = self.language_model.config.vocab_size
        else:
            # Create from scratch with custom config (only supports TinyLLM architecture)
            llm_config = TinyLLMConfig(
                vocab_size=self.config.language_vocab_size,
                hidden_size=self.config.language_hidden_size,
                num_hidden_layers=self.config.language_num_layers,
                num_attention_heads=self.config.language_num_heads,
                max_position_embeddings=self.config.language_max_length,
                hidden_dropout_prob=self.config.dropout,
                attention_probs_dropout_prob=self.config.dropout,
                use_qk_norm=self.config.use_qk_norm,
            )

            self.language_model = TinyLLMBackbone(
                config=llm_config,
                freeze_base=self.config.freeze_language_base,
                unfreeze_last_layer=self.config.unfreeze_last_layer,
            )

    def _build_fusion_module(self):
        """Initialize fusion module."""
        self.fusion_module = FusionModule(
            vision_dim=self.config.vision_output_dim,
            language_dim=self.config.language_hidden_size,
            bottleneck_dim=self.config.fusion_bottleneck_dim,
            num_visual_tokens=self.config.num_visual_tokens,
            dropout=self.config.fusion_dropout,
            use_qk_norm=self.config.use_qk_norm,
        )

    def _build_reasoning_module(self):
        """Initialize reasoning module."""
        self.reasoning_module = ReasoningModule(
            input_dim=self.config.language_hidden_size,
            hidden_dim=self.config.reasoning_hidden_dim,
            num_reasoning_layers=self.config.reasoning_num_layers,
            num_reasoning_heads=self.config.reasoning_num_heads,
            num_reasoning_steps=self.config.num_reasoning_steps,
            num_robots=self.config.num_robots,
            max_plan_steps=self.config.max_plan_steps,
            dropout=self.config.dropout,
        )

    def encode_image(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images to visual tokens.

        Args:
            pixel_values: Input images [B, C, H, W]

        Returns:
            Dictionary with visual tokens and pooled features
        """
        return self.vision_encoder(pixel_values)

    def fuse_features(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse visual tokens with language model space.

        Args:
            visual_tokens: Visual features [B, num_visual_tokens, vision_dim]
            text_embeds: Optional text embeddings for cross-attention

        Returns:
            Fused features in language model space
        """
        fusion_output = self.fusion_module(visual_tokens, text_embeds)
        return fusion_output['fused_features']

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input embeddings by merging text and visual tokens.

        Args:
            input_ids: Token IDs [B, seq_len]
            pixel_values: Optional images [B, C, H, W]
            image_positions: Positions where to insert image tokens [B]

        Returns:
            Tuple of (inputs_embeds, attention_mask)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # CRITICAL FIX: Use CPU round-trip for guaranteed safety against CUDA async issues
        vocab_size = self.language_model.get_input_embeddings().weight.shape[0]

        # Move to CPU, validate, clamp, then move back (fully synchronous)
        input_ids_cpu = input_ids.detach().cpu()
        max_val = input_ids_cpu.max().item()
        min_val = input_ids_cpu.min().item()

        if max_val >= vocab_size or min_val < 0:
            import logging
            logger = logging.getLogger(__name__)
            num_invalid = ((input_ids_cpu >= vocab_size) |
                           (input_ids_cpu < 0)).sum().item()
            logger.error(
                f"❌ prepare_inputs_embeds: {num_invalid} invalid token IDs! "
                f"Range: [{min_val}, {max_val}], Valid: [0, {vocab_size - 1}]. Clamping."
            )
            input_ids_cpu = torch.clamp(
                input_ids_cpu, min=0, max=vocab_size - 1)

        # Move back with blocking transfer
        input_ids = input_ids_cpu.to(device, non_blocking=False)

        # Sync before any embedding operations
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        # Get text embeddings - now safe to call
        text_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is None:
            # No images, return text embeddings directly
            pad_token_id = getattr(
                self.language_model.config, 'pad_token_id', 0) or 0
            attention_mask = (input_ids != pad_token_id).long()
            return text_embeds, attention_mask

        # Encode images
        vision_output = self.encode_image(pixel_values)
        visual_tokens = vision_output['visual_tokens']

        # Fuse visual tokens
        fused_visual = self.fuse_features(visual_tokens)
        num_visual = fused_visual.size(1)

        # CRITICAL: Validate vision token count to prevent index errors
        expected_num_visual = self.config.num_visual_tokens
        if num_visual != expected_num_visual:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"❌ VISION TOKEN MISMATCH: Got {num_visual}, expected {expected_num_visual}. "
                f"This will cause indexing errors!"
            )
            # Clamp to expected size to prevent crash
            if num_visual > expected_num_visual:
                fused_visual = fused_visual[:, :expected_num_visual, :]
                num_visual = expected_num_visual
            else:
                # Pad with zeros if too few
                padding = torch.zeros(
                    batch_size, expected_num_visual - num_visual,
                    fused_visual.size(2), dtype=fused_visual.dtype, device=device
                )
                fused_visual = torch.cat([fused_visual, padding], dim=1)
                num_visual = expected_num_visual

        # Determine image positions
        if image_positions is None:
            # Default: insert at beginning
            image_positions = torch.zeros(
                batch_size, dtype=torch.long, device=device)

        # CRITICAL: Validate image positions to prevent out-of-bounds indexing
        max_valid_pos = seq_len  # Can insert at any position in text sequence
        invalid_positions = (image_positions < 0) | (image_positions > max_valid_pos)
        if invalid_positions.any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"❌ INVALID IMAGE POSITIONS: min={image_positions.min().item()}, "
                f"max={image_positions.max().item()}, valid_range=[0, {max_valid_pos}]. "
                f"Clamping to prevent crash!"
            )
            image_positions = torch.clamp(image_positions, 0, max_valid_pos)

        # Create merged embeddings
        total_len = seq_len + num_visual
        inputs_embeds = torch.zeros(
            batch_size, total_len, self.config.language_hidden_size,
            dtype=text_embeds.dtype, device=device
        )
        attention_mask = torch.zeros(
            batch_size, total_len, dtype=torch.long, device=device)

        for i in range(batch_size):
            pos = image_positions[i].item()

            # CRITICAL: Assert bounds before any indexing
            assert 0 <= pos <= seq_len, \
                f"Image position {pos} out of bounds [0, {seq_len}]"
            assert pos + num_visual <= total_len, \
                f"Visual token range [{pos}, {pos + num_visual}) exceeds total_len {total_len}"
            assert num_visual == fused_visual.size(1), \
                f"num_visual {num_visual} != fused_visual.size(1) {fused_visual.size(1)}"

            # Insert visual tokens
            inputs_embeds[i, pos:pos + num_visual] = fused_visual[i]
            attention_mask[i, pos:pos + num_visual] = 1

            # Insert text before image
            if pos > 0:
                # Validate slice bounds
                assert pos <= text_embeds.size(1), \
                    f"Text slice end {pos} > text_embeds.size(1) {text_embeds.size(1)}"
                inputs_embeds[i, :pos] = text_embeds[i, :pos]
                attention_mask[i, :pos] = (
                    input_ids[i, :pos] != self.language_model.config.pad_token_id).long()

            # Insert text after image
            remaining = seq_len - pos
            text_start_idx = pos
            embed_start_idx = pos + num_visual

            # Validate text indexing bounds
            assert text_start_idx >= 0 and text_start_idx <= text_embeds.size(1), \
                f"Text start {text_start_idx} out of bounds [0, {text_embeds.size(1)}]"
            assert embed_start_idx + remaining <= total_len, \
                f"Embed range [{embed_start_idx}, {embed_start_idx + remaining}) exceeds {total_len}"

            inputs_embeds[i, embed_start_idx:embed_start_idx + remaining] = text_embeds[i, text_start_idx:]
            text_mask = (input_ids[i, text_start_idx:] !=
                         self.language_model.config.pad_token_id).long()
            attention_mask[i, embed_start_idx:embed_start_idx + remaining] = text_mask

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        robot_targets: Optional[torch.LongTensor] = None,
        action_targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_reasoning: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through EmberVLM.

        Args:
            input_ids: Token IDs [B, seq_len]
            pixel_values: Images [B, C, H, W]
            attention_mask: Attention mask [B, seq_len]
            labels: Target token IDs for LM loss
            image_positions: Where to insert image tokens
            robot_targets: Target robot indices for robot selection loss
            action_targets: Target action embeddings for action planning loss
            use_cache: Whether to cache key/values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_reasoning: Whether to return reasoning outputs

        Returns:
            Dictionary containing model outputs and losses
        """
        # Prepare inputs
        inputs_embeds, input_attention_mask = self.prepare_inputs_embeds(
            input_ids, pixel_values, image_positions
        )

        if attention_mask is not None:
            # Extend attention mask to include visual tokens
            num_visual = self.config.num_visual_tokens if pixel_values is not None else 0
            if attention_mask.size(1) < inputs_embeds.size(1):
                visual_mask = torch.ones(
                    attention_mask.size(0), num_visual,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat(
                    [visual_mask, attention_mask], dim=1)
        else:
            attention_mask = input_attention_mask

        # Adjust labels to match inputs_embeds length
        adjusted_labels = None
        if labels is not None:
            # CRITICAL SAFEGUARD: Validate labels before they are used
            # This prevents CUDA index out of bounds errors during loss computation
            vocab_size = self.language_model.get_input_embeddings(
            ).weight.shape[0]
            valid_labels_mask = labels != -100
            if valid_labels_mask.any():
                valid_labels = labels[valid_labels_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    if max_label >= vocab_size:
                        logger.error(
                            f"❌ CRITICAL: labels contain token ID {max_label} >= vocab_size {vocab_size}! "
                            f"Clamping to prevent CUDA crash."
                        )
                    if min_label < 0:
                        logger.error(
                            f"❌ CRITICAL: labels contain negative token ID {min_label}! "
                            f"Clamping to prevent CUDA crash."
                        )
                    # Clamp labels: preserve -100 for ignore, clamp everything else to valid range
                    labels = torch.where(
                        valid_labels_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )

            if labels.size(1) != inputs_embeds.size(1):
                # Labels need to be extended to match inputs_embeds
                batch_size = labels.size(0)
                num_visual = inputs_embeds.size(1) - labels.size(1)
                device = labels.device

                # Determine image positions
                if image_positions is None:
                    image_positions = torch.zeros(
                        batch_size, dtype=torch.long, device=device)

                # Create adjusted labels with -100 at visual token positions
                adjusted_labels = torch.full(
                    (batch_size, inputs_embeds.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )

                for i in range(batch_size):
                    pos = image_positions[i].item()
                    # Copy labels before image position
                    if pos > 0:
                        copy_len = min(pos, labels.size(1))
                        adjusted_labels[i, :copy_len] = labels[i, :copy_len]
                    # Visual tokens get -100 (already set)
                    # Copy labels after image position
                    start_pos = pos + num_visual
                    if start_pos < adjusted_labels.size(1) and pos < labels.size(1):
                        remaining = labels.size(1) - pos
                        end_pos = min(start_pos + remaining,
                                      adjusted_labels.size(1))
                        copy_len = end_pos - start_pos
                        adjusted_labels[i, start_pos:end_pos] = labels[i,
                                                                       pos:pos + copy_len]
            else:
                adjusted_labels = labels

        # CRITICAL: Validate attention_mask before forward pass
        # Attention mask should be 0 or 1, not used as indices
        if attention_mask is not None:
            if attention_mask.max() > 1 or attention_mask.min() < 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"❌ CRITICAL: attention_mask has invalid values! "
                    f"Range: [{attention_mask.min().item()}, {attention_mask.max().item()}]. "
                    f"Should be binary [0, 1]. Clamping!"
                )
                attention_mask = torch.clamp(attention_mask, 0, 1)

        # Forward through language model
        # Always get hidden states if reasoning is enabled and we need them
        need_hidden_states = output_hidden_states or (
            self.config.reasoning_enabled and (
                return_reasoning or robot_targets is not None)
        )

        lm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=adjusted_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=need_hidden_states,
        )

        outputs = {
            'logits': lm_outputs['logits'],
            'loss': lm_outputs.get('loss'),
            'hidden_states': lm_outputs.get('hidden_states'),
            'attentions': lm_outputs.get('attentions'),
            'past_key_values': lm_outputs.get('past_key_values'),
        }

        # --- Episodic memory conditioning (inference only) ---
        # Uses EdgeMemoryRuntime if available (precomputed tables, scope fast-path),
        # otherwise falls back to the standard controller path.
        if (self.config.use_episodic_memory
                and self.episodic_memory is not None
                and not self.training):
            _last_hidden = lm_outputs.get('last_hidden_state', None)
            if _last_hidden is None and 'hidden_states' in lm_outputs and lm_outputs['hidden_states']:
                _last_hidden = lm_outputs['hidden_states'][-1]
            if _last_hidden is not None:
                fused_repr = _last_hidden[:, -1, :]  # (B, C) last token

                # Fast path: use edge runtime if available
                if self._edge_runtime is not None:
                    lm_head = None
                    if hasattr(self.language_model, 'get_output_embeddings'):
                        lm_head = self.language_model.get_output_embeddings()
                    if lm_head is None and hasattr(self.language_model, 'model'):
                        lm_head = getattr(self.language_model.model, 'lm_head', None)
                    if lm_head is None:
                        lm_head = getattr(self.language_model, 'lm_head', None)

                    if lm_head is not None:
                        logit_bias = self._edge_runtime.condition_logits(fused_repr, lm_head)
                        if logit_bias is not None:
                            outputs['logits'] = outputs['logits'] + logit_bias.unsqueeze(1)
                else:
                    # Standard path (no edge runtime)
                    scope_probs = self.scope_detector(fused_repr)  # (B,)
                    scope_mask = scope_probs > 0.5

                    if scope_mask.any():
                        scoped_z = fused_repr[scope_mask]
                        memory_readout = self.episodic_memory.read(scoped_z)  # (B_scope, C)

                        full_readout = torch.zeros_like(fused_repr)
                        full_readout[scope_mask] = memory_readout

                        # Project readout through LM head to get logit-space bias
                        lm_head = None
                        if hasattr(self.language_model, 'get_output_embeddings'):
                            lm_head = self.language_model.get_output_embeddings()
                        if lm_head is None and hasattr(self.language_model, 'model'):
                            lm_head = getattr(self.language_model.model, 'lm_head', None)
                        if lm_head is None:
                            lm_head = getattr(self.language_model, 'lm_head', None)

                        if lm_head is not None:
                            readout_logits = lm_head(full_readout)  # (B, vocab)
                            outputs['logits'] = outputs['logits'] + readout_logits.unsqueeze(1)
                    outputs['episodic_scope_probs'] = scope_probs

        # Reasoning module
        if self.config.reasoning_enabled and (return_reasoning or robot_targets is not None):
            hidden_states = lm_outputs['last_hidden_state']

            reasoning_outputs = self.reasoning_module(
                hidden_states,
                attention_mask=attention_mask,
                generate_reasoning=return_reasoning,
                select_robot=True,
                plan_actions=True,
            )

            outputs.update({
                'robot_logits': reasoning_outputs.get('robot_logits'),
                'robot_probs': reasoning_outputs.get('robot_probs'),
                'robot_confidence': reasoning_outputs.get('robot_confidence'),
                'plan_steps': reasoning_outputs.get('plan_steps'),
                'plan_coherence': reasoning_outputs.get('plan_coherence'),
            })

            if return_reasoning and 'reasoning_chain' in reasoning_outputs:
                outputs['reasoning_chain'] = reasoning_outputs['reasoning_chain']

            # Compute reasoning losses
            if robot_targets is not None or action_targets is not None:
                targets = {}
                if robot_targets is not None:
                    targets['robot_target'] = robot_targets
                if action_targets is not None:
                    targets['action_target'] = action_targets

                reasoning_losses = self.reasoning_loss(
                    reasoning_outputs, targets)

                if outputs['loss'] is not None:
                    outputs['loss'] = outputs['loss'] + \
                        reasoning_losses['total_loss']
                else:
                    outputs['loss'] = reasoning_losses['total_loss']

                outputs['reasoning_losses'] = reasoning_losses

        return outputs

    def forward_vision_only(
        self,
        pixel_values: torch.Tensor,
        robot_targets: Optional[torch.LongTensor] = None,
        task_embeddings: Optional[torch.Tensor] = None,
        return_reasoning: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Vision-based forward pass for robot selection (Stage 3).

        This bypasses the language model's embedding lookup to avoid tokenization issues,
        but still supports reasoning based on visual understanding.

        The reasoning module will:
        1. Analyze visual features from the scene
        2. Apply chain-of-thought reasoning through ReasoningHead
        3. Select appropriate robot through RobotSelectionHead
        4. Generate action plan through ActionPlanningHead

        Args:
            pixel_values: Images [B, C, H, W]
            robot_targets: Target robot indices for classification loss
            task_embeddings: Optional pre-computed task embeddings [B, seq_len, hidden_dim]
                            (can be used to inject text context without tokenization)
            return_reasoning: Whether to return reasoning chain outputs

        Returns:
            Dictionary containing:
            - robot_logits: Raw classification scores [B, num_robots]
            - robot_probs: Softmax probabilities [B, num_robots]
            - robot_confidence: Confidence score [B, 1]
            - plan_steps: Action plan embeddings
            - plan_coherence: Plan quality score
            - reasoning_chain: (if return_reasoning=True) reasoning steps
            - loss: Classification loss (if robot_targets provided)
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. Encode images through vision encoder
        vision_output = self.encode_image(pixel_values)
        # [B, num_visual_tokens, vision_dim]
        visual_tokens = vision_output['visual_tokens']

        # 2. Project visual tokens to language model dimension through fusion
        # This preserves the learned visual-semantic alignment from Stages 1 & 2
        # [B, num_visual_tokens, language_hidden_size]
        fused_visual = self.fuse_features(visual_tokens)

        # 3. Optionally combine with task embeddings for richer context
        if task_embeddings is not None:
            # Concatenate visual and task features for joint reasoning
            combined_features = torch.cat(
                [fused_visual, task_embeddings], dim=1)
            seq_len = combined_features.size(1)
        else:
            combined_features = fused_visual
            seq_len = fused_visual.size(1)

        # 4. Create attention mask
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=device)

        # 5. Pass through reasoning module for robot selection
        # The reasoning module performs:
        # - Chain-of-thought reasoning (ReasoningHead)
        # - Robot selection (RobotSelectionHead)
        # - Action planning (ActionPlanningHead)
        outputs = {}

        if self.config.reasoning_enabled:
            reasoning_outputs = self.reasoning_module(
                combined_features,
                attention_mask=attention_mask,
                generate_reasoning=return_reasoning,
                select_robot=True,
                plan_actions=True,
            )

            outputs.update({
                'robot_logits': reasoning_outputs.get('robot_logits'),
                'robot_probs': reasoning_outputs.get('robot_probs'),
                'multi_robot_logits': reasoning_outputs.get('multi_robot_logits'),
                'multi_robot_probs': reasoning_outputs.get('multi_robot_probs'),
                'robot_confidence': reasoning_outputs.get('robot_confidence'),
                'top_k_indices': reasoning_outputs.get('top_k_indices'),
                'top_k_scores': reasoning_outputs.get('top_k_scores'),
                'robot_attention': reasoning_outputs.get('robot_attention'),
                'plan_steps': reasoning_outputs.get('plan_steps'),
                'plan_coherence': reasoning_outputs.get('plan_coherence'),
            })

            if return_reasoning and 'reasoning_chain' in reasoning_outputs:
                outputs['reasoning_chain'] = reasoning_outputs['reasoning_chain']

            # Compute robot selection loss
            loss = None
            if robot_targets is not None:
                targets = {'robot_target': robot_targets}
                reasoning_losses = self.reasoning_loss(
                    reasoning_outputs, targets)
                loss = reasoning_losses['total_loss']
                outputs['reasoning_losses'] = reasoning_losses

            outputs['loss'] = loss
        else:
            # Fallback: simple classification head on pooled visual features
            pooled = combined_features.mean(dim=1)  # [B, language_hidden_size]

            # Use a proper learned projection instead of random
            if not hasattr(self, '_fallback_classifier'):
                self._fallback_classifier = nn.Linear(
                    self.config.language_hidden_size,
                    self.config.num_robots
                ).to(device)

            robot_logits = self._fallback_classifier(pooled)
            outputs['robot_logits'] = robot_logits
            outputs['robot_probs'] = F.softmax(robot_logits, dim=-1)
            outputs['loss'] = None

            if robot_targets is not None:
                outputs['loss'] = F.cross_entropy(robot_logits, robot_targets)

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        use_va_refiner: Optional[bool] = None,
        tokenizer: Optional[Any] = None,
    ) -> torch.LongTensor:
        """
        Generate text given image and optional prompt.

        Uses a safe generation loop that validates all token IDs to prevent
        CUDA indexSelect assertions during autoregressive generation.

        Args:
            input_ids: Optional prompt token IDs
            pixel_values: Input images
            attention_mask: Attention mask
            image_positions: Where images are in sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: Prevent n-grams of this size from repeating
            use_va_refiner: Override config.use_va_refiner for this call.
                True = enable VA hallucination suppression, False = disable,
                None = defer to self.config.use_va_refiner.
            tokenizer: Tokenizer (required when use_va_refiner is active,
                used to identify visually-sensitive tokens).

        Returns:
            Generated token IDs
        """
        if input_ids is None and pixel_values is None:
            raise ValueError("Must provide either input_ids or pixel_values")

        # Get vocab size for validation
        vocab_size = self.language_model.get_input_embeddings().weight.shape[0]
        device = self.device

        # Prepare inputs
        if input_ids is not None:
            # DIAGNOSTIC: Log raw vision encoder output BEFORE fusion (per image)
            if pixel_values is not None:
                import logging
                logger = logging.getLogger(__name__)

                # Get raw vision encoder output
                raw_vision_output = self.encode_image(pixel_values)
                raw_visual_tokens = raw_vision_output['visual_tokens']

                # Log per-image stats (critical for debugging)
                if not hasattr(self, '_gen_raw_vision_logged'):
                    raw_mean = raw_visual_tokens.mean().item()
                    raw_std = raw_visual_tokens.std().item()
                    raw_var = raw_visual_tokens.var().item()
                    # Also compute per-batch-item variance to see if different
                    if raw_visual_tokens.size(0) > 1:
                        per_item_means = raw_visual_tokens.mean(dim=(1,2))
                        items_different = per_item_means.std().item()
                        logger.info(f"[Generate] RAW vision tokens: mean={raw_mean:.4f}, std={raw_std:.4f}, "
                                   f"var={raw_var:.4f}, cross-batch-std={items_different:.4f}")
                    else:
                        logger.info(f"[Generate] RAW vision tokens: mean={raw_mean:.4f}, std={raw_std:.4f}, var={raw_var:.4f}")
                    self._gen_raw_vision_logged = True

            inputs_embeds, attention_mask = self.prepare_inputs_embeds(
                input_ids, pixel_values, image_positions
            )
            # Keep track of initial sequence length for decoding
            initial_seq_len = input_ids.size(1)

            # DIAGNOSTIC: Log vision feature statistics to verify image is being used
            if pixel_values is not None:
                import logging
                logger = logging.getLogger(__name__)
                if not hasattr(self, '_gen_vision_logged'):
                    # Log pixel_values stats
                    pv_mean = pixel_values.mean().item()
                    pv_std = pixel_values.std().item()
                    logger.info(f"[Generate] pixel_values: shape={pixel_values.shape}, mean={pv_mean:.4f}, std={pv_std:.4f}")

                    # Log vision embedding stats (first num_visual tokens in inputs_embeds)
                    num_visual = self.config.num_visual_tokens
                    vision_embeds = inputs_embeds[:, :num_visual, :]
                    ve_mean = vision_embeds.mean().item()
                    ve_std = vision_embeds.std().item()
                    logger.info(f"[Generate] vision_embeds: shape={vision_embeds.shape}, mean={ve_mean:.4f}, std={ve_std:.4f}")
                    self._gen_vision_logged = True
        else:
            # Image only - create embeddings from image
            vision_output = self.encode_image(pixel_values)
            visual_tokens = vision_output['visual_tokens']
            inputs_embeds = self.fuse_features(visual_tokens)
            attention_mask = torch.ones(
                inputs_embeds.size(0), inputs_embeds.size(1),
                dtype=torch.long, device=inputs_embeds.device
            )
            initial_seq_len = 0

        batch_size = inputs_embeds.size(0)

        # Initialize generation with special tokens
        eos_token_id = getattr(self.language_model.config, 'eos_token_id', 50256)
        if eos_token_id >= vocab_size:
            eos_token_id = vocab_size - 1

        # Resolve VA refiner
        va_refiner = None
        _use_va = use_va_refiner if use_va_refiner is not None else self.config.use_va_refiner
        if _use_va:
            va_refiner = self.va_refiner  # lazy-init via property

        # Use safe autoregressive generation loop
        generated_ids = self._safe_autoregressive_generation(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            va_refiner=va_refiner,
            tokenizer=tokenizer,
        )

        return generated_ids

    def _safe_autoregressive_generation(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        eos_token_id: int,
        vocab_size: int,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        va_refiner: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> torch.LongTensor:
        """
        Safe autoregressive generation with per-step token validation.

        This method implements a controlled generation loop that validates
        every generated token ID before feeding it back to the model,
        preventing CUDA indexSelect assertion errors.

        Also validates position embeddings to prevent overflow.
        Includes repetition penalty and n-gram blocking to prevent degenerate outputs.
        """
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device

        # CRITICAL: Get max position embeddings to prevent overflow
        max_position_embeddings = getattr(
            self.language_model.config, 'max_position_embeddings',
            getattr(self.language_model.config, 'n_positions', 1024)
        )

        initial_seq_len = inputs_embeds.size(1)

        # Validate initial sequence doesn't exceed position limits
        if initial_seq_len >= max_position_embeddings:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"❌ POSITION OVERFLOW: initial_seq_len={initial_seq_len} >= "
                f"max_position_embeddings={max_position_embeddings}. Truncating inputs!"
            )
            inputs_embeds = inputs_embeds[:, :max_position_embeddings - max_new_tokens, :]
            attention_mask = attention_mask[:, :max_position_embeddings - max_new_tokens]
            initial_seq_len = inputs_embeds.size(1)

        # Adjust max_new_tokens to not exceed position limits
        max_allowed_new = max_position_embeddings - initial_seq_len
        if max_new_tokens > max_allowed_new:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Reducing max_new_tokens from {max_new_tokens} to {max_allowed_new} "
                f"to prevent position embedding overflow"
            )
            max_new_tokens = max(1, max_allowed_new)

        # Start with empty generated sequence
        generated_tokens = []
        current_embeds = inputs_embeds
        current_mask = attention_mask
        past_key_values = None

        for step in range(max_new_tokens):
            current_seq_len = current_embeds.size(1)

            # CRITICAL: Check position bounds before each forward pass
            if current_seq_len >= max_position_embeddings:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"⚠️ Stopping generation at step {step}: "
                    f"seq_len={current_seq_len} >= max_pos={max_position_embeddings}"
                )
                break

            # Forward pass
            outputs = self.language_model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get next token logits - CLAMP vocab dimension to prevent index errors
            logits = outputs['logits'][:, -1, :]
            if logits.size(-1) > vocab_size:
                next_token_logits = logits[:, :vocab_size]
            else:
                next_token_logits = logits

            # Apply repetition penalty to discourage repetitive outputs
            if repetition_penalty != 1.0 and generated_tokens:
                # Get all previously generated tokens
                prev_tokens = torch.cat(generated_tokens, dim=1)
                for i in range(batch_size):
                    # Get unique tokens this batch item has generated
                    for prev_token in prev_tokens[i].unique():
                        token_idx = prev_token.item()
                        if 0 <= token_idx < next_token_logits.size(-1):
                            # Apply penalty (reduce logit for previously generated tokens)
                            if next_token_logits[i, token_idx] > 0:
                                next_token_logits[i, token_idx] /= repetition_penalty
                            else:
                                next_token_logits[i, token_idx] *= repetition_penalty

            # Block n-grams that would repeat
            if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size - 1:
                # Check for n-gram repetition
                prev_tokens = torch.cat(generated_tokens, dim=1)
                for i in range(batch_size):
                    # Get the last (n-1) tokens
                    ngram_prefix = prev_tokens[i, -(no_repeat_ngram_size - 1):].tolist()
                    # Find all positions where this prefix occurred before
                    seq = prev_tokens[i].tolist()
                    for j in range(len(seq) - no_repeat_ngram_size + 1):
                        if seq[j:j + no_repeat_ngram_size - 1] == ngram_prefix:
                            # Block the token that would complete this n-gram
                            blocked_token = seq[j + no_repeat_ngram_size - 1]
                            if 0 <= blocked_token < next_token_logits.size(-1):
                                next_token_logits[i, blocked_token] = float('-inf')

            # ── VA hallucination suppression (optional) ──────────────
            if va_refiner is not None and tokenizer is not None:
                gen_ids = (
                    torch.cat(generated_tokens, dim=1)
                    if generated_tokens
                    else torch.zeros(batch_size, 0, dtype=torch.long, device=device)
                )
                next_token_logits = va_refiner.refine_logits(
                    next_token_logits, gen_ids, tokenizer
                )

            # Sample next token
            if do_sample and temperature > 0:
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # CRITICAL: Validate and clamp token IDs BEFORE using them
            next_tokens = torch.clamp(next_tokens, min=0, max=vocab_size - 1)

            # CRITICAL: Assert token validity before embedding lookup
            max_token = next_tokens.max().item()
            min_token = next_tokens.min().item()
            assert 0 <= min_token < vocab_size, \
                f"Generated token {min_token} < 0"
            assert 0 <= max_token < vocab_size, \
                f"Generated token {max_token} >= vocab_size {vocab_size}"

            # Check for EOS
            if (next_tokens == eos_token_id).all():
                break

            # Append to generated sequence
            generated_tokens.append(next_tokens.unsqueeze(1))

            # Get embeddings for next iteration (with validated token IDs)
            next_embeds = self.language_model.embed_tokens(next_tokens.unsqueeze(1))

            # Update for next iteration
            current_embeds = next_embeds
            current_mask = torch.cat([current_mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
            past_key_values = outputs.get('past_key_values')

        # Concatenate all generated tokens
        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=1)
        else:
            # No tokens generated, return empty
            generated_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        return generated_ids

    @torch.no_grad()
    def analyze_incident(
        self,
        pixel_values: torch.Tensor,
        instruction: Optional[str] = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """
        Analyze an incident image and select appropriate robot.

        Args:
            pixel_values: Input image [1, C, H, W]
            instruction: Optional instruction text
            tokenizer: Tokenizer for encoding instruction

        Returns:
            Dictionary with robot selection, confidence, and action plan
        """
        device = pixel_values.device

        # Prepare inputs
        if instruction is not None and tokenizer is not None:
            tokens = tokenizer(instruction, return_tensors="pt", padding=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
        else:
            input_ids = None
            attention_mask = None

        # Forward pass with reasoning
        outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_reasoning=True,
        )

        # Get robot selection
        robot_probs = outputs['robot_probs']
        robot_idx = robot_probs.argmax(dim=-1).item()
        confidence = outputs['robot_confidence'].item()

        result = {
            'selected_robot': self.config.robot_names[robot_idx],
            'robot_index': robot_idx,
            'confidence': confidence,
            'robot_probabilities': {
                name: prob.item()
                for name, prob in zip(self.config.robot_names, robot_probs[0])
            },
        }

        if 'plan_coherence' in outputs:
            result['plan_coherence'] = outputs['plan_coherence'].item()

        if 'reasoning_chain' in outputs:
            result['reasoning_chain'] = outputs['reasoning_chain']

        return result

    @torch.no_grad()
    def select_robots_topn(
        self,
        pixel_values: torch.Tensor,
        task_text: Optional[str] = None,
        tokenizer: Any = None,
        top_n: int = 3,
        return_reasoning: bool = True,
    ) -> Dict[str, Any]:
        """
        Select top-N robots for a task with reasoning explanation.

        This method provides ranked robot recommendations with confidence
        scores and optional reasoning for each selection.

        Args:
            pixel_values: Input image [1, C, H, W] or [B, C, H, W]
            task_text: Optional task description text
            tokenizer: Tokenizer for encoding task text
            top_n: Number of top robots to return (default: 3)
            return_reasoning: Whether to include reasoning chain

        Returns:
            Dictionary containing:
            - top_robots: List of (robot_name, score, confidence) tuples
            - selected_robot: Top-1 robot name (backward compatible)
            - robot_index: Top-1 robot index
            - all_scores: All robot scores as dict
            - reasoning_summary: Brief text explanation (if available)
            - reasoning_chain: Full reasoning tensors (if return_reasoning=True)
        """
        device = pixel_values.device
        batch_size = pixel_values.size(0)

        # Use vision-only forward for robustness
        outputs = self.forward_vision_only(
            pixel_values=pixel_values,
            return_reasoning=return_reasoning,
        )

        results = []

        for b in range(batch_size):
            result = {}

            # Get top-N robots
            if 'top_k_indices' in outputs and 'top_k_scores' in outputs:
                top_indices = outputs['top_k_indices'][b][:top_n]
                top_scores = outputs['top_k_scores'][b][:top_n]
            else:
                # Fallback to manual top-k
                probs = outputs['robot_probs'][b]
                top_scores, top_indices = torch.topk(
                    probs, min(top_n, self.config.num_robots))

            # Build top robots list
            top_robots = []
            for idx, (robot_idx, score) in enumerate(zip(top_indices, top_scores)):
                robot_name = self.config.robot_names[robot_idx.item()]
                confidence = outputs['robot_confidence'][b, robot_idx].item() if outputs.get(
                    'robot_confidence') is not None and outputs['robot_confidence'].dim() > 1 else score.item()
                top_robots.append({
                    'rank': idx + 1,
                    'robot_name': robot_name,
                    'robot_index': robot_idx.item(),
                    'score': score.item(),
                    'confidence': confidence,
                })

            result['top_robots'] = top_robots

            # Backward compatible fields
            if top_robots:
                result['selected_robot'] = top_robots[0]['robot_name']
                result['robot_index'] = top_robots[0]['robot_index']
                result['confidence'] = top_robots[0]['confidence']

            # All scores
            probs = outputs['robot_probs'][b]
            result['all_scores'] = {
                name: probs[i].item()
                for i, name in enumerate(self.config.robot_names)
            }

            # Robot attention for interpretability
            if 'robot_attention' in outputs:
                attention = outputs['robot_attention'][b]
                result['robot_attention'] = {
                    name: attention[i].item()
                    for i, name in enumerate(self.config.robot_names)
                }

            # Multi-robot selection (for tasks needing multiple robots)
            if 'multi_robot_probs' in outputs:
                multi_probs = outputs['multi_robot_probs'][b]
                selected_multi = (multi_probs > 0.5).nonzero().squeeze(-1)
                result['multi_robot_selection'] = [
                    self.config.robot_names[i.item()] for i in selected_multi
                ] if selected_multi.numel() > 0 else []

            # Generate reasoning summary
            result['reasoning_summary'] = self._generate_reasoning_summary(
                top_robots[0]['robot_name'] if top_robots else "Unknown",
                task_text,
                result.get('robot_attention', {})
            )

            # Include reasoning chain if requested
            if return_reasoning and 'reasoning_chain' in outputs:
                result['reasoning_chain'] = outputs['reasoning_chain'][b]

            # Plan coherence
            if 'plan_coherence' in outputs:
                result['plan_coherence'] = outputs['plan_coherence'][b].item()

            results.append(result)

        # Return single result for batch_size=1, otherwise list
        return results[0] if batch_size == 1 else results

    def _generate_reasoning_summary(
        self,
        selected_robot: str,
        task_text: Optional[str],
        robot_attention: Dict[str, float],
    ) -> str:
        """Generate a brief text summary explaining the robot selection."""
        # Simple template-based reasoning (can be enhanced with LLM generation)
        robot_capabilities = {
            "Drone": "aerial navigation, surveillance, and quick deployment",
            "Underwater Robot": "underwater exploration and marine operations",
            "Humanoid": "manipulation, tool use, and human interaction",
            "Robot with Wheels": "efficient transport on flat surfaces",
            "Robot with Legs": "rough terrain navigation and climbing",
        }

        capability = robot_capabilities.get(
            selected_robot, "specialized capabilities")

        # Sort by attention for explanation
        if robot_attention:
            sorted_robots = sorted(
                robot_attention.items(), key=lambda x: x[1], reverse=True)
            top_robot = sorted_robots[0][0]
            reason = f"Based on visual analysis, {selected_robot} is recommended for its {capability}."
        else:
            reason = f"{selected_robot} is selected for its {capability}."

        if task_text:
            reason += f" The task '{task_text[:50]}...' requires these specific abilities." if len(
                task_text) > 50 else f" The task '{task_text}' requires these specific abilities."

        return reason

    # ------------------------------------------------------------------
    #  Episodic Memory public APIs
    # ------------------------------------------------------------------

    def update_memory(
        self,
        pixel_values: torch.Tensor,
        text_input: torch.Tensor,
        tokenizer: Any = None,
        write: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode a multimodal episode through the fusion pipeline,
        compute novelty, and optionally write to episodic memory.

        Args:
            pixel_values: (B, C, H, W) images.
            text_input: (B, seq) token ids.
            tokenizer: tokenizer (used only for embedding lookup).
            write: if True, performs smart_write when novelty is high.

        Returns:
            dict with novelty_score, write_flags, nearest_indices, etc.
        """
        if self.episodic_memory is None:
            raise RuntimeError("Episodic memory is not enabled in config.")

        with torch.no_grad():
            # Get fused representation
            inputs_embeds, _ = self.prepare_inputs_embeds(text_input, pixel_values)
            lm_out = self.language_model(inputs_embeds=inputs_embeds, output_hidden_states=False)
            last_hidden = lm_out.get('last_hidden_state', lm_out.get('logits'))
            if last_hidden is not None and last_hidden.dim() == 3:
                fused_repr = last_hidden[:, -1, :]  # (B, C)
            else:
                raise RuntimeError("Could not extract hidden states for memory update.")

        sigma = self.episodic_memory.novelty_score(fused_repr)
        write_flags, _, nearest_idx = self.episodic_memory.should_write(fused_repr)

        result = {
            "novelty_score": sigma,
            "write_flags": write_flags,
            "nearest_indices": nearest_idx,
        }

        if write:
            write_stats = self.episodic_memory.smart_write(fused_repr)
            result.update(write_stats)

        return result

    def forget_memory(
        self,
        slot_index: Optional[int] = None,
        content: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forget a specific memory slot or nearest slot to given content."""
        if self.episodic_memory is None:
            raise RuntimeError("Episodic memory is not enabled in config.")
        return self.episodic_memory.forget(slot_index=slot_index, content=content)

    def save_memory_state(self, path: str):
        """Save episodic memory state (M, C, usage metadata)."""
        if self.episodic_memory is None:
            raise RuntimeError("Episodic memory is not enabled in config.")
        state = self.episodic_memory.get_state()
        torch.save(state, path)

    def load_memory_state(self, path: str, strict: bool = True):
        """Load episodic memory state (M, C, usage metadata)."""
        if self.episodic_memory is None:
            raise RuntimeError("Episodic memory is not enabled in config.")
        state = torch.load(path, map_location="cpu")
        self.episodic_memory.set_state(state, strict=strict)

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel()
                        for p in self.parameters() if p.requires_grad)

        # Per component
        vision_params = sum(p.numel()
                            for p in self.vision_encoder.parameters())
        language_params = sum(p.numel()
                              for p in self.language_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())

        result = {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'language_model': language_params,
            'fusion_module': fusion_params,
        }

        if self.config.reasoning_enabled:
            reasoning_params = sum(p.numel()
                                   for p in self.reasoning_module.parameters())
            result['reasoning_module'] = reasoning_params

        return result

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model weights
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: Optional[EmberVLMConfig] = None,
    ) -> 'EmberVLM':
        """Load model from directory."""
        import os
        import json
        import logging
        from pathlib import Path

        logger = logging.getLogger(__name__)

        # Search for config.json in pretrained_path and parent directories
        config_found = False
        pretrained_dir = Path(pretrained_path)

        # Try: checkpoint dir -> stage dir (parent) -> model dir (grandparent) -> final
        search_paths = [
            pretrained_dir / 'config.json',
            pretrained_dir.parent / 'config.json',          # e.g., stage2/config.json
            pretrained_dir.parent.parent / 'config.json',   # e.g., model_dir/config.json
            pretrained_dir / 'final' / 'config.json',       # e.g., checkpoint/final/config.json
        ]

        for config_path in search_paths:
            if config_path.exists():
                logger.info(f"[from_pretrained] Loading config from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    logger.info(f"[from_pretrained] Config dict: vision_backbone={config_dict.get('vision_backbone')}, "
                               f"language_backbone={config_dict.get('language_backbone')}")
                    config = EmberVLMConfig.from_dict(config_dict)
                    logger.info(f"[from_pretrained] Loaded config: vision_backbone={config.vision_backbone}, "
                               f"language_backbone={config.language_backbone}, "
                               f"pretrained_language_model={config.pretrained_language_model}")
                    config_found = True
                    break
                except Exception as e:
                    logger.warning(f"[from_pretrained] Failed to load config from {config_path}: {e}")

        # ═══════════════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: If no config found, infer architecture from checkpoint weights
        # ═══════════════════════════════════════════════════════════════════════════════
        if not config_found and config is None:
            logger.warning(f"[from_pretrained] No config.json found! Attempting to infer architecture from weights...")

            # Find pytorch_model.bin - search comprehensively
            model_paths = [
                pretrained_dir / 'pytorch_model.bin',
                pretrained_dir / 'final' / 'pytorch_model.bin',
                pretrained_dir.parent / 'pytorch_model.bin',  # Check parent directory
                pretrained_dir.parent / 'final' / 'pytorch_model.bin',
                pretrained_dir.parent.parent / 'pytorch_model.bin',  # Check grandparent
            ]

            # Also check for other checkpoint file patterns
            if pretrained_dir.exists():
                # Look for any .bin or .pt files
                for pattern in ['*.bin', '*.pt', 'model.safetensors']:
                    for found_file in pretrained_dir.rglob(pattern):
                        if 'pytorch_model' in found_file.name or 'model' in found_file.name:
                            model_paths.append(found_file)
                            if len(model_paths) > 20:  # Limit search
                                break

            inferred_config = None
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        # Load state dict to inspect shapes
                        state_dict = torch.load(str(model_path), map_location='cpu')

                        # Detect hidden_size from language model embeddings
                        hidden_size = None
                        vocab_size = None

                        # Check SmolLM-style keys
                        for key in ['language_model.model.model.embed_tokens.weight',
                                   'language_model.model.lm_head.weight']:
                            if key in state_dict:
                                vocab_size = state_dict[key].shape[0]
                                hidden_size = state_dict[key].shape[1]
                                logger.info(f"[from_pretrained] Detected from {key}: vocab={vocab_size}, hidden={hidden_size}")
                                break

                        # Check TinyLLM-style keys if not found
                        if hidden_size is None:
                            for key in ['language_model.model.transformer.wte.weight',
                                       'language_model.model.lm_head.weight']:
                                if key in state_dict:
                                    if 'wte' in key:
                                        vocab_size = state_dict[key].shape[0]
                                        hidden_size = state_dict[key].shape[1]
                                    else:
                                        vocab_size = state_dict[key].shape[0]
                                        hidden_size = state_dict[key].shape[1]
                                    logger.info(f"[from_pretrained] Detected from {key}: vocab={vocab_size}, hidden={hidden_size}")
                                    break

                        # Infer vision backbone from state_dict keys
                        inferred_vision = None
                        
                        # Check for DINOv2-specific keys
                        dinov2_keys = [k for k in state_dict.keys() if 'vision_encoder.backbone.embeddings.patch_embeddings' in k 
                                      or 'vision_encoder.backbone.encoder.layer' in k]
                        if dinov2_keys:
                            inferred_vision = 'dinov2_small'
                            logger.info(f"[from_pretrained] Detected DINOv2 from keys: {dinov2_keys[0]}")
                        
                        # Check for MobileViT-specific keys
                        mobilevit_keys = [k for k in state_dict.keys() if 'vision_encoder.backbone.encoder' in k 
                                         and 'mobilevit' in k.lower()]
                        if not inferred_vision and mobilevit_keys:
                            inferred_vision = 'mobilevit_xs'
                            logger.info(f"[from_pretrained] Detected MobileViT from keys: {mobilevit_keys[0]}")
                        
                        # Check for RepViT-specific keys  
                        repvit_keys = [k for k in state_dict.keys() if 'vision_encoder.stem' in k]
                        if not inferred_vision and repvit_keys:
                            inferred_vision = 'repvit'
                            logger.info(f"[from_pretrained] Detected RepViT from keys: {repvit_keys[0]}")

                        # Infer backbone from hidden_size
                        if hidden_size:
                            # Known hidden sizes:
                            # - TinyLLM-30M: 384
                            # - SmolLM-135M: 576
                            if hidden_size == 576:
                                inferred_language = 'smollm_135m'
                                # HARDCODED: SmolLM-135M is paired with DINOv2-Small
                                inferred_vision = 'dinov2_small'
                                logger.warning(f"[from_pretrained] ═══════════════════════════════════════════════")
                                logger.warning(f"[from_pretrained] INFERRED ARCHITECTURE FROM CHECKPOINT WEIGHTS!")
                                logger.warning(f"[from_pretrained]   hidden_size={hidden_size} → SmolLM-135M")
                                logger.warning(f"[from_pretrained]   HARDCODED: vision={inferred_vision}, language={inferred_language}")
                                logger.warning(f"[from_pretrained] ═══════════════════════════════════════════════")
                                inferred_config = EmberVLMConfig(
                                    vision_backbone=inferred_vision,
                                    language_backbone=inferred_language,
                                )
                                # CRITICAL: Set image_size explicitly for DINOv2 (224x224 -> 256 tokens)
                                if inferred_vision == 'dinov2_small':
                                    inferred_config.image_size = 224
                                if vocab_size:
                                    inferred_config.language_vocab_size = vocab_size
                            elif hidden_size == 384:
                                inferred_language = 'tinyllm'
                                # Use detected vision backbone, or default to repvit if detection failed
                                if not inferred_vision:
                                    inferred_vision = 'repvit'
                                    logger.warning(f"[from_pretrained] Could not detect vision backbone from state_dict keys, defaulting to repvit")
                                logger.warning(f"[from_pretrained] ═══════════════════════════════════════════════")
                                logger.warning(f"[from_pretrained] INFERRED ARCHITECTURE FROM CHECKPOINT WEIGHTS!")
                                logger.warning(f"[from_pretrained]   hidden_size={hidden_size} → TinyLLM-30M")
                                logger.warning(f"[from_pretrained]   Detected: vision={inferred_vision}, language={inferred_language}")
                                logger.warning(f"[from_pretrained] ═══════════════════════════════════════════════")
                                inferred_config = EmberVLMConfig(
                                    vision_backbone=inferred_vision,
                                    language_backbone=inferred_language,
                                )
                                # Set image_size explicitly based on vision backbone
                                if inferred_vision == 'dinov2_small':
                                    inferred_config.image_size = 224  # DINOv2: 224x224 -> 256 tokens
                                elif inferred_vision == 'mobilevit_xs':
                                    inferred_config.image_size = 256  # MobileViT: 256x256
                                elif inferred_vision == 'repvit':
                                    inferred_config.image_size = 224  # RepViT: 224x224
                                if vocab_size:
                                    inferred_config.language_vocab_size = vocab_size
                            else:
                                logger.error(f"[from_pretrained] Unknown hidden_size={hidden_size}, cannot infer architecture!")

                        # Clean up - don't keep state_dict in memory
                        del state_dict
                        break

                    except Exception as e:
                        logger.warning(f"[from_pretrained] Failed to infer from {model_path}: {e}")

            if inferred_config:
                config = inferred_config
                config_found = True  # Mark as found so we don't use defaults

                # IMPORTANT: Save the inferred config to prevent future issues
                try:
                    config_path = pretrained_dir / 'config.json'
                    with open(config_path, 'w') as f:
                        json.dump(config.to_dict(), f, indent=2)
                    logger.warning(f"[from_pretrained] ✓ Saved inferred config to {config_path}")
                except Exception as e:
                    logger.warning(f"[from_pretrained] Could not save inferred config: {e}")

        if not config_found:
            if config is None:
                logger.error(f"[from_pretrained] ════════════════════════════════════════════════════════════════")
                logger.error(f"[from_pretrained] CRITICAL: No config.json found and could not infer architecture!")
                logger.error(f"[from_pretrained] Using DEFAULTS (repvit + tinyllm) - THIS MAY BE WRONG!")
                logger.error(f"[from_pretrained] Searched: {[str(p) for p in search_paths]}")
                logger.error(f"[from_pretrained] ════════════════════════════════════════════════════════════════")
                config = EmberVLMConfig()
            else:
                logger.info(f"[from_pretrained] Using provided config")

        # Create model
        logger.info(f"[from_pretrained] Creating model with vision={config.vision_backbone}, "
                   f"language={config.language_backbone}")
        model = cls(config)

        # Search for pytorch_model.bin in pretrained_path and subdirectories
        model_paths = [
            pretrained_dir / 'pytorch_model.bin',
            pretrained_dir / 'final' / 'pytorch_model.bin',
            pretrained_dir.parent / 'pytorch_model.bin',  # Check parent directory
            pretrained_dir.parent / 'final' / 'pytorch_model.bin',
            pretrained_dir.parent.parent / 'pytorch_model.bin',  # Check grandparent
        ]

        # Also search for any model checkpoint files
        if pretrained_dir.exists():
            for pattern in ['*.bin', '*.pt']:
                for found_file in pretrained_dir.rglob(pattern):
                    if 'pytorch_model' in found_file.name or found_file.name == 'model.bin':
                        model_paths.append(found_file)
                        if len(model_paths) > 20:
                            break

        weights_loaded = False

        # Log what we're searching for
        logger.info(f"[from_pretrained] Searching for model weights in:")
        for mp in model_paths[:5]:  # Show first 5 paths
            exists_marker = "✓" if mp.exists() else "✗"
            logger.info(f"[from_pretrained]   [{exists_marker}] {mp}")

        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"[from_pretrained] Loading weights from {model_path}")
                state_dict = torch.load(str(model_path), map_location='cpu')

                # ═══════════════════════════════════════════════════════════════════════
                # COMPREHENSIVE ARCHITECTURE VALIDATION
                # ═══════════════════════════════════════════════════════════════════════
                # Prevent loading checkpoints from wrong architectures (e.g., TinyLLM vs SmolLM)

                checkpoint_vocab_size = None
                checkpoint_hidden_size = None
                current_vocab_size = None
                current_hidden_size = None

                # Detect checkpoint architecture from state_dict
                for key in ['language_model.model.model.embed_tokens.weight',
                           'language_model.model.lm_head.weight']:
                    if key in state_dict:
                        checkpoint_vocab_size = state_dict[key].shape[0]
                        checkpoint_hidden_size = state_dict[key].shape[1]
                        break

                # Detect current model architecture
                if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
                    lm = model.language_model.model
                    if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
                        current_vocab_size = lm.model.embed_tokens.weight.shape[0]
                        current_hidden_size = lm.model.embed_tokens.weight.shape[1]
                    elif hasattr(lm, 'embed_tokens'):
                        current_vocab_size = lm.embed_tokens.weight.shape[0]
                        current_hidden_size = lm.embed_tokens.weight.shape[1]

                # CRITICAL: Check hidden_size compatibility
                if checkpoint_hidden_size and current_hidden_size:
                    if checkpoint_hidden_size != current_hidden_size:
                        error_msg = (
                            f"\n{'═'*80}\n"
                            f"❌ FATAL: Architecture mismatch detected!\n"
                            f"{'═'*80}\n"
                            f"Checkpoint hidden_size: {checkpoint_hidden_size}\n"
                            f"Current model hidden_size: {current_hidden_size}\n\n"
                            f"This usually means:\n"
                            f"  - Checkpoint is from {config.vision_backbone} + {config.language_backbone}\n"
                            f"  - But config.json loading failed, defaulting to repvit + tinyllm\n\n"
                            f"Common hidden sizes:\n"
                            f"  - TinyLLM: 384\n"
                            f"  - SmolLM-135M: 576\n\n"
                            f"FIX: Ensure config.json exists and contains correct backbone names!\n"
                            f"{'═'*80}\n"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                # Vocab size mismatch is OK (can resize), but log it
                if checkpoint_vocab_size and current_vocab_size and checkpoint_vocab_size != current_vocab_size:
                    logger.warning(f"[from_pretrained] Vocab size mismatch detected!")
                    logger.warning(f"  Checkpoint vocab_size: {checkpoint_vocab_size}")
                    logger.warning(f"  Current model vocab_size: {current_vocab_size}")
                    logger.warning(f"  Resizing embeddings: {current_vocab_size} -> {checkpoint_vocab_size}")

                    try:
                        resize_successful = False

                        # Method 1: HuggingFace's resize_token_embeddings on language_model
                        if hasattr(model.language_model, 'resize_token_embeddings'):
                            model.language_model.resize_token_embeddings(checkpoint_vocab_size)
                            logger.info(f"✓ Resized language_model embeddings to {checkpoint_vocab_size}")
                            resize_successful = True

                        # Method 2: HuggingFace's resize_token_embeddings on language_model.model
                        elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'resize_token_embeddings'):
                            model.language_model.model.resize_token_embeddings(checkpoint_vocab_size)
                            logger.info(f"✓ Resized language_model.model embeddings to {checkpoint_vocab_size}")
                            resize_successful = True

                        # Method 3: Manual resize for SmolLM (model.model.embed_tokens + lm_head)
                        if not resize_successful:
                            lm = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model

                            # Check for SmolLM structure: model.model.embed_tokens
                            if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
                                embed = lm.model.embed_tokens
                                old_vocab, embed_dim = embed.weight.shape
                                new_embed = nn.Embedding(checkpoint_vocab_size, embed_dim)
                                nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
                                with torch.no_grad():
                                    copy_size = min(old_vocab, checkpoint_vocab_size)
                                    new_embed.weight[:copy_size] = embed.weight[:copy_size]
                                lm.model.embed_tokens = new_embed
                                logger.info(f"✓ Manually resized model.embed_tokens: {old_vocab} -> {checkpoint_vocab_size}")

                            # Also resize lm_head
                            if hasattr(lm, 'lm_head'):
                                old_head = lm.lm_head
                                new_head = nn.Linear(old_head.in_features, checkpoint_vocab_size, bias=old_head.bias is not None)
                                nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
                                with torch.no_grad():
                                    copy_size = min(old_head.out_features, checkpoint_vocab_size)
                                    new_head.weight[:copy_size] = old_head.weight[:copy_size]
                                    if old_head.bias is not None and new_head.bias is not None:
                                        new_head.bias[:copy_size] = old_head.bias[:copy_size]
                                lm.lm_head = new_head
                                logger.info(f"✓ Manually resized lm_head: {old_head.out_features} -> {checkpoint_vocab_size}")

                            resize_successful = True

                        if not resize_successful:
                            logger.error("❌ Cannot resize embeddings - no suitable method found")

                    except Exception as e:
                        logger.error(f"❌ Failed to resize embeddings: {e}")
                        raise
                        raise

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"[from_pretrained] Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"[from_pretrained] Unexpected keys: {len(unexpected)}")
                weights_loaded = True
                break

        if not weights_loaded:
            logger.warning(f"[from_pretrained] No pytorch_model.bin found in: {[str(p) for p in model_paths]}")

        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


# Convenience functions
def create_embervlm(
    config: Optional[EmberVLMConfig] = None,
    **kwargs,
) -> EmberVLM:
    """
    Create EmberVLM model with optional configuration overrides.

    Args:
        config: Optional base configuration
        **kwargs: Configuration overrides

    Returns:
        EmberVLM model instance
    """
    if config is None:
        config = EmberVLMConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return EmberVLM(config)


def load_embervlm(
    path_or_repo: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> EmberVLM:
    """
    Load EmberVLM model from local path or HuggingFace Hub.

    Args:
        path_or_repo: Local path or HuggingFace repo ID
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded EmberVLM model
    """
    import os

    if os.path.exists(path_or_repo):
        model = EmberVLM.from_pretrained(path_or_repo)
    else:
        # Try loading from HuggingFace Hub
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=path_or_repo)
        model = EmberVLM.from_pretrained(local_path)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model
