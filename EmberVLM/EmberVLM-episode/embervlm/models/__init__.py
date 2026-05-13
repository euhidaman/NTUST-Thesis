"""
EmberVLM Models Package

Provides model components for the EmberVLM multimodal VLM:
- Vision: DINOv2-Small (RECOMMENDED - best VL performance)
- Vision Alternative: RepViT-XXS, MobileViT-XS
- Language: SmolLM-135M (RECOMMENDED - frozen to preserve text)
- Language Alternative: TinyLLM-30M
- Fusion: Q-Former (RECOMMENDED) or Adapter-based
- Reasoning: Chain-of-Thought reasoning heads
"""

from embervlm.models.embervlm import EmberVLM, EmberVLMConfig
from embervlm.models.vision_encoder import (
    RepViTEncoder,
    MobileViTEncoder,
    DINOv2Encoder,
    ImagePreprocessor,
    create_vision_encoder,
    VISION_BACKBONE_REPVIT,
    VISION_BACKBONE_MOBILEVIT_XS,
    VISION_BACKBONE_DINOV2_SMALL,
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
from embervlm.models.fusion_module import FusionModule, QFormerFusion
from embervlm.models.reasoning_heads import ReasoningModule, ReasoningLoss
from embervlm.models.hallucination import (
    VARefiner,
    VAConfig,
    VAThresholds,
    VATemporalConfig,
    VAFeatureExtractor,
    VAClassifier,
)
from embervlm.models.episodic_memory import EpisodicMemoryController, ScopeDetector
from embervlm.models.edge_memory import (
    EdgeMemoryRuntime,
    EdgeMemoryConfig,
    FastGaussianAddressing,
    CachedPseudoinverseAddressing,
    FastScopeDetector,
    QuantisedMemory,
    benchmark_edge_latency,
    format_benchmark_report,
    compare_addressing_accuracy,
)

__all__ = [
    # Main model
    "EmberVLM",
    "EmberVLMConfig",
    # Vision
    "RepViTEncoder",
    "MobileViTEncoder",
    "DINOv2Encoder",
    "ImagePreprocessor",
    "create_vision_encoder",
    "VISION_BACKBONE_REPVIT",
    "VISION_BACKBONE_MOBILEVIT_XS",
    "VISION_BACKBONE_DINOV2_SMALL",
    # Language
    "TinyLLMBackbone",
    "TinyLLMConfig",
    "PretrainedTinyLLMBackbone",
    "SmolLMBackbone",
    "create_language_backbone",
    "PRETRAINED_TINYLLM_MODEL",
    "PRETRAINED_SMOLLM_135M",
    "BACKBONE_TINYLLM",
    "BACKBONE_SMOLLM_135M",
    # Fusion
    "FusionModule",
    "QFormerFusion",
    # Reasoning
    "ReasoningModule",
    "ReasoningLoss",
    # Hallucination mitigation
    "VARefiner",
    "VAConfig",
    "VAThresholds",
    "VATemporalConfig",
    "VAFeatureExtractor",
    "VAClassifier",
    # Episodic memory
    "EpisodicMemoryController",
    "ScopeDetector",
    # Edge-optimised episodic memory
    "EdgeMemoryRuntime",
    "EdgeMemoryConfig",
    "FastGaussianAddressing",
    "CachedPseudoinverseAddressing",
    "FastScopeDetector",
    "QuantisedMemory",
    "benchmark_edge_latency",
    "format_benchmark_report",
    "compare_addressing_accuracy",
]
