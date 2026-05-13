"""
EmberVLM - Tiny Multimodal VLM for Robot Fleet Selection with Incident Reasoning
"""

__version__ = "1.0.0"
__author__ = "EmberVLM Team"

from embervlm.models.embervlm import EmberVLM, EmberVLMConfig, create_embervlm, load_embervlm
from embervlm.models.vision_encoder import RepViTEncoder
from embervlm.models.language_model import TinyLLMBackbone
from embervlm.models.fusion_module import FusionModule
from embervlm.models.reasoning_heads import ReasoningModule

__all__ = [
    "EmberVLM",
    "EmberVLMConfig",
    "create_embervlm",
    "load_embervlm",
    "RepViTEncoder",
    "TinyLLMBackbone",
    "FusionModule",
    "ReasoningModule",
]

