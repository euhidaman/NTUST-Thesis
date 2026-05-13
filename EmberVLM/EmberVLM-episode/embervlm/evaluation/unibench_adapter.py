"""
UniBench Adapter for EmberVLM

This module provides a custom UniBench model adapter for EmberVLM,
enabling standardized evaluation on visual reasoning benchmarks.

CRITICAL CONSTRAINTS:
- NO tokenizer auto-resizing
- NO forced vocab expansion
- NO implicit embedding re-initialization
- NO half/float dtype mismatches
- NO image resizing that breaks vision encoder
- NO forced batch sizes exceeding memory
- Explicit device and dtype handling

Supported benchmarks (via UniBench):
- Object recognition
- Spatial reasoning
- Counting
- Visual reasoning
- Attribute detection

Author: EmberVLM Team
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# UniBench benchmarks that EmberVLM can handle
SUPPORTED_UNIBENCH_BENCHMARKS = {
    # Object recognition
    "imagenet_1k",
    "imagenet_a",
    "imagenet_r",
    "objectnet",
    # Visual reasoning
    "winoground",
    "vsr",
    "sugarcrepe",
    # Spatial reasoning
    "whatsup",
    "clevr_distance",
    "clevr_position",
    # Counting
    "countbench",
    "tallyqa",
    # Attribute detection
    "vaw",
    "paco",
}

# Benchmarks to skip (too large or incompatible)
UNSUPPORTED_BENCHMARKS = {
    "imagenet_21k",  # Too large
    "laion",  # Too large
}

# Default evaluation parameters
DEFAULT_BATCH_SIZE = 8
MAX_SAMPLES_PER_BENCHMARK = 5000


@dataclass
class UniBenchResult:
    """Result from UniBench evaluation."""
    benchmark_name: str
    score: float
    num_samples: int
    metric_name: str
    success: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "num_samples": self.num_samples,
            "metric_name": self.metric_name,
            "success": self.success,
            "error_message": self.error_message,
            "details": self.details,
        }


class EmberVLMUniBenchWrapper:
    """
    EmberVLM wrapper compatible with UniBench's AbstractModel interface.

    This wrapper allows EmberVLM to be evaluated using UniBench's
    standardized evaluation protocols while respecting EmberVLM's constraints.

    Implements the required methods:
    - get_image_embeddings(images) -> tensor
    - get_text_embeddings(texts) -> tensor
    - get_text_from_image(images, prompts) -> list[str]
    """

    def __init__(
        self,
        model,
        model_name: str,
        tokenizer,
        input_resolution: int = 224,
        norm_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        batch_per_gpu: int = 8,
        device: str = "cuda",
        max_new_tokens: int = 64,
        **kwargs,
    ):
        """
        Initialize UniBench wrapper for EmberVLM.

        Args:
            model: Loaded EmberVLM model
            model_name: Name identifier for the model
            tokenizer: Tokenizer for the model
            input_resolution: Image input resolution
            norm_mean: Image normalization mean
            norm_std: Image normalization std
            batch_per_gpu: Batch size per GPU
            device: Device to use
            max_new_tokens: Max tokens for generation
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_resolution = input_resolution
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.batch_per_gpu = batch_per_gpu
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Get vocab size for validation
        self._vocab_size = self._get_vocab_size()

        # Image preprocessor from model
        self.image_preprocessor = getattr(model, "image_preprocessor", None)

        # UniBench expected attributes
        self.use_transforms = False  # We handle our own transforms
        self.logit_scale = None
        self.context_length = 512
        self.zeroshot_weights = None
        self.classes = None
        self.templates = None
        self.prompt = None

        logger.info(f"[UniBench] EmberVLM wrapper initialized")
        logger.info(f"[UniBench] vocab_size={self._vocab_size}, device={device}")

    def _get_vocab_size(self) -> int:
        """Get actual embedding vocabulary size."""
        try:
            if hasattr(self.model, "language_model"):
                lm = self.model.language_model
                if hasattr(lm, "model") and hasattr(lm.model, "get_input_embeddings"):
                    return lm.model.get_input_embeddings().weight.shape[0]
                if hasattr(lm, "get_input_embeddings"):
                    return lm.get_input_embeddings().weight.shape[0]
        except Exception:
            pass
        return len(self.tokenizer) if self.tokenizer else 50257

    def _safe_tokenize(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Tokenize with safety checks - clamp to valid vocab range."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = inputs["input_ids"]

        # CRITICAL: Clamp to valid range
        if self._vocab_size > 0:
            invalid = (input_ids >= self._vocab_size) | (input_ids < 0)
            if invalid.any():
                safe_id = min(self.tokenizer.eos_token_id or 0, self._vocab_size - 1)
                input_ids = input_ids.clone()
                input_ids[invalid] = safe_id

        return input_ids.to(self.device)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for EmberVLM.

        Args:
            images: Tensor of shape [B, C, H, W] with values in [0, 1]

        Returns:
            Preprocessed tensor ready for model
        """
        # UniBench provides images as [B, C, H, W] tensors in [0, 1]
        images = images.to(self.device)

        if self.image_preprocessor is not None:
            # Use EmberVLM's preprocessor
            return self.image_preprocessor(images)
        else:
            # Basic preprocessing: resize and normalize
            from torchvision.transforms.functional import resize, normalize

            images = resize(images, [self.input_resolution, self.input_resolution])
            images = normalize(images, self.norm_mean, self.norm_std)

            return images

    @torch.no_grad()
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings from EmberVLM's vision encoder.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Image embeddings [B, D] or [B, N, D]
        """
        images = self._preprocess_images(images)

        # Encode through vision encoder
        vision_output = self.model.encode_image(images)
        visual_tokens = vision_output["visual_tokens"]

        # Pool to get single embedding per image
        # Use mean pooling over visual tokens
        embeddings = visual_tokens.mean(dim=1)  # [B, D]

        # Normalize for similarity computation
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def get_text_embeddings(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Get text embeddings from EmberVLM's language model.

        Args:
            texts: List of text strings or pre-tokenized tensor

        Returns:
            Text embeddings [B, D]
        """
        if isinstance(texts, torch.Tensor):
            input_ids = texts.to(self.device)
        else:
            input_ids = self._safe_tokenize(texts)

        # Get embeddings from language model
        if hasattr(self.model.language_model, "embed_tokens"):
            token_embeds = self.model.language_model.embed_tokens(input_ids)
        elif hasattr(self.model.language_model, "model"):
            token_embeds = self.model.language_model.model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embedding layer in language model")

        # Mean pool over sequence
        # Mask padding tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        attention_mask = attention_mask.unsqueeze(-1)

        embeddings = (token_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def get_text_from_image(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> List[str]:
        """
        Generate text descriptions for images given prompts.

        This is the main interface for VLM evaluation in UniBench.

        Args:
            images: Batch of images [B, C, H, W]
            prompts: List of text prompts

        Returns:
            List of generated text responses
        """
        images = self._preprocess_images(images)
        batch_size = images.size(0)

        responses = []

        for i in range(batch_size):
            # Get single image
            pixel_values = images[i:i+1]
            prompt = prompts[i] if i < len(prompts) else prompts[0]

            # Tokenize prompt
            input_ids = self._safe_tokenize([prompt])

            try:
                # Generate response
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=False,
                )

                # Decode
                if isinstance(outputs, torch.Tensor):
                    outputs = torch.clamp(outputs, 0, self._vocab_size - 1)
                    prompt_len = input_ids.size(1)
                    response = self.tokenizer.decode(
                        outputs[0][prompt_len:],
                        skip_special_tokens=True,
                    )
                else:
                    response = str(outputs)

                responses.append(response.strip())

            except Exception as e:
                logger.warning(f"[UniBench] Generation error: {e}")
                responses.append("")

        return responses

    def set_classes(self, classes: List[str]):
        """Set classification classes (for zero-shot classification)."""
        self.classes = classes
        self.zeroshot_weights = None  # Reset cached weights

    def set_templates(self, templates: List[str]):
        """Set prompt templates for zero-shot classification."""
        self.templates = templates
        self.zeroshot_weights = None

    def compute_zeroshot_weights(self) -> torch.Tensor:
        """
        Compute zero-shot classification weights.

        Creates text embeddings for each class using templates.
        """
        if self.classes is None:
            raise ValueError("Classes not set. Call set_classes() first.")

        if self.zeroshot_weights is not None:
            return self.zeroshot_weights

        templates = self.templates or ["a photo of a {}."]

        all_weights = []

        for classname in self.classes:
            texts = [t.format(classname) for t in templates]
            embeddings = self.get_text_embeddings(texts)

            # Average over templates
            class_embedding = embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, p=2, dim=0)

            all_weights.append(class_embedding)

        self.zeroshot_weights = torch.stack(all_weights, dim=1)  # [D, num_classes]

        return self.zeroshot_weights

    def get_zeroshot_predictions(
        self,
        images: torch.Tensor,
        zeroshot_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get zero-shot classification predictions.

        Args:
            images: Batch of images [B, C, H, W]
            zeroshot_weights: Optional precomputed weights [D, num_classes]

        Returns:
            Prediction logits [B, num_classes]
        """
        if zeroshot_weights is None:
            zeroshot_weights = self.compute_zeroshot_weights()

        image_embeddings = self.get_image_embeddings(images)

        # Compute similarity
        logits = image_embeddings @ zeroshot_weights  # [B, num_classes]

        # Scale (like CLIP)
        logits = logits * 100.0

        return logits


def create_embervlm_unibench_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    batch_per_gpu: int = 8,
    **kwargs,
) -> Tuple[EmberVLMUniBenchWrapper, List[str]]:
    """
    Create EmberVLM model wrapped for UniBench evaluation.

    Args:
        model_path: Path to EmberVLM checkpoint
        device: Device to use
        dtype: Model dtype
        batch_per_gpu: Batch size per GPU

    Returns:
        (model_wrapper, supported_tasks)
    """
    logger.info(f"[UniBench] Loading EmberVLM from {model_path}")

    # Add EmberVLM to path
    embervlm_root = os.environ.get("EMBERVLM_ROOT")
    if embervlm_root and embervlm_root not in sys.path:
        sys.path.insert(0, embervlm_root)

    from embervlm.models import EmberVLM as EmberVLMModel
    from transformers import AutoTokenizer

    # Load model
    model = EmberVLMModel.from_pretrained(str(model_path))
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load tokenizer
    model_path = Path(model_path)
    tokenizer_path = model_path / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = model_path.parent.parent / "tokenizer"

    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get image size from model config
    image_size = getattr(model.config, "image_size", 224)

    # Create wrapper
    wrapper = EmberVLMUniBenchWrapper(
        model=model,
        model_name="embervlm",
        tokenizer=tokenizer,
        input_resolution=image_size,
        batch_per_gpu=batch_per_gpu,
        device=device,
        **kwargs,
    )

    # Supported task types for EmberVLM
    supported_tasks = [
        "zeroshot_classification",
        "text_classification",
    ]

    return wrapper, supported_tasks


# =============================================================================
# UNIBENCH RUNNER
# =============================================================================

def run_unibench(
    model_path: str,
    benchmarks: List[str],
    output_dir: Path,
    timeout: int = 3600,
    batch_size: int = 8,
    device: str = "cuda",
    max_samples: int = 5000,
) -> Tuple[bool, Optional[Dict[str, float]], str]:
    """
    Run UniBench evaluation on EmberVLM.

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmarks: List of UniBench benchmark names
        output_dir: Output directory for results
        timeout: Timeout in seconds
        batch_size: Evaluation batch size
        device: Device to use
        max_samples: Maximum samples per benchmark

    Returns:
        (success, scores_dict, error_or_output)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[UniBench] Running benchmarks: {benchmarks}")
    logger.info(f"[UniBench] Model: {model_path}")

    try:
        # Check if unibench is installed
        try:
            from unibench.evaluator import Evaluator
            from unibench.models_zoo.registry import register_model
            HAS_UNIBENCH = True
        except ImportError:
            HAS_UNIBENCH = False
            return False, None, "UniBench not installed. Install with: pip install unibench"

        # Filter to supported benchmarks
        valid_benchmarks = [
            b for b in benchmarks
            if b in SUPPORTED_UNIBENCH_BENCHMARKS and b not in UNSUPPORTED_BENCHMARKS
        ]

        if not valid_benchmarks:
            return False, None, f"No supported benchmarks found. Supported: {SUPPORTED_UNIBENCH_BENCHMARKS}"

        logger.info(f"[UniBench] Evaluating: {valid_benchmarks}")

        # Create model
        wrapper, supported_tasks = create_embervlm_unibench_model(
            model_path=model_path,
            device=device,
            batch_per_gpu=batch_size,
        )

        # Register model with UniBench
        from functools import partial

        def embervlm_factory(model_name, **kwargs):
            return wrapper, supported_tasks

        register_model("embervlm", {})(embervlm_factory)

        # Create evaluator
        evaluator = Evaluator(
            models=["embervlm"],
            benchmarks=valid_benchmarks,
            output_dir=str(output_dir),
        )

        # Run evaluation
        evaluator.evaluate(
            device=device,
            batch_per_gpu=batch_size,
            max_num_samples=max_samples,
        )

        # Collect results
        results = {}

        try:
            evaluator.generate_aggregate_results()

            # Parse results from output handler
            for benchmark in valid_benchmarks:
                try:
                    df = evaluator.outputhandler.print_dataframe(
                        benchmark_name=[benchmark],
                        model_name=["embervlm"],
                    )
                    if not df.empty:
                        score = df.values[0][0] * 100  # Convert to percentage
                        results[benchmark] = score
                        logger.info(f"[UniBench] {benchmark}: {score:.2f}%")
                except Exception as e:
                    logger.warning(f"[UniBench] Failed to get results for {benchmark}: {e}")
        except Exception as e:
            logger.warning(f"[UniBench] Failed to aggregate results: {e}")

        if results:
            avg_score = sum(results.values()) / len(results)
            logger.info(f"[UniBench] Average score: {avg_score:.2f}%")
            return True, results, f"Successfully evaluated {len(results)} benchmarks"
        else:
            return False, None, "No benchmark results collected"

    except Exception as e:
        logger.error(f"[UniBench] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, str(e)


def check_unibench_available() -> bool:
    """Check if UniBench is installed and importable."""
    try:
        from unibench.evaluator import Evaluator
        return True
    except ImportError:
        return False


def list_unibench_benchmarks() -> List[str]:
    """List available UniBench benchmarks."""
    try:
        from unibench.benchmarks_zoo import list_benchmarks
        return list_benchmarks("all")
    except ImportError:
        return list(SUPPORTED_UNIBENCH_BENCHMARKS)

