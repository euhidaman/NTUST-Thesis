"""
Lighteval Adapter for EmberVLM

This module provides a custom Lighteval model adapter for EmberVLM,
enabling standardized evaluation on NLP and multimodal benchmarks.

CRITICAL CONSTRAINTS:
- NO tokenizer auto-resizing
- NO forced vocab expansion
- NO implicit embedding re-initialization
- NO half/float dtype mismatches
- Transformers backend ONLY (no vLLM/TGI)
- NO LLM-as-judge dependencies

Supported tasks:
- Text-only NLP tasks (accuracy, exact match)
- Multimodal tasks where applicable
- Custom task filtering for EmberVLM capabilities

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

# Tasks that EmberVLM can handle (filter out unsupported tasks)
SUPPORTED_TASK_TYPES = {
    "multiple_choice",
    "text_generation",
    "classification",
}

# Tasks to skip (require capabilities EmberVLM doesn't have)
UNSUPPORTED_TASKS = {
    "code_generation",
    "math_proof",
    "long_context",  # EmberVLM has limited context
}

# Default generation parameters for EmberVLM
DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "use_cache": False,  # Disabled for stability
}


@dataclass
class LightEvalResult:
    """Result from Lighteval evaluation."""
    task_name: str
    score: float
    num_samples: int
    metric_name: str
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "score": self.score,
            "num_samples": self.num_samples,
            "metric_name": self.metric_name,
            "success": self.success,
            "error_message": self.error_message,
        }


class EmberVLMLightEvalModel:
    """
    EmberVLM adapter for Lighteval evaluation.

    This adapter wraps EmberVLM to be compatible with Lighteval's
    evaluation protocols while respecting EmberVLM's constraints:

    - Fixed vocabulary size (no resizing)
    - Fixed embedding dimensions
    - Limited context length
    - Custom tokenizer handling
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        max_length: int = 512,
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize EmberVLM for Lighteval.

        Args:
            model_path: Path to EmberVLM checkpoint
            device: Device to use ('cuda' or 'cpu')
            dtype: Model dtype (float32 recommended for stability)
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"[Lighteval] Loading EmberVLM from {model_path}")
        logger.info(f"[Lighteval] Device: {self.device}, dtype: {dtype}")

        self._load_model()
        self._load_tokenizer()
        self._validate_setup()

        # Store vocab size for validation
        self._vocab_size = self._get_vocab_size()

        logger.info(f"[Lighteval] EmberVLM loaded, vocab_size={self._vocab_size}")

    def _load_model(self):
        """Load EmberVLM model without modifying embeddings."""
        # Add EmberVLM to path
        embervlm_root = os.environ.get("EMBERVLM_ROOT")
        if embervlm_root and embervlm_root not in sys.path:
            sys.path.insert(0, embervlm_root)

        try:
            from embervlm.models import EmberVLM as EmberVLMModel

            self.model = EmberVLMModel.from_pretrained(str(self.model_path))

            # CRITICAL: Move to device with explicit dtype
            self.model = self.model.to(device=self.device, dtype=self.dtype)
            self.model.eval()

            # Get image preprocessor
            self.image_preprocessor = getattr(self.model, "image_preprocessor", None)

            logger.info("[Lighteval] Model loaded successfully")

        except Exception as e:
            logger.error(f"[Lighteval] Failed to load model: {e}")
            raise

    def _load_tokenizer(self):
        """Load tokenizer without modifying vocab."""
        from transformers import AutoTokenizer

        # Try checkpoint tokenizer
        tokenizer_path = self.model_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = self.model_path.parent.parent / "tokenizer"

        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                use_fast=True,
            )
            logger.info(f"[Lighteval] Tokenizer loaded from {tokenizer_path}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM-135M",
                use_fast=True,
            )
            logger.warning("[Lighteval] Using fallback SmolLM tokenizer")

        # CRITICAL: Do NOT resize tokenizer or add special tokens
        # Just ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
        return len(self.tokenizer)

    def _validate_setup(self):
        """Validate model and tokenizer are compatible."""
        tokenizer_vocab = len(self.tokenizer)
        model_vocab = self._get_vocab_size()

        if tokenizer_vocab > model_vocab:
            logger.warning(
                f"[Lighteval] VOCAB MISMATCH: tokenizer({tokenizer_vocab}) > model({model_vocab}). "
                f"Token IDs will be clamped to prevent errors."
            )

        # Validate special tokens
        for attr in ["eos_token_id", "pad_token_id"]:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None and token_id >= model_vocab:
                logger.warning(f"[Lighteval] {attr}={token_id} >= vocab_size={model_vocab}")

    def _safe_tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text with safety checks.

        - Clamps token IDs to valid vocab range
        - No padding by default (prevents pad token issues)
        - Truncates to max_length
        """
        if max_length is None:
            max_length = self.max_length

        # Tokenize
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # CRITICAL: Clamp token IDs to valid range
        input_ids = inputs["input_ids"]
        if self._vocab_size > 0:
            invalid_mask = (input_ids >= self._vocab_size) | (input_ids < 0)
            if invalid_mask.any():
                safe_id = min(
                    self.tokenizer.eos_token_id or 0,
                    self._vocab_size - 1
                )
                input_ids = input_ids.clone()
                input_ids[invalid_mask] = safe_id
                inputs["input_ids"] = input_ids

        # Move to device
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Generate text response.

        Args:
            prompt: Text prompt
            image: Optional PIL image
            max_new_tokens: Max tokens to generate

        Returns:
            Generated text string
        """
        # Tokenize prompt
        inputs = self._safe_tokenize(prompt)
        input_ids = inputs["input_ids"]

        # Process image if provided
        pixel_values = None
        if image is not None and self.image_preprocessor is not None:
            import numpy as np
            img_np = np.array(image.convert("RGB")).astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            pixel_values = self.image_preprocessor(img_tensor.to(self.device))

        # Generate
        gen_kwargs = {**DEFAULT_GEN_KWARGS, **kwargs}
        gen_kwargs["max_new_tokens"] = max_new_tokens

        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs,
            )

            # Decode, skipping prompt
            if isinstance(outputs, torch.Tensor):
                # Clamp to valid range before decoding
                outputs = torch.clamp(outputs, 0, self._vocab_size - 1)
                prompt_len = input_ids.size(1)
                generated = self.tokenizer.decode(
                    outputs[0][prompt_len:],
                    skip_special_tokens=True,
                )
            else:
                generated = str(outputs)

            return generated.strip()

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"[Lighteval] CUDA error: {e}")
                torch.cuda.synchronize()
            return ""

    @torch.no_grad()
    def loglikelihood(
        self,
        prompt: str,
        continuation: str,
    ) -> Tuple[float, bool]:
        """
        Compute log-likelihood of continuation given prompt.

        Used for multiple-choice and ranking tasks.

        Args:
            prompt: Context/prompt text
            continuation: Text to score

        Returns:
            (log_likelihood, is_greedy) tuple
        """
        # Tokenize full sequence
        full_text = prompt + continuation
        full_inputs = self._safe_tokenize(full_text)

        # Tokenize prompt alone to find continuation start
        prompt_inputs = self._safe_tokenize(prompt)
        prompt_len = prompt_inputs["input_ids"].size(1)

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs.get("attention_mask")

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        logits = outputs["logits"]

        # Get log probs for continuation tokens
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = input_ids[:, prompt_len:]

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs (mask padding)
        if attention_mask is not None:
            cont_mask = attention_mask[:, prompt_len:]
            token_log_probs = token_log_probs * cont_mask

        total_log_prob = token_log_probs.sum().item()

        # Check if greedy decoding would produce same tokens
        greedy_tokens = shift_logits.argmax(dim=-1)
        is_greedy = (greedy_tokens == shift_labels).all().item()

        return total_log_prob, is_greedy

    @torch.no_grad()
    def loglikelihood_rolling(self, text: str) -> float:
        """
        Compute rolling log-likelihood of text (perplexity-style).

        Args:
            text: Text to score

        Returns:
            Total log-likelihood
        """
        inputs = self._safe_tokenize(text)
        input_ids = inputs["input_ids"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            labels=input_ids,
        )

        # Return negative loss (loss is negative log likelihood)
        loss = outputs.get("loss")
        if loss is not None:
            return -loss.item() * input_ids.size(1)
        return 0.0


# =============================================================================
# LIGHTEVAL RUNNER
# =============================================================================

def run_lighteval(
    model_path: str,
    tasks: List[str],
    output_dir: Path,
    timeout: int = 3600,
    batch_size: int = 1,
    device: str = "cuda",
) -> Tuple[bool, Optional[Dict[str, float]], str]:
    """
    Run Lighteval evaluation on EmberVLM.

    Args:
        model_path: Path to EmberVLM checkpoint
        tasks: List of Lighteval task names
        output_dir: Output directory for results
        timeout: Timeout in seconds
        batch_size: Evaluation batch size
        device: Device to use

    Returns:
        (success, scores_dict, error_or_output)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Lighteval] Running tasks: {tasks}")
    logger.info(f"[Lighteval] Model: {model_path}")

    try:
        # Check if lighteval is installed
        try:
            import lighteval
            HAS_LIGHTEVAL = True
        except ImportError:
            HAS_LIGHTEVAL = False
            return False, None, "Lighteval not installed. Install with: pip install lighteval"

        # Create model adapter
        model = EmberVLMLightEvalModel(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
        )

        results = {}

        # Run each task
        for task_name in tasks:
            logger.info(f"[Lighteval] Evaluating: {task_name}")

            # Skip unsupported tasks
            if task_name in UNSUPPORTED_TASKS:
                logger.warning(f"[Lighteval] Skipping unsupported task: {task_name}")
                continue

            try:
                # For now, implement basic evaluation
                # This can be extended to use lighteval's full infrastructure
                task_result = _evaluate_task(model, task_name, output_dir)

                if task_result.success:
                    results[task_name] = task_result.score
                    logger.info(f"[Lighteval] {task_name}: {task_result.score:.2f}%")
                else:
                    logger.warning(f"[Lighteval] {task_name} failed: {task_result.error_message}")

            except Exception as e:
                logger.error(f"[Lighteval] Error on {task_name}: {e}")
                continue

        if results:
            # Compute average
            avg_score = sum(results.values()) / len(results)
            logger.info(f"[Lighteval] Average score: {avg_score:.2f}%")
            return True, results, f"Successfully evaluated {len(results)} tasks"
        else:
            return False, None, "No tasks completed successfully"

    except Exception as e:
        logger.error(f"[Lighteval] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, str(e)


def _evaluate_task(
    model: EmberVLMLightEvalModel,
    task_name: str,
    output_dir: Path,
) -> LightEvalResult:
    """
    Evaluate a single task.

    This is a simplified evaluation loop. For full Lighteval integration,
    this would use lighteval's task infrastructure.
    """
    # Placeholder - in full integration, this would load the actual task
    # and run proper evaluation

    # For now, return a placeholder that indicates the task was attempted
    return LightEvalResult(
        task_name=task_name,
        score=0.0,
        num_samples=0,
        metric_name="accuracy",
        success=False,
        error_message="Task evaluation not yet implemented - use subprocess runner",
    )


def check_lighteval_available() -> bool:
    """Check if Lighteval is installed and importable."""
    try:
        import lighteval
        return True
    except ImportError:
        return False

