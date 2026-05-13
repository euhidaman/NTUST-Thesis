"""
EmberVLM Adapter for VLMEvalKit (FREE MODE ONLY)

This adapter allows EmberVLM to be evaluated using VLMEvalKit's rule-based
evaluation system WITHOUT any LLM judges or paid APIs.

CRITICAL CONSTRAINTS:
- NO padding during generation (prevents CUDA indexing errors)
- NO KV cache (use_cache=False for stability)
- NO tokenizer/config mutation at runtime
- Simple, explicit generation only

SUPPORTED BENCHMARKS (Rule-Based Only):
- MMStar: Multiple choice
- MMBench: Multiple choice
- MMMU: Multiple choice
- ScienceQA: Multiple choice
- AI2D: Multiple choice
- TextVQA: Exact match
- DocVQA: Answer match
- ChartQA: Numeric/exact match

NOT SUPPORTED (Require LLM Judge):
- MathVista (open-ended math)
- MMVet (GPT-4 judge)
- LLaVA-Bench (GPT-4 judge)

Author: EmberVLM Team
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class EmberVLMVLMEval:
    """
    EmberVLM adapter for VLMEvalKit evaluation.

    This class implements the VLMEvalKit model interface for EmberVLM,
    designed specifically for FREE/RULE-BASED evaluation only.

    Key design choices to prevent CUDA errors:
    1. NO padding during tokenization
    2. NO KV cache during generation
    3. Input ID clamping to valid vocab range
    4. Vision token count validation
    5. Safe CPU fallbacks for critical operations
    """

    INSTALL_REQ = False  # No additional installation required
    INTERLEAVE = False   # Does not support interleaved image-text

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """
        Initialize EmberVLM for VLMEvalKit.

        Args:
            model_path: Path to EmberVLM checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (ignored)
        """
        # Get model path from environment if not provided
        if model_path is None:
            model_path = os.environ.get("EMBERVLM_CHECKPOINT")
        if model_path is None:
            raise ValueError(
                "model_path must be provided or EMBERVLM_CHECKPOINT environment variable must be set"
            )

        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        logger.info(f"[EmberVLM-VLMEval] Loading model from {model_path}")
        logger.info(f"[EmberVLM-VLMEval] Device: {self.device}")
        logger.info(f"[EmberVLM-VLMEval] FREE MODE: No LLM judges will be used")

        self._load_model()
        self._load_tokenizer()
        self._validate_setup()

        logger.info(f"[EmberVLM-VLMEval] Model loaded successfully")
        logger.info(f"[EmberVLM-VLMEval] Vocab size: {self._vocab_size}")

    def _load_model(self):
        """Load EmberVLM model."""
        try:
            # Add EmberVLM to path if needed
            embervlm_root = Path(__file__).resolve().parents[2]
            if str(embervlm_root) not in sys.path:
                sys.path.insert(0, str(embervlm_root))

            from embervlm.models import EmberVLM as EmberVLMModel

            self.model = EmberVLMModel.from_pretrained(str(self.model_path))
            self.model = self.model.to(self.device).eval()

            # Get image preprocessor
            self.image_preprocessor = getattr(self.model, "image_preprocessor", None)

        except ImportError as e:
            logger.error(f"[EmberVLM-VLMEval] Failed to import EmberVLM: {e}")
            logger.error(
                "Make sure EmberVLM is installed or PYTHONPATH includes the EmberVLM directory"
            )
            raise
        except Exception as e:
            logger.error(f"[EmberVLM-VLMEval] Failed to load model: {e}")
            raise

    def _load_tokenizer(self):
        """Load tokenizer with fallback."""
        from transformers import AutoTokenizer

        # Try checkpoint tokenizer directory
        tokenizer_path = self.model_path / "tokenizer"
        if not tokenizer_path.exists():
            # Try parent directory structure
            tokenizer_path = self.model_path.parent.parent / "tokenizer"

        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"[EmberVLM-VLMEval] Loaded tokenizer from {tokenizer_path}")
        else:
            # Fallback to SmolLM tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            logger.warning(
                "[EmberVLM-VLMEval] Using fallback SmolLM-135M tokenizer"
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _validate_setup(self):
        """Validate model/tokenizer setup and get vocab size."""
        # Get actual vocab size from embeddings
        self._vocab_size = None
        try:
            if hasattr(self.model, "language_model"):
                lm = self.model.language_model
                if hasattr(lm, "model") and hasattr(lm.model, "get_input_embeddings"):
                    emb = lm.model.get_input_embeddings()
                    if emb is not None:
                        self._vocab_size = emb.weight.shape[0]
                elif hasattr(lm, "get_input_embeddings"):
                    emb = lm.get_input_embeddings()
                    if emb is not None:
                        self._vocab_size = emb.weight.shape[0]
        except Exception as e:
            logger.warning(f"[EmberVLM-VLMEval] Could not get embedding size: {e}")

        if self._vocab_size is None:
            self._vocab_size = len(self.tokenizer)
            logger.warning(
                f"[EmberVLM-VLMEval] Using tokenizer vocab size as fallback: {self._vocab_size}"
            )

        # Validate tokenizer/model alignment
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size != self._vocab_size:
            logger.warning(
                f"[EmberVLM-VLMEval] VOCAB MISMATCH: "
                f"tokenizer={tokenizer_size}, model={self._vocab_size}"
            )
            # Use smaller of the two to be safe
            self._vocab_size = min(self._vocab_size, tokenizer_size)

    def _safe_tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text with safety checks.

        NO PADDING to avoid CUDA indexing errors with pad tokens.
        Truncates to safe length and clamps any out-of-vocab IDs.
        """
        # Determine max length
        max_len = 1024
        if hasattr(self.model, "config"):
            cfg = self.model.config
            max_pos = getattr(cfg, "language_max_length", 1024)
            num_visual = getattr(cfg, "num_visual_tokens", 8)
            # Leave room for visual tokens and some generation
            max_len = min(max_len, max_pos - num_visual - 50)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,  # CRITICAL: No padding
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"]

        # Clamp to valid vocab range on CPU for safety
        input_ids_cpu = input_ids.cpu()
        oov_mask = (input_ids_cpu >= self._vocab_size) | (input_ids_cpu < 0)
        if oov_mask.any():
            num_oov = oov_mask.sum().item()
            safe_id = min(self.tokenizer.eos_token_id or 0, self._vocab_size - 1)
            logger.warning(
                f"[EmberVLM-VLMEval] Clamping {num_oov} out-of-vocab tokens to {safe_id}"
            )
            input_ids_cpu = input_ids_cpu.clone()
            input_ids_cpu[oov_mask] = safe_id

        return input_ids_cpu.to(self.device)

    def _load_image(self, image_input) -> Optional[torch.Tensor]:
        """
        Load and preprocess an image.

        Args:
            image_input: PIL Image, path string, or numpy array

        Returns:
            Preprocessed pixel values tensor or None
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert("RGB")
            else:
                logger.warning(f"[EmberVLM-VLMEval] Unknown image type: {type(image_input)}")
                return None

            if self.image_preprocessor is None:
                logger.warning("[EmberVLM-VLMEval] No image preprocessor available")
                return None

            # Convert to tensor
            img_np = np.array(img).astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device, dtype=torch.float32)

            # Preprocess
            pixel_values = self.image_preprocessor(img_tensor)
            return pixel_values.to(self.device)

        except Exception as e:
            logger.error(f"[EmberVLM-VLMEval] Image loading failed: {e}")
            return None

    def use_custom_prompt(self, dataset: str) -> bool:
        """Check if custom prompt should be used for dataset."""
        # Use simple prompt for all datasets
        return False

    def generate_inner(self, message: List[Dict], dataset: str = None) -> str:
        """
        Generate response for VLMEvalKit format input.

        Args:
            message: List of dicts with 'type' ('text' or 'image') and 'value'
            dataset: Name of the dataset (for logging)

        Returns:
            Generated text response
        """
        # Extract text and images from message
        text_parts = []
        image_input = None

        for item in message:
            item_type = item.get("type", "")
            item_value = item.get("value", "")

            if item_type == "text":
                text_parts.append(str(item_value))
            elif item_type == "image":
                image_input = item_value

        # Build prompt
        prompt = "\n".join(text_parts)
        # Remove any image placeholders
        prompt = prompt.replace("<|image|>", "").replace("<image>", "")
        prompt = prompt.replace("<img>", "").replace("</img>", "")
        prompt = prompt.strip()

        if not prompt:
            prompt = "Describe what you see in the image."

        # Tokenize
        input_ids = self._safe_tokenize(prompt)

        # Process image
        pixel_values = None
        image_positions = None
        if image_input is not None:
            pixel_values = self._load_image(image_input)
            if pixel_values is not None:
                image_positions = torch.zeros(1, dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=None,  # No attention mask (no padding)
                    image_positions=image_positions,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic for evaluation
                    temperature=1.0,
                    top_k=50,
                    top_p=1.0,
                    use_cache=False,  # Disable KV cache for stability
                )

                # Decode output
                if isinstance(outputs, torch.Tensor):
                    # Clamp outputs to valid range
                    outputs = torch.clamp(outputs, 0, self._vocab_size - 1)

                    prompt_len = input_ids.size(1)
                    generated = self.tokenizer.decode(
                        outputs[0][prompt_len:], skip_special_tokens=True
                    )
                else:
                    generated = str(outputs)

                return generated.strip()

            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e).lower():
                    logger.error(f"[EmberVLM-VLMEval] CUDA error: {e}")
                    # Try to recover
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except:
                        pass
                    return ""
                raise

    def generate(self, message: List[Dict], dataset: str = None) -> str:
        """
        VLMEvalKit interface for generation.

        This is the main entry point called by VLMEvalKit.
        """
        return self.generate_inner(message, dataset)


# For VLMEvalKit registration
def get_model_class():
    """Return the model class for VLMEvalKit registration."""
    return EmberVLMVLMEval

