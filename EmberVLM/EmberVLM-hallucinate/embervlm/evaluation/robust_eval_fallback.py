"""
Robust Evaluation Fallback System for EmberVLM

This module implements a two-tier evaluation system:
1. PRIMARY: lmms-eval framework
2. FALLBACK: VLMEvalKit (FREE MODE ONLY - no paid APIs, no LLM judges)

CRITICAL CONSTRAINTS:
- NO paid APIs (OpenAI, Anthropic, Google, etc.)
- NO LLM-as-a-judge evaluation
- VLMEvalKit runs ONLY in rule-based mode (exact match, regex, multiple choice)
- Everything runs FULLY OFFLINE

Author: EmberVLM Team
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Failure detection patterns for lmms-eval
CUDA_ERROR_PATTERNS = [
    "CUDA error",
    "device-side assert triggered",
    "cudaErrorAssert",
    "RuntimeError: CUDA",
    "torch.cuda.OutOfMemoryError",
    "indexSelectSmallIndex",
    "Assertion `srcIndex < srcSelectDimSize` failed",
]

KNOWN_INCOMPATIBILITY_PATTERNS = [
    "token ID out of bounds",
    "embedding index out of range",
    "pad_token_id",
    "eos_token_id",
    "IndexError: index",
    "KeyError:",
]

# Benchmark mapping between lmms-eval and VLMEvalKit
BENCHMARK_MAPPING = {
    # lmms-eval task name -> VLMEvalKit dataset name
    "mathvista_testmini": "MathVista_MINI",
    "mmmu_val": "MMMU_DEV_VAL",
    "mmstar": "MMStar",
    "mmbench_en_dev": "MMBench_DEV_EN",
    "textvqa_val": "TextVQA_VAL",
    "docvqa_test": "DocVQA_TEST",
    "ai2d": "AI2D_TEST",
    "scienceqa_img": "ScienceQA_VAL",
    "chartqa": "ChartQA_TEST",
}

# VLMEvalKit benchmarks that support FREE (rule-based) evaluation
# These do NOT require LLM judges
VLMEVAL_FREE_BENCHMARKS = {
    "MMStar",           # Multiple choice - rule based
    "MMBench_DEV_EN",   # Multiple choice - rule based
    "MMBench_TEST_EN",  # Multiple choice - rule based
    "MMMU_DEV_VAL",     # Multiple choice - rule based
    "ScienceQA_VAL",    # Multiple choice - rule based
    "AI2D_TEST",        # Multiple choice - rule based
    "TextVQA_VAL",      # Exact match scoring
    "DocVQA_VAL",       # Answer match scoring
    "DocVQA_TEST",      # Answer match scoring
    "ChartQA_TEST",     # Numeric/exact match
}

# Benchmarks that REQUIRE LLM judge (must be SKIPPED in fallback)
VLMEVAL_JUDGE_REQUIRED = {
    "MathVista_MINI",   # Requires LLM judge for open-ended math
    "MMVet",            # Requires GPT-4 judge
    "LLaVA-Bench",      # Requires GPT-4 judge
    "POPE",             # Can use rule-based, but often uses judge
}


@dataclass
class EvalResult:
    """Result from a single benchmark evaluation."""
    benchmark: str
    score: float
    framework: str  # 'lmms-eval' or 'vlmeval'
    success: bool
    error_message: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "score": self.score,
            "framework": self.framework,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class FallbackEvalResult:
    """Aggregated result from multi-tier evaluation system."""
    results: Dict[str, EvalResult]
    aggregate_score: float
    vlmeval_used: int          # Primary: VLMEvalKit
    lighteval_used: int        # Tier 2: Lighteval
    unibench_used: int         # Tier 3: UniBench
    lmms_eval_used: int        # Fallback: lmms-eval
    total_benchmarks: int
    failed_benchmarks: List[str]
    framework_summary: Dict[str, List[str]]

    # Legacy alias for backwards compatibility
    @property
    def vlmeval_fallback_used(self) -> int:
        return self.vlmeval_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "aggregate_score": self.aggregate_score,
            "vlmeval_used": self.vlmeval_used,
            "lighteval_used": self.lighteval_used,
            "unibench_used": self.unibench_used,
            "lmms_eval_used": self.lmms_eval_used,
            "total_benchmarks": self.total_benchmarks,
            "failed_benchmarks": self.failed_benchmarks,
            "framework_summary": self.framework_summary,
        }


# =============================================================================
# FAILURE DETECTION
# =============================================================================

def detect_lmms_eval_failure(
    returncode: int,
    stdout: str,
    stderr: str,
    output_dir: Path,
) -> Tuple[bool, str]:
    """
    Detect if lmms-eval has failed and determine the reason.

    Returns:
        (failed: bool, reason: str)
    """
    # Check exit code
    if returncode != 0:
        return True, f"Non-zero exit code: {returncode}"

    # Check for CUDA errors
    combined_output = (stdout or "") + (stderr or "")
    for pattern in CUDA_ERROR_PATTERNS:
        if pattern.lower() in combined_output.lower():
            return True, f"CUDA error detected: {pattern}"

    # Check for known incompatibility errors
    for pattern in KNOWN_INCOMPATIBILITY_PATTERNS:
        if pattern.lower() in combined_output.lower():
            return True, f"Incompatibility error: {pattern}"

    # Check if result files were generated
    if output_dir.exists():
        result_files = list(output_dir.rglob("*_results.json")) + \
                      list(output_dir.rglob("results.json"))
        if not result_files:
            return True, "No result files generated"

    return False, ""


# =============================================================================
# LMMS-EVAL RUNNER
# =============================================================================

def run_lmms_eval(
    model_path: str,
    benchmark: str,
    output_dir: Path,
    lmms_eval_repo: str,
    timeout: int = 3600,
) -> Tuple[bool, Optional[float], str]:
    """
    Run a single benchmark using lmms-eval.

    Returns:
        (success: bool, score: Optional[float], error_or_output: str)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model", "embervlm",
        "--tasks", benchmark,
        "--model_args", f"pretrained={model_path}",
        "--batch_size", "1",
        "--log_samples",
        "--output_path", str(output_dir),
    ]

    env = os.environ.copy()
    env["EMBERVLM_CHECKPOINT"] = str(model_path)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Disable distributed training environment variables
    for var in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        env.pop(var, None)

    logger.info(f"[lmms-eval] Running: {benchmark}")
    logger.info(f"[lmms-eval] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=lmms_eval_repo,
            timeout=timeout,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Detect failure
        failed, reason = detect_lmms_eval_failure(
            result.returncode, stdout, stderr, output_dir
        )

        if failed:
            logger.warning(f"[lmms-eval] {benchmark} FAILED: {reason}")
            return False, None, reason

        # Parse results
        result_files = list(output_dir.rglob("*_results.json"))
        if not result_files:
            result_files = list(output_dir.rglob("results.json"))

        if not result_files:
            return False, None, "No result files found after successful run"

        result_file = max(result_files, key=lambda p: p.stat().st_mtime)
        with open(result_file, "r") as f:
            results_data = json.load(f)

        # Extract score
        score = None
        if "results" in results_data:
            task_results = results_data["results"].get(benchmark, {})
            for metric in ["accuracy", "acc", "score", "exact_match", "average", "overall"]:
                if metric in task_results:
                    score = float(task_results[metric])
                    # Convert to percentage if needed
                    if score <= 1.0:
                        score *= 100
                    break

            # Fallback: try any numeric value
            if score is None:
                for key, value in task_results.items():
                    if isinstance(value, (int, float)):
                        score = float(value)
                        if score <= 1.0:
                            score *= 100
                        break

        if score is not None:
            logger.info(f"[lmms-eval] {benchmark} SUCCESS: {score:.2f}%")
            return True, score, stdout

        return False, None, "Could not parse score from results"

    except subprocess.TimeoutExpired:
        return False, None, f"Timeout after {timeout}s"
    except Exception as e:
        return False, None, str(e)


# =============================================================================
# VLMEVALKIT ADAPTER FOR EMBERVLM (FREE MODE)
# =============================================================================

class EmberVLMAdapter:
    """
    VLMEvalKit adapter for EmberVLM.

    IMPORTANT: This adapter is designed for FREE/RULE-BASED evaluation only.
    - NO padding during generation (prevents pad token issues)
    - NO KV cache (use_cache=False for stability)
    - NO tokenizer/config mutation
    - Simple, explicit generation
    """

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize EmberVLM for VLMEvalKit evaluation.

        Args:
            model_path: Path to EmberVLM checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        import torch
        from transformers import AutoTokenizer
        from PIL import Image

        self.device = torch.device(device)
        self.model_path = model_path

        logger.info(f"[VLMEvalKit] Loading EmberVLM from {model_path}")

        # Load model
        try:
            from embervlm.models import EmberVLM as EmberVLMModel
            self.model = EmberVLMModel.from_pretrained(model_path)
            self.model = self.model.to(self.device).eval()
        except Exception as e:
            logger.error(f"[VLMEvalKit] Failed to load model: {e}")
            raise

        # Load tokenizer
        tokenizer_path = Path(model_path) / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = Path(model_path).parent.parent / "tokenizer"

        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            logger.warning("[VLMEvalKit] Using fallback tokenizer")

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get vocab size for validation
        self._vocab_size = self._get_vocab_size()

        # Image preprocessor
        self.image_preprocessor = getattr(self.model, "image_preprocessor", None)

        # Generation kwargs - EXPLICIT, NO HIDDEN STATE
        self.gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "do_sample": False,  # Deterministic for evaluation
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "use_cache": False,  # Disable KV cache for stability
        }

        logger.info(f"[VLMEvalKit] EmberVLM loaded successfully on {device}")
        logger.info(f"[VLMEvalKit] Vocab size: {self._vocab_size}")

    def _get_vocab_size(self) -> int:
        """Get the actual embedding vocabulary size."""
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

    def _safe_tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text with safety checks.

        - No padding (prevents pad token issues)
        - Truncation to prevent position overflow
        - Clamps any out-of-vocab tokens
        """
        # Get max length
        max_len = 1024
        if hasattr(self.model, "config"):
            max_pos = getattr(self.model.config, "language_max_length", 1024)
            num_visual = getattr(self.model.config, "num_visual_tokens", 8)
            max_len = min(max_len, max_pos - num_visual - 10)  # Leave room for visual tokens

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,  # NO PADDING
            truncation=True,
            max_length=max_len,
        )

        input_ids = inputs["input_ids"]

        # Clamp to valid vocab range
        if self._vocab_size > 0:
            oov_mask = (input_ids >= self._vocab_size) | (input_ids < 0)
            if oov_mask.any():
                safe_id = min(self.tokenizer.eos_token_id or 0, self._vocab_size - 1)
                input_ids = input_ids.clone()
                input_ids[oov_mask] = safe_id

        return input_ids

    def _load_image(self, image_path: str):
        """Load and preprocess image."""
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")

        if self.image_preprocessor is not None:
            img_np = np.array(img).astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            pixel_values = self.image_preprocessor(img_tensor.to(self.device))
            return pixel_values

        return None

    def generate_inner(self, message: List[Dict], dataset: str = None) -> str:
        """
        Generate response for a single input.

        Args:
            message: List of dicts with 'type' ('text' or 'image') and 'value'
            dataset: Name of the dataset (for dataset-specific prompting)

        Returns:
            Generated text response
        """
        # Extract text and images from message
        text_parts = []
        image_path = None

        for item in message:
            if item["type"] == "text":
                text_parts.append(item["value"])
            elif item["type"] == "image":
                image_path = item["value"]

        prompt = "\n".join(text_parts)

        # Clean up prompt (remove image placeholders)
        prompt = prompt.replace("<|image|>", "").replace("<image>", "").strip()

        # Tokenize
        input_ids = self._safe_tokenize(prompt).to(self.device)

        # Load image if present
        pixel_values = None
        if image_path:
            pixel_values = self._load_image(image_path)

        # Prepare image positions
        image_positions = None
        if pixel_values is not None:
            image_positions = torch.zeros(1, dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=None,  # No explicit attention mask (no padding)
                    image_positions=image_positions,
                    **self.gen_kwargs,
                )

                # Decode
                if isinstance(outputs, torch.Tensor):
                    # Clamp outputs to valid range
                    if self._vocab_size > 0:
                        outputs = torch.clamp(outputs, 0, self._vocab_size - 1)

                    prompt_len = input_ids.size(1)
                    generated_text = self.tokenizer.decode(
                        outputs[0][prompt_len:], skip_special_tokens=True
                    )
                else:
                    generated_text = str(outputs)

                return generated_text.strip()

            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"[VLMEvalKit] CUDA error in generation: {e}")
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass
                    return ""
                raise

    def generate(self, message: List[Dict], dataset: str = None) -> str:
        """Generate response (VLMEvalKit interface)."""
        return self.generate_inner(message, dataset)


# =============================================================================
# VLMEVALKIT RUNNER (FREE MODE ONLY)
# =============================================================================

def run_vlmeval_free_mode(
    model_path: str,
    benchmark: str,
    output_dir: Path,
    vlmeval_repo: str,
    timeout: int = 3600,
) -> Tuple[bool, Optional[float], str]:
    """
    Run benchmark using VLMEvalKit in FREE mode (no LLM judge).

    IMPORTANT: This function ONLY runs benchmarks that support rule-based evaluation.
    Benchmarks requiring LLM judges are REJECTED.

    Returns:
        (success: bool, score: Optional[float], error_or_output: str)
    """
    # Map lmms-eval benchmark name to VLMEvalKit name
    vlmeval_benchmark = BENCHMARK_MAPPING.get(benchmark, benchmark)

    # Check if this benchmark supports free evaluation
    if vlmeval_benchmark not in VLMEVAL_FREE_BENCHMARKS:
        if vlmeval_benchmark in VLMEVAL_JUDGE_REQUIRED:
            return False, None, (
                f"Benchmark {vlmeval_benchmark} requires LLM judge (paid API). "
                f"Skipping per FREE MODE constraint."
            )
        return False, None, f"Benchmark {vlmeval_benchmark} not supported in VLMEvalKit free mode"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command - VLMEvalKit expects model name that's registered in config.py
    # EmberVLM reads model_path from EMBERVLM_CHECKPOINT environment variable
    cmd = [
        sys.executable, "run.py",
        "--data", vlmeval_benchmark,
        "--model", "EmberVLM",
        "--work-dir", str(output_dir),
        "--mode", "all",  # Run inference and eval
        # NO --judge flag = no LLM judge (FREE MODE)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # CRITICAL: Set model path for EmberVLM to load
    env["EMBERVLM_CHECKPOINT"] = str(model_path)

    # Set PYTHONPATH to include EmberVLM
    embervlm_root = str(Path(__file__).resolve().parents[2])
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{embervlm_root}{os.pathsep}{existing_path}" if existing_path else embervlm_root

    # CRITICAL: Disable any automatic LLM judge environment variables
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)

    logger.info(f"[VLMEvalKit-FREE] Running: {vlmeval_benchmark}")
    logger.info(f"[VLMEvalKit-FREE] NO LLM judge will be used (free mode)")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=vlmeval_repo,
            timeout=timeout,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0:
            logger.warning(f"[VLMEvalKit-FREE] {vlmeval_benchmark} failed with code {result.returncode}")
            return False, None, f"Exit code {result.returncode}: {stderr[:500]}"

        # Parse results from VLMEvalKit output
        # VLMEvalKit saves results to work-dir/{model_name}/{benchmark}_*.csv
        result_files = list(output_dir.rglob(f"*{vlmeval_benchmark}*.csv"))
        if not result_files:
            result_files = list(output_dir.rglob("*.csv"))

        score = None
        if result_files:
            import csv
            result_file = max(result_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(result_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try common score columns
                        for col in ["Overall", "accuracy", "Accuracy", "score", "Score"]:
                            if col in row and row[col]:
                                try:
                                    score = float(row[col])
                                    if score <= 1.0:
                                        score *= 100
                                    break
                                except ValueError:
                                    continue
                        if score is not None:
                            break
            except Exception as e:
                logger.warning(f"[VLMEvalKit-FREE] Failed to parse CSV: {e}")

        # Also try to extract from stdout
        if score is None:
            # Look for patterns like "Overall: 0.xxx" or "accuracy: 0.xxx"
            patterns = [
                r"Overall[:\s]+(\d+\.?\d*)",
                r"[Aa]ccuracy[:\s]+(\d+\.?\d*)",
                r"[Ss]core[:\s]+(\d+\.?\d*)",
            ]
            for pattern in patterns:
                match = re.search(pattern, stdout)
                if match:
                    try:
                        score = float(match.group(1))
                        if score <= 1.0:
                            score *= 100
                        break
                    except ValueError:
                        continue

        if score is not None:
            logger.info(f"[VLMEvalKit-FREE] {vlmeval_benchmark} SUCCESS: {score:.2f}%")
            return True, score, stdout

        return False, None, "Could not parse score from VLMEvalKit output"

    except subprocess.TimeoutExpired:
        return False, None, f"Timeout after {timeout}s"
    except Exception as e:
        return False, None, str(e)


# =============================================================================
# LIGHTEVAL RUNNER (TIER 2)
# =============================================================================

def run_lighteval_benchmark(
    model_path: str,
    benchmark: str,
    output_dir: Path,
    timeout: int = 3600,
) -> Tuple[bool, Optional[float], str]:
    """
    Run benchmark using Lighteval (Tier 2).

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmark: Benchmark name
        output_dir: Output directory
        timeout: Timeout in seconds

    Returns:
        (success, score, error_or_output)
    """
    if not HAS_LIGHTEVAL:
        return False, None, "Lighteval not available"

    # Map to lighteval task name if needed
    lighteval_task = LIGHTEVAL_TASK_MAPPING.get(benchmark, benchmark)

    logger.info(f"[Lighteval] Running: {benchmark}")

    try:
        success, results, output = run_lighteval(
            model_path=model_path,
            tasks=[lighteval_task],
            output_dir=output_dir,
            timeout=timeout,
        )

        if success and results:
            score = list(results.values())[0] if results else None
            if score is not None:
                logger.info(f"[Lighteval] {benchmark} SUCCESS: {score:.2f}%")
                return True, score, output

        return False, None, output or "No results from Lighteval"

    except Exception as e:
        logger.error(f"[Lighteval] {benchmark} error: {e}")
        return False, None, str(e)


# =============================================================================
# UNIBENCH RUNNER (TIER 3)
# =============================================================================

def run_unibench_benchmark(
    model_path: str,
    benchmark: str,
    output_dir: Path,
    timeout: int = 3600,
) -> Tuple[bool, Optional[float], str]:
    """
    Run benchmark using UniBench (Tier 3).

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmark: Benchmark name
        output_dir: Output directory
        timeout: Timeout in seconds

    Returns:
        (success, score, error_or_output)
    """
    if not HAS_UNIBENCH:
        return False, None, "UniBench not available"

    # Map to unibench benchmark name
    unibench_name = UNIBENCH_BENCHMARK_MAPPING.get(benchmark, benchmark)

    # Check if benchmark is supported by UniBench
    if unibench_name not in SUPPORTED_UNIBENCH_BENCHMARKS:
        return False, None, f"Benchmark {benchmark} not supported by UniBench"

    logger.info(f"[UniBench] Running: {benchmark}")

    try:
        success, results, output = run_unibench(
            model_path=model_path,
            benchmarks=[unibench_name],
            output_dir=output_dir,
            timeout=timeout,
        )

        if success and results:
            score = list(results.values())[0] if results else None
            if score is not None:
                logger.info(f"[UniBench] {benchmark} SUCCESS: {score:.2f}%")
                return True, score, output

        return False, None, output or "No results from UniBench"

    except Exception as e:
        logger.error(f"[UniBench] {benchmark} error: {e}")
        return False, None, str(e)


# =============================================================================
# MAIN FOUR-TIER EVALUATION RUNNER
# =============================================================================

def run_evaluation_with_fallback(
    model_path: str,
    benchmarks: List[str],
    output_dir: str,
    lmms_eval_repo: str,
    vlmeval_repo: Optional[str] = None,
    timeout_per_benchmark: int = 3600,
    skip_lmms_eval_fallback: bool = False,
    enable_lighteval: bool = True,
    enable_unibench: bool = True,
) -> FallbackEvalResult:
    """
    Run evaluation with four-tier fallback system.

    EVALUATION ORDER:
    1. VLMEvalKit (FREE MODE) - Primary, rule-based scoring
    2. Lighteval - NLP and multimodal benchmarks
    3. UniBench - Visual reasoning benchmarks
    4. lmms-eval - Final fallback

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmarks: List of benchmark names (lmms-eval format)
        output_dir: Output directory for all results
        lmms_eval_repo: Path to lmms-eval repository
        vlmeval_repo: Path to VLMEvalKit repository (optional)
        timeout_per_benchmark: Timeout in seconds per benchmark
        skip_lmms_eval_fallback: If True, don't use lmms-eval fallback
        enable_lighteval: If True, use Lighteval as tier 2
        enable_unibench: If True, use UniBench as tier 3

    Returns:
        FallbackEvalResult with all benchmark results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, EvalResult] = {}
    vlmeval_count = 0
    lighteval_count = 0
    unibench_count = 0
    lmms_eval_count = 0
    failed_benchmarks = []
    framework_summary = {
        "vlmeval": [],
        "lighteval": [],
        "unibench": [],
        "lmms-eval": [],
        "failed": []
    }

    logger.info("=" * 80)
    logger.info("FOUR-TIER EVALUATION SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Benchmarks: {benchmarks}")
    logger.info("")
    logger.info("EVALUATION ORDER:")
    logger.info("  1. VLMEvalKit (FREE MODE) - Primary")
    if enable_lighteval and HAS_LIGHTEVAL:
        logger.info("  2. Lighteval - Tier 2")
    if enable_unibench and HAS_UNIBENCH:
        logger.info("  3. UniBench - Tier 3")
    if not skip_lmms_eval_fallback:
        logger.info("  4. lmms-eval - Final fallback")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️  IMPORTANT CONSTRAINTS:")
    logger.info("    - NO paid APIs (OpenAI, Anthropic, Google)")
    logger.info("    - NO LLM-as-a-judge evaluation")
    logger.info("    - All frameworks use rule-based scoring")
    logger.info("    - Everything runs FULLY OFFLINE")
    logger.info("")

    for benchmark in benchmarks:
        logger.info("-" * 60)
        logger.info(f"Evaluating: {benchmark}")
        logger.info("-" * 60)

        tier = 1
        total_tiers = 1 + int(enable_lighteval and HAS_LIGHTEVAL) + \
                      int(enable_unibench and HAS_UNIBENCH) + \
                      int(not skip_lmms_eval_fallback)

        # === TIER 1: VLMEVALKIT (PRIMARY) ===
        vlmeval_success = False
        vlmeval_score = None
        vlmeval_output = "VLMEvalKit repo not provided"

        if vlmeval_repo is not None:
            logger.info(f"[{tier}/{total_tiers}] Trying VLMEvalKit (FREE MODE)...")

            vlmeval_output_dir = output_dir / f"vlmeval_{benchmark}"

            vlmeval_success, vlmeval_score, vlmeval_output = run_vlmeval_free_mode(
                model_path=model_path,
                benchmark=benchmark,
                output_dir=vlmeval_output_dir,
                vlmeval_repo=vlmeval_repo,
                timeout=timeout_per_benchmark,
            )

            if vlmeval_success and vlmeval_score is not None:
                results[benchmark] = EvalResult(
                    benchmark=benchmark,
                    score=vlmeval_score,
                    framework="vlmeval",
                    success=True,
                )
                vlmeval_count += 1
                framework_summary["vlmeval"].append(benchmark)
                logger.info(f"✓ {benchmark}: {vlmeval_score:.2f}% (VLMEvalKit)")
                continue
            else:
                logger.warning(f"[VLMEvalKit] {benchmark} failed: {vlmeval_output}")
        else:
            logger.info(f"[{tier}/{total_tiers}] VLMEvalKit repo not provided, skipping...")

        tier += 1

        # === TIER 2: LIGHTEVAL ===
        if enable_lighteval and HAS_LIGHTEVAL:
            logger.info(f"[{tier}/{total_tiers}] Trying Lighteval...")

            lighteval_output_dir = output_dir / f"lighteval_{benchmark}"

            lighteval_success, lighteval_score, lighteval_output = run_lighteval_benchmark(
                model_path=model_path,
                benchmark=benchmark,
                output_dir=lighteval_output_dir,
                timeout=timeout_per_benchmark,
            )

            if lighteval_success and lighteval_score is not None:
                results[benchmark] = EvalResult(
                    benchmark=benchmark,
                    score=lighteval_score,
                    framework="lighteval",
                    success=True,
                )
                lighteval_count += 1
                framework_summary["lighteval"].append(benchmark)
                logger.info(f"✓ {benchmark}: {lighteval_score:.2f}% (Lighteval)")
                continue
            else:
                logger.warning(f"[Lighteval] {benchmark} failed: {lighteval_output}")

            tier += 1

        # === TIER 3: UNIBENCH ===
        if enable_unibench and HAS_UNIBENCH:
            logger.info(f"[{tier}/{total_tiers}] Trying UniBench...")

            unibench_output_dir = output_dir / f"unibench_{benchmark}"

            unibench_success, unibench_score, unibench_output = run_unibench_benchmark(
                model_path=model_path,
                benchmark=benchmark,
                output_dir=unibench_output_dir,
                timeout=timeout_per_benchmark,
            )

            if unibench_success and unibench_score is not None:
                results[benchmark] = EvalResult(
                    benchmark=benchmark,
                    score=unibench_score,
                    framework="unibench",
                    success=True,
                )
                unibench_count += 1
                framework_summary["unibench"].append(benchmark)
                logger.info(f"✓ {benchmark}: {unibench_score:.2f}% (UniBench)")
                continue
            else:
                logger.warning(f"[UniBench] {benchmark} failed: {unibench_output}")

            tier += 1

        # === TIER 4: LMMS-EVAL (FINAL FALLBACK) ===
        if skip_lmms_eval_fallback:
            logger.warning(f"[SKIP] lmms-eval fallback disabled")
            results[benchmark] = EvalResult(
                benchmark=benchmark,
                score=0.0,
                framework="none",
                success=False,
                error_message=f"All tiers failed. Last error: {vlmeval_output}",
            )
            failed_benchmarks.append(benchmark)
            framework_summary["failed"].append(benchmark)
            continue

        logger.info(f"[{tier}/{total_tiers}] Falling back to lmms-eval...")

        lmms_output_dir = output_dir / f"lmms_eval_{benchmark}"

        lmms_success, lmms_score, lmms_output = run_lmms_eval(
            model_path=model_path,
            benchmark=benchmark,
            output_dir=lmms_output_dir,
            lmms_eval_repo=lmms_eval_repo,
            timeout=timeout_per_benchmark,
        )

        if lmms_success and lmms_score is not None:
            results[benchmark] = EvalResult(
                benchmark=benchmark,
                score=lmms_score,
                framework="lmms-eval",
                success=True,
            )
            lmms_eval_count += 1
            framework_summary["lmms-eval"].append(benchmark)
            logger.info(f"✓ {benchmark}: {lmms_score:.2f}% (lmms-eval)")
        else:
            logger.warning(f"[lmms-eval] {benchmark} failed: {lmms_output}")
            # All tiers failed
            results[benchmark] = EvalResult(
                benchmark=benchmark,
                score=0.0,
                framework="none",
                success=False,
                error_message=f"All tiers failed. Last error: {lmms_output}",
            )
            failed_benchmarks.append(benchmark)
            framework_summary["failed"].append(benchmark)
            logger.error(f"✗ {benchmark}: All evaluation tiers failed")

    # === COMPUTE AGGREGATE ===
    successful_scores = [r.score for r in results.values() if r.success]
    aggregate_score = sum(successful_scores) / len(successful_scores) if successful_scores else 0.0

    # === LOG SUMMARY ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY (FOUR-TIER SYSTEM)")
    logger.info("=" * 80)
    logger.info(f"Total benchmarks: {len(benchmarks)}")
    logger.info(f"VLMEvalKit (Tier 1) succeeded: {vlmeval_count}")
    logger.info(f"Lighteval (Tier 2) succeeded: {lighteval_count}")
    logger.info(f"UniBench (Tier 3) succeeded: {unibench_count}")
    logger.info(f"lmms-eval (Tier 4) succeeded: {lmms_eval_count}")
    logger.info(f"Failed: {len(failed_benchmarks)}")
    logger.info(f"Aggregate score: {aggregate_score:.2f}%")
    logger.info("")

    if framework_summary["vlmeval"]:
        logger.info(f"[VLMEvalKit] {', '.join(framework_summary['vlmeval'])}")
    if framework_summary["lighteval"]:
        logger.info(f"[Lighteval] {', '.join(framework_summary['lighteval'])}")
    if framework_summary["unibench"]:
        logger.info(f"[UniBench] {', '.join(framework_summary['unibench'])}")
    if framework_summary["lmms-eval"]:
        logger.info(f"[lmms-eval] {', '.join(framework_summary['lmms-eval'])}")
    if framework_summary["failed"]:
        logger.info(f"[FAILED] {', '.join(framework_summary['failed'])}")
    logger.info("=" * 80)

    # === SAVE RESULTS ===
    final_result = FallbackEvalResult(
        results=results,
        aggregate_score=aggregate_score,
        vlmeval_used=vlmeval_count,
        lighteval_used=lighteval_count,
        unibench_used=unibench_count,
        lmms_eval_used=lmms_eval_count,
        total_benchmarks=len(benchmarks),
        failed_benchmarks=failed_benchmarks,
        framework_summary=framework_summary,
    )

    results_path = output_dir / "fallback_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(final_result.to_dict(), f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return final_result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for robust evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust Evaluation with Fallback (EmberVLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT CONSTRAINTS:
  - NO paid APIs (OpenAI, Anthropic, Google, etc.)
  - NO LLM-as-a-judge evaluation
  - VLMEvalKit runs ONLY in rule-based mode
  - Everything runs FULLY OFFLINE

Examples:
  # Run with lmms-eval only
  python robust_eval_fallback.py --model outputs/checkpoint --benchmarks mmstar mmmu_val

  # Run with VLMEvalKit fallback
  python robust_eval_fallback.py --model outputs/checkpoint --benchmarks mmstar \\
      --vlmeval-repo /path/to/VLMEvalKit

  # Skip VLMEvalKit fallback
  python robust_eval_fallback.py --model outputs/checkpoint --benchmarks mmstar \\
      --skip-vlmeval-fallback
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to EmberVLM checkpoint"
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+", required=True,
        help="Benchmark names (lmms-eval format)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--lmms-eval-repo", type=str, default="/root/lmms-eval",
        help="Path to lmms-eval repository"
    )
    parser.add_argument(
        "--vlmeval-repo", type=str, default=None,
        help="Path to VLMEvalKit repository (optional)"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout per benchmark in seconds"
    )
    parser.add_argument(
        "--skip-vlmeval-fallback", action="store_true",
        help="Skip VLMEvalKit fallback (lmms-eval only)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_evaluation_with_fallback(
        model_path=args.model,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
        lmms_eval_repo=args.lmms_eval_repo,
        vlmeval_repo=args.vlmeval_repo,
        timeout_per_benchmark=args.timeout,
        skip_vlmeval_fallback=args.skip_vlmeval_fallback,
    )

    # Exit with error code if all benchmarks failed
    if result.lmms_eval_used == 0 and result.vlmeval_fallback_used == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

