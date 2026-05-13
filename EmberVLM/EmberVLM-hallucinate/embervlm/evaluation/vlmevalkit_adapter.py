"""
EmberVLM Model Adapter for lmms-eval

This module provides lmms-eval-compatible wrappers for EmberVLM,
enabling standardized evaluation on common VLM benchmarks.
"""

import torch
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

# This adapter is now a reference/stub - the main lmms-eval integration
# is handled in lmms-eval/lmms_eval/models/simple/tinyllava.py (EmberVLM class)
# This file is kept for backwards compatibility but is no longer actively used.


def get_embervlm_for_lmms_eval(checkpoint_path: str):
    """
    Helper function to load EmberVLM for lmms-eval.

    Args:
        checkpoint_path: Path to EmberVLM checkpoint

    Returns:
        Loaded model and tokenizer

    Note:
        For evaluation, use lmms-eval directly:
        ```
        python -m lmms_eval --model embervlm --tasks mmbench_en_dev \
            --model_args pretrained=/path/to/checkpoint
        ```
    """
    import sys
    import json
    from transformers import AutoTokenizer

    # Add EmberVLM to path
    embervlm_root = Path(__file__).parent.parent.parent
    if str(embervlm_root) not in sys.path:
        sys.path.insert(0, str(embervlm_root))

    from embervlm.models import EmberVLM, EmberVLMConfig

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = EmberVLMConfig(**config_dict)
    else:
        config = EmberVLMConfig()

    # Load tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Load model
    model = EmberVLM(config)

    # Load checkpoint
    checkpoint_file = checkpoint_path / "pytorch_model.bin"
    if checkpoint_file.exists():
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded EmberVLM from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint file not found at {checkpoint_file}")

    model.eval()

    return model, tokenizer


# Benchmark configuration for different evaluation stages
STAGE1_BENCHMARKS = [
    'mmbench_en_dev',
]

STAGE2_BENCHMARKS = [
    'mmbench_en_dev',
    'textvqa_val',
]

STAGE3_BENCHMARKS = [
    'mmbench_en_dev',
    'textvqa_val',
]

STAGE4_BENCHMARKS = [
    'mmbench_en_dev',
    'textvqa_val',
]


def get_benchmarks_for_stage(stage: int) -> List[str]:
    """Get appropriate benchmarks for a training stage."""
    if stage == 1:
        return STAGE1_BENCHMARKS
    elif stage == 2:
        return STAGE2_BENCHMARKS
    elif stage == 3:
        return STAGE3_BENCHMARKS
    elif stage == 4:
        return STAGE4_BENCHMARKS
    else:
        return STAGE2_BENCHMARKS

