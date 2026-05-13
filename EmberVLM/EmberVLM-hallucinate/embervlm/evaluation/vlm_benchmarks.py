"""
[DEPRECATED] VLM Benchmark Evaluation using VLMEvalKit.

This file is deprecated and no longer used. EmberVLM now uses lmms-eval for
VLM benchmark evaluation. See embervlm/training/stage2_5_eval.py for the
current implementation.
"""

import logging
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

logger = logging.getLogger(__name__)


# Benchmark configurations
BENCHMARK_PRESETS = {
    'quick': {
        'benchmarks': ['MMBench_DEV_EN', 'ScienceQA_VAL', 'TextVQA_VAL'],
        'description': 'Quick validation (~30 mins)',
        'expected_time_minutes': 30,
    },
    'standard': {
        'benchmarks': [
            'MMBench_DEV_EN',  # General VQA
            'TextVQA_VAL',      # OCR+VQA
            'ScienceQA_IMG',    # Reasoning
            'AI2D_TEST',        # Diagram understanding
            'ChartQA_TEST',     # Chart reasoning
            'SEED_IMG',         # General visual understanding
        ],
        'description': 'Standard VLM evaluation suite (5-6 benchmarks, ~1-2 hours)',
    },
    'full': {
        'benchmarks': [
            'MMBench_DEV_EN', 'SEED_IMG', 'TextVQA_VAL', 'VQAv2_VAL',
            'OCRBench', 'ChartQA', 'AI2D', 'ScienceQA_IMG', 'MMStar'
        ],
        'description': 'Full evaluation suite (~4-6 hours)',
    }
}

# Baseline scores for SmolVLM-256M equivalent (for comparison)
SMOLVLM_256M_BASELINES = {
    'MMBench_DEV_EN': 35.2,
    'TextVQA_VAL': 28.5,
    'ScienceQA_IMG': 42.3,
    'AI2D_TEST': 35.7,
    'ChartQA_TEST': 18.5,
}


def compute_quality_score(results: Dict[str, float]) -> float:
    """
    Compute aggregate quality score from benchmark results.
    
    Args:
        results: Dictionary of benchmark scores
        
    Returns:
        Normalized quality score (0-100)
    """
    # Weights for different benchmark categories
    weights = {
        'mmbench': 0.25,
        'textvqa': 0.20,
        'scienceqa': 0.20,
        'ai2d': 0.15,
        'chartqa': 0.15,
        'ocrbench': 0.15,
        'seed_bench': 0.15,
    }
    
    weighted_score = 0.0
    total_weight = 0.0
    
    for benchmark, weight in weights.items():
        if benchmark in results:
            weighted_score += results[benchmark] * weight
            total_weight += weight
            logger.info(f"  {benchmark:20s}: {results[benchmark]:6.2f}% (weight: {weight:.2f})")
    
    if total_weight > 0:
        return weighted_score / total_weight
    return 0.0


class EmberVLMEvalAdapter:
    """Adapter to make EmberVLM compatible with VLMEvalKit."""
    
    def __init__(self, model_path: str, tokenizer, device: str = 'cuda'):
        """Initialize adapter with EmberVLM model."""
        from embervlm.models import EmberVLM, EmberVLMConfig
        
        logger.info(f"Loading EmberVLM from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(Path(model_path) / 'pytorch_model.bin', map_location='cpu')
        
        # Create config and model
        config = EmberVLMConfig.from_pretrained(model_path)
        self.model = EmberVLM(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device).eval()
        
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_inner(self, message, dataset=None):
        """
        VLMEvalKit required interface.
        
        Args:
            message: List of dicts with 'type' and 'value' keys
                     [{"type": "image", "value": <PIL.Image>}, 
                      {"type": "text", "value": "Question?"}]
        
        Returns:
            Generated text response
        """
        # Extract image and text from message
        image = None
        text_parts = []
        
        for item in message:
            if isinstance(item, dict):
                if item.get('type') == 'image':
                    image = item.get('value')
                elif item.get('type') == 'text':
                    text_parts.append(item['text'])
        
        # Combine text parts
        prompt = ' '.join(text_parts)
        
        # Generate response
        response = self.model.generate(
            image=image,
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
        )
        
        return response