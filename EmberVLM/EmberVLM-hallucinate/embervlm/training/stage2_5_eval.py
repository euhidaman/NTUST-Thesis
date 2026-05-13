"""
Stage 2.5: VLM Evaluation (Lighteval + Coherence Checks)

This module has been SIMPLIFIED to use ONLY:
1. Coherence checks (always runs) - with REAL images
2. Lighteval benchmarks (if available)

NO lmms-eval, NO VLMEvalKit dependencies.
All evaluation runs FULLY OFFLINE with NO paid APIs.

For the full implementation, see stage2_5_eval_lighteval.py
"""

import logging

logger = logging.getLogger(__name__)

# Re-export everything from the Lighteval-only implementation
try:
    from embervlm.training.stage2_5_eval_lighteval import (
        run_stage2_5_evaluation,
        CoherenceChecker,
        run_lighteval_simple,
        run_unibench_evaluation,
        log_results_to_wandb,
        get_sample_images_from_unibench,
        download_sample_test_images,
        generate_test_images_with_content,
        COHERENCE_TEST_PROMPTS,
        VISION_COHERENCE_TEST_PROMPTS,
        BASELINE_SCORES,
        QUALITY_THRESHOLDS,
    )
    HAS_LIGHTEVAL_IMPL = True
except ImportError as e:
    logger.warning(f"Could not import from stage2_5_eval_lighteval: {e}")
    HAS_LIGHTEVAL_IMPL = False
    # Provide stub implementations
    run_stage2_5_evaluation = None
    CoherenceChecker = None
    run_lighteval_simple = None
    run_unibench_evaluation = None
    log_results_to_wandb = None
    get_sample_images_from_unibench = None
    download_sample_test_images = None
    generate_test_images_with_content = None
    COHERENCE_TEST_PROMPTS = []
    VISION_COHERENCE_TEST_PROMPTS = []
    BASELINE_SCORES = {}
    QUALITY_THRESHOLDS = {}

# Keep backward compatibility for old imports
BENCHMARK_PRESETS = {
    'mini': {
        'benchmarks': ['coherence_check'],
        'description': 'Coherence checks only (fast)',
        'expected_minutes': 5,
    },
    'standard': {
        'benchmarks': ['coherence_check'],
        'description': 'Coherence checks (standard)',
        'expected_minutes': 5,
    },
    'full': {
        'benchmarks': ['coherence_check'],
        'description': 'Coherence checks (full)',
        'expected_minutes': 10,
    }
}


def check_lmms_eval_installation():
    """Legacy function - lmms-eval is no longer used."""
    logger.info("lmms-eval is no longer required - using Lighteval + coherence checks")
    return True


__all__ = [
    'run_stage2_5_evaluation',
    'CoherenceChecker',
    'run_lighteval_simple',
    'run_unibench_evaluation',
    'log_results_to_wandb',
    'get_sample_images_from_unibench',
    'download_sample_test_images',
    'generate_test_images_with_content',
    'COHERENCE_TEST_PROMPTS',
    'VISION_COHERENCE_TEST_PROMPTS',
    'BASELINE_SCORES',
    'QUALITY_THRESHOLDS',
    'BENCHMARK_PRESETS',
    'check_lmms_eval_installation',
    'HAS_LIGHTEVAL_IMPL',
]
