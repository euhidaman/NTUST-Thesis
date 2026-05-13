"""
EmberVLM Evaluation Package

This package provides evaluation utilities for EmberVLM including:
- Robot selection metrics
- Action plan evaluation
- Reasoning chain evaluation
- Coherence checking (primary)
- Lighteval integration (if available)

SIMPLIFIED EVALUATION:
- Primary: Coherence checks (always available)
- Secondary: Lighteval benchmarks (if installed)
- NO lmms-eval required
- NO VLMEvalKit required

CRITICAL CONSTRAINTS:
- NO paid APIs
- NO LLM-as-judge evaluation
- All evaluation is FULLY OFFLINE
"""

from embervlm.evaluation.metrics import (
    compute_robot_selection_metrics,
    compute_action_plan_metrics,
    compute_reasoning_metrics,
)

# Lighteval adapter (primary)
try:
    from embervlm.evaluation.lighteval_adapter import (
        EmberVLMLightEvalModel,
        run_lighteval,
        check_lighteval_available,
        LightEvalResult,
    )
    HAS_LIGHTEVAL = True
except ImportError:
    HAS_LIGHTEVAL = False
    EmberVLMLightEvalModel = None
    run_lighteval = None
    check_lighteval_available = None
    LightEvalResult = None

# Legacy robust eval fallback (kept for backward compatibility, but not required)
try:
    from embervlm.evaluation.robust_eval_fallback import (
        run_evaluation_with_fallback,
        run_lmms_eval,
        run_vlmeval_free_mode,
        run_lighteval_benchmark,
        run_unibench_benchmark,
        detect_lmms_eval_failure,
        BENCHMARK_MAPPING,
        VLMEVAL_FREE_BENCHMARKS,
        LIGHTEVAL_TASK_MAPPING,
        UNIBENCH_BENCHMARK_MAPPING,
        EvalResult,
        FallbackEvalResult,
    )
    HAS_ROBUST_EVAL = True
except ImportError:
    HAS_ROBUST_EVAL = False
    run_evaluation_with_fallback = None
    run_lmms_eval = None
    run_vlmeval_free_mode = None
    run_lighteval_benchmark = None
    run_unibench_benchmark = None
    detect_lmms_eval_failure = None
    BENCHMARK_MAPPING = None
    VLMEVAL_FREE_BENCHMARKS = None
    LIGHTEVAL_TASK_MAPPING = None
    UNIBENCH_BENCHMARK_MAPPING = None
    EvalResult = None
    FallbackEvalResult = None

# UniBench adapter (optional)
try:
    from embervlm.evaluation.unibench_adapter import (
        EmberVLMUniBenchWrapper,
        create_embervlm_unibench_model,
        run_unibench,
        check_unibench_available,
        list_unibench_benchmarks,
        SUPPORTED_UNIBENCH_BENCHMARKS,
    )
    HAS_UNIBENCH = True
except ImportError:
    HAS_UNIBENCH = False
    EmberVLMUniBenchWrapper = None
    create_embervlm_unibench_model = None
    run_unibench = None
    check_unibench_available = None
    list_unibench_benchmarks = None
    SUPPORTED_UNIBENCH_BENCHMARKS = None

# VLMEvalKit adapter (optional, FREE mode only)
try:
    from embervlm.evaluation.vlmeval_adapter import (
        EmberVLMVLMEval,
        get_model_class as get_vlmeval_model_class,
    )
    HAS_VLMEVAL_ADAPTER = True
except ImportError:
    HAS_VLMEVAL_ADAPTER = False
    EmberVLMVLMEval = None
    get_vlmeval_model_class = None

# lmms-eval integration (optional import)
try:
    from embervlm.evaluation.vlmevalkit_adapter import (
        get_embervlm_for_lmms_eval,
        get_benchmarks_for_stage,
        STAGE1_BENCHMARKS,
        STAGE2_BENCHMARKS,
        STAGE3_BENCHMARKS,
        STAGE4_BENCHMARKS,
    )
    HAS_LMMS_EVAL = True
except ImportError:
    HAS_LMMS_EVAL = False
    get_embervlm_for_lmms_eval = None
    get_benchmarks_for_stage = None
    STAGE1_BENCHMARKS = None
    STAGE2_BENCHMARKS = None
    STAGE3_BENCHMARKS = None
    STAGE4_BENCHMARKS = None

__all__ = [
    # Metrics
    "compute_robot_selection_metrics",
    "compute_action_plan_metrics",
    "compute_reasoning_metrics",
    # lmms-eval
    "get_embervlm_for_lmms_eval",
    "get_benchmarks_for_stage",
    "STAGE1_BENCHMARKS",
    "STAGE2_BENCHMARKS",
    "STAGE3_BENCHMARKS",
    "STAGE4_BENCHMARKS",
    "HAS_LMMS_EVAL",
    # Robust four-tier evaluation
    "run_evaluation_with_fallback",
    "run_lmms_eval",
    "run_vlmeval_free_mode",
    "run_lighteval_benchmark",
    "run_unibench_benchmark",
    "detect_lmms_eval_failure",
    "BENCHMARK_MAPPING",
    "VLMEVAL_FREE_BENCHMARKS",
    "LIGHTEVAL_TASK_MAPPING",
    "UNIBENCH_BENCHMARK_MAPPING",
    "EvalResult",
    "FallbackEvalResult",
    "HAS_ROBUST_EVAL",
    # Lighteval adapter
    "EmberVLMLightEvalModel",
    "run_lighteval",
    "check_lighteval_available",
    "HAS_LIGHTEVAL",
    # UniBench adapter
    "EmberVLMUniBenchWrapper",
    "create_embervlm_unibench_model",
    "run_unibench",
    "check_unibench_available",
    "list_unibench_benchmarks",
    "SUPPORTED_UNIBENCH_BENCHMARKS",
    "HAS_UNIBENCH",
    # VLMEvalKit adapter
    "EmberVLMVLMEval",
    "get_vlmeval_model_class",
    "HAS_VLMEVAL_ADAPTER",
]

