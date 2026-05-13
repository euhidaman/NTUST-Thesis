"""
EmberVLM Training Package
"""

from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
)

# Stage 2.5 evaluation (Lighteval + Coherence checks)
# These are optional - evaluation can be skipped
try:
    from embervlm.training.stage2_5_eval import (
        run_stage2_5_evaluation,
        CoherenceChecker,
        BENCHMARK_PRESETS,
    )
except ImportError:
    run_stage2_5_evaluation = None
    CoherenceChecker = None
    BENCHMARK_PRESETS = {}

__all__ = [
    "TrainingConfig",
    "setup_distributed",
    "cleanup_distributed",
    "get_optimizer",
    "get_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "run_stage2_5_evaluation",
    "CoherenceChecker",
    "BENCHMARK_PRESETS",
]

