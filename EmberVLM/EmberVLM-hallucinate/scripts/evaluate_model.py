"""
EmberVLM Standalone Evaluation Script

Run benchmarks on a trained EmberVLM model using the four-tier evaluation system:
1. VLMEvalKit (FREE MODE) - Primary, rule-based scoring
2. Lighteval - NLP and multimodal benchmarks
3. UniBench - Visual reasoning benchmarks
4. lmms-eval - Final fallback

Usage:
    # Evaluate a checkpoint
    python scripts/evaluate_model.py --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-1

    # Evaluate with specific preset
    python scripts/evaluate_model.py --model_path outputs/final --preset mini

    # Evaluate without Lighteval/UniBench (only VLMEvalKit + lmms-eval)
    python scripts/evaluate_model.py --model_path outputs/final --disable_lighteval --disable_unibench
"""

import argparse
import logging
from pathlib import Path
import sys

# Add EmberVLM to path
embervlm_root = Path(__file__).parent.parent
sys.path.insert(0, str(embervlm_root))

from embervlm.training.stage2_5_eval import run_stage2_5_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EmberVLM model on VLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Stage 2 checkpoint
  python scripts/evaluate_model.py --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-1
  
  # Evaluate final model with mini preset (fast)
  python scripts/evaluate_model.py --model_path outputs/trial_run/mobilevit_xs_smollm_135m/final --preset mini
  
  # Full evaluation with all benchmarks
  python scripts/evaluate_model.py --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage3/final --preset full
  
  # Evaluate with only VLMEvalKit and lmms-eval (skip Lighteval/UniBench)
  python scripts/evaluate_model.py --model_path outputs/final --disable_lighteval --disable_unibench
        """
    )

    # Required arguments
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to model checkpoint (e.g., outputs/stage2/checkpoint-epoch-1 or outputs/final)'
    )

    # Optional arguments
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for evaluation results (default: same as model_path parent)'
    )

    parser.add_argument(
        '--preset', type=str, default='standard',
        choices=['mini', 'standard', 'full'],
        help='Benchmark preset: mini (~30min), standard (~1-2hr), full (~4-6hr). Default: standard'
    )

    parser.add_argument(
        '--threshold_mode', type=str, default='skip',
        choices=['strict', 'standard', 'permissive', 'auto', 'skip'],
        help='Quality threshold mode. Use "skip" for evaluation-only (no gating). Default: skip'
    )

    parser.add_argument(
        '--lmms_eval_repo', type=str, default=None,
        help='Path to lmms-eval repository (auto-detected if not specified)'
    )

    parser.add_argument(
        '--vlmeval_repo', type=str, default=None,
        help='Path to VLMEvalKit repository (auto-detected if not specified)'
    )

    parser.add_argument(
        '--disable_lighteval', action='store_true',
        help='Disable Lighteval (Tier 2) evaluation'
    )

    parser.add_argument(
        '--disable_unibench', action='store_true',
        help='Disable UniBench (Tier 3) evaluation'
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    # Auto-detect output directory
    if args.output_dir is None:
        # Use parent of model path for output
        if model_path.name.startswith('checkpoint-'):
            output_dir = model_path.parent.parent
        elif model_path.name == 'final':
            output_dir = model_path.parent
        else:
            output_dir = model_path.parent
    else:
        output_dir = Path(args.output_dir)

    logger.info("="*80)
    logger.info("EmberVLM Standalone Evaluation")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Threshold mode: {args.threshold_mode}")
    logger.info("")
    logger.info("Evaluation tiers:")
    logger.info("  1. VLMEvalKit (FREE MODE) - Primary")
    logger.info(f"  2. Lighteval - {'ENABLED' if not args.disable_lighteval else 'DISABLED'}")
    logger.info(f"  3. UniBench - {'ENABLED' if not args.disable_unibench else 'DISABLED'}")
    logger.info("  4. lmms-eval - Final fallback")
    logger.info("="*80)
    logger.info("")

    # Run evaluation
    try:
        passed, results_summary = run_stage2_5_evaluation(
            model_path=str(model_path),
            output_dir=str(output_dir),
            preset=args.preset,
            threshold_mode=args.threshold_mode,
            lmms_eval_repo=args.lmms_eval_repo,
            vlmeval_repo=args.vlmeval_repo,
            skip_on_error=False,
            use_robust_fallback=True,
            enable_lighteval=not args.disable_lighteval,
            enable_unibench=not args.disable_unibench,
        )

        logger.info("")
        logger.info("="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {output_dir / 'stage2_5_evaluation'}")
        logger.info("")
        logger.info("Benchmark results:")
        for benchmark, score in results_summary.get('benchmarks', {}).items():
            logger.info(f"  {benchmark}: {score:.2f}%")
        logger.info("")
        logger.info(f"Aggregate score: {results_summary.get('aggregate_score', 0):.2f}%")
        logger.info(f"Quality check: {'PASSED' if passed else 'FAILED'}")
        logger.info("="*80)

        # Exit with appropriate code
        sys.exit(0 if passed or args.threshold_mode == 'skip' else 1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

