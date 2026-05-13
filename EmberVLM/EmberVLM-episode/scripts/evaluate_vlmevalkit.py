#!/usr/bin/env python
"""
EmberVLM Standalone Benchmark Evaluation Script

Evaluates trained EmberVLM checkpoints on standard VLM benchmarks using lmms-eval.
This script runs benchmarks independently of the training pipeline.

Usage:
    # Run on Stage 2 checkpoint with standard benchmarks
    python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10
    
    # Run specific benchmarks
    python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/final --benchmarks mmbench_en_dev textvqa_val
    
    # Run with custom preset
    python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10 --preset standard
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_lmms_eval():
    """Check if lmms-eval is available."""
    try:
        import lmms_eval
        logger.info("✅ lmms-eval is installed and available")
        return True
    except ImportError:
        logger.error("""
╔══════════════════════════════════════════════════════════╗
║   ⚠️  lmms-eval Not Installed                           ║
╚══════════════════════════════════════════════════════════╝

lmms-eval is required for benchmarking but is not installed.

To install lmms-eval:

1. Run the setup script from EmberVLM directory:
   python setup_vlmeval.py

2. Or install manually:
   cd /root/lmms-eval  (or d:\\BabyLM\\lmms-eval on Windows)
   pip install -e .

3. Verify installation:
   python -c "import lmms_eval; print('lmms-eval OK')"

For more info: https://github.com/EvolvingLMMs-Lab/lmms-eval
""")
        return False


def run_evaluation(
    model_path: str,
    benchmarks: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run lmms-eval evaluation on EmberVLM using the stage2_5_eval infrastructure.

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmarks: List of benchmark task names
        output_dir: Output directory for results

    Returns:
        Dictionary of benchmark results with scores and metadata
    """
    from embervlm.training.stage2_5_eval import run_lmms_eval_benchmark, compute_aggregate_score, BASELINE_SCORES
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("EmberVLM Benchmark Evaluation")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Benchmarks: {', '.join(benchmarks)}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)
    
    results = {}
    failed = []
    
    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f"\n[{i}/{len(benchmarks)}] Running {benchmark}...")
        
        try:
            score = run_lmms_eval_benchmark(
                model_path=model_path,
                task_name=benchmark,
                output_dir=str(output_path)
            )
            
            if score is not None:
                results[benchmark] = float(score)
                baseline = BASELINE_SCORES.get(benchmark, 30.0)
                delta = score - baseline
                logger.info(f"✅ {benchmark}: {score:.2f}% (baseline: {baseline:.1f}%, Δ{delta:+.1f}%)")
            else:
                failed.append(benchmark)
                logger.error(f"❌ {benchmark}: Failed to get score")
                
        except Exception as e:
            failed.append(benchmark)
            logger.error(f"❌ {benchmark}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Compute aggregate score
    aggregate = compute_aggregate_score(results) if results else 0.0
    
    # Create results summary
    summary = {
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat(),
        'total_benchmarks': len(benchmarks),
        'successful': len(results),
        'failed': len(failed),
        'failed_benchmarks': failed,
        'aggregate_score': aggregate,
        'individual_scores': results,
        'baselines': {k: BASELINE_SCORES.get(k, 30.0) for k in benchmarks},
    }
    
    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"Aggregate Score: {aggregate:.2f}%")
    logger.info(f"Successful: {len(results)}/{len(benchmarks)} benchmarks")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EmberVLM on VLM benchmarks using lmms-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Stage 2 checkpoint with standard preset
  python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10

  # Evaluate with specific benchmarks
  python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/final --benchmarks mmbench_en_dev textvqa_val

  # Evaluate with custom output directory
  python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10 --output_dir ./benchmark_results

  # Use a specific benchmark preset
  python evaluate_vlmevalkit.py --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10 --preset mini
"""
    )

    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to EmberVLM checkpoint directory (e.g., outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10)'
    )
    parser.add_argument(
        '--benchmarks', type=str, nargs='+', default=None,
        help='Specific benchmarks to evaluate (default: uses preset)'
    )
    parser.add_argument(
        '--preset', type=str, default='standard',
        choices=['mini', 'standard', 'full'],
        help='Benchmark preset: mini (2 benchmarks, ~30 min), standard (6 benchmarks, ~90 min), full (10+ benchmarks, ~5 hours)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for results (default: model_path/../stage2_5_evaluation)'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent
        args.output_dir = str(model_dir.parent / "stage2_5_evaluation")
    
    # Determine benchmarks
    if args.benchmarks is None:
        from embervlm.training.stage2_5_eval import BENCHMARK_PRESETS
        preset_info = BENCHMARK_PRESETS.get(args.preset, BENCHMARK_PRESETS['standard'])
        args.benchmarks = preset_info['benchmarks']
        
        logger.info(f"Using '{args.preset}' preset:")
        logger.info(f"  {preset_info['description']}")
        logger.info(f"  Expected time: ~{preset_info['expected_minutes']} minutes")

    # Check lmms-eval
    if not check_lmms_eval():
        logger.error("lmms-eval is required for benchmarking. Please install it first.")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_evaluation(
            model_path=args.model_path,
            benchmarks=args.benchmarks,
            output_dir=args.output_dir,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Aggregate Score: {results['aggregate_score']:.2f}%")
        print(f"Successful: {results['successful']}/{results['total_benchmarks']} benchmarks")
        print("\nIndividual Scores:")
        for bench, score in sorted(results['individual_scores'].items()):
            baseline = results['baselines'].get(bench, 30.0)
            delta = score - baseline
            print(f"  {bench:25s}: {score:5.2f}% (baseline: {baseline:4.1f}%, Δ{delta:+5.1f}%)")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

    # Determine output directory
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent
        args.output_dir = str(model_dir.parent / "stage2_5_evaluation")
    
    # Determine benchmarks
    if args.benchmarks is None:
        from embervlm.training.stage2_5_eval import BENCHMARK_PRESETS
        preset_info = BENCHMARK_PRESETS.get(args.preset, BENCHMARK_PRESETS['standard'])
        args.benchmarks = preset_info['benchmarks']
        
        logger.info(f"Using '{args.preset}' preset:")
        logger.info(f"  {preset_info['description']}")
        logger.info(f"  Expected time: ~{preset_info['expected_minutes']} minutes")

    # Check lmms-eval
    if not check_lmms_eval():
        logger.error("lmms-eval is required for benchmarking. Please install it first.")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_evaluation(
            model_path=args.model_path,
            benchmarks=args.benchmarks,
            output_dir=args.output_dir,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Aggregate Score: {results['aggregate_score']:.2f}%")
        print(f"Successful: {results['successful']}/{results['total_benchmarks']} benchmarks")
        print("\nIndividual Scores:")
        for bench, score in sorted(results['individual_scores'].items()):
            baseline = results['baselines'].get(bench, 30.0)
            delta = score - baseline
            print(f"  {bench:25s}: {score:5.2f}% (baseline: {baseline:4.1f}%, Δ{delta:+5.1f}%)")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Stage: {args.stage}")
    print("-"*60)

    if 'results' in results:
        for benchmark, score in results['results'].items():
            print(f"{benchmark}: {score}")

    if 'robot_selection' in results:
        print("-"*60)
        print("Robot Selection:")
        for metric, value in results['robot_selection'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

    print("="*60)


if __name__ == '__main__':
    sys.exit(main())
    main()

