"""
EmberVLM Evaluation Script

Run comprehensive evaluation on trained model.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import logging

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from embervlm.models import EmberVLM
from embervlm.evaluation.metrics import (
    compute_robot_selection_metrics,
    compute_action_plan_metrics,
    compute_reasoning_metrics,
    compute_confidence_calibration,
    EmberVLMEvaluator,
)
from embervlm.data.robot_loader import RobotSelectionDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cuda') -> EmberVLM:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    model = EmberVLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_path: str = None) -> AutoTokenizer:
    """Load tokenizer."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def evaluate_robot_selection(
    model: EmberVLM,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Evaluate robot selection performance."""
    logger.info("Evaluating robot selection...")

    all_predictions = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['robot_target'].to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_reasoning=True,
            )

            if 'robot_logits' in outputs:
                preds = outputs['robot_logits'].argmax(dim=-1)
                all_predictions.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

            if 'robot_confidence' in outputs:
                confidences = outputs['robot_confidence'].squeeze()
                if confidences.dim() == 0:
                    all_confidences.append(confidences.cpu().item())
                else:
                    all_confidences.extend(confidences.cpu().tolist())

    # Compute metrics
    robot_names = ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"]

    metrics = compute_robot_selection_metrics(
        all_predictions, all_targets, robot_names
    )

    if all_confidences:
        calibration = compute_confidence_calibration(
            all_predictions, all_targets, all_confidences
        )
        metrics['calibration'] = calibration

    return metrics


def evaluate_inference_speed(
    model: EmberVLM,
    device: str = 'cuda',
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Benchmark inference speed."""
    logger.info("Benchmarking inference speed...")

    import time

    # Create dummy inputs
    batch_size = 1
    seq_len = 128

    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()

            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

            if device == 'cuda':
                torch.cuda.synchronize()

            latencies.append(time.perf_counter() - start)

    import numpy as np

    return {
        'avg_latency_ms': np.mean(latencies) * 1000,
        'std_latency_ms': np.std(latencies) * 1000,
        'min_latency_ms': np.min(latencies) * 1000,
        'max_latency_ms': np.max(latencies) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'throughput_samples_per_sec': batch_size / np.mean(latencies),
    }


def evaluate_model_size(model: EmberVLM) -> Dict[str, Any]:
    """Analyze model size."""
    logger.info("Analyzing model size...")

    param_counts = model.count_parameters()

    # Estimate sizes
    fp32_size = param_counts['total'] * 4 / 1e6
    fp16_size = param_counts['total'] * 2 / 1e6
    int8_size = param_counts['total'] * 1 / 1e6
    int4_size = param_counts['total'] * 0.5 / 1e6

    return {
        **param_counts,
        'estimated_fp32_mb': fp32_size,
        'estimated_fp16_mb': fp16_size,
        'estimated_int8_mb': int8_size,
        'estimated_int4_mb': int4_size,
    }


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete evaluation."""

    # Load model
    device = args.device
    model = load_model(args.model_path, device)
    tokenizer = load_tokenizer(args.tokenizer_path)

    results = {}

    # Model size
    results['model_size'] = evaluate_model_size(model)
    logger.info(f"Model size: {results['model_size']['estimated_fp16_mb']:.1f} MB (FP16)")

    # Inference speed
    if args.benchmark_speed:
        results['inference_speed'] = evaluate_inference_speed(model, device)
        logger.info(f"Avg latency: {results['inference_speed']['avg_latency_ms']:.2f} ms")

    # Robot selection evaluation
    if args.eval_data:
        dataset = RobotSelectionDataset(
            data_dir=args.eval_data,
            tokenizer=tokenizer,
            split='val',
            augment_data=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        results['robot_selection'] = evaluate_robot_selection(model, dataloader, device)
        logger.info(f"Robot selection accuracy: {results['robot_selection']['accuracy']:.2%}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="EmberVLM Evaluation")

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                       help='Path to tokenizer')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='Path to evaluation data')
    parser.add_argument('--output_path', type=str, default='./evaluation_results.json',
                       help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Evaluation batch size')
    parser.add_argument('--benchmark_speed', action='store_true',
                       help='Run speed benchmark')

    args = parser.parse_args()

    results = run_evaluation(args)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print(f"\nModel Size:")
    print(f"  Total Parameters: {results['model_size']['total']:,}")
    print(f"  Trainable: {results['model_size']['trainable']:,}")
    print(f"  FP16 Size: {results['model_size']['estimated_fp16_mb']:.1f} MB")
    print(f"  INT8 Size: {results['model_size']['estimated_int8_mb']:.1f} MB")

    if 'inference_speed' in results:
        print(f"\nInference Speed:")
        print(f"  Avg Latency: {results['inference_speed']['avg_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['inference_speed']['throughput_samples_per_sec']:.1f} samples/sec")

    if 'robot_selection' in results:
        print(f"\nRobot Selection:")
        print(f"  Accuracy: {results['robot_selection']['accuracy']:.2%}")
        print(f"  Macro F1: {results['robot_selection']['macro_f1']:.2%}")


if __name__ == "__main__":
    main()

