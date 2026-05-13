"""
Evaluation Metrics for EmberVLM

Computes metrics for robot selection, action planning,
and reasoning quality evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def compute_robot_selection_metrics(
    predictions: List[int],
    targets: List[int],
    robot_names: List[str] = None,
) -> Dict[str, float]:
    """
    Compute metrics for robot selection task.

    Args:
        predictions: Predicted robot indices
        targets: Target robot indices
        robot_names: Optional robot names for per-class metrics

    Returns:
        Dictionary of metrics
    """
    if robot_names is None:
        robot_names = ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"]

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Overall accuracy
    accuracy = (predictions == targets).mean()

    # Per-class metrics
    num_classes = len(robot_names)
    per_class_acc = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for i, name in enumerate(robot_names):
        # True positives, false positives, false negatives
        tp = ((predictions == i) & (targets == i)).sum()
        fp = ((predictions == i) & (targets != i)).sum()
        fn = ((predictions != i) & (targets == i)).sum()

        # Accuracy for this class
        class_mask = targets == i
        if class_mask.sum() > 0:
            per_class_acc[name] = (predictions[class_mask] == i).mean()
        else:
            per_class_acc[name] = 0.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_precision[name] = precision

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall[name] = recall

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = f1

    # Macro averages
    macro_precision = np.mean(list(per_class_precision.values()))
    macro_recall = np.mean(list(per_class_recall.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))

    # Weighted F1
    class_counts = Counter(targets)
    total = len(targets)
    weighted_f1 = sum(
        per_class_f1[robot_names[i]] * class_counts.get(i, 0) / total
        for i in range(num_classes)
    )

    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_accuracy': per_class_acc,
        'per_class_f1': per_class_f1,
    }


def compute_confidence_calibration(
    predictions: List[int],
    targets: List[int],
    confidences: List[float],
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        predictions: Predicted classes
        targets: Target classes
        confidences: Prediction confidences
        num_bins: Number of bins for calibration

    Returns:
        Calibration metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    confidences = np.array(confidences)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0  # Maximum Calibration Error

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }


def compute_action_plan_metrics(
    generated_plans: List[str],
    reference_plans: List[str],
) -> Dict[str, float]:
    """
    Compute metrics for action plan generation.

    Args:
        generated_plans: Generated action plans
        reference_plans: Reference action plans

    Returns:
        Dictionary of metrics
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        nltk_available = True
    except ImportError:
        logger.warning("NLTK/rouge_score not available. Using basic metrics.")
        nltk_available = False

    bleu_scores = []
    rouge_l_scores = []

    if nltk_available:
        smoother = SmoothingFunction()
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for gen, ref in zip(generated_plans, reference_plans):
            # BLEU
            gen_tokens = gen.split()
            ref_tokens = [ref.split()]

            if gen_tokens and ref_tokens[0]:
                bleu = sentence_bleu(
                    ref_tokens, gen_tokens,
                    smoothing_function=smoother.method1
                )
                bleu_scores.append(bleu)

            # ROUGE-L
            rouge_scores = scorer.score(ref, gen)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
    else:
        # Basic word overlap metric
        for gen, ref in zip(generated_plans, reference_plans):
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())

            if gen_words and ref_words:
                overlap = len(gen_words & ref_words)
                precision = overlap / len(gen_words)
                recall = overlap / len(ref_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                bleu_scores.append(precision)
                rouge_l_scores.append(f1)

    return {
        'bleu4': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        'rouge_l': float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        'num_samples': len(generated_plans),
    }


def compute_reasoning_metrics(
    reasoning_chains: List[List[str]],
    reference_chains: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for reasoning chain quality.

    Args:
        reasoning_chains: Generated reasoning chains
        reference_chains: Optional reference chains

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'avg_chain_length': 0.0,
        'step_completeness': 0.0,
        'logical_consistency': 0.0,
    }

    if not reasoning_chains:
        return metrics

    # Chain length statistics
    chain_lengths = [len(chain) for chain in reasoning_chains]
    metrics['avg_chain_length'] = np.mean(chain_lengths)
    metrics['min_chain_length'] = min(chain_lengths)
    metrics['max_chain_length'] = max(chain_lengths)

    # Step completeness (check for key components)
    key_components = [
        'identify', 'requirement', 'analyze',
        'evaluate', 'consider', 'match',
        'select', 'capability', 'terrain'
    ]

    completeness_scores = []
    for chain in reasoning_chains:
        chain_text = ' '.join(chain).lower()
        found = sum(1 for comp in key_components if comp in chain_text)
        completeness_scores.append(found / len(key_components))

    metrics['step_completeness'] = np.mean(completeness_scores)

    # Logical consistency (basic heuristic)
    # Check if reasoning flows logically (later steps reference earlier concepts)
    consistency_scores = []
    for chain in reasoning_chains:
        if len(chain) < 2:
            consistency_scores.append(1.0)
            continue

        # Check for progression indicators
        progression_words = ['then', 'therefore', 'thus', 'so', 'hence', 'based on']
        chain_text = ' '.join(chain).lower()

        progression_score = sum(1 for w in progression_words if w in chain_text)
        consistency_scores.append(min(1.0, progression_score / 2))

    metrics['logical_consistency'] = np.mean(consistency_scores)

    # Compare with reference if available
    if reference_chains:
        similarity_scores = []
        for gen_chain, ref_chain in zip(reasoning_chains, reference_chains):
            gen_text = ' '.join(gen_chain).lower()
            ref_text = ' '.join(ref_chain).lower()

            gen_words = set(gen_text.split())
            ref_words = set(ref_text.split())

            if gen_words and ref_words:
                overlap = len(gen_words & ref_words)
                jaccard = overlap / len(gen_words | ref_words)
                similarity_scores.append(jaccard)

        metrics['reference_similarity'] = np.mean(similarity_scores) if similarity_scores else 0.0

    return metrics


def compute_hallucination_rate(
    generated_texts: List[str],
    source_texts: List[str],
) -> Dict[str, float]:
    """
    Estimate hallucination rate.

    Args:
        generated_texts: Generated text outputs
        source_texts: Source/context texts

    Returns:
        Hallucination metrics
    """
    hallucination_scores = []

    for gen, src in zip(generated_texts, source_texts):
        gen_words = set(gen.lower().split())
        src_words = set(src.lower().split())

        # Words in generated text not in source
        novel_words = gen_words - src_words

        # Filter common words
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'and', 'or', 'but', 'if', 'then', 'else',
            'when', 'where', 'why', 'how', 'what', 'which', 'who', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'to', 'of', 'in', 'for', 'on',
            'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        }

        novel_words = novel_words - common_words

        # Hallucination rate
        if gen_words:
            halluc_rate = len(novel_words) / len(gen_words)
        else:
            halluc_rate = 0.0

        hallucination_scores.append(halluc_rate)

    return {
        'hallucination_rate': float(np.mean(hallucination_scores)),
        'min_hallucination': float(min(hallucination_scores)) if hallucination_scores else 0.0,
        'max_hallucination': float(max(hallucination_scores)) if hallucination_scores else 0.0,
    }


class EmberVLMEvaluator:
    """
    Complete evaluator for EmberVLM model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        robot_names: List[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.robot_names = robot_names or [
            "Drone", "Humanoid", "Wheeled", "Legged", "Underwater"
        ]

    def evaluate(
        self,
        dataloader,
        device: str = 'cuda',
    ) -> Dict[str, Any]:
        """
        Run full evaluation.

        Args:
            dataloader: Evaluation data loader
            device: Device to run on

        Returns:
            Complete evaluation results
        """
        import torch

        self.model.eval()
        self.model.to(device)

        all_predictions = []
        all_targets = []
        all_confidences = []
        all_generated_plans = []
        all_reference_plans = []
        all_reasoning_chains = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_reasoning=True,
                )

                # Robot selection
                if 'robot_logits' in outputs:
                    preds = outputs['robot_logits'].argmax(dim=-1)
                    all_predictions.extend(preds.cpu().tolist())

                if 'robot_target' in batch:
                    all_targets.extend(batch['robot_target'].tolist())

                if 'robot_confidence' in outputs:
                    all_confidences.extend(outputs['robot_confidence'].squeeze().cpu().tolist())

        results = {}

        # Robot selection metrics
        if all_predictions and all_targets:
            results['robot_selection'] = compute_robot_selection_metrics(
                all_predictions, all_targets, self.robot_names
            )

        # Calibration metrics
        if all_confidences and all_predictions and all_targets:
            results['calibration'] = compute_confidence_calibration(
                all_predictions, all_targets, all_confidences
            )

        return results

