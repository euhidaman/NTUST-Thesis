"""
Enhanced Metrics for Robot Selection Training

Tracks:
- Per-robot precision/recall/F1
- Reasoning quality metrics
- Multi-robot coordination efficiency
- Confusion matrices
- Confidence calibration
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RobotSelectionMetrics:
    """Comprehensive metrics for robot selection evaluation."""

    def __init__(self, num_robots: int = 5, robot_names: Optional[List[str]] = None):
        self.num_robots = num_robots
        self.robot_names = robot_names or [f"Robot_{i}" for i in range(num_robots)]
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Per-robot confusion matrix
        self.confusion_matrix = np.zeros((self.num_robots, self.num_robots), dtype=np.int64)

        # Confidence calibration
        self.confidences = []
        self.correct_predictions = []

        # Multi-robot metrics
        self.multi_robot_accuracy = []
        self.multi_robot_jaccard = []

        # Reasoning quality (placeholder - would need NLP metrics)
        self.reasoning_lengths = []

        # Overall stats
        self.total_samples = 0
        self.correct_predictions_count = 0

        # Store all predictions and targets for visualization
        self.all_predictions = []
        self.all_targets = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        multi_robot_preds: Optional[torch.Tensor] = None,
        multi_robot_targets: Optional[torch.Tensor] = None,
    ):
        """Update metrics with batch results."""
        batch_size = predictions.size(0)
        self.total_samples += batch_size

        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Store for visualization
        self.all_predictions.extend(preds_np.tolist())
        self.all_targets.extend(targets_np.tolist())

        # Update confusion matrix
        for pred, target in zip(preds_np, targets_np):
            if 0 <= pred < self.num_robots and 0 <= target < self.num_robots:
                self.confusion_matrix[target, pred] += 1
                if pred == target:
                    self.correct_predictions_count += 1

        # Confidence calibration
        if confidences is not None:
            conf_np = confidences.cpu().numpy()
            correct = (preds_np == targets_np).astype(np.float32)
            self.confidences.extend(conf_np.tolist())
            self.correct_predictions.extend(correct.tolist())

        # Multi-robot metrics
        if multi_robot_preds is not None and multi_robot_targets is not None:
            multi_preds_np = (multi_robot_preds > 0.5).cpu().numpy()
            multi_targets_np = (multi_robot_targets > 0.5).cpu().numpy()

            for pred, target in zip(multi_preds_np, multi_targets_np):
                # Exact match accuracy
                exact_match = np.all(pred == target)
                self.multi_robot_accuracy.append(float(exact_match))

                # Jaccard similarity (IoU)
                intersection = np.logical_and(pred, target).sum()
                union = np.logical_or(pred, target).sum()
                jaccard = intersection / union if union > 0 else 1.0
                self.multi_robot_jaccard.append(jaccard)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = self.correct_predictions_count / max(self.total_samples, 1)

        # Per-robot metrics
        for i, robot_name in enumerate(self.robot_names):
            # True positives, false positives, false negatives
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[f'{robot_name}_precision'] = precision
            metrics[f'{robot_name}_recall'] = recall
            metrics[f'{robot_name}_f1'] = f1
            metrics[f'{robot_name}_support'] = int(self.confusion_matrix[i, :].sum())

        # Macro-averaged metrics
        precisions = [metrics[f'{name}_precision'] for name in self.robot_names]
        recalls = [metrics[f'{name}_recall'] for name in self.robot_names]
        f1s = [metrics[f'{name}_f1'] for name in self.robot_names]

        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1'] = np.mean(f1s)

        # Confidence calibration (Expected Calibration Error)
        if self.confidences:
            ece = self._compute_ece(self.confidences, self.correct_predictions)
            metrics['expected_calibration_error'] = ece

        # Multi-robot metrics
        if self.multi_robot_accuracy:
            metrics['multi_robot_exact_match'] = np.mean(self.multi_robot_accuracy)
            metrics['multi_robot_jaccard'] = np.mean(self.multi_robot_jaccard)

        return metrics

    def _compute_ece(self, confidences: List[float], correct: List[float], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.array(confidences)
        correct = np.array(correct)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidence = confidences[mask].mean()
                bin_accuracy = correct[mask].mean()
                bin_weight = mask.sum() / len(confidences)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return ece

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_matrix.copy()

    def get_per_class_accuracy(self) -> Dict[str, float]:
        """Get per-class accuracy."""
        accuracies = {}
        for i, name in enumerate(self.robot_names):
            total = self.confusion_matrix[i, :].sum()
            correct = self.confusion_matrix[i, i]
            accuracies[name] = correct / total if total > 0 else 0.0
        return accuracies

    def get_all_predictions(self) -> Optional[np.ndarray]:
        """Get all predictions as numpy array."""
        if self.all_predictions:
            return np.array(self.all_predictions)
        return None

    def get_all_targets(self) -> Optional[np.ndarray]:
        """Get all targets as numpy array."""
        if self.all_targets:
            return np.array(self.all_targets)
        return None

    def get_all_confidences(self) -> Optional[np.ndarray]:
        """Get all confidence scores as numpy array."""
        if self.confidences:
            return np.array(self.confidences)
        return None

    def get_all_correct(self) -> Optional[np.ndarray]:
        """Get all correct prediction flags as numpy array."""
        if self.correct_predictions:
            return np.array(self.correct_predictions)
        return None

    def get_per_robot_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get per-robot precision, recall, F1 for radar chart visualization."""
        metrics = {'precision': {}, 'recall': {}, 'f1': {}}

        for i, robot_name in enumerate(self.robot_names):
            # True positives, false positives, false negatives
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Use shorter names for visualization
            short_name = robot_name.replace("Robot with ", "").replace(" Robot", "")
            metrics['precision'][short_name] = precision
            metrics['recall'][short_name] = recall
            metrics['f1'][short_name] = f1

        return metrics


class ReasoningQualityMetrics:
    """Metrics for evaluating reasoning quality."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.reasoning_steps_count = []
        self.reasoning_coherence_scores = []
        self.reasoning_relevance_scores = []

    def update(
        self,
        reasoning_texts: List[str],
        task_descriptions: List[str],
        robot_selections: List[str],
    ):
        """Update reasoning quality metrics."""
        for reasoning, task, robot in zip(reasoning_texts, task_descriptions, robot_selections):
            # Count reasoning steps
            steps = reasoning.count('Step ')
            self.reasoning_steps_count.append(steps)

            # Simple heuristic: check if robot name appears in reasoning
            coherence = 1.0 if robot.lower() in reasoning.lower() else 0.0
            self.reasoning_coherence_scores.append(coherence)

            # Simple heuristic: check keyword overlap
            task_keywords = set(task.lower().split())
            reasoning_keywords = set(reasoning.lower().split())
            overlap = len(task_keywords & reasoning_keywords)
            relevance = min(overlap / max(len(task_keywords), 1), 1.0)
            self.reasoning_relevance_scores.append(relevance)

    def compute(self) -> Dict[str, float]:
        """Compute reasoning metrics."""
        if not self.reasoning_steps_count:
            return {}

        return {
            'avg_reasoning_steps': np.mean(self.reasoning_steps_count),
            'reasoning_coherence': np.mean(self.reasoning_coherence_scores),
            'reasoning_relevance': np.mean(self.reasoning_relevance_scores),
        }

