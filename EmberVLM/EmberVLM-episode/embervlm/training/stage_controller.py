"""
Dynamic Stage Controller for EmberVLM

Implements intelligent stage transition logic based on multiple convergence signals,
behavioral validation, and stability criteria. Designed for conference-quality
research with full transparency and interpretability.

Reference: CVPR/AAAI/ICML/EMNLP standards for multimodal model training
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Tracks metrics for convergence detection."""

    # Loss tracking
    loss_history: deque = field(default_factory=lambda: deque(maxlen=100))
    loss_variance_window: int = 20
    loss_plateau_threshold: float = 0.01  # 1% improvement threshold
    loss_plateau_patience: int = 5  # Number of windows to check

    # Accuracy tracking
    accuracy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_plateau_threshold: float = 0.005  # 0.5% improvement
    accuracy_plateau_patience: int = 5

    # Gradient tracking
    gradient_norm_history: deque = field(default_factory=lambda: deque(maxlen=100))
    gradient_variance_threshold: float = 0.3  # Max allowed variance

    # Stability tracking
    validation_loss_history: deque = field(default_factory=lambda: deque(maxlen=50))
    validation_oscillation_threshold: float = 0.1  # Max oscillation amplitude

    def add_metrics(self, loss: float, accuracy: float, grad_norm: float, val_loss: Optional[float] = None):
        """Add new metrics to history."""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.gradient_norm_history.append(grad_norm)
        if val_loss is not None:
            self.validation_loss_history.append(val_loss)

    def is_loss_plateaued(self) -> bool:
        """Check if loss has plateaued across multiple windows."""
        if len(self.loss_history) < self.loss_variance_window * self.loss_plateau_patience:
            return False

        # Split into windows and check improvement
        history = list(self.loss_history)
        plateau_count = 0

        for i in range(self.loss_plateau_patience):
            window_start = -(i + 1) * self.loss_variance_window
            window_end = -i * self.loss_variance_window if i > 0 else None

            current_window = history[window_start:window_end]
            previous_window_start = window_start - self.loss_variance_window
            previous_window = history[previous_window_start:window_start]

            current_mean = np.mean(current_window)
            previous_mean = np.mean(previous_window)

            improvement = (previous_mean - current_mean) / previous_mean
            if improvement < self.loss_plateau_threshold:
                plateau_count += 1

        return plateau_count >= self.loss_plateau_patience - 1

    def is_accuracy_plateaued(self) -> bool:
        """Check if accuracy has plateaued."""
        if len(self.accuracy_history) < self.loss_variance_window * self.accuracy_plateau_patience:
            return False

        history = list(self.accuracy_history)
        plateau_count = 0

        for i in range(self.accuracy_plateau_patience):
            window_start = -(i + 1) * self.loss_variance_window
            window_end = -i * self.loss_variance_window if i > 0 else None

            current_window = history[window_start:window_end]
            previous_window_start = window_start - self.loss_variance_window
            previous_window = history[previous_window_start:window_start]

            current_mean = np.mean(current_window)
            previous_mean = np.mean(previous_window)

            improvement = (current_mean - previous_mean) / previous_mean
            if improvement < self.accuracy_plateau_threshold:
                plateau_count += 1

        return plateau_count >= self.accuracy_plateau_patience - 1

    def is_gradient_stable(self) -> bool:
        """Check if gradients are stable (not exploding or vanishing)."""
        if len(self.gradient_norm_history) < 20:
            return True  # Not enough data

        recent_grads = list(self.gradient_norm_history)[-20:]
        variance = np.var(recent_grads) / (np.mean(recent_grads) ** 2 + 1e-8)

        return variance < self.gradient_variance_threshold

    def is_validation_stable(self) -> bool:
        """Check if validation loss is stable (not oscillating)."""
        if len(self.validation_loss_history) < 10:
            return True

        recent_vals = list(self.validation_loss_history)[-10:]
        mean_val = np.mean(recent_vals)
        oscillation = (np.max(recent_vals) - np.min(recent_vals)) / (mean_val + 1e-8)

        return oscillation < self.validation_oscillation_threshold


@dataclass
class Stage1Criteria:
    """Convergence criteria for Stage 1: Visual-Language Alignment."""

    # Quantitative thresholds
    contrastive_loss_plateau: bool = False
    i2t_accuracy_plateau: bool = False
    t2i_accuracy_plateau: bool = False
    captioning_perplexity_plateau: bool = False
    cosine_similarity_stable: bool = False

    # Behavioral checks
    captions_reference_visual: bool = False
    visual_tokens_attended: bool = False
    no_mode_collapse: bool = False

    # Stability checks
    gradients_stable: bool = False
    adapter_activations_healthy: bool = False
    validation_stable: bool = False

    def all_quantitative_met(self) -> bool:
        return (self.contrastive_loss_plateau and
                self.i2t_accuracy_plateau and
                self.t2i_accuracy_plateau)

    def all_behavioral_met(self) -> bool:
        return (self.captions_reference_visual and
                self.visual_tokens_attended and
                self.no_mode_collapse)

    def all_stability_met(self) -> bool:
        return (self.gradients_stable and
                self.adapter_activations_healthy and
                self.validation_stable)

    def ready_for_transition(self) -> bool:
        """Check if all criteria are met for Stage 1 → 2 transition."""
        return (self.all_quantitative_met() and
                self.all_behavioral_met() and
                self.all_stability_met())


@dataclass
class Stage2Criteria:
    """Convergence criteria for Stage 2: Instruction Tuning."""

    # Quantitative thresholds
    instruction_accuracy_plateau: bool = False
    distillation_loss_converged: bool = False
    kl_divergence_stable: bool = False
    vqa_accuracy_saturated: bool = False

    # Behavioral checks
    responses_instruction_faithful: bool = False
    image_tokens_attended: bool = False
    failures_semantic_not_format: bool = False

    # Representation checks
    attention_maps_relevant: bool = False
    stage1_metrics_maintained: bool = False

    def all_quantitative_met(self) -> bool:
        return (self.instruction_accuracy_plateau and
                self.distillation_loss_converged and
                self.vqa_accuracy_saturated)

    def all_behavioral_met(self) -> bool:
        return (self.responses_instruction_faithful and
                self.image_tokens_attended and
                self.failures_semantic_not_format)

    def representation_quality_maintained(self) -> bool:
        return (self.attention_maps_relevant and
                self.stage1_metrics_maintained)

    def ready_for_transition(self) -> bool:
        """Check if ready for Stage 2 → 3 transition."""
        return (self.all_quantitative_met() and
                self.all_behavioral_met() and
                self.representation_quality_maintained())


@dataclass
class Stage3Criteria:
    """Convergence criteria for Stage 3: Robot Selection."""

    # Quantitative thresholds
    robot_accuracy_plateau: bool = False
    confidence_calibration_improved: bool = False
    consistency_loss_stable: bool = False
    per_class_f1_converged: bool = False

    # Reasoning alignment
    reasoning_justifies_choice: bool = False
    no_reasoning_contradiction: bool = False
    reasoning_smoothness_improved: bool = False

    # Behavioral audits
    counterfactual_robust: bool = False
    paraphrase_robust: bool = False
    capability_sensitive: bool = False

    def all_quantitative_met(self) -> bool:
        return (self.robot_accuracy_plateau and
                self.confidence_calibration_improved and
                self.per_class_f1_converged)

    def reasoning_aligned(self) -> bool:
        return (self.reasoning_justifies_choice and
                not self.no_reasoning_contradiction and
                self.reasoning_smoothness_improved)

    def behavioral_robust(self) -> bool:
        return (self.counterfactual_robust and
                self.paraphrase_robust and
                self.capability_sensitive)

    def ready_for_transition(self) -> bool:
        """Check if ready for Stage 3 → 4 transition."""
        return (self.all_quantitative_met() and
                self.reasoning_aligned() and
                self.behavioral_robust())


class StageController:
    """
    Controls training stage transitions based on multi-metric convergence.

    Implements intelligent decision-making for when to transition between
    training stages, ensuring the model has truly learned before progressing.
    """

    def __init__(
        self,
        output_dir: str,
        wandb_logger: Optional[Any] = None,
        enable_visualization: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_logger = wandb_logger
        self.enable_visualization = enable_visualization

        # Initialize convergence trackers
        self.convergence_metrics = ConvergenceMetrics()

        # Initialize stage criteria
        self.stage1_criteria = Stage1Criteria()
        self.stage2_criteria = Stage2Criteria()
        self.stage3_criteria = Stage3Criteria()

        # History for visualization
        self.full_history = {
            'loss': [],
            'accuracy': [],
            'grad_norm': [],
            'val_loss': [],
            'step': [],
        }

        logger.info(f"Initialized StageController with output_dir={output_dir}")

    def update_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
    ):
        """Update metrics and check convergence criteria."""
        # Extract core metrics
        loss = metrics.get('loss', 0.0)
        accuracy = metrics.get('accuracy', metrics.get('acc', 0.0))
        grad_norm = metrics.get('grad_norm', 0.0)
        val_loss = metrics.get('val_loss', None)

        # Update convergence tracker
        self.convergence_metrics.add_metrics(loss, accuracy, grad_norm, val_loss)

        # Update full history
        self.full_history['loss'].append(loss)
        self.full_history['accuracy'].append(accuracy)
        self.full_history['grad_norm'].append(grad_norm)
        if val_loss is not None:
            self.full_history['val_loss'].append(val_loss)
        self.full_history['step'].append(step)

    def check_stage1_convergence(
        self,
        metrics: Dict[str, float],
        behavioral_checks: Dict[str, bool],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if Stage 1 (Visual-Language Alignment) has converged.

        Returns:
            (ready_to_transition, detailed_report)
        """
        # Update quantitative criteria
        self.stage1_criteria.contrastive_loss_plateau = self.convergence_metrics.is_loss_plateaued()
        self.stage1_criteria.i2t_accuracy_plateau = self.convergence_metrics.is_accuracy_plateaued()
        self.stage1_criteria.t2i_accuracy_plateau = self.convergence_metrics.is_accuracy_plateaued()

        # Update behavioral checks from input
        self.stage1_criteria.captions_reference_visual = behavioral_checks.get('captions_reference_visual', False)
        self.stage1_criteria.visual_tokens_attended = behavioral_checks.get('visual_tokens_attended', False)
        self.stage1_criteria.no_mode_collapse = behavioral_checks.get('no_mode_collapse', False)

        # Update stability checks
        self.stage1_criteria.gradients_stable = self.convergence_metrics.is_gradient_stable()
        self.stage1_criteria.adapter_activations_healthy = behavioral_checks.get('adapter_healthy', True)
        self.stage1_criteria.validation_stable = self.convergence_metrics.is_validation_stable()

        # Generate detailed report
        report = {
            'stage': 'Stage 1',
            'ready_for_transition': self.stage1_criteria.ready_for_transition(),
            'quantitative_criteria': {
                'contrastive_loss_plateau': self.stage1_criteria.contrastive_loss_plateau,
                'i2t_accuracy_plateau': self.stage1_criteria.i2t_accuracy_plateau,
                't2i_accuracy_plateau': self.stage1_criteria.t2i_accuracy_plateau,
                'all_met': self.stage1_criteria.all_quantitative_met(),
            },
            'behavioral_criteria': {
                'captions_reference_visual': self.stage1_criteria.captions_reference_visual,
                'visual_tokens_attended': self.stage1_criteria.visual_tokens_attended,
                'no_mode_collapse': self.stage1_criteria.no_mode_collapse,
                'all_met': self.stage1_criteria.all_behavioral_met(),
            },
            'stability_criteria': {
                'gradients_stable': self.stage1_criteria.gradients_stable,
                'adapter_activations_healthy': self.stage1_criteria.adapter_activations_healthy,
                'validation_stable': self.stage1_criteria.validation_stable,
                'all_met': self.stage1_criteria.all_stability_met(),
            },
            'metrics': metrics,
        }

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log({
                'stage1/convergence/quantitative': self.stage1_criteria.all_quantitative_met(),
                'stage1/convergence/behavioral': self.stage1_criteria.all_behavioral_met(),
                'stage1/convergence/stability': self.stage1_criteria.all_stability_met(),
                'stage1/convergence/ready_for_transition': self.stage1_criteria.ready_for_transition(),
            })

        # Save report
        self._save_convergence_report(report)

        return self.stage1_criteria.ready_for_transition(), report

    def check_stage2_convergence(
        self,
        metrics: Dict[str, float],
        behavioral_checks: Dict[str, bool],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if Stage 2 (Instruction Tuning) has converged."""
        # Update quantitative criteria
        self.stage2_criteria.instruction_accuracy_plateau = self.convergence_metrics.is_accuracy_plateaued()
        self.stage2_criteria.distillation_loss_converged = self.convergence_metrics.is_loss_plateaued()

        # Update behavioral checks
        self.stage2_criteria.responses_instruction_faithful = behavioral_checks.get('instruction_faithful', False)
        self.stage2_criteria.image_tokens_attended = behavioral_checks.get('image_tokens_attended', False)
        self.stage2_criteria.failures_semantic_not_format = behavioral_checks.get('semantic_failures', False)

        # Update representation checks
        self.stage2_criteria.attention_maps_relevant = behavioral_checks.get('attention_relevant', False)
        self.stage2_criteria.stage1_metrics_maintained = behavioral_checks.get('stage1_maintained', True)

        # Generate report
        report = {
            'stage': 'Stage 2',
            'ready_for_transition': self.stage2_criteria.ready_for_transition(),
            'quantitative_criteria': {
                'instruction_accuracy_plateau': self.stage2_criteria.instruction_accuracy_plateau,
                'distillation_loss_converged': self.stage2_criteria.distillation_loss_converged,
                'all_met': self.stage2_criteria.all_quantitative_met(),
            },
            'behavioral_criteria': {
                'responses_instruction_faithful': self.stage2_criteria.responses_instruction_faithful,
                'image_tokens_attended': self.stage2_criteria.image_tokens_attended,
                'failures_semantic': self.stage2_criteria.failures_semantic_not_format,
                'all_met': self.stage2_criteria.all_behavioral_met(),
            },
            'representation_quality': {
                'attention_maps_relevant': self.stage2_criteria.attention_maps_relevant,
                'stage1_metrics_maintained': self.stage2_criteria.stage1_metrics_maintained,
                'maintained': self.stage2_criteria.representation_quality_maintained(),
            },
            'metrics': metrics,
        }

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log({
                'stage2/convergence/quantitative': self.stage2_criteria.all_quantitative_met(),
                'stage2/convergence/behavioral': self.stage2_criteria.all_behavioral_met(),
                'stage2/convergence/representation': self.stage2_criteria.representation_quality_maintained(),
                'stage2/convergence/ready_for_transition': self.stage2_criteria.ready_for_transition(),
            })

        self._save_convergence_report(report)

        return self.stage2_criteria.ready_for_transition(), report

    def check_stage3_convergence(
        self,
        metrics: Dict[str, float],
        behavioral_checks: Dict[str, bool],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if Stage 3 (Robot Selection) has converged."""
        # Update quantitative criteria
        self.stage3_criteria.robot_accuracy_plateau = self.convergence_metrics.is_accuracy_plateaued()

        # Update reasoning alignment
        self.stage3_criteria.reasoning_justifies_choice = behavioral_checks.get('reasoning_justifies', False)
        self.stage3_criteria.no_reasoning_contradiction = not behavioral_checks.get('reasoning_contradicts', False)
        self.stage3_criteria.reasoning_smoothness_improved = behavioral_checks.get('reasoning_smooth', False)

        # Update behavioral audits
        self.stage3_criteria.counterfactual_robust = behavioral_checks.get('counterfactual_robust', False)
        self.stage3_criteria.paraphrase_robust = behavioral_checks.get('paraphrase_robust', False)

        # Generate report
        report = {
            'stage': 'Stage 3',
            'ready_for_transition': self.stage3_criteria.ready_for_transition(),
            'quantitative_criteria': {
                'robot_accuracy_plateau': self.stage3_criteria.robot_accuracy_plateau,
                'all_met': self.stage3_criteria.all_quantitative_met(),
            },
            'reasoning_alignment': {
                'justifies_choice': self.stage3_criteria.reasoning_justifies_choice,
                'no_contradiction': self.stage3_criteria.no_reasoning_contradiction,
                'smoothness_improved': self.stage3_criteria.reasoning_smoothness_improved,
                'aligned': self.stage3_criteria.reasoning_aligned(),
            },
            'behavioral_robustness': {
                'counterfactual': self.stage3_criteria.counterfactual_robust,
                'paraphrase': self.stage3_criteria.paraphrase_robust,
                'robust': self.stage3_criteria.behavioral_robust(),
            },
            'metrics': metrics,
        }

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log({
                'stage3/convergence/quantitative': self.stage3_criteria.all_quantitative_met(),
                'stage3/convergence/reasoning': self.stage3_criteria.reasoning_aligned(),
                'stage3/convergence/behavioral': self.stage3_criteria.behavioral_robust(),
                'stage3/convergence/ready_for_transition': self.stage3_criteria.ready_for_transition(),
            })

        self._save_convergence_report(report)

        return self.stage3_criteria.ready_for_transition(), report

    def _save_convergence_report(self, report: Dict[str, Any]):
        """Save convergence report to disk."""
        stage = report['stage'].replace(' ', '_').lower()
        report_path = self.output_dir / f'{stage}_convergence_report.json'

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved convergence report to {report_path}")

    def generate_training_visualizations(
        self,
        stage_name: str,
        save_dir: Optional[Path] = None,
    ):
        """Generate publication-quality visualizations."""
        if not self.enable_visualization:
            return

        if save_dir is None:
            save_dir = self.output_dir / 'visualizations' / stage_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Set style for publication quality
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

        # Plot 1: Loss trajectory
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = self.full_history['step']
        losses = self.full_history['loss']

        ax.plot(steps, losses, linewidth=2, label='Training Loss', color='#2E86AB')
        if self.full_history['val_loss']:
            ax.plot(steps[:len(self.full_history['val_loss'])],
                   self.full_history['val_loss'],
                   linewidth=2, label='Validation Loss',
                   color='#A23B72', linestyle='--')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{stage_name}: Loss Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'loss_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Gradient norm stability
        if self.full_history['grad_norm']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, self.full_history['grad_norm'],
                   linewidth=1.5, color='#F18F01', alpha=0.7)

            # Add rolling mean
            window = 20
            if len(self.full_history['grad_norm']) >= window:
                rolling_mean = np.convolve(
                    self.full_history['grad_norm'],
                    np.ones(window)/window,
                    mode='valid'
                )
                ax.plot(steps[window-1:], rolling_mean,
                       linewidth=2, color='#C73E1D', label=f'Rolling Mean ({window})')

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(f'{stage_name}: Gradient Stability')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_dir / 'gradient_stability.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Saved visualizations to {save_dir}")

        # Log to wandb if available
        if self.wandb_logger:
            import wandb
            self.wandb_logger.log({
                f'{stage_name}/viz/loss_trajectory': wandb.Image(str(save_dir / 'loss_trajectory.png')),
                f'{stage_name}/viz/gradient_stability': wandb.Image(str(save_dir / 'gradient_stability.png')),
            })


def create_stage_controller(
    output_dir: str,
    wandb_logger: Optional[Any] = None,
    enable_visualization: bool = True,
) -> StageController:
    """Factory function to create a StageController."""
    return StageController(
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        enable_visualization=enable_visualization,
    )

