"""
Weights & Biases Logger for EmberVLM

Provides comprehensive logging for training metrics,
model artifacts, and visualizations.
"""

import os
import threading
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

from embervlm.monitoring.stage_visualizations import _viz_save_path

logger = logging.getLogger(__name__)


def _init_wandb_with_timeout(timeout=30, **kwargs):
    """
    Initialize W&B with a timeout to prevent hanging.

    Args:
        timeout: Maximum seconds to wait for initialization
        **kwargs: Arguments to pass to wandb.init()

    Returns:
        W&B run object or None if timeout/failure
    """
    import wandb

    result = [None]
    exception = [None]

    def init_wandb():
        try:
            result[0] = wandb.init(**kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=init_wandb, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"W&B initialization timed out after {timeout}s")
        return None

    if exception[0]:
        raise exception[0]

    return result[0]


class WandbLogger:
    """
    Weights & Biases logger for EmberVLM training.

    Features:
    - Training metrics logging
    - Model checkpointing
    - Gradient visualization
    - Attention heatmaps
    - Custom dashboards
    """

    def __init__(
        self,
        project: str = "embervlm",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        resume: bool = False,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        **kwargs,  # Accept extra kwargs like output_dir for compatibility
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.entity = entity
        self.enabled = True

        # Check if W&B is disabled via environment variable
        import os as _os
        if _os.environ.get('DISABLE_WANDB', '').lower() in ('1', 'true', 'yes'):
            logger.info("W&B disabled via DISABLE_WANDB environment variable")
            self.wandb = None
            self.run = None
            self.enabled = False
            return

        try:
            import wandb
            self.wandb = wandb

            # Initialize run with timeout protection and fork-safe settings
            logger.info(f"Initializing W&B run: {name}")

            # Use thread-safe initialization and disable certain features that can hang
            wandb_settings = wandb.Settings(
                start_method="thread",
                _disable_stats=True,  # Disable system stats collection
                _disable_meta=True,   # Disable metadata collection
            )

            # Set timeout environment variable
            import os as _os
            _os.environ.setdefault('WANDB_INIT_TIMEOUT', '60')

            # Try to initialize with timeout
            logger.info("Calling wandb.init with 30s timeout...")
            self.run = _init_wandb_with_timeout(
                timeout=30,
                project=project,
                name=name,
                config=config,
                entity=entity,
                resume="allow" if resume else None,
                tags=tags,
                notes=notes,
                mode=mode,
                settings=wandb_settings,
            )

            if self.run is None:
                raise RuntimeError("W&B init timed out or returned None")

            logger.info(f"Successfully initialized W&B run: {self.run.name}")

        except ImportError:
            logger.warning("wandb not installed. Logging disabled.")
            self.wandb = None
            self.run = None
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
            self.wandb = None
            self.run = None
            self.enabled = False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            commit: Whether to commit the log
        """
        if not self.enabled:
            return

        try:
            self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_image(
        self,
        key: str,
        images: Union[List, Any],
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Log images to W&B."""
        if not self.enabled:
            logger.warning(f"log_image called but W&B is not enabled, skipping {key}")
            return

        try:
            if isinstance(images, list):
                wandb_images = [
                    self.wandb.Image(img, caption=caption)
                    for img in images
                ]
            else:
                wandb_images = self.wandb.Image(images, caption=caption)

            self.wandb.log({key: wandb_images}, step=step)
            logger.info(f"✓ W&B image logged: {key} at step {step}")
        except Exception as e:
            logger.error(f"Failed to log image {key}: {e}", exc_info=True)

    def log_attention_map(
        self,
        key: str,
        attention_weights: Any,
        step: Optional[int] = None,
    ):
        """
        Log attention heatmap visualization.

        Args:
            key: Metric key name
            attention_weights: Attention tensor [B, H, T, T] or [H, T, T]
            step: Training step
        """
        if not self.enabled:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert to numpy
            if hasattr(attention_weights, 'cpu'):
                attn = attention_weights.cpu().detach().numpy()
            else:
                attn = np.array(attention_weights)

            # Average over batch and heads if needed
            if attn.ndim == 4:
                attn = attn[0].mean(axis=0)  # [T, T]
            elif attn.ndim == 3:
                attn = attn.mean(axis=0)  # [T, T]

            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(attn, cmap='viridis')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title('Attention Weights')
            plt.colorbar(im, ax=ax)

            self.wandb.log({key: self.wandb.Image(fig)}, step=step)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to log attention map: {e}")

    def log_gradient_histogram(
        self,
        model: Any,
        step: Optional[int] = None,
    ):
        """Log gradient histograms for model parameters."""
        if not self.enabled:
            return

        try:
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.cpu().detach()
                    gradients[f"gradients/{name}"] = self.wandb.Histogram(grad.numpy())

            self.wandb.log(gradients, step=step)
        except Exception as e:
            logger.warning(f"Failed to log gradients: {e}")

    def log_model(
        self,
        model_path: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model checkpoint as artifact.

        Args:
            model_path: Path to saved model
            name: Artifact name
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        try:
            artifact = self.wandb.Artifact(
                name=name or "model",
                type="model",
                metadata=metadata,
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            logger.info(f"Logged model artifact: {name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ):
        """Log tabular data."""
        if not self.enabled:
            return

        try:
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({key: table}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log table: {e}")

    def log_confusion_matrix(
        self,
        key: str,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        step: Optional[int] = None,
    ):
        """Log confusion matrix."""
        if not self.enabled:
            return

        try:
            self.wandb.log({
                key: self.wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            }, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def define_metric(
        self,
        name: str,
        step_metric: str = "step",
        summary: str = "best",
        goal: str = "maximize",
    ):
        """Define custom metric with summary."""
        if not self.enabled:
            return

        try:
            self.wandb.define_metric(
                name,
                step_metric=step_metric,
                summary=summary,
                goal=goal,
            )
        except Exception as e:
            logger.warning(f"Failed to define metric: {e}")

    def watch(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """Watch model for automatic gradient logging."""
        if not self.enabled:
            return

        try:
            self.wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ):
        """Send alert."""
        if not self.enabled:
            return

        try:
            level_map = {
                "INFO": self.wandb.AlertLevel.INFO,
                "WARN": self.wandb.AlertLevel.WARN,
                "ERROR": self.wandb.AlertLevel.ERROR,
            }
            self.wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level, self.wandb.AlertLevel.INFO),
            )
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

    def finish(self):
        """Finish the W&B run."""
        if not self.enabled:
            return

        try:
            self.run.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

    @property
    def url(self) -> Optional[str]:
        """Get run URL."""
        if self.run:
            return self.run.url
        return None


def create_training_dashboard(
    logger: WandbLogger,
    stages: List[str] = None,
):
    """
    Configure W&B dashboard for training.

    Args:
        logger: WandbLogger instance
        stages: Training stages to track
    """
    if not logger.enabled:
        return

    stages = stages or ['stage1', 'stage2', 'stage3', 'stage4']

    # Define metrics
    metrics = [
        ('loss', 'minimize'),
        ('accuracy', 'maximize'),
        ('robot_accuracy', 'maximize'),
        ('lr', 'last'),
    ]

    for metric, goal in metrics:
        logger.define_metric(
            metric,
            summary='best' if goal != 'last' else 'last',
            goal=goal if goal != 'last' else 'maximize',
        )

    # Validation metrics
    for stage in stages:
        logger.define_metric(
            f'{stage}/val_loss',
            summary='min',
            goal='minimize',
        )


class EnhancedWandbLogger(WandbLogger):
    """
    Enhanced W&B logger with comprehensive visualizations.

    Adds publication-quality plots, attention maps, and convergence analysis.
    Supports stage-specific visualizations for EmberVLM training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize visualizer with error handling
        self.visualizer = None
        self.stage_visualizers = {}
        output_dir = kwargs.get('output_dir', './outputs/visualizations')
        self.current_stage = kwargs.get('stage', 1)

        try:
            from embervlm.monitoring.visualization import TrainingVisualizer
            logger.info(f"Initializing TrainingVisualizer with output_dir: {output_dir}")
            self.visualizer = TrainingVisualizer(output_dir=output_dir)
            logger.info(f"✓ TrainingVisualizer initialized successfully! Visualizations will be saved to: {output_dir}")

            # Initialize stage-specific visualizers
            try:
                from embervlm.monitoring.stage_visualizations import (
                    Stage1Visualizer, Stage2Visualizer,
                    Stage3Visualizer, Stage4Visualizer,
                    CrossStageVisualizer
                )
                self.stage_visualizers = {
                    1: Stage1Visualizer(output_dir=f"{output_dir}/stage1"),
                    2: Stage2Visualizer(output_dir=f"{output_dir}/stage2"),
                    3: Stage3Visualizer(output_dir=f"{output_dir}/stage3"),
                    4: Stage4Visualizer(output_dir=f"{output_dir}/stage4"),
                }
                self.cross_stage_viz = CrossStageVisualizer(output_dir=output_dir)
                logger.info("✓ Stage visualizers initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize stage visualizers: {e}")
                self.stage_visualizers = {}
        except Exception as e:
            logger.error(f"✗ Failed to initialize TrainingVisualizer: {e}", exc_info=True)
            logger.warning("Visualizations will be disabled. Only basic metrics will be logged.")
            self.visualizer = None

        # Track metrics history for plotting
        self.metrics_history = {}
        logger.info(f"EnhancedWandbLogger initialized. Visualizer available: {self.visualizer is not None}")

    def log_with_visualization(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        stage_name: str = "training",
        commit: bool = True,
    ):
        """
        Log metrics and generate visualizations.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            stage_name: Name of training stage
            commit: Whether to commit
        """
        # Update history for all numeric metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)

        # Log basic metrics
        self.log(metrics, step=step, commit=False)

        # Generate visualizations periodically (only if visualizer is available)
        # Check for any loss-like metric (loss, robot_loss, sft_loss, etc.)
        loss_keys = [k for k in self.metrics_history.keys() if 'loss' in k.lower()]
        has_enough_samples = any(len(self.metrics_history.get(k, [])) >= 10 for k in loss_keys)

        if self.visualizer and step and step % 500 == 0 and has_enough_samples:
            try:
                # Find the primary loss metric
                primary_loss_key = 'loss'
                for candidate in ['loss', 'robot_loss', 'sft_loss', 'total_loss']:
                    if candidate in self.metrics_history and len(self.metrics_history[candidate]) >= 10:
                        primary_loss_key = candidate
                        break

                num_samples = len(self.metrics_history.get(primary_loss_key, []))
                logger.info(f"[{stage_name}] Generating visualizations at step {step} with {num_samples} {primary_loss_key} samples")

                # Loss decomposition plot
                try:
                    _, loss_img = self.visualizer.plot_loss_decomposition(
                        self.metrics_history,
                        stage_name,
                        save_path=str(_viz_save_path(self.visualizer.output_dir, "loss_decomposition", f"{stage_name}_step{step}.png"))
                    )
                    self.log_image(f"{stage_name}/loss_decomposition", loss_img, step=step)
                    logger.info(f"✓ Logged loss decomposition to W&B for {stage_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate loss decomposition for {stage_name}: {e}")

                # Convergence analysis
                try:
                    _, conv_img = self.visualizer.plot_convergence_analysis(
                        self.metrics_history,
                        stage_name,
                        save_path=str(_viz_save_path(self.visualizer.output_dir, "convergence", f"{stage_name}_step{step}.png"))
                    )
                    self.log_image(f"{stage_name}/convergence_analysis", conv_img, step=step)
                    logger.info(f"✓ Logged convergence analysis to W&B for {stage_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate convergence analysis for {stage_name}: {e}")

                # Log metrics summary
                logger.info(f"[{stage_name}] Metrics tracked: {list(self.metrics_history.keys())}")

            except Exception as e:
                logger.error(f"Failed to generate visualization for {stage_name}: {e}", exc_info=True)

        if commit:
            try:
                self.wandb.log({}, commit=True)
            except Exception as e:
                logger.warning(f"Failed to commit W&B log: {e}")

    def log_attention_visualization(
        self,
        image: Any,
        attention_map: Any,
        text_tokens: List[str],
        step: int,
        stage_name: str = "training",
    ):
        """
        Log attention map overlaid on image.

        Args:
            image: Input image tensor
            attention_map: Attention weights
            text_tokens: List of text tokens
            step: Training step
            stage_name: Stage name
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            _, attn_img = self.visualizer.visualize_attention_on_image(
                image, attention_map, text_tokens,
                save_path=str(_viz_save_path(self.visualizer.output_dir, "attention", f"step{step}.png"))
            )
            self.log_image(f"{stage_name}/attention_visualization", attn_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log attention visualization: {e}")

    def log_gradient_distribution(
        self,
        gradients: Dict[str, Any],
        step: int,
        stage_name: str = "training",
    ):
        """
        Log gradient distribution across modules.

        Args:
            gradients: Dictionary of module gradients
            step: Training step
            stage_name: Stage name
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            _, grad_img = self.visualizer.plot_gradient_distribution(
                gradients, step,
                save_path=str(_viz_save_path(self.visualizer.output_dir, "gradients", f"step{step}.png"))
            )
            self.log_image(f"{stage_name}/gradient_distribution", grad_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log gradient distribution: {e}")

    def log_confusion_matrix(
        self,
        predictions: Any,
        labels: Any,
        class_names: List[str],
        step: int,
        stage_name: str = "training",
        confusion_matrix: Any = None,
    ):
        """
        Log confusion matrix for classification tasks.

        Args:
            predictions: Predicted class indices (or None if confusion_matrix provided)
            labels: True class indices (or None if confusion_matrix provided)
            class_names: List of class names
            step: Training step
            stage_name: Stage name
            confusion_matrix: Pre-computed confusion matrix (optional)
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            import numpy as np

            if confusion_matrix is not None:
                # Use pre-computed confusion matrix
                cm = np.array(confusion_matrix)
            else:
                # Convert to numpy and compute confusion matrix
                if hasattr(predictions, 'cpu'):
                    pred_np = predictions.cpu().numpy()
                else:
                    pred_np = np.array(predictions)

                if hasattr(labels, 'cpu'):
                    label_np = labels.cpu().numpy()
                else:
                    label_np = np.array(labels)

                # Compute confusion matrix
                from sklearn.metrics import confusion_matrix as sklearn_cm
                cm = sklearn_cm(label_np, pred_np)

            _, cm_img = self.visualizer.plot_confusion_matrix(
                None, None, class_names, stage_name,
                save_path=str(_viz_save_path(self.visualizer.output_dir, "confusion_matrix", f"step{step}.png")),
                confusion_matrix=cm,
            )
            self.log_image(f"{stage_name}/confusion_matrix", cm_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def generate_final_visualizations(self, stage_name: str, step: int):
        """
        Generate end-of-training visualizations (loss decomposition + convergence).

        Called at the end of each training stage to ensure visualizations are
        saved to disk even for short runs that never hit the periodic threshold.

        Args:
            stage_name: Name of the stage (e.g., 'stage1', 'stage2')
            step: Final training step
        """
        if not self.enabled or not self.visualizer:
            return

        loss_keys = [k for k in self.metrics_history.keys() if 'loss' in k.lower()]
        has_data = any(len(self.metrics_history.get(k, [])) >= 2 for k in loss_keys)

        if not has_data:
            logger.info(f"[{stage_name}] Not enough loss data for final visualizations")
            return

        logger.info(f"[{stage_name}] Generating end-of-training visualizations at step {step}...")

        # Loss decomposition
        try:
            save_path = str(_viz_save_path(self.visualizer.output_dir, "loss_decomposition", f"{stage_name}_final.png"))
            _, loss_img = self.visualizer.plot_loss_decomposition(
                self.metrics_history, stage_name, save_path=save_path
            )
            self.log_image(f"{stage_name}/loss_decomposition_final", loss_img, step=step)
            logger.info(f"  ✓ Saved loss decomposition → {save_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate loss decomposition: {e}")

        # Convergence analysis
        try:
            save_path = str(_viz_save_path(self.visualizer.output_dir, "convergence", f"{stage_name}_final.png"))
            _, conv_img = self.visualizer.plot_convergence_analysis(
                self.metrics_history, stage_name, save_path=save_path
            )
            self.log_image(f"{stage_name}/convergence_analysis_final", conv_img, step=step)
            logger.info(f"  ✓ Saved convergence analysis → {save_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate convergence analysis: {e}")

        logger.info(f"[{stage_name}] End-of-training visualizations complete")

    def finish(self):
        """Clean up resources and finish run."""
        if self.visualizer:
            self.visualizer.close()
        super().finish()

    # ==================== STAGE-SPECIFIC LOGGING ====================

    def set_stage(self, stage: int):
        """Set the current training stage."""
        self.current_stage = stage
        logger.info(f"W&B logger set to Stage {stage}")

    def get_stage_visualizer(self, stage: int = None):
        """Get visualizer for specific stage."""
        stage = stage or self.current_stage
        return self.stage_visualizers.get(stage)

    # Stage 1: Visual-Language Alignment
    def log_similarity_matrix(
        self,
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log image-text similarity matrix for Stage 1."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_similarity_matrix(image_embeds, text_embeds, step)
            self.log_image("stage1/similarity_matrix", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log similarity matrix: {e}")

    def log_embedding_tsne(
        self,
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log t-SNE visualization of embeddings."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_embedding_tsne(image_embeds, text_embeds, step)
            self.log_image("stage1/embedding_tsne", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log t-SNE: {e}")

    def log_retrieval_examples(
        self,
        images: List,
        captions: List[str],
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log retrieval examples for Stage 1."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_retrieval_examples(
                images, captions, image_embeds, text_embeds, step
            )
            self.log_image("stage1/retrieval_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log retrieval examples: {e}")

    # Stage 2: Instruction Tuning
    def log_generation_examples(
        self,
        images: List,
        instructions: List[str],
        generated: List[str],
        ground_truth: List[str],
        step: int,
    ):
        """Log generation examples for Stage 2."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(2)
        if viz is None:
            return

        try:
            _, img = viz.plot_generation_examples(
                images, instructions, generated, ground_truth, step
            )
            self.log_image("stage2/generation_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log generation examples: {e}")

    def log_token_probabilities(
        self,
        logits: Any,
        labels: Any,
        step: int,
    ):
        """Log token probability distribution."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(2)
        if viz is None:
            return

        try:
            _, img = viz.plot_token_probability_distribution(logits, labels, step)
            self.log_image("stage2/token_probabilities", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log token probabilities: {e}")

    # Stage 3: Robot Selection
    def log_robot_confusion_matrix(
        self,
        predictions: Any,
        labels: Any,
        step: int,
    ):
        """Log robot selection confusion matrix."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_confusion_matrix(predictions, labels, step)
            self.log_image("stage3/confusion_matrix", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def log_robot_radar_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: int,
    ):
        """Log per-robot performance radar chart."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_per_robot_radar(metrics, step)
            self.log_image("stage3/robot_radar", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log radar chart: {e}")

    def log_calibration_plot(
        self,
        confidences: Any,
        correct: Any,
        step: int,
    ):
        """Log confidence calibration plot."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_confidence_calibration(confidences, correct, step)
            self.log_image("stage3/calibration", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log calibration: {e}")

    def log_reasoning_examples(
        self,
        tasks: List[str],
        reasoning_chains: List[str],
        predictions: List[str],
        ground_truth: List[str],
        correct: List[bool],
        step: int,
    ):
        """Log reasoning chain examples."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_reasoning_examples(
                tasks, reasoning_chains, predictions, ground_truth, correct, step
            )
            self.log_image("stage3/reasoning_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reasoning examples: {e}")

    # Stage 4: Chain-of-Thought Reasoning
    def log_reasoning_quality(
        self,
        coherence_scores: List[float],
        consistency_scores: List[float],
        step_counts: List[int],
        step: int,
    ):
        """Log reasoning quality metrics."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_reasoning_quality_metrics(
                coherence_scores, consistency_scores, step_counts, step
            )
            self.log_image("stage4/reasoning_quality", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reasoning quality: {e}")

    def log_phase_comparison(
        self,
        phase1_history: Dict[str, List[float]],
        phase2_history: Dict[str, List[float]],
        step: int,
    ):
        """Log Phase 1 vs Phase 2 comparison."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_phase_comparison(phase1_history, phase2_history, step)
            self.log_image("stage4/phase_comparison", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log phase comparison: {e}")

    def log_cot_comparison(
        self,
        tasks: List[str],
        without_cot: List,
        with_cot: List,
        step: int,
    ):
        """Log CoT comparison examples."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_cot_examples(tasks, without_cot, with_cot, step)
            self.log_image("stage4/cot_comparison", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log CoT comparison: {e}")

    # Cross-Stage Visualizations
    def log_stage_progression(
        self,
        eval_results: Dict[int, Dict[str, float]],
    ):
        """Log benchmark progression across stages."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_stage_progression(eval_results)
            self.log_image("cross_stage/progression", img)
        except Exception as e:
            logger.warning(f"Failed to log stage progression: {e}")

    def log_training_summary(
        self,
        all_metrics: Dict[int, Dict[str, List[float]]],
    ):
        """Log comprehensive training summary."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_training_summary(all_metrics)
            self.log_image("cross_stage/training_summary", img)
        except Exception as e:
            logger.warning(f"Failed to log training summary: {e}")

    def log_carbon_footprint(
        self,
        emissions_per_stage: Dict[int, float],
    ):
        """Log carbon footprint analysis."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_carbon_footprint(emissions_per_stage)
            self.log_image("cross_stage/carbon_footprint", img)
        except Exception as e:
            logger.warning(f"Failed to log carbon footprint: {e}")

    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        stage: int,
    ):
        """Log lmms-eval evaluation results."""
        if not self.enabled:
            return

        try:
            # Log individual benchmark scores
            if 'results' in results:
                for benchmark, score in results['results'].items():
                    if isinstance(score, dict):
                        for metric, value in score.items():
                            if isinstance(value, (int, float)):
                                self.log({f"eval/{benchmark}/{metric}": value})
                    elif isinstance(score, (int, float)):
                        self.log({f"eval/{benchmark}": score})

            # Log robot selection metrics
            if 'robot_selection' in results:
                for metric, value in results['robot_selection'].items():
                    if isinstance(value, (int, float)):
                        self.log({f"eval/robot_selection/{metric}": value})

            # Log calibration
            if 'calibration' in results:
                for metric, value in results['calibration'].items():
                    if isinstance(value, (int, float)):
                        self.log({f"eval/calibration/{metric}": value})

            logger.info(f"Logged evaluation results for Stage {stage}")

        except Exception as e:
            logger.warning(f"Failed to log evaluation results: {e}")

    # ==================== COMPREHENSIVE DASHBOARD VISUALIZATIONS ====================

    def log_training_dashboard(
        self,
        stage: int,
        metrics_history: Dict[str, List[float]],
        step: int,
    ):
        """
        Log a comprehensive training dashboard for the given stage.

        Args:
            stage: Training stage (1-4)
            metrics_history: Dictionary of metric name -> list of values
            step: Current training step
        """
        if not self.enabled:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image
            import io

            stage_colors = {
                1: ['#2E86AB', '#A23B72', '#45B7D1'],  # Blues/Purples
                2: ['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Reds/Teals
                3: ['#96CEB4', '#FFEAA7', '#DDA0DD'],  # Greens/Yellows
                4: ['#6B5B95', '#88B04B', '#F7CAC9'],  # Purples/Greens
            }

            colors = stage_colors.get(stage, ['#2E86AB', '#A23B72', '#45B7D1'])

            fig = plt.figure(figsize=(20, 12))

            # Create subplots grid
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

            # 1. Loss over time (top-left, spans 2 cols)
            ax1 = fig.add_subplot(gs[0, :2])
            loss_keys = [k for k in metrics_history.keys() if 'loss' in k.lower()]
            for i, key in enumerate(loss_keys[:3]):
                if metrics_history.get(key):
                    ax1.plot(metrics_history[key], label=key, color=colors[i % len(colors)], linewidth=2)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Stage {stage} Loss Curves', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # 2. Accuracy/Performance metrics (top-right, spans 2 cols)
            ax2 = fig.add_subplot(gs[0, 2:])
            acc_keys = [k for k in metrics_history.keys() if 'acc' in k.lower() or 'f1' in k.lower()]
            for i, key in enumerate(acc_keys[:3]):
                if metrics_history.get(key):
                    ax2.plot(metrics_history[key], label=key, color=colors[i % len(colors)], linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Score')
            ax2.set_title(f'Stage {stage} Performance Metrics', fontsize=12, fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)

            # 3. Learning rate schedule (middle-left)
            ax3 = fig.add_subplot(gs[1, 0])
            if 'lr' in metrics_history and metrics_history['lr']:
                ax3.plot(metrics_history['lr'], color=colors[0], linewidth=2)
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Learning Rate')
                ax3.set_title('LR Schedule')
                ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax3.grid(True, alpha=0.3)

            # 4. Loss distribution histogram (middle-center)
            ax4 = fig.add_subplot(gs[1, 1])
            if 'loss' in metrics_history and len(metrics_history['loss']) > 10:
                ax4.hist(metrics_history['loss'][-100:], bins=20, color=colors[1], alpha=0.7, edgecolor='white')
                ax4.axvline(np.mean(metrics_history['loss'][-100:]), color='red', linestyle='--', label='Mean')
                ax4.set_xlabel('Loss Value')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Recent Loss Distribution')
                ax4.legend()

            # 5. Moving average smoothed loss (middle-right, spans 2)
            ax5 = fig.add_subplot(gs[1, 2:])
            for key in loss_keys[:2]:
                if key in metrics_history and len(metrics_history[key]) > 5:
                    values = np.array(metrics_history[key])
                    window = min(50, len(values) // 5)
                    if window > 1:
                        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                        ax5.plot(range(window-1, len(values)), smoothed, label=f'{key} (smoothed)', linewidth=2)
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Smoothed Loss')
            ax5.set_title('Smoothed Loss (Moving Average)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 6. Summary statistics table (bottom, spans all cols)
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')

            # Create summary table
            table_data = []
            headers = ['Metric', 'Current', 'Best', 'Mean', 'Std']
            for key in list(metrics_history.keys())[:8]:
                if metrics_history[key]:
                    values = metrics_history[key]
                    current = values[-1]
                    best = min(values) if 'loss' in key.lower() else max(values)
                    mean = np.mean(values)
                    std = np.std(values)
                    table_data.append([
                        key,
                        f'{current:.4f}',
                        f'{best:.4f}',
                        f'{mean:.4f}',
                        f'{std:.4f}'
                    ])

            if table_data:
                table = ax6.table(
                    cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colColours=['#E8E8E8'] * 5,
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)

            stage_names = {
                1: 'Visual-Language Alignment',
                2: 'Instruction Tuning',
                3: 'Robot Selection',
                4: 'Chain-of-Thought Reasoning'
            }
            fig.suptitle(
                f'Stage {stage}: {stage_names.get(stage, "")} Training Dashboard (Step {step})',
                fontsize=14, fontweight='bold', y=0.98
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Convert to image and log
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img = Image.open(buf).copy()
            buf.close()
            plt.close(fig)

            self.log_image(f"stage{stage}/training_dashboard", img, step=step)
            logger.info(f"✓ Logged training dashboard to W&B for Stage {stage} at step {step}")

        except Exception as e:
            logger.warning(f"Failed to log training dashboard: {e}")

    def log_model_analysis(
        self,
        model,
        stage: int,
        step: int,
    ):
        """
        Log model analysis visualizations including parameter distributions.

        Args:
            model: The model to analyze
            stage: Training stage
            step: Current step
        """
        if not self.enabled:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image
            import io

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # Flatten axes
            axes = axes.flatten()

            # Collect parameter stats by module
            module_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    module = name.split('.')[0]
                    if module not in module_stats:
                        module_stats[module] = {'params': [], 'grads': []}
                    module_stats[module]['params'].append(param.data.cpu().numpy().flatten())
                    if param.grad is not None:
                        module_stats[module]['grads'].append(param.grad.cpu().numpy().flatten())

            # 1. Parameter magnitude by module
            ax = axes[0]
            module_names = list(module_stats.keys())[:6]
            param_means = [np.mean([np.abs(p).mean() for p in module_stats[m]['params']]) for m in module_names]
            ax.barh(module_names, param_means, color='#2E86AB')
            ax.set_xlabel('Mean Absolute Parameter Value')
            ax.set_title('Parameter Magnitude by Module')

            # 2. Gradient magnitude by module
            ax = axes[1]
            grad_means = []
            for m in module_names:
                if module_stats[m]['grads']:
                    grad_means.append(np.mean([np.abs(g).mean() for g in module_stats[m]['grads']]))
                else:
                    grad_means.append(0)
            ax.barh(module_names, grad_means, color='#A23B72')
            ax.set_xlabel('Mean Absolute Gradient Value')
            ax.set_title('Gradient Magnitude by Module')

            # 3. Parameter distribution (all params combined)
            ax = axes[2]
            all_params = np.concatenate([np.concatenate(v['params']) for v in module_stats.values() if v['params']])
            ax.hist(all_params[::max(1, len(all_params)//10000)], bins=50, alpha=0.7, color='#4ECDC4')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Count')
            ax.set_title('Overall Parameter Distribution')
            ax.set_yscale('log')

            # 4. Gradient distribution
            ax = axes[3]
            all_grads = []
            for v in module_stats.values():
                if v['grads']:
                    all_grads.extend([np.concatenate(v['grads'])])
            if all_grads:
                all_grads = np.concatenate(all_grads)
                ax.hist(all_grads[::max(1, len(all_grads)//10000)], bins=50, alpha=0.7, color='#FF6B6B')
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Count')
            ax.set_title('Overall Gradient Distribution')
            ax.set_yscale('log')

            # 5. Trainable parameter count by module
            ax = axes[4]
            param_counts = {m: sum(p.size for p in v['params']) for m, v in module_stats.items()}
            sorted_modules = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:6]
            ax.pie([c for _, c in sorted_modules], labels=[m for m, _ in sorted_modules], autopct='%1.1f%%')
            ax.set_title('Trainable Parameters by Module')

            # 6. Training info text
            ax = axes[5]
            ax.axis('off')
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info_text = f"""
Model Analysis - Stage {stage}, Step {step}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Trainable Ratio: {trainable_params/total_params*100:.2f}%

Number of Modules: {len(module_stats)}
Modules with Gradients: {sum(1 for v in module_stats.values() if v['grads'])}
            """
            ax.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=11, va='center')

            fig.suptitle(f'Model Analysis - Stage {stage}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Convert and log
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img = Image.open(buf).copy()
            buf.close()
            plt.close(fig)

            self.log_image(f"stage{stage}/model_analysis", img, step=step)
            logger.info(f"✓ Logged model analysis to W&B for Stage {stage}")

        except Exception as e:
            logger.warning(f"Failed to log model analysis: {e}")

    def log_comprehensive_metrics_table(
        self,
        metrics: Dict[str, Any],
        stage: int,
        step: int,
    ):
        """
        Log a comprehensive metrics table to W&B.

        Args:
            metrics: Dictionary of all metrics
            stage: Training stage
            step: Current step
        """
        if not self.enabled:
            return

        try:
            # Create W&B table
            columns = ['Metric', 'Value', 'Category']
            data = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Categorize metrics
                    if 'loss' in key.lower():
                        category = 'Loss'
                    elif 'acc' in key.lower():
                        category = 'Accuracy'
                    elif 'f1' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                        category = 'Classification'
                    elif 'lr' in key.lower():
                        category = 'Training'
                    else:
                        category = 'Other'

                    data.append([key, float(value), category])

            if data:
                table = self.wandb.Table(columns=columns, data=data)
                self.wandb.log({f"stage{stage}/metrics_table": table}, step=step)
                logger.info(f"✓ Logged metrics table to W&B for Stage {stage}")

        except Exception as e:
            logger.warning(f"Failed to log metrics table: {e}")

    # ==================== ADVANCED VISUALIZATIONS ====================

    def log_layer_training_dynamics_3d(
        self,
        accuracy_data,
        layer_names: List[str] = None,
        step_labels: List[int] = None,
        robot_name: str = None,
        step: int = None,
    ):
        """
        Log 3D surface plot of layer-wise training dynamics to W&B.

        Args:
            accuracy_data: 2D array [n_layers, n_steps]
            layer_names: Layer names
            step_labels: Training step labels
            robot_name: Robot type for coloring
            step: Current step for logging
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer
            import numpy as np

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            title = f"Layer-wise Training Dynamics"
            if robot_name:
                title += f" - {robot_name}"

            _, img = self.advanced_viz.plot_layer_training_dynamics_3d(
                np.array(accuracy_data), layer_names, step_labels,
                title=title, robot_name=robot_name, save=True
            )

            key = f"advanced/layer_dynamics_3d_{robot_name}" if robot_name else "advanced/layer_dynamics_3d"
            self.log_image(key, img, step=step)
            logger.info(f"✓ Logged 3D layer dynamics to W&B")

        except Exception as e:
            logger.warning(f"Failed to log 3D layer dynamics: {e}")

    def log_multi_robot_3d_surfaces(
        self,
        robot_accuracy_data: Dict[str, Any],
        layer_names: List[str] = None,
        step_labels: List[int] = None,
        step: int = None,
    ):
        """
        Log multi-panel 3D surface plots for all robot types.

        Args:
            robot_accuracy_data: {robot_name: [n_layers, n_steps] array}
            layer_names: Layer names
            step_labels: Training step labels
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_multi_robot_3d_surfaces(
                robot_accuracy_data, layer_names, step_labels, save=True
            )
            self.log_image("advanced/multi_robot_3d_surfaces", img, step=step)
            logger.info(f"✓ Logged multi-robot 3D surfaces to W&B")

        except Exception as e:
            logger.warning(f"Failed to log multi-robot 3D surfaces: {e}")

    def log_frozen_trained_comparison(
        self,
        frozen_metrics: Dict[str, float],
        trained_metrics: Dict[str, float],
        categories: List[str] = None,
        step: int = None,
    ):
        """
        Log frozen vs trained comparison chart.

        Args:
            frozen_metrics: Metrics with frozen backbone
            trained_metrics: Metrics with trained backbone
            categories: Categories to compare
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_frozen_trained_comparison(
                frozen_metrics, trained_metrics, categories, save=True
            )
            self.log_image("advanced/frozen_trained_comparison", img, step=step)
            logger.info(f"✓ Logged frozen vs trained comparison to W&B")

        except Exception as e:
            logger.warning(f"Failed to log frozen/trained comparison: {e}")

    def log_layer_importance_ranks(
        self,
        importance_scores: Dict[str, Any],
        layer_names: List[str] = None,
        step: int = None,
    ):
        """
        Log layer importance rank distribution.

        Args:
            importance_scores: {method: [layer_scores]}
            layer_names: Layer names
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_layer_budget_rank_distribution(
                importance_scores, layer_names, save=True
            )
            self.log_image("advanced/layer_importance_ranks", img, step=step)
            logger.info(f"✓ Logged layer importance ranks to W&B")

        except Exception as e:
            logger.warning(f"Failed to log layer importance ranks: {e}")

    def log_pareto_frontier(
        self,
        models: Dict[str, Dict[str, float]],
        highlight_model: str = "EmberVLM",
        step: int = None,
    ):
        """
        Log Pareto frontier analysis.

        Args:
            models: {model_name: {metric: value}}
            highlight_model: Model to highlight
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_pareto_frontier(
                models, highlight_model=highlight_model, save=True
            )
            self.log_image("advanced/pareto_frontier", img, step=step)
            logger.info(f"✓ Logged Pareto frontier to W&B")

        except Exception as e:
            logger.warning(f"Failed to log Pareto frontier: {e}")

    def log_ablation_tornado(
        self,
        ablation_results: Dict[str, float],
        baseline: float,
        step: int = None,
    ):
        """
        Log ablation study tornado chart.

        Args:
            ablation_results: {component_removed: accuracy}
            baseline: Baseline accuracy
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_ablation_tornado(
                ablation_results, baseline, save=True
            )
            self.log_image("advanced/ablation_tornado", img, step=step)
            logger.info(f"✓ Logged ablation tornado to W&B")

        except Exception as e:
            logger.warning(f"Failed to log ablation tornado: {e}")

    def log_confusion_evolution(
        self,
        confusion_matrices: List,
        step_labels: List[int],
        class_names: List[str],
        step: int = None,
    ):
        """
        Log confusion matrix evolution over training.

        Args:
            confusion_matrices: List of confusion matrices
            step_labels: Step for each matrix
            class_names: Class names
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_confusion_evolution(
                confusion_matrices, step_labels, class_names, save=True
            )
            self.log_image("advanced/confusion_evolution", img, step=step)
            logger.info(f"✓ Logged confusion evolution to W&B")

        except Exception as e:
            logger.warning(f"Failed to log confusion evolution: {e}")

    def log_benchmark_heatmap(
        self,
        results: Dict[int, Dict[str, float]],
        step: int = None,
    ):
        """
        Log benchmark performance heatmap across stages.

        Args:
            results: {stage: {benchmark: score}}
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_benchmark_heatmap(results, save=True)
            self.log_image("advanced/benchmark_heatmap", img, step=step)
            logger.info(f"✓ Logged benchmark heatmap to W&B")

        except Exception as e:
            logger.warning(f"Failed to log benchmark heatmap: {e}")

    def log_task_robot_heatmap(
        self,
        performance: Dict[str, Dict[str, float]],
        step: int = None,
    ):
        """
        Log task type vs robot performance heatmap.

        Args:
            performance: {task_type: {robot: accuracy}}
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_task_robot_performance_heatmap(
                performance, save=True
            )
            self.log_image("advanced/task_robot_heatmap", img, step=step)
            logger.info(f"✓ Logged task-robot heatmap to W&B")

        except Exception as e:
            logger.warning(f"Failed to log task-robot heatmap: {e}")

    def log_image_text_similarity_3d(
        self,
        similarity_history: List,
        step_labels: List[int] = None,
        step: int = None,
    ):
        """
        Log 3D surface of image-text similarity evolution.

        Args:
            similarity_history: List of similarity arrays per step
            step_labels: Training step labels
            step: Current step
        """
        if not self.enabled:
            return

        try:
            from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

            if not hasattr(self, 'advanced_viz'):
                self.advanced_viz = AdvancedVisualizer()

            _, img = self.advanced_viz.plot_image_text_similarity_evolution_3d(
                similarity_history, step_labels, save=True
            )
            self.log_image("advanced/similarity_evolution_3d", img, step=step)
            logger.info(f"✓ Logged similarity evolution 3D to W&B")

        except Exception as e:
            logger.warning(f"Failed to log similarity evolution 3D: {e}")

    def log_comprehensive_stage_summary(
        self,
        stage: int,
        metrics: Dict[str, Any],
        step: int,
        confusion_matrix = None,
        class_names: List[str] = None,
        extra_visualizations: Dict[str, Any] = None,
    ):
        """
        Log comprehensive summary for a training stage with all visualizations.

        Args:
            stage: Training stage (1-4)
            metrics: Dictionary of metrics
            step: Current step
            confusion_matrix: Optional confusion matrix
            class_names: Class names for confusion matrix
            extra_visualizations: Additional data for visualizations
        """
        if not self.enabled:
            return

        try:
            # Log basic metrics
            stage_metrics = {f"stage{stage}/{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float))}
            self.log(stage_metrics, step=step)

            # Log training dashboard
            self.log_training_dashboard(stage, self.metrics_history, step)

            # Log confusion matrix if provided
            if confusion_matrix is not None and class_names is not None:
                self.log_confusion_matrix(
                    None, None, class_names, step, f"stage{stage}", confusion_matrix
                )

            # Log metrics table
            self.log_comprehensive_metrics_table(metrics, stage, step)

            # Stage-specific visualizations
            if stage == 1 and extra_visualizations:
                # Stage 1: Visual-Language Alignment
                if 'similarity_matrix' in extra_visualizations:
                    viz = self.get_stage_visualizer(1)
                    if viz:
                        try:
                            _, img = viz.plot_similarity_matrix(
                                extra_visualizations['image_embeds'],
                                extra_visualizations['text_embeds'],
                                step
                            )
                            self.log_image("stage1/similarity_matrix", img, step=step)
                        except Exception as e:
                            logger.warning(f"Failed to log similarity matrix: {e}")

            elif stage == 3 and extra_visualizations:
                # Stage 3: Robot Selection
                if 'per_robot_metrics' in extra_visualizations:
                    viz = self.get_stage_visualizer(3)
                    if viz:
                        try:
                            _, img = viz.plot_per_robot_radar(
                                extra_visualizations['per_robot_metrics'], step
                            )
                            self.log_image("stage3/robot_radar", img, step=step)
                        except Exception as e:
                            logger.warning(f"Failed to log robot radar: {e}")

                if 'confidences' in extra_visualizations and 'correct' in extra_visualizations:
                    viz = self.get_stage_visualizer(3)
                    if viz:
                        try:
                            _, img = viz.plot_confidence_calibration(
                                extra_visualizations['confidences'],
                                extra_visualizations['correct'],
                                step
                            )
                            self.log_image("stage3/calibration", img, step=step)
                        except Exception as e:
                            logger.warning(f"Failed to log calibration: {e}")

            elif stage == 4 and extra_visualizations:
                # Stage 4: CoT Reasoning
                if 'reasoning_quality' in extra_visualizations:
                    viz = self.get_stage_visualizer(4)
                    if viz:
                        try:
                            rq = extra_visualizations['reasoning_quality']
                            _, img = viz.plot_reasoning_quality_metrics(
                                rq.get('coherence', []),
                                rq.get('consistency', []),
                                rq.get('step_counts', []),
                                step
                            )
                            self.log_image("stage4/reasoning_quality", img, step=step)
                        except Exception as e:
                            logger.warning(f"Failed to log reasoning quality: {e}")

            logger.info(f"✓ Logged comprehensive stage {stage} summary to W&B at step {step}")

        except Exception as e:
            logger.warning(f"Failed to log comprehensive stage summary: {e}")

