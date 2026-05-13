"""
Stage 3: Robot Fleet Selection Training

Trains the model for robot fleet selection based on task requirements.
Uses the robot-selection-dataset with augmentation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.models.reasoning_heads import ReasoningLoss
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    MetricTracker,
    print_trainable_parameters,
    push_checkpoint_to_hub,
    enable_gradient_checkpointing,
)
from embervlm.data.robot_loader import get_robot_selection_dataloader
from embervlm.training.robot_metrics import RobotSelectionMetrics, ReasoningQualityMetrics
from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker
from embervlm.training.early_stopping import EarlyStopping, BestCheckpointTracker, log_stage_summary, save_stage_metrics_json

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage3Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage3Visualizer = None


class Stage3Trainer:
    """Trainer for Stage 3: Robot Fleet Selection."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        robot_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        hub_repo_id: Optional[str] = None,
        vision_backbone: str = "repvit",
        language_backbone: str = "tinyllm",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.hub_repo_id = hub_repo_id
        self.vision_backbone = vision_backbone
        self.language_backbone = language_backbone

        # Early stopping: monitor macro F1 (higher is better)
        self.early_stopping = EarlyStopping(
            patience=5,
            mode='max',
            min_delta=0.01,
            verbose=True
        )
        self.best_checkpoint_tracker = BestCheckpointTracker(
            save_dir=config.output_dir,
            metric_name='val_macro_f1',
            mode='max'
        )
        self.metrics_history = []

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        set_seed(config.seed, self.rank)

        # Unwrap model if it was previously wrapped with DDP
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # CRITICAL: Validate embedding size matches tokenizer BEFORE any training
        # This prevents cryptic CUDA index out of bounds errors
        if tokenizer is not None:
            required_vocab_size = len(tokenizer)
            current_vocab_size = None

            if hasattr(model.language_model, 'get_input_embeddings'):
                current_vocab_size = model.language_model.get_input_embeddings().weight.shape[0]
            elif hasattr(model.language_model, 'model'):
                if hasattr(model.language_model.model, 'get_input_embeddings'):
                    current_vocab_size = model.language_model.model.get_input_embeddings().weight.shape[0]

            if current_vocab_size is not None:
                if current_vocab_size != required_vocab_size:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"❌ CRITICAL ERROR: Token embedding size mismatch!\n"
                        f"{'='*80}\n"
                        f"  Tokenizer vocabulary size: {required_vocab_size}\n"
                        f"  Model embedding layer size: {current_vocab_size}\n"
                        f"  Difference: {required_vocab_size - current_vocab_size} tokens\n"
                        f"\n"
                        f"This mismatch causes index out of bounds errors during training.\n"
                        f"The model's embedding layer must be resized to match the tokenizer.\n"
                        f"\n"
                        f"Special tokens in tokenizer:\n"
                    )
                    for token in tokenizer.additional_special_tokens:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        error_msg += f"    {token} → ID {token_id}\n"
                    error_msg += f"\n"
                    error_msg += f"SOLUTION: The train_all.py script should have resized the embeddings.\n"
                    error_msg += f"If this error persists, manually resize embeddings in train_all.py\n"
                    error_msg += f"before calling Stage 3 training.\n"
                    error_msg += f"{'='*80}\n"

                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.info(f"✓ Embedding size validation passed: {current_vocab_size} tokens")

                    # Additional validation: Check if special tokens are actually within bounds
                    special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.additional_special_tokens]
                    max_special_id = max(special_token_ids) if special_token_ids else 0

                    if max_special_id >= current_vocab_size:
                        error_msg = (
                            f"\n{'='*80}\n"
                            f"❌ CRITICAL ERROR: Special token ID out of bounds!\n"
                            f"{'='*80}\n"
                            f"  Maximum special token ID: {max_special_id}\n"
                            f"  Model embedding layer size: {current_vocab_size}\n"
                            f"\n"
                            f"Special token IDs:\n"
                        )
                        for token, token_id in zip(tokenizer.additional_special_tokens, special_token_ids):
                            status = "❌ OUT OF BOUNDS" if token_id >= current_vocab_size else "✓ OK"
                            error_msg += f"    {token} → ID {token_id} {status}\n"
                        error_msg += f"{'='*80}\n"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        logger.info(f"✓ All special tokens within valid range (max ID: {max_special_id})")


        # Enable gradient checkpointing for memory efficiency
        if getattr(config, 'gradient_checkpointing', True):
            logger.info("Enabling gradient checkpointing for memory efficiency...")
            if hasattr(model, 'language_model'):
                enable_gradient_checkpointing(model.language_model)
                logger.info("✓ Gradient checkpointing enabled on language model")

        # Ensure model is on correct device before DDP
        try:
            model = model.to(self.device)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        except RuntimeError as e:
            logger.error(f"[Rank {self.rank}] Failed to move model to device: {e}")
            raise

        # Synchronize to ensure all ranks have model loaded
        if torch.distributed.is_initialized() and self.world_size > 1:
            try:
                torch.distributed.barrier()
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Barrier failed: {e}")
                raise

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data loaders
        self.robot_dataloader = robot_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Reasoning loss with focal loss for class balancing
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        self.reasoning_loss = ReasoningLoss(
            num_robots=model_ref.config.num_robots,
            reasoning_weight=1.0,
            robot_weight=1.0,
            action_weight=1.0,
            consistency_weight=0.5,
            use_focal_loss=True,  # Use focal loss for class imbalance
            focal_gamma=2.0,
            label_smoothing=0.1,
        )

        # Compute class weights from training data
        self._compute_class_weights()

        # Loss weights
        self.ce_weight = 0.6
        self.reasoning_consistency_weight = 0.4

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger with visualizations (main process)...")
            try:
                wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                self.wandb_logger = EnhancedWandbLogger(
                    project=wandb_project,
                    name="stage3_robot_selection",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info(f"Enhanced W&B logger initialized with project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced W&B logger: {e}")
                self.wandb_logger = None

            try:
                self.carbon_tracker = CarbonTracker(output_dir=config.output_dir)
                logger.info("Carbon tracker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize carbon tracker: {e}")
                self.carbon_tracker = None

        # Synchronize all ranks after logging initialization
        if torch.distributed.is_initialized() and self.world_size > 1:
            logger.info(f"[Rank {self.rank}] Waiting at post-logging barrier...")
            torch.distributed.barrier()
            logger.info(f"[Rank {self.rank}] Passed post-logging barrier")

        self.metric_tracker = MetricTracker()
        self.robot_metrics = RobotSelectionMetrics(
            num_robots=5,
            robot_names=["Drone", "Underwater Robot", "Humanoid", "Robot with Wheels", "Robot with Legs"]
        )
        self.reasoning_metrics = ReasoningQualityMetrics()
        self.global_step = 0

        # Stage 3 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage3Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage3Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage3Visualizer: {e}")

        # Track data for visualizations
        self.last_robot_preds = None
        self.last_robot_targets = None
        self.last_confidences = None

        # VA neuron heatmap tracker (generates every ~50 steps)
        self.va_tracker = None
        if is_main_process():
            try:
                from embervlm.monitoring.va_neuron_analysis import VAHeatmapTracker
                self.va_tracker = VAHeatmapTracker(
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                    interval=getattr(config, 'va_heatmap_interval', 50),
                )
                logger.info("✓ VAHeatmapTracker initialised (interval=%d)",
                            self.va_tracker.interval)
            except Exception as e:
                logger.warning(f"Failed to initialise VAHeatmapTracker: {e}")

        # Cache vocab size for validation
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_ref.language_model, 'get_input_embeddings'):
            self._vocab_size = model_ref.language_model.get_input_embeddings().weight.shape[0]
        elif hasattr(model_ref.language_model, 'model'):
            if hasattr(model_ref.language_model.model, 'get_input_embeddings'):
                self._vocab_size = model_ref.language_model.model.get_input_embeddings().weight.shape[0]
            else:
                self._vocab_size = 50262  # fallback
        else:
            self._vocab_size = 50262  # fallback
        logger.info(f"Cached vocab size for validation: {self._vocab_size}")

        if is_main_process():
            print_trainable_parameters(self.model)

    def _compute_class_weights(self):
        """Compute class weights from training data for focal loss."""
        try:
            # Count samples per class
            class_counts = torch.zeros(5, dtype=torch.float32)
            
            # Robot names for logging
            robot_names = ["Drone", "Underwater Robot", "Humanoid", "Robot with Wheels", "Robot with Legs"]

            # Iterate through all batches to get full distribution
            for batch in self.robot_dataloader:
                targets = batch.get('robot_target')
                if targets is not None:
                    for t in targets:
                        if 0 <= t < 5:
                            class_counts[t] += 1

            # Set class weights in reasoning loss
            if class_counts.sum() > 0:
                self.reasoning_loss.set_class_weights(class_counts)

                if is_main_process():
                    logger.info("="*60)
                    logger.info("Class Distribution in Training Data:")
                    for i, (name, count) in enumerate(zip(robot_names, class_counts)):
                        pct = (count / class_counts.sum() * 100).item()
                        logger.info(f"  {name}: {int(count)} samples ({pct:.1f}%)")
                    
                    if hasattr(self.reasoning_loss.robot_criterion, 'alpha'):
                        logger.info("\nComputed Class Weights (inverse frequency):")
                        weights = self.reasoning_loss.robot_criterion.alpha
                        for i, (name, weight) in enumerate(zip(robot_names, weights)):
                            logger.info(f"  {name}: {weight:.3f}")
                    logger.info("="*60)
            else:
                logger.warning("⚠️  No samples found for class weight computation")
                
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")

    def _validate_and_fix_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Validate and fix token IDs before model forward pass.

        This is a critical safeguard to prevent CUDA index out of bounds errors.
        Uses .clone() to ensure we're working with fresh tensors that CUDA can't
        have already queued operations on.
        """
        vocab_size = self._vocab_size
        device = input_ids.device

        # CRITICAL: Clone tensors FIRST to avoid modifying originals and CUDA async issues
        input_ids = input_ids.clone()
        labels = labels.clone()

        # Check input_ids
        invalid_input_mask = (input_ids >= vocab_size) | (input_ids < 0)
        if invalid_input_mask.any():
            num_invalid = invalid_input_mask.sum().item()
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            logger.warning(
                f"⚠️ {num_invalid} invalid input_ids detected. "
                f"Max: {max_id}, Min: {min_id}, Vocab: {vocab_size}. Replacing with 0."
            )
            input_ids = torch.where(invalid_input_mask, torch.zeros_like(input_ids), input_ids)

            # Force CUDA sync after modification
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        # Check labels (skip -100 which is ignore index)
        valid_labels_mask = labels != -100
        if valid_labels_mask.any():
            # Check only the valid (non -100) positions
            invalid_label_positions = valid_labels_mask & ((labels >= vocab_size) | (labels < 0))
            if invalid_label_positions.any():
                num_invalid = invalid_label_positions.sum().item()
                logger.warning(
                    f"⚠️ {num_invalid} invalid labels detected. Replacing with 0."
                )
                # Replace invalid labels with 0
                labels = torch.where(invalid_label_positions, torch.zeros_like(labels), labels)

                # Force CUDA sync after modification
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

        return input_ids, labels

    def train_robot_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for robot selection using vision-only forward pass.

        This uses forward_vision_only() which bypasses the language model entirely,
        avoiding all tokenization-related CUDA index out of bounds errors.
        Robot selection is primarily a vision-based task anyway.
        """
        pixel_values = batch['pixel_values'].to(self.device)
        robot_targets = batch['robot_target'].to(self.device)

        multi_robot_targets = batch.get('multi_robot_target')
        if multi_robot_targets is not None:
            multi_robot_targets = multi_robot_targets.to(self.device)

        # Get model reference for config access
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # Clamp robot targets to valid range to avoid gather OOB in losses
        num_robots = getattr(model_ref.config, 'num_robots', None)
        if num_robots is not None:
            if robot_targets is not None:
                max_robot = robot_targets.max().item()
                min_robot = robot_targets.min().item()
                if max_robot >= num_robots or min_robot < 0:
                    logger.warning(
                        f"⚠️ robot_target out of bounds detected (min={min_robot}, max={max_robot}, num_robots={num_robots}); clamping"
                    )
                    robot_targets = torch.clamp(robot_targets, 0, num_robots - 1)

            if multi_robot_targets is not None:
                # Ensure multi-hot vector width matches num_robots
                if multi_robot_targets.size(-1) > num_robots:
                    multi_robot_targets = multi_robot_targets[..., :num_robots]
                elif multi_robot_targets.size(-1) < num_robots:
                    pad_width = num_robots - multi_robot_targets.size(-1)
                    pad_shape = list(multi_robot_targets.shape[:-1]) + [pad_width]
                    pad_tensor = torch.zeros(pad_shape, device=multi_robot_targets.device, dtype=multi_robot_targets.dtype)
                    multi_robot_targets = torch.cat([multi_robot_targets, pad_tensor], dim=-1)

                # Clamp to [0,1]
                multi_robot_targets = multi_robot_targets.clamp(0.0, 1.0)

        with get_autocast_context(self.config):
            # Use vision-only forward pass - bypasses language model entirely
            # Use model_ref to access underlying model when using DDP
            outputs = model_ref.forward_vision_only(
                pixel_values=pixel_values,
                robot_targets=robot_targets,
                return_reasoning=True,
            )

            # Validate robot logits width matches num_robots
            if num_robots is not None and 'robot_logits' in outputs:
                if outputs['robot_logits'].shape[-1] != num_robots:
                    logger.error(
                        f"❌ robot_logits dim mismatch: got {outputs['robot_logits'].shape[-1]}, expected {num_robots}; trimming for safety"
                    )
                    outputs['robot_logits'] = outputs['robot_logits'][..., :num_robots]

            loss = outputs['loss']

            # Robot selection accuracy
            if 'robot_logits' in outputs:
                robot_preds = outputs['robot_logits'].argmax(dim=-1)
                robot_acc = (robot_preds == robot_targets).float().mean()

                # Store for visualization (detach + CPU to avoid GPU memory buildup)
                self.last_robot_preds = robot_preds.detach().cpu()
                self.last_robot_targets = robot_targets.detach().cpu()
                self.last_confidences = torch.softmax(outputs['robot_logits'], dim=-1).max(dim=-1)[0].detach().cpu()

                # Store top-k predictions if available
                if 'top_k_indices' in outputs:
                    self.last_topk_indices = outputs['top_k_indices'].detach().cpu()
                    self.last_topk_scores = outputs['top_k_scores'].detach().cpu()

                # Store multi-robot outputs for multi-label evaluation
                if 'multi_robot_logits' in outputs and multi_robot_targets is not None:
                    multi_preds = torch.sigmoid(outputs['multi_robot_logits']) > 0.5
                    multi_acc = ((multi_preds == multi_robot_targets.bool()).float().mean())
                    self.last_multi_robot_preds = outputs['multi_robot_logits'].detach().cpu()
                    self.last_multi_robot_targets = multi_robot_targets.detach().cpu()
            else:
                robot_acc = torch.tensor(0.0)

        return loss, {
            'robot_loss': loss.item(),
            'robot_accuracy': robot_acc.item(),
        }

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        dataloader = self.robot_dataloader
        desc = f"Robot Selection Epoch {epoch}"

        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # TRIAL MODE: Stop early if we hit max training steps
            if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                if self.global_step >= self.config.num_training_steps:
                    logger.info(f"🛑 Reached max training steps ({self.config.num_training_steps}), stopping epoch early")
                    break
            
            loss, metrics = self.train_robot_step(batch)

            # Backward
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                metrics['lr'] = self.scheduler.get_last_lr()[0]
                self.metric_tracker.update(metrics)

                # Always track per-step metrics for collated plots
                if self.wandb_logger and hasattr(self.wandb_logger, 'metrics_history'):
                    for _mk, _mv in metrics.items():
                        if isinstance(_mv, (int, float)):
                            self.wandb_logger.metrics_history.setdefault(_mk, []).append(float(_mv))

                # Periodic GPU memory management to prevent OOM from fragmentation
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache()

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        if self.wandb_logger is not None:
                            if hasattr(self.wandb_logger, 'log_with_visualization'):
                                self.wandb_logger.log_with_visualization(
                                    avg_metrics,
                                    step=self.global_step,
                                    stage_name="stage3",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Stage 3 specific visualizations every 500 steps
                            if self.global_step % 500 == 0:
                                logger.info(f"[Stage3] Step {self.global_step}: Attempting visualizations...")
                                logger.info(f"  stage_visualizer: {self.stage_visualizer is not None}")
                                logger.info(f"  last_robot_preds: {self.last_robot_preds is not None}")
                                logger.info(f"  last_robot_targets: {self.last_robot_targets is not None}")
                                logger.info(f"  last_confidences: {self.last_confidences is not None}")

                                if self.stage_visualizer is not None:
                                    if self.last_robot_preds is not None and self.last_robot_targets is not None:
                                        try:
                                            logger.info(f"  Generating confusion matrix...")
                                            _, cm_img = self.stage_visualizer.plot_confusion_matrix(
                                                self.last_robot_preds,
                                                self.last_robot_targets,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage3/confusion_matrix", cm_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged confusion matrix to W&B")

                                            if self.last_confidences is not None:
                                                logger.info(f"  Generating calibration plot...")
                                                correct = (self.last_robot_preds == self.last_robot_targets)
                                                _, cal_img = self.stage_visualizer.plot_confidence_calibration(
                                                    self.last_confidences,
                                                    correct,
                                                    self.global_step
                                                )
                                                self.wandb_logger.log_image(
                                                    "stage3/calibration", cal_img, step=self.global_step
                                                )
                                                logger.info(f"  ✓ Logged calibration plot to W&B")
                                        except Exception as e:
                                            logger.error(f"  ✗ Failed to generate Stage 3 visualizations: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"  Skipping visualizations: predictions not available")
                                else:
                                    logger.warning(f"  Skipping visualizations: stage_visualizer is None")

                            # Gradient distribution every 500 steps
                            if self.global_step % 500 == 0 and hasattr(self.wandb_logger, 'log_gradient_distribution'):
                                gradients = {}
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None and param.requires_grad:
                                        gradients[name.split('.')[-1]] = param.grad
                                if gradients:
                                    self.wandb_logger.log_gradient_distribution(
                                        gradients, self.global_step, "stage3"
                                    )

                    # VA neuron heatmap (runs on its own interval, e.g. every 50 steps)
                    if is_main_process() and self.va_tracker is not None:
                        try:
                            self.va_tracker.maybe_generate(
                                model=self.model,
                                tokenizer=self.tokenizer,
                                global_step=self.global_step,
                                device=self.device,
                            )
                        except Exception as e:
                            logger.warning(f"VA heatmap generation skipped: {e}")

                    display_metrics = {k: f"{v:.4f}" for k, v in avg_metrics.items()
                                      if k != 'lr'}
                    progress_bar.set_postfix(display_metrics)

                    self.metric_tracker.reset()

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.val_dataloader is not None and \
                   self.global_step % self.config.eval_steps == 0:
                    self.evaluate()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate robot selection performance with comprehensive metrics.

        IMPORTANT: Uses vision-only forward pass to avoid tokenization issues.
        Stage 3 is primarily a vision-based robot selection task, so this is
        the most robust approach. Language reasoning evaluation happens in Stage 4.
        """
        self.model.eval()

        # Reset metrics
        self.robot_metrics.reset()
        self.reasoning_metrics.reset()

        val_data = self.val_dataloader if self.val_dataloader else self.robot_dataloader

        # Get model reference (unwrap DDP if needed)
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        num_robots = getattr(model_ref.config, 'num_robots', 5)

        for batch in tqdm(
            val_data,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)

            robot_targets = batch.get('robot_target')
            multi_robot_targets = batch.get('multi_robot_target')

            if robot_targets is not None:
                robot_targets = robot_targets.to(self.device)
                # Clamp targets to valid range
                robot_targets = torch.clamp(robot_targets, 0, num_robots - 1)

            if multi_robot_targets is not None:
                multi_robot_targets = multi_robot_targets.to(self.device)
                # Ensure multi-hot vector width matches num_robots
                if multi_robot_targets.size(-1) > num_robots:
                    multi_robot_targets = multi_robot_targets[..., :num_robots]
                elif multi_robot_targets.size(-1) < num_robots:
                    pad_width = num_robots - multi_robot_targets.size(-1)
                    pad_shape = list(multi_robot_targets.shape[:-1]) + [pad_width]
                    pad_tensor = torch.zeros(pad_shape, device=multi_robot_targets.device, dtype=multi_robot_targets.dtype)
                    multi_robot_targets = torch.cat([multi_robot_targets, pad_tensor], dim=-1)
                multi_robot_targets = multi_robot_targets.clamp(0.0, 1.0)

            with get_autocast_context(self.config):
                # Use vision-only forward pass - bypasses language model entirely
                # This avoids all tokenization-related CUDA index out of bounds errors
                # and is appropriate since Stage 3 is primarily vision-based robot selection
                outputs = model_ref.forward_vision_only(
                    pixel_values=pixel_values,
                    robot_targets=robot_targets,
                    return_reasoning=True,
                )

                if 'robot_logits' in outputs and robot_targets is not None:
                    if num_robots is not None and outputs['robot_logits'].shape[-1] != num_robots:
                        outputs['robot_logits'] = outputs['robot_logits'][..., :num_robots]

                    robot_preds = outputs['robot_logits'].argmax(dim=-1)

                    # Get confidence scores (softmax probabilities)
                    confidences = torch.softmax(outputs['robot_logits'], dim=-1)
                    pred_confidences = confidences.gather(1, robot_preds.unsqueeze(1)).squeeze(1)

                    # Multi-robot predictions (if available)
                    multi_robot_preds = None
                    if 'multi_robot_logits' in outputs and multi_robot_targets is not None:
                        multi_robot_preds = torch.sigmoid(outputs['multi_robot_logits'])
                        if multi_robot_preds.size(-1) > num_robots:
                            multi_robot_preds = multi_robot_preds[..., :num_robots]

                    # Update comprehensive metrics
                    self.robot_metrics.update(
                        predictions=robot_preds,
                        targets=robot_targets,
                        confidences=pred_confidences,
                        multi_robot_preds=multi_robot_preds,
                        multi_robot_targets=multi_robot_targets,
                    )

        # Compute all metrics
        metrics = self.robot_metrics.compute()

        if is_main_process():
            logger.info(f"Validation: {metrics}")

            if self.wandb_logger is not None:
                self.wandb_logger.log(metrics, step=self.global_step)

                # Log confusion matrix visualization
                if hasattr(self.wandb_logger, 'log_robot_confusion_matrix'):
                    all_preds = self.robot_metrics.get_all_predictions()
                    all_targets = self.robot_metrics.get_all_targets()
                    if all_preds is not None and all_targets is not None:
                        self.wandb_logger.log_robot_confusion_matrix(
                            predictions=all_preds,
                            labels=all_targets,
                            step=self.global_step,
                        )

                # Log per-robot radar chart
                if hasattr(self.wandb_logger, 'log_robot_radar_chart'):
                    per_robot_metrics = self.robot_metrics.get_per_robot_metrics()
                    if per_robot_metrics:
                        self.wandb_logger.log_robot_radar_chart(
                            metrics=per_robot_metrics,
                            step=self.global_step,
                        )

                # Log calibration plot
                if hasattr(self.wandb_logger, 'log_calibration_plot'):
                    all_confidences = self.robot_metrics.get_all_confidences()
                    all_correct = self.robot_metrics.get_all_correct()
                    if all_confidences is not None and all_correct is not None:
                        self.wandb_logger.log_calibration_plot(
                            confidences=all_confidences,
                            correct=all_correct,
                            step=self.global_step,
                        )

                # Log comprehensive stage summary with advanced visualizations
                if hasattr(self.wandb_logger, 'log_comprehensive_stage_summary'):
                    per_robot_metrics = self.robot_metrics.get_per_robot_metrics()
                    all_preds = self.robot_metrics.get_all_predictions()
                    all_targets = self.robot_metrics.get_all_targets()
                    all_confidences = self.robot_metrics.get_all_confidences()
                    all_correct = self.robot_metrics.get_all_correct()

                    # Build confusion matrix
                    cm = None
                    if all_preds is not None and all_targets is not None:
                        try:
                            from sklearn.metrics import confusion_matrix as sklearn_cm
                            cm = sklearn_cm(
                                all_targets.cpu().numpy() if hasattr(all_targets, 'cpu') else all_targets,
                                all_preds.cpu().numpy() if hasattr(all_preds, 'cpu') else all_preds,
                                labels=list(range(5))
                            )
                        except Exception:
                            pass

                    extra_viz = {
                        'per_robot_metrics': per_robot_metrics,
                        'confidences': all_confidences,
                        'correct': all_correct,
                    }

                    self.wandb_logger.log_comprehensive_stage_summary(
                        stage=3,
                        metrics=metrics,
                        step=self.global_step,
                        confusion_matrix=cm,
                        class_names=["Drone", "Underwater", "Humanoid", "Wheeled", "Legged"],
                        extra_visualizations=extra_viz,
                    )

                # Log training dashboard every epoch
                if hasattr(self.wandb_logger, 'log_training_dashboard'):
                    self.wandb_logger.log_training_dashboard(
                        stage=3,
                        metrics_history=self.wandb_logger.metrics_history if hasattr(self.wandb_logger, 'metrics_history') else {},
                        step=self.global_step,
                    )

        self.model.train()
        return metrics

    def save_checkpoint(self):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir) / f'checkpoint-{self.global_step}'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            step=self.global_step,
            metrics=self.metric_tracker.get_average(),
            output_dir=str(output_dir),
            scaler=self.scaler,
        )

        # Clean old checkpoints
        if is_main_process():
            # Get all checkpoint directories (both checkpoint-{step} and checkpoint-epoch-{n} formats)
            checkpoints = []
            for cp in Path(self.config.output_dir).glob('checkpoint-*'):
                # Skip best_checkpoint markers or symlinks
                if 'best' in cp.name.lower():
                    continue
                # Extract numeric part for sorting
                parts = cp.name.split('-')
                try:
                    if 'epoch' in cp.name.lower():
                        num = int(parts[-1])
                    else:
                        num = int(parts[-1])
                    checkpoints.append((num, cp))
                except (ValueError, IndexError):
                    continue

            # Sort by numeric value
            checkpoints.sort(key=lambda x: x[0])
            checkpoints = [cp for _, cp in checkpoints]

            while len(checkpoints) > self.config.max_checkpoints:
                oldest = checkpoints.pop(0)
                import shutil
                try:
                    if oldest.is_symlink():
                        # Remove symbolic link directly - don't use rmtree
                        oldest.unlink()
                    elif oldest.is_dir():
                        # Use rmtree only for actual directories
                        shutil.rmtree(str(oldest), ignore_errors=True)
                    elif oldest.is_file():
                        oldest.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {oldest}: {e}")

    def train(self, robot_epochs: int = 20):
        """Run Stage 3 robot selection training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            logger.info("Stage 3: Robot Fleet Selection Training")
            for epoch in range(robot_epochs):
                # TRIAL MODE: Check if we've already hit max steps
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        break
                
                if hasattr(self.robot_dataloader.sampler, 'set_epoch'):
                    self.robot_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)
                
                # TRIAL MODE: Check again after epoch
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        break

                # Evaluate after each epoch
                val_metrics = {}
                improved = False
                if self.val_dataloader is not None:
                    val_metrics = self.evaluate()
                    
                    # Track metrics history
                    self.metrics_history.append(val_metrics)
                    
                    # Monitor macro F1 (higher is better)
                    monitor_metric = val_metrics.get('macro_f1', 0.0)
                    
                    # Check early stopping
                    improved = self.early_stopping(monitor_metric, epoch)
                    
                    if improved:
                        # Save best checkpoint
                        checkpoint_path = str(Path(self.config.output_dir) / f'checkpoint-epoch-{epoch+1}')
                        self.save_checkpoint()
                        self.best_checkpoint_tracker.update(monitor_metric, checkpoint_path)
                    
                    # Check if we should stop
                    if self.early_stopping.early_stop:
                        logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                        break

                # Push to HuggingFace Hub after each epoch
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'accuracy': val_metrics.get('accuracy', 0.0),
                            'Drone_f1': val_metrics.get('Drone_f1', 0.0),
                            'Underwater_Robot_f1': val_metrics.get('Underwater Robot_f1', 0.0),
                            'Humanoid_f1': val_metrics.get('Humanoid_f1', 0.0),
                            'Robot_with_Wheels_f1': val_metrics.get('Robot with Wheels_f1', 0.0),
                            'Robot_with_Legs_f1': val_metrics.get('Robot with Legs_f1', 0.0),
                            'macro_f1': val_metrics.get('macro_f1', 0.0),
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=epoch + 1,
                            stage="stage3",
                            metrics=metrics,
                            vision_backbone=self.vision_backbone,
                            language_backbone=self.language_backbone,
                            carbon_emissions=carbon_emissions,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push to HuggingFace Hub: {e}")

                barrier()

            # Final save (if not already saved as best)
            if not improved:
                self.save_checkpoint()
            
            # Log training summary
            if is_main_process():
                log_stage_summary("Stage 3: Robot Fleet Selection", self.metrics_history)
                logger.info(f"✅ Stage 3 training completed: {len(self.metrics_history)}/{robot_epochs} epochs")
                logger.info(f"   Best checkpoint: {self.best_checkpoint_tracker.load_best_checkpoint()}")

                # Generate end-of-training visualizations
                logger.info("📊 Generating Stage 3 end-of-training visualizations...")
                try:
                    # Base visualizations (loss decomposition + convergence)
                    if self.wandb_logger is not None:
                        self.wandb_logger.generate_final_visualizations("stage3", self.global_step)

                    # Stage-specific: confusion matrix, radar, calibration
                    if self.stage_visualizer is not None:
                        if self.last_robot_preds is not None and self.last_robot_targets is not None:
                            try:
                                import torch
                                preds = self.last_robot_preds
                                targets = self.last_robot_targets
                                robot_names = getattr(self, 'robot_names', None)
                                if robot_names is None:
                                    robot_names = [f"Robot_{i}" for i in range(max(targets.max().item() + 1, 2))]
                                _, cm_img = self.stage_visualizer.plot_confusion_matrix(
                                    preds, targets, robot_names, self.global_step
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage3/confusion_matrix_final", cm_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final confusion matrix")
                            except Exception as e:
                                logger.warning(f"  Failed to generate confusion matrix: {e}")

                            # Calibration plot
                            if self.last_confidences is not None:
                                try:
                                    correct = (self.last_robot_preds == self.last_robot_targets).float()
                                    _, cal_img = self.stage_visualizer.plot_confidence_calibration(
                                        self.last_confidences, correct, self.global_step
                                    )
                                    if self.wandb_logger is not None:
                                        self.wandb_logger.log_image(
                                            "stage3/calibration_final", cal_img, step=self.global_step
                                        )
                                    logger.info("  ✓ Saved final confidence calibration")
                                except Exception as e:
                                    logger.warning(f"  Failed to generate calibration plot: {e}")
                        else:
                            logger.info("  Skipping stage-specific viz: no cached predictions")
                    logger.info("📊 Stage 3 visualizations complete")
                except Exception as e:
                    logger.warning(f"Failed to generate Stage 3 end-of-training visualizations: {e}")

                # Save metrics JSON for cross-stage collation
                try:
                    step_metrics = {}
                    if self.wandb_logger is not None and hasattr(self.wandb_logger, 'metrics_history'):
                        step_metrics = {k: v for k, v in self.wandb_logger.metrics_history.items()
                                       if isinstance(v, list) and v}
                    save_stage_metrics_json(
                        stage_name='stage3',
                        output_dir=self.config.output_dir,
                        metrics_history=self.metrics_history,
                        step_metrics=step_metrics,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save stage3 metrics JSON: {e}")

        finally:
            if is_main_process():
                if self.carbon_tracker is not None:
                    emissions = self.carbon_tracker.stop()
                    logger.info(f"Total emissions: {emissions:.4f} kg CO2eq")

                if self.wandb_logger is not None:
                    self.wandb_logger.finish()

            # Note: Do NOT call cleanup_distributed() here
            # The process group should persist across stages
            # cleanup_distributed() should only be called at the end of all training


def run_stage3_training(
    model: EmberVLM,
    config: TrainingConfig,
    robot_data_dir: str,
    tokenizer: Any,
    robot_epochs: int = 20,
    hub_repo_id: Optional[str] = None,
    vision_backbone: str = "repvit",
    language_backbone: str = "tinyllm",
):
    """Run Stage 3 robot selection training with HuggingFace Hub push support."""
    robot_dataloader = get_robot_selection_dataloader(
        data_dir=robot_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_robot_selection_dataloader(
        data_dir=robot_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    trainer = Stage3Trainer(
        model=model,
        config=config,
        robot_dataloader=robot_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        hub_repo_id=hub_repo_id,
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
    )

    trainer.train(robot_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage3')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--robot_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)
    else:
        model = EmberVLM()

    run_stage3_training(
        model=model,
        config=config,
        robot_data_dir=args.robot_dir,
        tokenizer=tokenizer,
        robot_epochs=args.robot_epochs,
    )
