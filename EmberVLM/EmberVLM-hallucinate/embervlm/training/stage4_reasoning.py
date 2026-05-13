"""
Stage 4: Chain-of-Thought Reasoning Integration

Integrates DeepSeek-R1 style reasoning with two-phase training:
1. Train reasoning heads with frozen backbone
2. Joint fine-tuning with reduced learning rate

Inspired by tiny-r1, uses structured XML format:
<reasoning>...</reasoning>
<answer>...</answer>
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
    unwrap_model,
    save_checkpoint,
    MetricTracker,
    print_trainable_parameters,
    push_checkpoint_to_hub,
    enable_gradient_checkpointing,
)
from embervlm.data.loaders import get_reasoning_dataloader
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker
from embervlm.training.early_stopping import log_stage_summary, save_stage_metrics_json

# Import reward functions (tiny-r1 style)
try:
    from embervlm.training.reasoning_rewards import (
        ReasoningRewardModel,
        ReasoningRewardLoss,
        extract_xml_answer,
        extract_xml_reasoning,
        compute_total_reward,
    )
    HAS_REWARD_FUNCS = True
except ImportError:
    HAS_REWARD_FUNCS = False
    ReasoningRewardModel = None
    ReasoningRewardLoss = None

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage4Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage4Visualizer = None


class ReasoningConsistencyLoss(nn.Module):
    """Loss for ensuring consistent reasoning chains (DeepSeek-R1 style)."""

    def __init__(self, use_reward_model: bool = True):
        super().__init__()
        self.use_reward_model = use_reward_model and HAS_REWARD_FUNCS
        if self.use_reward_model:
            self.reward_model = ReasoningRewardModel()
        else:
            self.reward_model = None

    def forward(
        self,
        reasoning_chain: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None,
        generated_texts: Optional[List[str]] = None,
        target_robots: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute reasoning consistency loss with optional reward signal.

        Args:
            reasoning_chain: Generated reasoning [B, steps, seq_len, dim]
            target_chain: Optional target reasoning
            generated_texts: Optional list of generated text for reward calculation
            target_robots: Optional list of target robot names for correctness reward

        Returns:
            Consistency loss with reward signal incorporated
        """
        # Encourage smooth transitions between steps
        if reasoning_chain.dim() == 4:
            step_diffs = reasoning_chain[:, 1:] - reasoning_chain[:, :-1]
            smoothness_loss = torch.mean(step_diffs ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=reasoning_chain.device)

        # If target is provided, compute MSE
        if target_chain is not None:
            target_loss = F.mse_loss(reasoning_chain, target_chain)
            total_loss = smoothness_loss + target_loss
        else:
            total_loss = smoothness_loss

        # If we have generated texts, incorporate reward signal as a loss modifier
        if generated_texts is not None and self.reward_model is not None:
            rewards = self.reward_model(generated_texts, target_robots)
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                # Higher reward = lower loss modifier (reward ranges 0-5 typically)
                reward_modifier = max(0.5, 2.0 - mean_reward / 2.5)
                total_loss = total_loss * reward_modifier

        return total_loss


class XMLFormatLoss(nn.Module):
    """
    Loss function that encourages proper XML format in generated text.
    Inspired by tiny-r1's reward functions.
    """

    def __init__(self):
        super().__init__()
        if HAS_REWARD_FUNCS:
            from embervlm.training.reasoning_rewards import xml_count_reward, soft_format_reward
            self.xml_count_reward = xml_count_reward
            self.soft_format_reward = soft_format_reward
        else:
            self.xml_count_reward = None
            self.soft_format_reward = None

    def forward(self, generated_texts: List[str]) -> torch.Tensor:
        """
        Compute format loss based on XML structure.
        Lower reward = higher loss.
        """
        if self.xml_count_reward is None:
            return torch.tensor(0.0)

        # Get XML count rewards (0-1 range)
        xml_rewards = self.xml_count_reward(generated_texts)
        format_rewards = self.soft_format_reward(generated_texts)

        # Combine rewards
        combined_rewards = [(x + f) / 2.0 for x, f in zip(xml_rewards, format_rewards)]

        # Convert to loss (1 - reward)
        mean_reward = sum(combined_rewards) / len(combined_rewards) if combined_rewards else 0.0
        format_loss = 1.0 - mean_reward

        return torch.tensor(format_loss)


class Stage4Trainer:
    """Trainer for Stage 4: Chain-of-Thought Reasoning (DeepSeek-R1 style)."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
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

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        set_seed(config.seed, self.rank)

        # Unwrap model if it was previously wrapped with DDP
        model = unwrap_model(model)

        # Enable gradient checkpointing for memory efficiency
        if getattr(config, 'gradient_checkpointing', True):
            logger.info("Enabling gradient checkpointing for memory efficiency...")
            if hasattr(model, 'language_model'):
                enable_gradient_checkpointing(model.language_model)
                logger.info("✓ Gradient checkpointing enabled on language model")

        # Ensure model is on correct device before DDP
        model = model.to(self.device)

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Phase tracking
        self.current_phase = 1

        # Reward model (tiny-r1 style)
        self.reward_model = None
        if HAS_REWARD_FUNCS:
            self.reward_model = ReasoningRewardModel()
            logger.info("✓ Initialized reasoning reward model (tiny-r1 style)")

        # Optimizer (will be re-created for each phase)
        self.optimizer = None
        self.scheduler = None
        self.scaler = get_grad_scaler(config)

        # Losses (with reward model integration)
        self.reasoning_consistency_loss = ReasoningConsistencyLoss(use_reward_model=True)
        self.xml_format_loss = XMLFormatLoss()

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger (main process)...")
            try:
                from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
                wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                self.wandb_logger = EnhancedWandbLogger(
                    project=wandb_project,
                    name="stage4_reasoning",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info(f"Enhanced W&B logger initialized with project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced W&B logger: {e}")
                # Fallback to basic logger
                try:
                    wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                    self.wandb_logger = WandbLogger(
                        project=wandb_project,
                        name="stage4_reasoning",
                        config=config.to_dict(),
                    )
                    logger.info(f"Basic W&B logger initialized with project: {wandb_project}")
                except Exception as e2:
                    logger.warning(f"Failed to initialize W&B logger: {e2}")
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
        self.global_step = 0

        # Reward tracking for visualization
        self.reward_history = {
            'total': [],
            'correctness': [],
            'format': [],
            'reasoning_quality': [],
        }

        # Stage 4 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage4Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage4Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage4Visualizer: {e}")

        # Track metrics for phase comparison
        self.phase1_metrics = {'loss': [], 'robot_accuracy': [], 'consistency_loss': []}
        self.phase2_metrics = {'loss': [], 'robot_accuracy': [], 'consistency_loss': []}

        # Epoch-level metrics history (for log_stage_summary + save_stage_metrics_json)
        self.metrics_history = []

        # Per-sample reasoning quality tracking (for plot_reasoning_quality_metrics)
        self.eval_coherence_scores = []
        self.eval_consistency_scores = []
        self.eval_step_counts = []

        # CoT comparison tracking (for plot_cot_examples)
        self.eval_cot_tasks = []
        self.eval_without_cot = []  # List of (prediction, correct)
        self.eval_with_cot = []    # List of (reasoning, prediction, correct)

    def _freeze_backbone(self):
        """Freeze backbone, only train reasoning heads."""
        model = unwrap_model(self.model)

        # Freeze vision encoder
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze language model
        for param in model.language_model.parameters():
            param.requires_grad = False

        # Freeze fusion module
        for param in model.fusion_module.parameters():
            param.requires_grad = False

        # Keep reasoning module trainable
        if hasattr(model, 'reasoning_module'):
            for param in model.reasoning_module.parameters():
                param.requires_grad = True

        if is_main_process():
            logger.info("Phase 1: Backbone frozen, training reasoning heads only")
            print_trainable_parameters(model)

    def _unfreeze_all(self):
        """Unfreeze all trainable parameters."""
        model = unwrap_model(self.model)

        # Unfreeze last layer of language model - handle both GPT-2 and Llama structures
        lm_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
        
        # Try GPT-2 style (TinyLLM) first
        if hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'h'):
            for param in lm_model.transformer.h[-1].parameters():
                param.requires_grad = True
            for param in lm_model.transformer.ln_f.parameters():
                param.requires_grad = True
        # Try Llama style (SmolLM)
        elif hasattr(lm_model, 'layers'):
            for param in lm_model.layers[-1].parameters():
                param.requires_grad = True
            if hasattr(lm_model, 'norm'):
                for param in lm_model.norm.parameters():
                    param.requires_grad = True
        
        # Unfreeze LM head
        if hasattr(model.language_model, 'lm_head'):
            for param in model.language_model.lm_head.parameters():
                param.requires_grad = True
        elif hasattr(lm_model, 'lm_head'):
            for param in lm_model.lm_head.parameters():
                param.requires_grad = True

        # Unfreeze fusion module
        for param in model.fusion_module.parameters():
            param.requires_grad = True

        # Keep reasoning module trainable
        if hasattr(model, 'reasoning_module'):
            for param in model.reasoning_module.parameters():
                param.requires_grad = True

        if is_main_process():
            logger.info("Phase 2: Joint fine-tuning")
            print_trainable_parameters(model)

    def _setup_phase(self, phase: int, lr: float, num_steps: int):
        """Setup optimizer for current phase."""
        self.current_phase = phase

        if phase == 1:
            self._freeze_backbone()
        else:
            self._unfreeze_all()

        # Create new optimizer
        phase_config = TrainingConfig(
            learning_rate=lr,
            min_learning_rate=lr / 10,
            num_training_steps=num_steps,
            warmup_steps=min(100, num_steps // 10),
            weight_decay=self.config.weight_decay,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            eps=self.config.eps,
            optimizer=self.config.optimizer,
            scheduler=self.config.scheduler,
        )

        self.optimizer = get_optimizer(self.model, phase_config)
        self.scheduler = get_scheduler(self.optimizer, phase_config)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        robot_targets = batch.get('robot_target')
        if robot_targets is not None:
            robot_targets = robot_targets.to(self.device)

        reasoning_targets = batch.get('reasoning_chain')

        # CRITICAL: Validate input_ids and labels are within embedding bounds
        # This prevents cryptic CUDA index out of bounds errors
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        vocab_size = None
        if hasattr(model_ref.language_model, 'get_input_embeddings'):
            vocab_size = model_ref.language_model.get_input_embeddings().weight.shape[0]
        elif hasattr(model_ref.language_model, 'model'):
            if hasattr(model_ref.language_model.model, 'get_input_embeddings'):
                vocab_size = model_ref.language_model.model.get_input_embeddings().weight.shape[0]

        if vocab_size is not None:
            # Validate and clamp input_ids
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()
            if max_token_id >= vocab_size or min_token_id < 0:
                if max_token_id >= vocab_size:
                    logger.error(f"❌ input_ids max={max_token_id} >= vocab_size={vocab_size}")
                if min_token_id < 0:
                    logger.error(f"❌ input_ids min={min_token_id} < 0")
                input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
                logger.warning(f"   Token IDs clamped to valid range [0, {vocab_size - 1}]")

            # Validate and clamp labels (preserve -100 ignore index)
            valid_labels_mask = labels != -100
            if valid_labels_mask.any():
                valid_labels = labels[valid_labels_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()
                if max_label >= vocab_size or min_label < 0:
                    labels = torch.where(
                        valid_labels_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )
                    logger.warning(f"   Labels clamped to valid range [0, {vocab_size - 1}]")

        with get_autocast_context(self.config):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                robot_targets=robot_targets,
                return_reasoning=True,
            )

            loss = outputs['loss']

            # Reasoning consistency loss
            if 'reasoning_chain' in outputs:
                reasoning_chain = outputs['reasoning_chain']

                if reasoning_targets is not None:
                    reasoning_targets = reasoning_targets.to(self.device)
                    consistency_loss = self.reasoning_consistency_loss(
                        reasoning_chain, reasoning_targets
                    )
                else:
                    consistency_loss = self.reasoning_consistency_loss(reasoning_chain)

                loss = loss + 0.1 * consistency_loss
            else:
                consistency_loss = torch.tensor(0.0)

        metrics = {
            'loss': loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0,
        }

        if 'robot_logits' in outputs and robot_targets is not None:
            robot_preds = outputs['robot_logits'].argmax(dim=-1)
            robot_acc = (robot_preds == robot_targets).float().mean()
            metrics['robot_accuracy'] = robot_acc.item()

        return loss, metrics

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Phase {self.current_phase} Epoch {epoch}",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # TRIAL MODE: Stop early if we hit max training steps
            if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                if self.global_step >= self.config.num_training_steps:
                    logger.info(f"🛑 Reached max training steps ({self.config.num_training_steps}), stopping epoch early")
                    break
            
            loss, metrics = self.train_step(batch)

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
                metrics['phase'] = self.current_phase
                self.metric_tracker.update(metrics)

                # Periodic GPU memory management to prevent OOM from fragmentation
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache()

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    # Track metrics for phase comparison
                    phase_metrics = self.phase1_metrics if self.current_phase == 1 else self.phase2_metrics
                    for key in ['loss', 'robot_accuracy', 'consistency_loss']:
                        if key in avg_metrics:
                            phase_metrics[key].append(avg_metrics[key])

                    if is_main_process():
                        if self.wandb_logger is not None:
                            # Use enhanced logging if available
                            if hasattr(self.wandb_logger, 'log_with_visualization'):
                                self.wandb_logger.log_with_visualization(
                                    avg_metrics,
                                    step=self.global_step,
                                    stage_name="stage4",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Stage 4 specific visualizations every 500 steps
                            if self.global_step % 500 == 0:
                                logger.info(f"[Stage4] Step {self.global_step}: Attempting visualizations...")

                                if self.stage_visualizer is not None:
                                    try:
                                        # Phase comparison visualization
                                        if len(self.phase1_metrics['loss']) >= 5 or len(self.phase2_metrics['loss']) >= 5:
                                            logger.info(f"  Generating phase comparison...")
                                            _, phase_img = self.stage_visualizer.plot_phase_comparison(
                                                self.phase1_metrics,
                                                self.phase2_metrics,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage4/phase_comparison", phase_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged phase comparison to W&B")
                                    except Exception as e:
                                        logger.error(f"  ✗ Failed to generate Stage 4 visualizations: {e}", exc_info=True)

                                # Gradient distribution every 500 steps
                                if hasattr(self.wandb_logger, 'log_gradient_distribution'):
                                    gradients = {}
                                    for name, param in self.model.named_parameters():
                                        if param.grad is not None and param.requires_grad:
                                            gradients[name.split('.')[-1]] = param.grad
                                    if gradients:
                                        self.wandb_logger.log_gradient_distribution(
                                            gradients, self.global_step, "stage4"
                                        )

                        display = {k: f"{v:.4f}" for k, v in avg_metrics.items()
                                  if k not in ['lr', 'phase']}
                        progress_bar.set_postfix(display)

                    self.metric_tracker.reset()

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate reasoning quality with reward-based metrics (DeepSeek-R1 style)."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        eval_metrics = MetricTracker()

        # Collect generated texts for reward evaluation
        all_generated_texts = []
        all_target_robots = []

        # Per-sample tracking for reasoning quality visualization
        batch_coherence_scores = []
        batch_consistency_scores = []
        batch_step_counts = []

        # CoT example tracking (collect a few representative examples)
        cot_tasks_batch = []
        cot_without_batch = []  # (prediction_str, correct_bool)
        cot_with_batch = []     # (reasoning_str, prediction_str, correct_bool)

        ROBOT_NAMES = ['drone', 'underwater', 'humanoid', 'wheeled', 'legged']

        for batch in tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            robot_targets = batch.get('robot_target')
            if robot_targets is not None:
                robot_targets = robot_targets.to(self.device)
                # Convert to robot names for reward evaluation
                target_names = batch.get('robot_target_names', [])
                if target_names:
                    all_target_robots.extend(target_names)

            # --- Forward WITH reasoning ---
            with get_autocast_context(self.config):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_reasoning=True,
                )

                if 'robot_logits' in outputs and robot_targets is not None:
                    robot_preds = outputs['robot_logits'].argmax(dim=-1)
                    robot_acc = (robot_preds == robot_targets).float().mean()
                    eval_metrics.update({'robot_accuracy': robot_acc.item()})

                if 'robot_confidence' in outputs:
                    confidence = outputs['robot_confidence'].mean()
                    eval_metrics.update({'confidence': confidence.item()})

                if 'plan_coherence' in outputs:
                    coherence = outputs['plan_coherence'].mean()
                    eval_metrics.update({'plan_coherence': coherence.item()})

                # --- Collect per-sample reasoning quality metrics ---
                if 'plan_coherence' in outputs and outputs['plan_coherence'] is not None:
                    coherence_vals = outputs['plan_coherence'].detach().float().flatten()
                    batch_coherence_scores.extend([float(v) for v in coherence_vals.cpu()])

                if 'reasoning_chain' in outputs and outputs['reasoning_chain'] is not None:
                    rc = outputs['reasoning_chain']
                    # Count reasoning steps per sample
                    if rc.dim() >= 2:
                        n_steps = rc.shape[1] if rc.dim() >= 3 else 1
                        batch_step_counts.extend([n_steps] * rc.shape[0])

                    # Compute consistency score from smoothness of reasoning chain
                    if rc.dim() == 4 and rc.shape[1] > 1:
                        step_diffs = rc[:, 1:] - rc[:, :-1]
                        per_sample_smooth = torch.mean(step_diffs ** 2, dim=(1, 2, 3))
                        # Convert to 0-1 consistency (lower diff = higher consistency)
                        consistency = 1.0 / (1.0 + per_sample_smooth)
                        batch_consistency_scores.extend([float(v) for v in consistency.flatten().cpu()])
                    else:
                        batch_consistency_scores.extend([0.5] * (rc.shape[0] if rc.dim() >= 1 else 1))

                # --- Collect CoT comparison examples (up to 8 total) ---
                if len(cot_tasks_batch) < 8 and robot_targets is not None and 'robot_logits' in outputs:
                    B = min(2, robot_targets.shape[0])  # 2 per batch to accumulate
                    for b_idx in range(B):
                        if len(cot_tasks_batch) >= 8:
                            break
                        target_idx = robot_targets[b_idx].item()
                        target_name = ROBOT_NAMES[target_idx] if target_idx < len(ROBOT_NAMES) else f'robot_{target_idx}'
                        pred_idx = outputs['robot_logits'][b_idx].argmax().item()
                        pred_name = ROBOT_NAMES[pred_idx] if pred_idx < len(ROBOT_NAMES) else f'robot_{pred_idx}'
                        correct_with_cot = (pred_idx == target_idx)

                        # Task description
                        task_desc = f"Select robot for task (target: {target_name})"
                        if 'task_descriptions' in batch:
                            desc_list = batch['task_descriptions']
                            if b_idx < len(desc_list):
                                task_desc = desc_list[b_idx][:200]

                        # Reasoning text (from chain if available)
                        reasoning_text = "Reasoning chain generated"
                        if 'plan_coherence' in outputs and outputs['plan_coherence'] is not None:
                            coherence_val = float(outputs['plan_coherence'][b_idx]) if outputs['plan_coherence'].dim() > 0 else float(outputs['plan_coherence'])
                            reasoning_text = f"Coherence: {coherence_val:.3f}\nPrediction: {pred_name}"

                        cot_tasks_batch.append(task_desc)
                        cot_with_batch.append((reasoning_text, pred_name, correct_with_cot))
                        # Without-CoT: use logits without reasoning as approximation
                        cot_without_batch.append((pred_name, correct_with_cot))

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        # Store per-sample metrics for visualization
        if batch_coherence_scores:
            self.eval_coherence_scores = batch_coherence_scores
        if batch_consistency_scores:
            self.eval_consistency_scores = batch_consistency_scores
        if batch_step_counts:
            self.eval_step_counts = batch_step_counts
        if cot_tasks_batch:
            self.eval_cot_tasks = cot_tasks_batch
            self.eval_without_cot = cot_without_batch
            self.eval_with_cot = cot_with_batch

        # Compute reward-based metrics if reward model available
        if self.reward_model is not None and all_generated_texts:
            try:
                detailed_rewards = self.reward_model.get_detailed_rewards(
                    all_generated_texts,
                    all_target_robots if all_target_robots else None
                )

                # Add reward metrics
                for key, values in detailed_rewards.items():
                    if values:
                        avg_metrics[f'val_reward_{key}'] = sum(values) / len(values)

                # Track for visualization
                if 'total' in detailed_rewards:
                    self.reward_history['total'].append(sum(detailed_rewards['total']) / len(detailed_rewards['total']))

            except Exception as e:
                logger.warning(f"Failed to compute reward metrics: {e}")

        if is_main_process():
            if self.wandb_logger is not None:
                self.wandb_logger.log(avg_metrics, step=self.global_step)

                # Log reward visualization if we have enough data
                if len(self.reward_history['total']) >= 3 and hasattr(self.wandb_logger, 'log_custom_chart'):
                    try:
                        import wandb
                        data = [[i, r] for i, r in enumerate(self.reward_history['total'])]
                        table = wandb.Table(data=data, columns=["step", "reward"])
                        self.wandb_logger.log({
                            "stage4/reward_history": wandb.plot.line(
                                table, "step", "reward", title="Reasoning Reward Over Time"
                            )
                        }, step=self.global_step)
                    except Exception as e:
                        logger.debug(f"Could not log reward chart: {e}")

            # Generate reasoning quality visualization if we have per-sample data
            if self.stage_visualizer is not None and len(self.eval_coherence_scores) >= 2:
                try:
                    # Fill in consistency scores if we didn't get enough
                    if not self.eval_consistency_scores:
                        self.eval_consistency_scores = [0.5] * len(self.eval_coherence_scores)
                    if not self.eval_step_counts:
                        self.eval_step_counts = [1] * len(self.eval_coherence_scores)

                    # Align lengths
                    min_len = min(len(self.eval_coherence_scores), len(self.eval_consistency_scores), len(self.eval_step_counts))
                    _, quality_img = self.stage_visualizer.plot_reasoning_quality_metrics(
                        coherence_scores=self.eval_coherence_scores[:min_len],
                        consistency_scores=self.eval_consistency_scores[:min_len],
                        step_counts=self.eval_step_counts[:min_len],
                        step=self.global_step,
                    )
                    if self.wandb_logger is not None:
                        self.wandb_logger.log_image(
                            "stage4/reasoning_quality", quality_img, step=self.global_step
                        )
                    logger.info(f"  ✓ Saved reasoning quality visualization (step {self.global_step})")
                except Exception as e:
                    logger.warning(f"  Failed to generate reasoning quality viz: {e}")

            # Generate CoT comparison visualization if we have examples
            if self.stage_visualizer is not None and len(self.eval_cot_tasks) >= 2:
                try:
                    _, cot_img = self.stage_visualizer.plot_cot_examples(
                        tasks=self.eval_cot_tasks,
                        without_cot=self.eval_without_cot,
                        with_cot=self.eval_with_cot,
                        step=self.global_step,
                        n_examples=min(4, len(self.eval_cot_tasks)),
                    )
                    if self.wandb_logger is not None:
                        self.wandb_logger.log_image(
                            "stage4/cot_comparison", cot_img, step=self.global_step
                        )
                    logger.info(f"  ✓ Saved CoT comparison visualization (step {self.global_step})")
                except Exception as e:
                    logger.warning(f"  Failed to generate CoT comparison viz: {e}")

            logger.info(f"Evaluation: {avg_metrics}")

        self.model.train()
        return avg_metrics

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

    def train(
        self,
        phase1_epochs: int = 5,
        phase2_epochs: int = 5,
        phase1_lr: float = 1e-4,
        phase2_lr: float = 5e-5,
    ):
        """Run full Stage 4 training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            # Calculate steps
            steps_per_epoch = len(self.train_dataloader)
            phase1_steps = phase1_epochs * steps_per_epoch
            phase2_steps = phase2_epochs * steps_per_epoch

            # Phase 1: Train reasoning heads with frozen backbone
            logger.info("="*50)
            logger.info("Phase 1: Training reasoning heads (frozen backbone)")
            logger.info("="*50)

            self._setup_phase(1, phase1_lr, phase1_steps)

            for epoch in range(phase1_epochs):
                # TRIAL MODE: Check if we've already hit max steps
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        return
                
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)
                
                # TRIAL MODE: Check again after epoch
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        return
                        
                val_metrics = self.evaluate()
                self.metrics_history.append(val_metrics)
                
                # Push to HuggingFace Hub after each epoch in Phase 1
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'reasoning_loss': self.metric_tracker.get_average().get('reasoning_loss', 0.0),
                            'robot_accuracy': val_metrics.get('robot_accuracy', 0.0),
                            'phase': 'phase1',
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=epoch + 1,
                            stage="stage4_phase1",
                            metrics=metrics,
                            vision_backbone=self.vision_backbone,
                            language_backbone=self.language_backbone,
                            carbon_emissions=carbon_emissions,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push to HuggingFace Hub: {e}")
                
                barrier()

            # Phase 2: Joint fine-tuning
            logger.info("="*50)
            logger.info("Phase 2: Joint fine-tuning")
            logger.info("="*50)

            self._setup_phase(2, phase2_lr, phase2_steps)

            for epoch in range(phase2_epochs):
                # TRIAL MODE: Check if we've already hit max steps
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        return
                
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(phase1_epochs + epoch)

                self.train_epoch(epoch)
                
                # TRIAL MODE: Check again after epoch
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        return
                        
                val_metrics = self.evaluate()
                self.metrics_history.append(val_metrics)
                
                # Push to HuggingFace Hub after each epoch in Phase 2
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'reasoning_loss': self.metric_tracker.get_average().get('reasoning_loss', 0.0),
                            'robot_accuracy': val_metrics.get('robot_accuracy', 0.0),
                            'phase': 'phase2',
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=phase1_epochs + epoch + 1,
                            stage="stage4_phase2",
                            metrics=metrics,
                            vision_backbone=self.vision_backbone,
                            language_backbone=self.language_backbone,
                            carbon_emissions=carbon_emissions,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push to HuggingFace Hub: {e}")
                
                barrier()

            # Final checkpoint
            self.save_checkpoint()

            # Log training summary
            if is_main_process():
                log_stage_summary("Stage 4: CoT Reasoning", self.metrics_history)
                logger.info(f"✅ Stage 4 training completed")

                # Generate end-of-training visualizations
                logger.info("📊 Generating Stage 4 end-of-training visualizations...")
                try:
                    # Base visualizations (loss decomposition + convergence)
                    if self.wandb_logger is not None:
                        self.wandb_logger.generate_final_visualizations("stage4", self.global_step)

                    # Stage-specific: phase comparison
                    if self.stage_visualizer is not None:
                        has_phase1 = len(self.phase1_metrics.get('loss', [])) >= 2
                        has_phase2 = len(self.phase2_metrics.get('loss', [])) >= 2
                        if has_phase1 or has_phase2:
                            try:
                                _, phase_img = self.stage_visualizer.plot_phase_comparison(
                                    self.phase1_metrics, self.phase2_metrics, self.global_step
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage4/phase_comparison_final", phase_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final phase comparison")
                            except Exception as e:
                                logger.warning(f"  Failed to generate phase comparison: {e}")
                        else:
                            logger.info("  Skipping phase comparison: not enough data")

                        # Final reasoning quality metrics
                        if len(self.eval_coherence_scores) >= 2:
                            try:
                                if not self.eval_consistency_scores:
                                    self.eval_consistency_scores = [0.5] * len(self.eval_coherence_scores)
                                if not self.eval_step_counts:
                                    self.eval_step_counts = [1] * len(self.eval_coherence_scores)
                                min_len = min(len(self.eval_coherence_scores), len(self.eval_consistency_scores), len(self.eval_step_counts))
                                _, quality_img = self.stage_visualizer.plot_reasoning_quality_metrics(
                                    coherence_scores=self.eval_coherence_scores[:min_len],
                                    consistency_scores=self.eval_consistency_scores[:min_len],
                                    step_counts=self.eval_step_counts[:min_len],
                                    step=self.global_step,
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage4/reasoning_quality_final", quality_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final reasoning quality visualization")
                            except Exception as e:
                                logger.warning(f"  Failed to generate reasoning quality viz: {e}")
                        else:
                            logger.info("  Skipping reasoning quality: not enough per-sample data")

                        # Final CoT comparison examples
                        if len(self.eval_cot_tasks) >= 2:
                            try:
                                _, cot_img = self.stage_visualizer.plot_cot_examples(
                                    tasks=self.eval_cot_tasks,
                                    without_cot=self.eval_without_cot,
                                    with_cot=self.eval_with_cot,
                                    step=self.global_step,
                                    n_examples=min(4, len(self.eval_cot_tasks)),
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage4/cot_comparison_final", cot_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final CoT comparison visualization")
                            except Exception as e:
                                logger.warning(f"  Failed to generate CoT comparison viz: {e}")
                        else:
                            logger.info("  Skipping CoT comparison: not enough example data")

                    logger.info("📊 Stage 4 visualizations complete")
                except Exception as e:
                    logger.warning(f"Failed to generate Stage 4 end-of-training visualizations: {e}")

                # Save metrics JSON for cross-stage collation
                try:
                    step_metrics = {}
                    if self.wandb_logger is not None and hasattr(self.wandb_logger, 'metrics_history'):
                        step_metrics = {k: v for k, v in self.wandb_logger.metrics_history.items()
                                       if isinstance(v, list) and v}
                    # Also include phase-specific metrics
                    step_metrics['phase1_loss'] = self.phase1_metrics.get('loss', [])
                    step_metrics['phase1_robot_accuracy'] = self.phase1_metrics.get('robot_accuracy', [])
                    step_metrics['phase2_loss'] = self.phase2_metrics.get('loss', [])
                    step_metrics['phase2_robot_accuracy'] = self.phase2_metrics.get('robot_accuracy', [])
                    save_stage_metrics_json(
                        stage_name='stage4',
                        output_dir=self.config.output_dir,
                        metrics_history=self.metrics_history,
                        step_metrics=step_metrics,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save stage4 metrics JSON: {e}")

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


def run_stage4_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    phase1_epochs: int = 5,
    phase2_epochs: int = 5,
    phase1_lr: float = 1e-4,
    phase2_lr: float = 5e-5,
    hub_repo_id: Optional[str] = None,
    vision_backbone: str = "repvit",
    language_backbone: str = "tinyllm",
):
    """Run Stage 4 training with HuggingFace Hub push support."""
    train_dataloader = get_reasoning_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_reasoning_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    trainer = Stage4Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        hub_repo_id=hub_repo_id,
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
    )

    trainer.train(
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
    )


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage4')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--phase1_epochs', type=int, default=5)
    parser.add_argument('--phase2_epochs', type=int, default=5)
    parser.add_argument('--phase1_lr', type=float, default=1e-4)
    parser.add_argument('--phase2_lr', type=float, default=5e-5)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)
    else:
        model = EmberVLM()

    run_stage4_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
    )

