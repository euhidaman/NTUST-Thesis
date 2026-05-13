"""
Stage 2: Multimodal Instruction Tuning

Teaches task-following capabilities with teacher distillation
from larger VLM models.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
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
    unwrap_model,
    set_seed,
    get_rank,
    barrier,
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
from embervlm.training.early_stopping import (
    EarlyStopping,
    BestCheckpointTracker,
    log_stage_summary,
    save_stage_metrics_json,
)
from embervlm.training.ema_teacher import EMATeacher
from embervlm.training.multi_scale_vision import MultiScaleVisionExtractor
from embervlm.data.loaders import get_instruction_dataloader
from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage2Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage2Visualizer = None


class RepetitionPenaltyLoss(nn.Module):
    """
    Loss that penalizes repetitive token predictions.

    Helps prevent degenerate outputs like 'formformformform'.
    Works by penalizing when consecutive predictions are identical.
    """

    def __init__(self, penalty_weight: float = 0.1):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute repetition penalty.

        Args:
            logits: [B, seq_len, vocab_size] model predictions
            labels: [B, seq_len] target labels (used for masking)

        Returns:
            Scalar penalty loss
        """
        # Get predicted tokens
        preds = logits.argmax(dim=-1)  # [B, seq_len]

        # Create mask for valid positions (not padding, not -100)
        valid_mask = labels != -100  # [B, seq_len]

        # Check for consecutive repetitions
        # Compare pred[t] with pred[t-1]
        same_as_prev = (preds[:, 1:] == preds[:, :-1]).float()  # [B, seq_len-1]

        # Only penalize valid positions
        valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]  # [B, seq_len-1]

        # Compute penalty (mean of repetitions in valid positions)
        if valid_pairs.sum() > 0:
            penalty = (same_as_prev * valid_pairs.float()).sum() / valid_pairs.sum()
        else:
            penalty = torch.tensor(0.0, device=logits.device)

        return self.penalty_weight * penalty


class HiddenStateAlignmentLoss(nn.Module):
    """
    Per-token hidden-state alignment loss (vocab-agnostic distillation).

    Instead of pooling teacher/student representations to a single vector,
    this loss aligns hidden states token-by-token on a shared prefix of
    length min(128, T_teacher, T_student).  A learned linear projection maps
    the teacher dimension to the student dimension so that arbitrary teacher
    models can be used without vocabulary alignment.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # Project teacher -> student space for per-token comparison
        self.teacher_proj = nn.Linear(teacher_dim, student_dim)
        # Small init to avoid destabilising early training
        nn.init.xavier_uniform_(self.teacher_proj.weight, gain=0.1)
        nn.init.zeros_(self.teacher_proj.bias)

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token L2 alignment loss on a shared prefix.

        Args:
            student_hidden: Student hidden states [B, T_s, D_student]
            teacher_hidden: Teacher hidden states [B, T_t, D_teacher]
        """
        # Align on a shared prefix of at most 128 tokens
        L = min(128, teacher_hidden.size(1), student_hidden.size(1))
        teacher_slice = teacher_hidden[:, :L, :]  # [B, L, D_teacher]
        student_slice = student_hidden[:, :L, :]  # [B, L, D_student]

        teacher_proj = self.teacher_proj(teacher_slice)  # [B, L, D_student]

        return F.mse_loss(student_slice, teacher_proj)


class VisionGroundingLoss(nn.Module):
    """
    Loss that encourages the model to actually use visual information.

    This addresses the core issue where models generate fluent text but ignore
    the actual image content. The loss works by:
    1. Ensuring vision-conditioned hidden states are different from text-only states
    2. Ensuring different images produce different hidden state patterns

    Without this, models can learn to generate generic descriptions that "sound right"
    but don't actually describe the image (hallucination).
    """

    def __init__(self, margin: float = 0.1, weight: float = 0.1):
        super().__init__()
        self.margin = margin  # Minimum difference required between vision/no-vision
        self.weight = weight

    def forward(
        self,
        vision_hidden: torch.Tensor,  # Hidden states WITH vision [B, seq, dim]
        text_hidden: torch.Tensor,    # Hidden states WITHOUT vision (or from text-only tokens)
        different_images: bool = True,  # Whether batch has different images
    ) -> torch.Tensor:
        """
        Compute vision grounding loss.

        Args:
            vision_hidden: Hidden states from vision-conditioned forward [B, seq, dim]
            text_hidden: Hidden states from text tokens (after vision) [B, seq, dim]
            different_images: If True, penalize if different images give similar outputs

        Returns:
            Vision grounding loss (lower = better grounding)
        """
        # Pool to get single representation per sample
        vision_pooled = vision_hidden.mean(dim=1)  # [B, dim]
        text_pooled = text_hidden.mean(dim=1)      # [B, dim]

        # Normalize for cosine similarity
        vision_norm = F.normalize(vision_pooled, dim=-1)
        text_norm = F.normalize(text_pooled, dim=-1)

        # Loss 1: Vision should contribute meaningfully (not be ignored)
        # If vision_hidden ≈ text_hidden, vision is being ignored
        # We want cosine_sim(vision, text) to be less than 1 - margin
        same_sample_sim = (vision_norm * text_norm).sum(dim=-1)  # [B]
        # We DON'T penalize this directly - they should be related
        # Instead, we focus on Loss 2

        # Loss 2: Different images should produce different hidden states
        # This is the key - if all images produce similar outputs, model is hallucinating
        if different_images and vision_pooled.size(0) > 1:
            # Compute pairwise cosine similarities between vision features
            # sim[i,j] = cos_sim(vision_i, vision_j)
            sim_matrix = torch.mm(vision_norm, vision_norm.t())  # [B, B]

            # Mask out diagonal (self-similarity = 1)
            batch_size = sim_matrix.size(0)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

            # Off-diagonal similarities (between different images)
            off_diag_sims = sim_matrix[mask]  # [B*(B-1)]

            # Penalize if off-diagonal sims are too high
            # We want different images to have similarity < (1 - margin)
            # loss = ReLU(sim - (1 - margin))
            threshold = 1.0 - self.margin
            grounding_loss = F.relu(off_diag_sims - threshold).mean()

            return self.weight * grounding_loss

        return torch.tensor(0.0, device=vision_hidden.device)


class Stage2Trainer:
    """Trainer for Stage 2: Multimodal Instruction Tuning."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        teacher_model: Optional[nn.Module] = None,
        teacher_tokenizer: Any = None,
        teacher_processor: Any = None,
        distillation_config: Optional[Dict[str, Any]] = None,
        hub_repo_id: Optional[str] = None,
        vision_backbone: str = "repvit",
        language_backbone: str = "tinyllm",
        use_ema_teacher: bool = False,
        use_multi_scale: bool = False,
        use_layer_wise_lr: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_processor = teacher_processor
        self.hub_repo_id = hub_repo_id
        self.vision_backbone = vision_backbone
        self.language_backbone = language_backbone
        self.use_ema_teacher = use_ema_teacher
        self.use_multi_scale = use_multi_scale
        self.use_layer_wise_lr = use_layer_wise_lr
        
        # Label smoothing to reduce overconfident predictions and repetition
        self.label_smoothing = 0.1

        # Repetition penalty to discourage repeated token predictions
        self.repetition_penalty_loss = RepetitionPenaltyLoss(penalty_weight=0.15)

        # Vision grounding loss to ensure model actually uses visual information
        # This is CRITICAL - without it, model can learn to generate fluent but
        # ungrounded descriptions that don't match the actual image.
        # Weight 0.5: strong enough to differentiate images, but not so large
        # that it dominates over the SFT loss (0.85) and causes instability.
        self.vision_grounding_loss = VisionGroundingLoss(margin=0.2, weight=0.5)

        # EMA teacher initialization (will be done after model setup)
        self.ema_teacher = None

        # Early stopping: monitor perplexity (lower is better)
        self.early_stopping = EarlyStopping(
            patience=5,
            mode='min',
            min_delta=0.05,
            verbose=True
        )
        self.best_checkpoint_tracker = BestCheckpointTracker(
            save_dir=config.output_dir,
            metric_name='val_perplexity',
            mode='min'
        )
        self.metrics_history = []

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Set seed
        set_seed(config.seed, self.rank)

        # Unwrap model if it was previously wrapped with DDP
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # CRITICAL: Validate and FIX embedding size to match tokenizer
        # CRITICAL: Sync tokenizer, embeddings, and special token IDs
        if tokenizer is not None and hasattr(model, 'sync_tokenizer_and_embeddings'):
            try:
                model.sync_tokenizer_and_embeddings(
                    tokenizer,
                    add_special_tokens=True,
                    force_resize=False,
                    logger=logger,
                )
            except TypeError:
                model.sync_tokenizer_and_embeddings(
                    tokenizer,
                    add_special_tokens=True,
                    force_resize=False,
                )

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

        # CRITICAL: Log exact trainable groups and validate LLM unfreezing state
        self._log_and_validate_trainable_parameters()

        # Initialize distillation flags early (before any checks)
        self.use_distillation = False
        self.use_hidden_state_distillation = False
        
        # Teacher model (frozen)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # CRITICAL: Check if teacher and student have compatible vocabularies
            # If not, disable distillation (vocab-agnostic methods needed instead)
            teacher_vocab_size = None
            try:
                if hasattr(self.teacher_model, 'language_model'):
                    if hasattr(self.teacher_model.language_model, 'config'):
                        teacher_vocab_size = self.teacher_model.language_model.config.vocab_size
                elif hasattr(self.teacher_model, 'config'):
                    teacher_vocab_size = self.teacher_model.config.vocab_size
            except:
                pass
            
            student_vocab_size = len(tokenizer) if tokenizer else None
            
            # Check if we need vocab-agnostic distillation (update the flag set earlier)
            if teacher_vocab_size and student_vocab_size:
                if teacher_vocab_size != student_vocab_size:
                    logger.info("="*60)
                    logger.info("✓ USING HIDDEN STATE DISTILLATION (Vocab-Agnostic)")
                    logger.info(f"   Teacher vocab size: {teacher_vocab_size}")
                    logger.info(f"   Student vocab size: {student_vocab_size}")
                    logger.info("")
                    logger.info("   Hidden state distillation is ACTIVE and working correctly.")
                    logger.info("   This distills internal representations (vocab-agnostic),")
                    logger.info("   which is more effective than logit-based distillation.")
                    logger.info("="*60)
                    self.use_distillation = True
                    self.use_hidden_state_distillation = True

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer and scheduler with optional layer-wise learning rates
        self.optimizer = get_optimizer(self.model, config, use_layer_wise_lr=self.use_layer_wise_lr)
        if self.use_layer_wise_lr and is_main_process():
            logger.info("✓ Layer-wise learning rates enabled:")
            logger.info("  - Vision encoder: 1e-5")
            logger.info("  - Projection/Fusion: 2e-4")
            logger.info("  - Language model: 5e-5")
        
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)
        
        # Initialize multi-scale vision extractor FIRST (before EMA teacher)
        # This is critical because multi-scale changes the model architecture,
        # and EMA teacher needs to copy the final architecture
        if self.use_multi_scale:
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, 'vision_encoder'):
                self.multi_scale_extractor = MultiScaleVisionExtractor(model_ref.vision_encoder)
                # Count fusion parameters
                fusion_params = sum(p.numel() for p in self.multi_scale_extractor.fusion.parameters())
                if is_main_process():
                    logger.info(f"✓ Multi-scale vision features enabled:")
                    logger.info(f"  - Extracting from layers [3, 6, 9, last]")
                    logger.info(f"  - Fusion parameters: {fusion_params:,} (~1K)")
            else:
                logger.warning("⚠️ Multi-scale features requested but model has no vision_encoder")
                self.use_multi_scale = False
        
        # Initialize EMA teacher AFTER multi-scale (so it copies the correct architecture)
        # IMPORTANT: EMA warmup is increased to 500 steps to let the model
        # stabilize before self-distillation kicks in. This prevents early
        # garbage outputs from being amplified through self-teaching.
        if self.use_ema_teacher:
            # Use the underlying model (not the DDP wrapper)
            model_for_ema = self.model.module if hasattr(self.model, 'module') else self.model
            self.ema_teacher = EMATeacher(model_for_ema, decay=0.9995, update_after_step=500)
            if is_main_process():
                logger.info("✓ EMA teacher enabled (decay=0.9995, warmup=500 steps)")
                logger.info("  - Self-distillation from exponential moving average")
                logger.info("  - Warmup prevents early garbage from being amplified")

        # Distillation flags combine external teacher or EMA
        self.use_distillation = self.use_distillation or self.use_ema_teacher
        self.use_hidden_state_distillation = self.use_distillation

        # Distillation
        distillation_config = distillation_config or {}
        self.ema_hidden_state_loss = None
        self.external_hidden_state_loss = None
        
        if self.use_distillation:
            if self.use_hidden_state_distillation or self.use_ema_teacher:
                # Get student hidden dimension
                student_dim = 576  # default
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model_ref.language_model, 'config'):
                    student_dim = model_ref.language_model.config.hidden_size
                
                # EMA teacher loss (same dimension as student)
                if self.use_ema_teacher:
                    self.ema_hidden_state_loss = HiddenStateAlignmentLoss(student_dim, student_dim).to(self.device)
                    logger.info(f"✓ EMA distillation loss initialized (student_dim={student_dim}, teacher_dim={student_dim})")
                
                # External teacher loss (potentially different dimension)
                if self.teacher_model is not None:
                    external_teacher_dim = 4096  # default
                    try:
                        # Try various config paths for hidden dimension
                        if hasattr(self.teacher_model, 'language_model'):
                            if hasattr(self.teacher_model.language_model, 'config'):
                                external_teacher_dim = self.teacher_model.language_model.config.hidden_size
                        elif hasattr(self.teacher_model, 'config'):
                            # Qwen2-VL: config.hidden_size
                            if hasattr(self.teacher_model.config, 'hidden_size'):
                                external_teacher_dim = self.teacher_model.config.hidden_size
                            # Idefics3/SmolVLM: config.text_config.hidden_size
                            elif hasattr(self.teacher_model.config, 'text_config'):
                                if hasattr(self.teacher_model.config.text_config, 'hidden_size'):
                                    external_teacher_dim = self.teacher_model.config.text_config.hidden_size
                    except Exception as e:
                        logger.warning(f"Could not auto-detect external teacher hidden_size: {e}, using default {external_teacher_dim}")
                    
                    self.external_hidden_state_loss = HiddenStateAlignmentLoss(student_dim, external_teacher_dim).to(self.device)
                    logger.info(f"✓ External teacher distillation loss initialized (student_dim={student_dim}, teacher_dim={external_teacher_dim})")
            else:
                logger.warning("❌ No distillation configured - neither hidden state alignment nor teacher logits available")

        # Loss weights - IMPORTANT: Keep distillation weight LOW to prevent
        # destabilizing the language model. Vision/distillation gradients can
        # overwhelm the LM causing repetitive/degenerate outputs.
        # Start with: L = 0.85 * L_sft + 0.1 * L_distill + 0.05 * L_rep
        self.sft_weight = distillation_config.get('sft_weight', 0.85)
        self.distill_weight = distillation_config.get('distill_weight', 0.1)

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger with visualizations (main process)...")
            try:
                wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                self.wandb_logger = EnhancedWandbLogger(
                    project=wandb_project,
                    name="stage2_instruction",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info(f"Enhanced W&B logger initialized with project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B logger: {e}")
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

        # Metrics
        self.metric_tracker = MetricTracker()
        self.global_step = 0

        # Stage 2 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage2Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage2Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage2Visualizer: {e}")

        # Track data for visualizations
        self.last_logits = None
        self.last_labels = None

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

        if is_main_process():
            print_trainable_parameters(self.model)

    def _log_and_validate_trainable_parameters(self):
        """Log exact trainable parameter groups and validate expected LLM trainability."""
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        trainable_by_group = {}
        trainable_names = []
        total_params = 0
        trainable_params = 0
        llm_total = 0
        llm_trainable = 0

        for name, param in model_ref.named_parameters():
            count = param.numel()
            total_params += count

            if name.startswith('language_model'):
                llm_total += count
                if param.requires_grad:
                    llm_trainable += count

            if param.requires_grad:
                trainable_params += count
                trainable_names.append(name)

                parts = name.split('.')
                group = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
                trainable_by_group[group] = trainable_by_group.get(group, 0) + count

        if is_main_process():
            logger.info("=" * 60)
            logger.info("Stage 2 trainable parameter groups")
            logger.info("=" * 60)
            logger.info(f"Total params: {total_params:,}")
            logger.info(f"Trainable params: {trainable_params:,}")
            logger.info(f"LLM trainable params: {llm_trainable:,}/{llm_total:,}")

            if trainable_by_group:
                for group, count in sorted(trainable_by_group.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  - {group}: {count:,}")
            else:
                logger.warning("No trainable parameter groups found")

            if trainable_names:
                logger.info("Trainable parameter names (first 50):")
                for name in trainable_names[:50]:
                    logger.info(f"    • {name}")
                if len(trainable_names) > 50:
                    logger.info(f"    ... and {len(trainable_names) - 50} more")

            logger.info("=" * 60)

        # Hard guard: if config expects last LM layer unfreezing but none are trainable, fail fast.
        model_cfg = getattr(model_ref, 'config', None)
        expect_unfrozen_llm = bool(getattr(model_cfg, 'freeze_language_base', False)) and bool(
            getattr(model_cfg, 'unfreeze_last_layer', False)
        )

        if expect_unfrozen_llm and llm_total > 0 and llm_trainable == 0:
            msg = (
                "LLM has 0 trainable parameters in Stage 2 while config expects last-layer unfreezing "
                "(freeze_language_base=True, unfreeze_last_layer=True). "
                "Set EMBERVLM_ALLOW_ZERO_LLM_TRAINABLE=1 to bypass this guard if intentional."
            )
            if os.getenv('EMBERVLM_ALLOW_ZERO_LLM_TRAINABLE', '0') == '1':
                logger.warning(f"⚠️ {msg}")
            else:
                raise RuntimeError(msg)

    def compute_teacher_outputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        raw_texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get outputs from teacher model (supports both external and EMA teacher).
        
        Returns combined outputs from BOTH teachers when available:
        - external_hidden_states: from external teacher (vocab-agnostic)
        - ema_hidden_states: from EMA self-distillation teacher
        """
        if not self.use_distillation:
            return {}

        result = {}

        # EMA teacher path (uses student tokenization)
        if self.use_ema_teacher and self.ema_teacher is not None:
            try:
                with torch.no_grad():
                    ema_outputs = self.ema_teacher.get_teacher_outputs(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    if ema_outputs.get('hidden_states') is not None:
                        result['ema_hidden_states'] = ema_outputs['hidden_states']
                        # Debug: Check if hidden states are valid
                        if not hasattr(self, '_ema_debug_logged'):
                            logger.info(f"✓ EMA teacher hidden states extracted: {len(ema_outputs['hidden_states'])} layers")
                            logger.info(f"   Last layer shape: {ema_outputs['hidden_states'][-1].shape}")
                            self._ema_debug_logged = True
                    else:
                        if not hasattr(self, '_ema_no_hidden_logged'):
                            logger.warning("⚠️ EMA teacher outputs don't contain hidden_states")
                            self._ema_no_hidden_logged = True
            except Exception as e:
                if not hasattr(self, '_ema_teacher_failed'):
                    logger.error(f"❌ EMA teacher failed: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"   Full traceback: {traceback.format_exc()}")
                    self._ema_teacher_failed = True

        # External teacher path (vocab-agnostic via re-tokenizing raw text)
        if self.teacher_model is None or self.teacher_tokenizer is None:
            return result  # Return EMA results if available

        if raw_texts is None:
            if not hasattr(self, '_missing_raw_text_logged'):
                logger.warning("⚠️ raw_text missing in batch; skipping external teacher distillation")
                self._missing_raw_text_logged = True
            return result  # Return EMA results if available

        try:
            # Detect Qwen2-VL by checking model config
            is_qwen2vl = False
            if hasattr(self.teacher_model, 'config'):
                model_type = getattr(self.teacher_model.config, 'model_type', '')
                if 'qwen2_vl' in model_type.lower():
                    is_qwen2vl = True
            
            if is_qwen2vl and self.teacher_processor is not None:
                # CRITICAL: Qwen2-VL requires proper image preprocessing through its processor
                # Raw pixel_values cannot be passed directly - they need special formatting
                from PIL import Image
                
                # Convert tensor pixel_values back to PIL images for processor
                # pixel_values shape: (B, C, H, W) with values normalized to [-1, 1] or [0, 1]
                batch_size = pixel_values.shape[0]
                pil_images = []
                for i in range(batch_size):
                    img_tensor = pixel_values[i].cpu()  # (C, H, W)
                    # Denormalize: assume ImageNet normalization was used
                    # Standard denorm: img = img * std + mean
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    # Convert to PIL
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                    pil_images.append(Image.fromarray(img_np))
                
                # Build messages for Qwen2-VL processor
                messages_batch = []
                for i, text in enumerate(raw_texts):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_images[i]},
                                {"type": "text", "text": text[:512]},  # Truncate text
                            ],
                        }
                    ]
                    messages_batch.append(messages)
                
                # Process each sample through Qwen2-VL processor
                # Process one at a time to handle variable image sizes
                all_hidden_states = []
                for i, messages in enumerate(messages_batch):
                    try:
                        text = self.teacher_processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = self.teacher_processor(
                            text=[text],
                            images=[pil_images[i]],
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
                        
                        with torch.no_grad():
                            outputs = self.teacher_model(
                                output_hidden_states=True,
                                **inputs,
                            )
                        
                        # Extract last hidden state and pool to fixed size
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            hs = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                            all_hidden_states.append(hs)
                    except Exception as e:
                        if not hasattr(self, '_qwen_sample_failed'):
                            logger.warning(f"⚠️ Qwen2-VL sample {i} failed: {e}")
                            self._qwen_sample_failed = True
                        continue
                
                if len(all_hidden_states) > 0:
                    # Average pool each hidden state to fixed length, then stack
                    # Student hidden: (B, student_seq_len, dim)
                    # We need to align teacher seq_len with student seq_len
                    # Use adaptive pooling to match student sequence length
                    teacher_dim = all_hidden_states[0].shape[-1]
                    
                    # Get target length from student (will be determined in train_step)
                    # For now, use min of all teacher seq lengths
                    min_seq_len = min(h.shape[1] for h in all_hidden_states)
                    
                    pooled_states = []
                    for hs in all_hidden_states:
                        # Adaptive average pool to target length
                        # Shape: (1, seq_len, dim) -> (1, min_seq_len, dim)
                        hs_permuted = hs.permute(0, 2, 1)  # (1, dim, seq_len)
                        pooled = F.adaptive_avg_pool1d(hs_permuted, min_seq_len)
                        pooled = pooled.permute(0, 2, 1)  # (1, min_seq_len, dim)
                        pooled_states.append(pooled)
                    
                    # Stack batch: (B, min_seq_len, dim)
                    teacher_hidden = torch.cat(pooled_states, dim=0)
                    result['external_hidden_states'] = teacher_hidden
                    
                    if not hasattr(self, '_qwen2vl_success_logged'):
                        logger.info(f"✓ Qwen2-VL external teacher: hidden_states shape={teacher_hidden.shape}")
                        self._qwen2vl_success_logged = True
            else:
                # Non-Qwen2-VL teacher or no processor: use simple tokenization approach
                teacher_inputs = self.teacher_tokenizer(
                    raw_texts,
                    padding=True,
                    truncation=True,
                    max_length=min(getattr(self.teacher_tokenizer, 'model_max_length', 512), 512),
                    return_tensors='pt',
                )
                teacher_inputs = {k: v.to(self.device) for k, v in teacher_inputs.items() if torch.is_tensor(v)}

                # Prepare pixel_values for teacher model
                teacher_pixel_values = pixel_values
                if pixel_values.ndim == 4:
                    teacher_pixel_values = pixel_values.unsqueeze(1)
                teacher_inputs['pixel_values'] = teacher_pixel_values

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        output_hidden_states=True,
                        **teacher_inputs,
                    )

                hidden_states = None
                if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
                    hidden_states = teacher_outputs.hidden_states[-1]
                elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
                    hs = teacher_outputs['hidden_states']
                    if isinstance(hs, (list, tuple)) and len(hs) > 0:
                        hidden_states = hs[-1]

                if hidden_states is not None:
                    result['external_hidden_states'] = hidden_states
            
            return result
        except Exception as e:
            if not hasattr(self, '_teacher_failed'):
                logger.error(f"❌ External teacher failed: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"   Full traceback: {traceback.format_exc()}")
                logger.error("   External teacher distillation will be skipped for the rest of this run")
                self._teacher_failed = True
            return result  # Return EMA results if available

    def _align_labels_for_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Align labels to logits length, inserting -100 for visual tokens and trimming if needed."""
        num_visual = logits.size(1) - labels.size(1)
        if num_visual > 0:
            # Prepend -100 for visual tokens
            batch_size = labels.size(0)
            adjusted = torch.full(
                (batch_size, logits.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            adjusted[:, num_visual:] = labels
        elif num_visual < 0:
            # Labels longer than logits, truncate to logits length
            adjusted = labels[:, :logits.size(1)]
        else:
            adjusted = labels

        # Final safety trim to the min common length
        if adjusted.size(1) != logits.size(1):
            min_len = min(adjusted.size(1), logits.size(1))
            adjusted = adjusted[:, :min_len]
        return adjusted

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        raw_texts = batch.get('raw_text') if isinstance(batch, dict) else None

        # CRITICAL: Validate input_ids and labels are within embedding bounds
        # This prevents cryptic CUDA index out of bounds errors
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # Get vocab size with multiple fallback methods
        vocab_size = None
        try:
            if hasattr(model_ref, 'language_model'):
                lm = model_ref.language_model
                if hasattr(lm, 'get_input_embeddings') and lm.get_input_embeddings() is not None:
                    vocab_size = lm.get_input_embeddings().weight.shape[0]
                elif hasattr(lm, 'model') and hasattr(lm.model, 'get_input_embeddings'):
                    emb = lm.model.get_input_embeddings()
                    if emb is not None:
                        vocab_size = emb.weight.shape[0]
                elif hasattr(lm, 'config') and hasattr(lm.config, 'vocab_size'):
                    vocab_size = lm.config.vocab_size
        except Exception as e:
            logger.warning(f"Could not determine vocab_size: {e}")

        # ALWAYS clamp to a safe range even if we couldn't detect exact vocab_size
        if vocab_size is None:
            vocab_size = 49157  # Known embedding size from logs
            logger.warning(f"Using fallback vocab_size={vocab_size}")

        # Validate and clamp input_ids BEFORE any forward pass
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()
        needs_clamping = max_token_id >= vocab_size or min_token_id < 0
        
        if needs_clamping:
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

        # Forward pass
        with get_autocast_context(self.config):
            # Student forward
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=self.use_distillation,
            )

            # SFT loss with label smoothing to reduce overconfidence and repetition
            if self.label_smoothing > 0 and 'logits' in outputs:
                logits = outputs['logits']

                # Align labels to logits length once and reuse (handles visual tokens)
                adjusted_labels = self._align_labels_for_logits(logits, labels)

                # Shift for causal LM (predict next token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = adjusted_labels[..., 1:].contiguous()

                # SAFETY CHECK: Ensure dimensions match before cross_entropy
                if shift_logits.size(0) != shift_labels.size(0) or shift_logits.size(1) != shift_labels.size(1):
                    if not hasattr(self, '_dim_mismatch_warned'):
                        logger.warning(f"⚠️ Dimension mismatch: logits {shift_logits.shape} vs labels {shift_labels.shape}")
                        self._dim_mismatch_warned = True
                    min_seq_len = min(shift_logits.size(1), shift_labels.size(1))
                    shift_logits = shift_logits[:, :min_seq_len, :].contiguous()
                    shift_labels = shift_labels[:, :min_seq_len].contiguous()

                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=-100,
                    label_smoothing=self.label_smoothing,
                    reduction='mean'
                )
                sft_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                # Fallback to model's built-in loss
                sft_loss = outputs['loss']

            # Distillation loss (supports both EMA and external teacher simultaneously)
            ema_distill_loss = torch.tensor(0.0, device=self.device)
            external_distill_loss = torch.tensor(0.0, device=self.device)
            vision_grounding_loss = torch.tensor(0.0, device=self.device)

            if self.use_distillation:
                teacher_outputs = self.compute_teacher_outputs(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    raw_texts=raw_texts,
                )

                if self.use_hidden_state_distillation and outputs.get('hidden_states') is not None:
                    student_hidden = outputs['hidden_states'][-1]  # [B, seq_len, student_dim]
                    
                    # Debug logging (only once)
                    if not hasattr(self, '_distill_debug_logged'):
                        logger.info(f"🔍 Distillation Debug:")
                        logger.info(f"   Student hidden shape: {student_hidden.shape}")
                        logger.info(f"   Teacher outputs keys: {list(teacher_outputs.keys())}")
                        logger.info(f"   use_distillation: {self.use_distillation}")
                        logger.info(f"   use_hidden_state_distillation: {self.use_hidden_state_distillation}")
                        self._distill_debug_logged = True
                    
                    # EMA teacher distillation
                    if 'ema_hidden_states' in teacher_outputs and self.ema_hidden_state_loss is not None:
                        try:
                            ema_teacher_hidden = teacher_outputs['ema_hidden_states'][-1]
                            ema_distill_loss = self.ema_hidden_state_loss(student_hidden, ema_teacher_hidden)
                            if not hasattr(self, '_ema_loss_computed_logged'):
                                logger.info(f"✓ EMA distillation loss computed: {ema_distill_loss.item():.4f}")
                                self._ema_loss_computed_logged = True
                        except Exception as e:
                            if not hasattr(self, '_ema_loss_failed'):
                                logger.error(f"❌ EMA distillation loss failed: {e}")
                                self._ema_loss_failed = True
                    else:
                        if not hasattr(self, '_ema_not_in_outputs_logged'):
                            if 'ema_hidden_states' not in teacher_outputs:
                                logger.warning("⚠️ 'ema_hidden_states' not in teacher_outputs")
                            self._ema_not_in_outputs_logged = True
                    
                    # External teacher distillation (vocab-agnostic)
                    if 'external_hidden_states' in teacher_outputs and self.external_hidden_state_loss is not None:
                        try:
                            external_teacher_hidden = teacher_outputs['external_hidden_states']
                            # Handle sequence length mismatch between student and teacher
                            # Student: [B, seq_student, dim_student], Teacher: [B, seq_teacher, dim_teacher]
                            student_seq_len = student_hidden.shape[1]
                            teacher_seq_len = external_teacher_hidden.shape[1]
                            
                            if student_seq_len != teacher_seq_len:
                                # Pool both to the minimum sequence length
                                target_len = min(student_seq_len, teacher_seq_len)
                                
                                if student_seq_len != target_len:
                                    # Pool student: (B, seq, dim) -> permute -> pool -> permute
                                    student_permuted = student_hidden.permute(0, 2, 1)  # (B, dim, seq)
                                    student_pooled = F.adaptive_avg_pool1d(student_permuted, target_len)
                                    student_for_loss = student_pooled.permute(0, 2, 1)  # (B, target_len, dim)
                                else:
                                    student_for_loss = student_hidden
                                
                                if teacher_seq_len != target_len:
                                    # Pool teacher: (B, seq, dim) -> permute -> pool -> permute
                                    teacher_permuted = external_teacher_hidden.permute(0, 2, 1)
                                    teacher_pooled = F.adaptive_avg_pool1d(teacher_permuted, target_len)
                                    teacher_for_loss = teacher_pooled.permute(0, 2, 1)
                                else:
                                    teacher_for_loss = external_teacher_hidden
                                
                                if not hasattr(self, '_seq_len_pooling_logged'):
                                    logger.info(f"🔄 Pooled hidden states: student {student_seq_len}->{target_len}, teacher {teacher_seq_len}->{target_len}")
                                    self._seq_len_pooling_logged = True
                            else:
                                student_for_loss = student_hidden
                                teacher_for_loss = external_teacher_hidden
                            
                            external_distill_loss = self.external_hidden_state_loss(student_for_loss, teacher_for_loss)
                            if not hasattr(self, '_external_loss_computed_logged'):
                                logger.info(f"✓ External distillation loss computed: {external_distill_loss.item():.4f}")
                                self._external_loss_computed_logged = True
                        except Exception as e:
                            if not hasattr(self, '_external_loss_failed'):
                                logger.error(f"❌ External distillation loss failed: {e}")
                                import traceback
                                logger.error(f"   Traceback: {traceback.format_exc()}")
                                self._external_loss_failed = True
                    else:
                        if not hasattr(self, '_external_not_in_outputs_logged'):
                            logger.warning("⚠️ 'external_hidden_states' not in teacher_outputs (external teacher may have failed)")
                            self._external_not_in_outputs_logged = True
                    
                    # Combined distillation loss (weighted sum of both)
                    total_distill_loss = ema_distill_loss + external_distill_loss
                    
                    # VISION GROUNDING LOSS: Ensure model actually uses visual information
                    # This is CRITICAL - without it, model can learn to generate fluent but
                    # ungrounded descriptions that don't match the actual image
                    vision_grounding_loss = torch.tensor(0.0, device=self.device)
                    if hasattr(self, 'vision_grounding_loss') and student_hidden is not None:
                        try:
                            # Use first num_visual_tokens as vision-conditioned features
                            # and remaining as text features
                            num_visual = getattr(model_ref.config, 'num_visual_tokens', 8)
                            if student_hidden.size(1) > num_visual:
                                vision_hidden = student_hidden[:, :num_visual, :]
                                text_hidden = student_hidden[:, num_visual:, :]
                                vision_grounding_loss = self.vision_grounding_loss(
                                    vision_hidden, text_hidden, different_images=True
                                )
                                if not hasattr(self, '_grounding_loss_logged'):
                                    logger.info(f"✓ Vision grounding loss: {vision_grounding_loss.item():.4f}")
                                    self._grounding_loss_logged = True
                        except Exception as e:
                            if not hasattr(self, '_grounding_loss_failed'):
                                logger.warning(f"⚠️ Vision grounding loss failed: {e}")
                                self._grounding_loss_failed = True

                    # Combined loss with repetition penalty AND grounding
                    aligned_labels_for_penalty = self._align_labels_for_logits(outputs['logits'], labels)
                    rep_penalty = self.repetition_penalty_loss(outputs['logits'], aligned_labels_for_penalty)

                    # --- Length penalty: discourage over-long hallucinated responses ---
                    # Penalise only the *generated* portion (after visual tokens),
                    # not the prompt itself.
                    max_len = 64
                    num_visual = getattr(model_ref.config, 'num_visual_tokens', 8)
                    total_seq_len = adjusted_labels.size(1)
                    # Count non-ignored label positions as the generated portion
                    gen_lengths = (adjusted_labels != -100).sum(dim=1).float()
                    length_over = torch.clamp(gen_lengths - max_len, min=0.0)
                    lambda_len = 0.01
                    length_penalty = lambda_len * length_over.mean()

                    loss = (self.sft_weight * sft_loss +
                            self.distill_weight * total_distill_loss +
                            rep_penalty +
                            vision_grounding_loss +
                            length_penalty)
                else:
                    aligned_labels_for_penalty = self._align_labels_for_logits(outputs['logits'], labels)
                    rep_penalty = self.repetition_penalty_loss(outputs['logits'], aligned_labels_for_penalty)
                    loss = sft_loss + rep_penalty
                    total_distill_loss = torch.tensor(0.0, device=self.device)
            else:
                aligned_labels_for_penalty = self._align_labels_for_logits(outputs['logits'], labels)
                rep_penalty = self.repetition_penalty_loss(outputs['logits'], aligned_labels_for_penalty)
                loss = sft_loss + rep_penalty
                total_distill_loss = torch.tensor(0.0, device=self.device)

        metrics = {
            'loss': loss.item(),
            'sft_loss': sft_loss.item(),
            'distill_loss': total_distill_loss.item() if isinstance(total_distill_loss, torch.Tensor) else 0.0,
            'ema_distill_loss': ema_distill_loss.item() if isinstance(ema_distill_loss, torch.Tensor) else 0.0,
            'external_distill_loss': external_distill_loss.item() if isinstance(external_distill_loss, torch.Tensor) else 0.0,
            'rep_penalty': rep_penalty.item() if isinstance(rep_penalty, torch.Tensor) else 0.0,
            'vision_grounding_loss': vision_grounding_loss.item() if isinstance(vision_grounding_loss, torch.Tensor) else 0.0,
        }

        # Store for visualization (detach + CPU to avoid GPU memory buildup)
        self.last_logits = outputs['logits'].detach().cpu()
        self.last_labels = labels.detach().cpu()

        return loss, metrics

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
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

            # Gradient accumulation
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
                
                # Update EMA teacher after optimizer step
                if self.use_ema_teacher and self.ema_teacher is not None:
                    model_for_ema = self.model.module if hasattr(self.model, 'module') else self.model
                    self.ema_teacher.update(model_for_ema)

                # Add lr to metrics
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
                                    stage_name="stage2",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Stage 2 specific visualizations every 500 steps
                            if self.global_step % 500 == 0:
                                logger.info(f"[Stage2] Step {self.global_step}: Attempting visualizations...")
                                logger.info(f"  stage_visualizer: {self.stage_visualizer is not None}")
                                logger.info(f"  last_logits: {self.last_logits is not None}")
                                logger.info(f"  last_labels: {self.last_labels is not None}")

                                if self.stage_visualizer is not None:
                                    if self.last_logits is not None and self.last_labels is not None:
                                        try:
                                            logger.info(f"  Generating token probability distribution...")
                                            _, prob_img = self.stage_visualizer.plot_token_probability_distribution(
                                                self.last_logits,
                                                self.last_labels,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage2/token_probabilities", prob_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged token probability distribution to W&B")
                                        except Exception as e:
                                            logger.error(f"  ✗ Failed to generate Stage 2 visualizations: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"  Skipping visualizations: logits/labels not available")
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
                                        gradients, self.global_step, "stage2"
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

                    progress_bar.set_postfix({
                        'loss': f"{avg_metrics['loss']:.4f}",
                    })

                    self.metric_tracker.reset()

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    barrier()  # Sync before checkpoint
                    self.save_checkpoint()
                    barrier()  # Sync after checkpoint

        # Sync at end of epoch before evaluation
        barrier()

    @torch.no_grad()
    def evaluate(self, eval_step: Optional[int] = None):
        """Evaluate on validation set."""
        if eval_step is None:
            eval_step = self.global_step

        self.model.eval()
        eval_metrics = MetricTracker()

        # Track for visualization (sample 3 random indices)
        num_batches = len(self.val_dataloader)
        vis_indices = set(np.random.choice(num_batches, min(3, num_batches), replace=False)) if is_main_process() else set()

        for batch_idx, batch in enumerate(tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        )):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with get_autocast_context(self.config):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=True,  # For visualization
                )

                loss = outputs['loss']

                # Compute accuracy (token-level)
                logits = outputs['logits']
                predictions = logits.argmax(dim=-1)

                # Align predictions with labels length (handle visual token offset)
                if predictions.size(1) != labels.size(1):
                    min_len = min(predictions.size(1), labels.size(1))
                    # Truncate both to same length to avoid mismatch
                    predictions = predictions[:, :min_len]
                    labels_eval = labels[:, :min_len]
                else:
                    labels_eval = labels

                mask = labels_eval != -100
                correct = ((predictions == labels_eval) & mask).sum()
                total = mask.sum()
                accuracy = correct.float() / total.float() if total > 0 else torch.tensor(0.0)

                # Visualize attention for random samples
                if batch_idx in vis_indices and is_main_process():
                    try:
                        # Get attention weights if available
                        if 'attentions' in outputs or hasattr(outputs, 'attentions'):
                            attentions = outputs.get('attentions') or outputs.attentions
                            if attentions is not None and len(attentions) > 0:
                                # Use last layer attention
                                attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]

                                # Decode tokens for caption
                                text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:20].tolist())

                                self.wandb_logger.log_attention_visualization(
                                    image=pixel_values[0],
                                    attention_map=attn,
                                    text_tokens=text_tokens,
                                    step=eval_step,
                                    stage_name="stage2",
                                )
                    except Exception as e:
                        logger.debug(f"Failed to visualize attention: {e}")

            eval_metrics.update({
                'loss': loss.item(),
                'accuracy': accuracy.item(),
            })

        avg_metrics = eval_metrics.get_average()
        
        # Calculate perplexity from average loss
        perplexity = torch.exp(torch.tensor(avg_metrics['loss'])).item()
        avg_metrics['perplexity'] = perplexity
        
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            if self.wandb_logger is not None:
                self.wandb_logger.log(avg_metrics, step=eval_step)
            logger.info(f"Validation: {avg_metrics}")

        self.model.train()
        return avg_metrics

    def save_checkpoint(self, custom_path: Optional[str] = None):
        """Save checkpoint.

        Args:
            custom_path: Optional custom path for checkpoint. If None, uses checkpoint-{step}.
        """
        if custom_path:
            output_dir = Path(custom_path)
        else:
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
                        # checkpoint-epoch-3 format
                        num = int(parts[-1])
                    else:
                        # checkpoint-37 format
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
                    # If none of the above, skip it
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {oldest}: {e}")

    def train(self, num_epochs: int):
        """Run full training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            for epoch in range(num_epochs):
                # TRIAL MODE: Check if we've already hit max steps
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        break
                
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)
                
                # TRIAL MODE: Check again after epoch
                if hasattr(self.config, 'num_training_steps') and self.config.num_training_steps is not None:
                    if self.global_step >= self.config.num_training_steps:
                        logger.info(f"✅ Reached max training steps ({self.config.num_training_steps}), stopping training")
                        break

                val_metrics = {}
                improved = False
                if self.val_dataloader is not None:
                    val_metrics = self.evaluate()
                    
                    # Track metrics history
                    self.metrics_history.append(val_metrics)
                    
                    # Monitor perplexity (lower is better)
                    monitor_metric = val_metrics.get('val_perplexity', float('inf'))
                    
                    # Check early stopping
                    improved = self.early_stopping(monitor_metric, epoch)
                    
                    if improved:
                        # Save best checkpoint to epoch-specific path
                        checkpoint_path = str(Path(self.config.output_dir) / f'checkpoint-epoch-{epoch+1}')
                        self.save_checkpoint(custom_path=checkpoint_path)
                        self.best_checkpoint_tracker.update(monitor_metric, checkpoint_path)
                    
                    # Check if we should stop
                    if self.early_stopping.early_stop:
                        logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                        break

                # Save regular checkpoint if not best
                if not improved:
                    self.save_checkpoint()
                
                # Push to HuggingFace Hub after each epoch
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'instruction_loss': self.metric_tracker.get_average().get('instruction_loss', 0.0),
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=epoch + 1,
                            stage="stage2",
                            metrics=metrics,
                            vision_backbone=self.vision_backbone,
                            language_backbone=self.language_backbone,
                            carbon_emissions=carbon_emissions,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push to HuggingFace Hub: {e}")
                
                barrier()
            
            # Log training summary
            if is_main_process():
                log_stage_summary("Stage 2: Instruction Tuning", self.metrics_history)
                logger.info(f"✅ Stage 2 training completed: {len(self.metrics_history)}/{num_epochs} epochs")
                logger.info(f"   Best checkpoint: {self.best_checkpoint_tracker.load_best_checkpoint()}")

                # Generate end-of-training visualizations
                logger.info("📊 Generating Stage 2 end-of-training visualizations...")
                try:
                    # Base visualizations (loss decomposition + convergence)
                    if self.wandb_logger is not None:
                        self.wandb_logger.generate_final_visualizations("stage2", self.global_step)

                    # Stage-specific: token probability distribution
                    if self.stage_visualizer is not None:
                        if self.last_logits is not None and self.last_labels is not None:
                            try:
                                _, prob_img = self.stage_visualizer.plot_token_probability_distribution(
                                    self.last_logits, self.last_labels, self.global_step
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage2/token_probabilities_final", prob_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final token probability distribution")
                            except Exception as e:
                                logger.warning(f"  Failed to generate token probabilities: {e}")
                        else:
                            logger.info("  Skipping stage-specific viz: no cached logits/labels")
                    logger.info("📊 Stage 2 visualizations complete")

                    # Save metrics JSON for cross-stage collation
                    try:
                        step_metrics = {}
                        if self.wandb_logger is not None and hasattr(self.wandb_logger, 'metrics_history'):
                            step_metrics = {k: v for k, v in self.wandb_logger.metrics_history.items()
                                           if isinstance(v, list) and v}
                        save_stage_metrics_json(
                            stage_name='stage2',
                            output_dir=self.config.output_dir,
                            metrics_history=self.metrics_history,
                            step_metrics=step_metrics,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save stage2 metrics JSON: {e}")
                except Exception as e:
                    logger.warning(f"Failed to generate Stage 2 end-of-training visualizations: {e}")

        except Exception as e:
            # Log error to WandB before crashing
            error_msg = f"Training failed at epoch {epoch}, step {self.global_step}: {str(e)}"
            logger.error(error_msg)

            if is_main_process() and self.wandb_logger is not None:
                self.wandb_logger.log({
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'failed_at_step': self.global_step,
                }, step=self.global_step)

            raise  # Re-raise to preserve stack trace

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


def run_stage2_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    num_epochs: int = 5,
    teacher_model: Optional[nn.Module] = None,
    teacher_tokenizer: Any = None,
    teacher_processor: Any = None,
    hub_repo_id: Optional[str] = None,
    vision_backbone: str = "repvit",
    language_backbone: str = "tinyllm",
    use_ema_teacher: bool = False,
    use_multi_scale: bool = False,
    use_layer_wise_lr: bool = False,
):
    """
    Run Stage 2 training with HuggingFace Hub push support and optional improvements.
    
    Args:
        model: EmberVLM model
        config: Training configuration
        data_dir: Path to training data
        tokenizer: Tokenizer for text processing
        num_epochs: Number of training epochs
        teacher_model: Optional external teacher model for distillation
        teacher_tokenizer: Optional teacher tokenizer for vocab-agnostic distillation
        teacher_processor: Optional teacher processor for image preprocessing (required for Qwen2-VL)
        hub_repo_id: Optional HuggingFace Hub repository ID
        vision_backbone: Vision backbone name
        language_backbone: Language backbone name
        use_ema_teacher: Enable EMA teacher for self-distillation (no external teacher needed)
        use_multi_scale: Enable multi-scale vision feature extraction
        use_layer_wise_lr: Enable layer-wise learning rates (vision=1e-5, projection=2e-4, language=5e-5)
    """
    train_dataloader = get_instruction_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
        trial_mode=config.trial_mode,
    )

    val_dataloader = get_instruction_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
        trial_mode=config.trial_mode,
    )

    trainer = Stage2Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_processor=teacher_processor,
        distillation_config={
            'temperature': 2.0,
            'alpha': 0.3,
            'sft_weight': 0.7,
            'distill_weight': 0.3,
        },
        hub_repo_id=hub_repo_id,
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
        use_ema_teacher=use_ema_teacher,
        use_multi_scale=use_multi_scale,
        use_layer_wise_lr=use_layer_wise_lr,
    )

    trainer.train(num_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = EmberVLM()

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)

    run_stage2_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
    )



