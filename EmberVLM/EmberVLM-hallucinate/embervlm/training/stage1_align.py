"""
Stage 1: Visual-Language Alignment Training

Aligns RepViT vision features with TinyLLM text space using
contrastive learning and image captioning.
"""

import os
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    load_checkpoint,
    MetricTracker,
    print_trainable_parameters,
    push_checkpoint_to_hub,
    enable_gradient_checkpointing,
)
from embervlm.data.loaders import get_alignment_dataloader
from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker
from embervlm.training.contrastive_metrics import compute_enhanced_contrastive_metrics
from embervlm.training.early_stopping import EarlyStopping, BestCheckpointTracker, log_stage_summary, save_stage_metrics_json

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage1Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage1Visualizer = None


# ---------------------------------------------------------------------------
# Cross-GPU feature gathering (gradient-preserving)
# ---------------------------------------------------------------------------

class _AllGatherWithGrad(torch.autograd.Function):
    """All-gather that keeps gradients flowing to the local shard."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return grad_output
        rank = dist.get_rank()
        chunk = grad_output.size(0) // world_size
        return grad_output[rank * chunk : (rank + 1) * chunk]


def gather_features(tensor: torch.Tensor) -> torch.Tensor:
    """Gather embeddings from all GPUs (gradient-preserving).

    In DDP each GPU only sees its local micro-batch.  For contrastive
    learning we want as many negatives as possible, so we all-gather
    image and text embeddings across all ranks before computing the
    similarity matrix.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    return _AllGatherWithGrad.apply(tensor)


class ContrastiveLoss(nn.Module):
    """
    Image-text contrastive loss (CLIP-style) with hard-negative shifting.

    After computing cosine-similarity logits, the hardest in-batch negative
    score is subtracted from every row so that the cross-entropy focuses on
    discriminating positives from the most confusing negatives.

    Temperature is *learnable* (log-parameterized for stability, clamped to
    [0.01, 0.5]).  A higher/softer temperature early in training encourages
    better generalisation of the alignment signal beyond in-batch negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # Learnable log-temperature (CLIP-style).  init ≈ 0.07 which is
        # softer than the previous fixed 0.05, reducing overfitting to
        # in-batch negatives.  Clamped to [0.01, 0.5] during forward.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature))
        )

    @property
    def temperature(self) -> float:
        """Current temperature value (for logging)."""
        return self.log_temperature.exp().clamp(0.01, 0.5).item()

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss with hard-negative shifting.

        Args:
            image_features: [B, D] normalized image features
            text_features: [B, D] normalized text features

        Returns:
            Loss and metrics dictionary
        """
        # Normalize features
        img_embeds = F.normalize(image_features, dim=-1)
        txt_embeds = F.normalize(text_features, dim=-1)

        # Learnable temperature clamped to a safe range
        tau = self.log_temperature.exp().clamp(0.01, 0.5)
        logits_i2t = img_embeds @ txt_embeds.T / tau  # [B, B]
        logits_t2i = logits_i2t.T                     # [B, B]

        B = logits_i2t.size(0)
        eye = torch.eye(B, device=logits_i2t.device, dtype=torch.bool)

        # ---- Hard-negative shift ------------------------------------
        # For each anchor, find the hardest negative (highest sim among
        # non-positives) and subtract it so CE focuses on that contrast.
        neg_i2t = logits_i2t.detach().masked_fill(eye, -1e9)
        max_neg_i2t, _ = neg_i2t.max(dim=1, keepdim=True)  # [B, 1]
        logits_i2t = logits_i2t - max_neg_i2t

        neg_t2i = logits_t2i.detach().masked_fill(eye, -1e9)
        max_neg_t2i, _ = neg_t2i.max(dim=1, keepdim=True)  # [B, 1]
        logits_t2i = logits_t2i - max_neg_t2i

        # Labels (diagonal is positive)
        labels = torch.arange(B, device=img_embeds.device)

        # Cross entropy loss
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy (on original un-shifted logits for interpretability)
        with torch.no_grad():
            orig_logits_i2t = img_embeds @ txt_embeds.T / tau
            orig_logits_t2i = orig_logits_i2t.T
            pred_i2t = orig_logits_i2t.argmax(dim=-1)
            pred_t2i = orig_logits_t2i.argmax(dim=-1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()

        metrics = {
            'contrastive_loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'temperature': tau.item(),
        }

        return loss, metrics


class ContrastiveProjectionHead(nn.Module):
    """
    Lightweight MLP projection head for contrastive learning.

    Decouples the contrastive alignment objective from the captioning objective
    by projecting pooled features into a dedicated contrastive embedding space.
    This avoids the fusion module being pulled in two competing directions
    (LM-input space for captioning vs cosine-similarity space for retrieval).

    Architecture: Linear → GELU → Linear → L2-normalize
    """

    def __init__(self, input_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class Stage1Trainer:
    """Trainer for Stage 1: Visual-Language Alignment."""

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

        # Early stopping: monitor average of top-5 retrieval accuracies
        self.early_stopping = EarlyStopping(
            patience=5,
            mode='max',
            min_delta=0.001,
            verbose=True
        )
        self.best_checkpoint_tracker = BestCheckpointTracker(
            save_dir=config.output_dir,
            metric_name='avg_retrieval_top5',
            mode='max'
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

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Losses
        logger.info("Initializing contrastive loss (learnable tau, init=0.07, hard-negative shift)...")
        self.contrastive_loss = ContrastiveLoss(temperature=0.07).to(self.device)
        logger.info("Contrastive loss initialized (learnable temperature)")

        # Contrastive projection heads — project pooled features into a
        # dedicated 256-d contrastive space so that the fusion module is not
        # forced to simultaneously optimise for LM-input AND cosine-similarity.
        language_dim = getattr(model.config, 'language_hidden_size', 576)
        self.image_proj_head = ContrastiveProjectionHead(
            input_dim=language_dim, proj_dim=256
        ).to(self.device)
        self.text_proj_head = ContrastiveProjectionHead(
            input_dim=language_dim, proj_dim=256
        ).to(self.device)
        logger.info(f"Contrastive projection heads initialized (dim {language_dim} → 256)")

        # Loss weights — contrastive-heavy because alignment is the
        # *primary* goal of Stage 1.  Captioning keeps the LM warm but
        # should not dominate gradients.
        self.contrastive_weight = 0.7
        self.captioning_weight = 0.3
        logger.info(f"Stage 1 loss weights: contrastive={self.contrastive_weight}, captioning={self.captioning_weight}")

        # Add contrastive-specific parameters to the optimizer so they
        # actually receive gradients.  The main optimizer was created before
        # these modules existed, so we append a new param group.
        contrastive_params = (
            list(self.contrastive_loss.parameters()) +
            list(self.image_proj_head.parameters()) +
            list(self.text_proj_head.parameters())
        )
        self.optimizer.add_param_group({
            'params': contrastive_params,
            'lr': config.learning_rate,
            'weight_decay': 0.0,  # no decay on contrastive heads
        })
        n_contrastive_params = sum(p.numel() for p in contrastive_params)
        logger.info(f"Added {n_contrastive_params:,} contrastive params to optimizer")

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger with visualizations (main process)...")
            try:
                wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                self.wandb_logger = EnhancedWandbLogger(
                    project=wandb_project,
                    name="stage1_alignment",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info(f"Enhanced W&B logger initialized with project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B logger: {e}")
                self.wandb_logger = None

            logger.info("Initializing carbon tracker...")
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

        # Stage 1 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage1Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage1Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage1Visualizer: {e}")

        # Store embeddings for visualization (updated each batch)
        self.last_image_embeds = None
        self.last_text_embeds = None

        # Print info
        if is_main_process():
            print_trainable_parameters(self.model)

        logger.info(f"[Rank {self.rank}] Stage1Trainer initialization complete")

    # ------------------------------------------------------------------
    # Dynamic loss weighting — captioning warmup
    # ------------------------------------------------------------------
    def _get_loss_weights(self) -> Tuple[float, float]:
        """Return (contrastive_weight, captioning_weight) for current step.

        For the first 20 % of training, captioning weight linearly ramps
        from 0 → ``self.captioning_weight`` while the contrastive weight
        fills the remainder (so total weight = 1.0).  This lets the
        contrastive objective establish alignment before captioning starts
        pulling representations toward LM-input space.
        """
        total_steps = getattr(self.config, 'num_training_steps', None) or 25000
        warmup_frac = 0.20
        warmup_steps = int(total_steps * warmup_frac)

        if self.global_step < warmup_steps:
            alpha = self.global_step / max(warmup_steps, 1)  # 0 → 1
            cap_w = self.captioning_weight * alpha
            con_w = 1.0 - cap_w
        else:
            con_w = self.contrastive_weight
            cap_w = self.captioning_weight
        return con_w, cap_w

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
            
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)

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

            # Validate and clamp labels if present (preserve -100 ignore index)
            if labels is not None:
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
                # Get visual features
                vision_output = self.model.module.encode_image(pixel_values) \
                    if hasattr(self.model, 'module') else self.model.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']

                # Fuse to language space
                fused_visual = self.model.module.fuse_features(visual_tokens) \
                    if hasattr(self.model, 'module') else self.model.fuse_features(visual_tokens)

                # Get text embeddings
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool features for contrastive loss
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Project pooled features through dedicated contrastive heads
                # so that contrastive and captioning objectives don't fight
                # over the same representation space.
                image_proj = self.image_proj_head(image_pooled)
                text_proj = self.text_proj_head(text_pooled)

                # ── Cross-GPU gathering ──────────────────────────────
                # In DDP each GPU only sees its local micro-batch.  We
                # all-gather the projected embeddings so every GPU
                # computes similarity against ALL negatives across ranks,
                # greatly improving contrastive learning quality.
                image_proj_all = gather_features(image_proj)
                text_proj_all = gather_features(text_proj)

                # Contrastive loss (on gathered projected features)
                contrastive_loss, contrastive_metrics = self.contrastive_loss(
                    image_proj_all, text_proj_all
                )

                # Store embeddings for visualization (detach + CPU to avoid GPU memory buildup)
                self.last_image_embeds = image_pooled.detach().cpu()
                self.last_text_embeds = text_pooled.detach().cpu()

                # Free pooled tensors no longer needed
                del image_pooled, text_pooled, image_proj, text_proj
                del image_proj_all, text_proj_all

                # Captioning loss (language modeling)
                # MEMORY FIX: Reuse already-computed fused_visual instead of
                # calling prepare_inputs_embeds which re-encodes the image.
                # Build combined embeddings: [visual_tokens, text_tokens]
                num_visual = fused_visual.size(1)
                inputs_embeds = torch.cat([fused_visual, text_embeds], dim=1)

                # Free intermediate vision tensors
                del visual_tokens, fused_visual, vision_output

                # Adjust labels to match inputs_embeds length (account for visual tokens)
                batch_size_lbl = labels.size(0)
                adjusted_labels = torch.full(
                    (batch_size_lbl, inputs_embeds.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                # Visual tokens get -100, copy original labels after them
                adjusted_labels[:, num_visual:num_visual + labels.size(1)] = labels

                lm_outputs = model_ref.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones(
                        inputs_embeds.size(0), inputs_embeds.size(1),
                        device=self.device, dtype=torch.long
                    ),
                    labels=adjusted_labels,
                )

                # Captioning loss with label smoothing for better generalization
                if 'logits' in lm_outputs:
                    logits = lm_outputs['logits']
                    # Shift for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = adjusted_labels[..., 1:].contiguous()

                    # SAFETY CHECK: Ensure dimensions match before cross_entropy
                    # This prevents cryptic "batch_size mismatch" errors
                    if shift_logits.size(0) != shift_labels.size(0) or shift_logits.size(1) != shift_labels.size(1):
                        # Log the mismatch (only once per training)
                        if not hasattr(self, '_dim_mismatch_warned'):
                            logger.warning(f"⚠️ Dimension mismatch: logits {shift_logits.shape} vs labels {shift_labels.shape}")
                            self._dim_mismatch_warned = True

                        # Align to the minimum sequence length
                        min_seq_len = min(shift_logits.size(1), shift_labels.size(1))
                        shift_logits = shift_logits[:, :min_seq_len, :].contiguous()
                        shift_labels = shift_labels[:, :min_seq_len].contiguous()

                    # Use label smoothing (0.1) to prevent overconfident predictions
                    loss_fct = nn.CrossEntropyLoss(
                        ignore_index=-100,
                        label_smoothing=0.1,
                        reduction='mean'
                    )
                    captioning_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    # Free large LM output tensors before backward
                    del logits, shift_logits, shift_labels, lm_outputs, inputs_embeds
                else:
                    captioning_loss = lm_outputs.get('loss', torch.tensor(0.0, device=self.device))
                    del lm_outputs, inputs_embeds

                # Total loss — dynamic weighting with captioning warmup
                con_w, cap_w = self._get_loss_weights()
                loss = con_w * contrastive_loss + cap_w * captioning_loss

            # Backward pass
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

                # Update metrics
                metrics = {
                    'loss': loss.item() * self.config.gradient_accumulation_steps,
                    'contrastive_loss': contrastive_metrics['contrastive_loss'],
                    'captioning_loss': captioning_loss.item() if isinstance(captioning_loss, torch.Tensor) else 0.0,
                    'contrastive_weight': con_w,
                    'captioning_weight': cap_w,
                    'lr': self.scheduler.get_last_lr()[0],
                    **contrastive_metrics,
                }
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        if self.wandb_logger is not None:
                            # Use enhanced logging if available
                            if hasattr(self.wandb_logger, 'log_with_visualization'):
                                self.wandb_logger.log_with_visualization(
                                    avg_metrics,
                                    step=self.global_step,
                                    stage_name="stage1",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Stage 1 specific visualizations every 500 steps
                            if self.global_step % 500 == 0:
                                logger.info(f"[Stage1] Step {self.global_step}: Attempting visualizations...")
                                logger.info(f"  stage_visualizer: {self.stage_visualizer is not None}")
                                logger.info(f"  last_image_embeds: {self.last_image_embeds is not None}")
                                logger.info(f"  last_text_embeds: {self.last_text_embeds is not None}")

                                if self.stage_visualizer is not None:
                                    if self.last_image_embeds is not None and self.last_text_embeds is not None:
                                        try:
                                            logger.info(f"  Generating similarity matrix...")
                                            _, sim_img = self.stage_visualizer.plot_similarity_matrix(
                                                self.last_image_embeds,
                                                self.last_text_embeds,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage1/similarity_matrix", sim_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged similarity matrix to W&B")

                                            logger.info(f"  Generating t-SNE...")
                                            _, tsne_img = self.stage_visualizer.plot_embedding_tsne(
                                                self.last_image_embeds,
                                                self.last_text_embeds,
                                                self.global_step,
                                                n_samples=min(100, self.last_image_embeds.size(0))
                                            )
                                            self.wandb_logger.log_image(
                                                "stage1/embedding_tsne", tsne_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged t-SNE to W&B")
                                        except Exception as e:
                                            logger.error(f"  ✗ Failed to generate Stage 1 visualizations: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"  Skipping visualizations: embeddings not available")
                                else:
                                    logger.warning(f"  Skipping visualizations: stage_visualizer is None")

                            # Log loss curve and metrics summary every 100 steps
                            if self.global_step % 100 == 0:
                                try:
                                    # Log detailed loss components as charts
                                    self.wandb_logger.log({
                                        'stage1/contrastive_loss_chart': self.wandb_logger.wandb.plot.line_series(
                                            xs=[list(range(len(self.metric_tracker.history.get('contrastive_loss', []))))],
                                            ys=[self.metric_tracker.history.get('contrastive_loss', [])],
                                            keys=['Contrastive'],
                                            title='Stage 1 Contrastive Loss',
                                            xname='Step'
                                        ),
                                    }, step=self.global_step) if hasattr(self.metric_tracker, 'history') and self.metric_tracker.history.get('contrastive_loss') else None
                                except Exception as e:
                                    pass  # Silently ignore chart errors

                            # Gradient distribution every 500 steps
                            if self.global_step % 500 == 0 and hasattr(self.wandb_logger, 'log_gradient_distribution'):
                                gradients = {}
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None:
                                        gradients[name.split('.')[-1]] = param.grad
                                if gradients:
                                    self.wandb_logger.log_gradient_distribution(
                                        gradients, self.global_step, "stage1"
                                    )

                        progress_bar.set_postfix({
                            'loss': f"{avg_metrics['loss']:.4f}",
                            'acc': f"{avg_metrics.get('acc_i2t', 0):.3f}",
                        })

                    self.metric_tracker.reset()

                # Periodic GPU memory management to prevent OOM from fragmentation
                if self.global_step % 50 == 0:
                    torch.cuda.empty_cache()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.val_dataloader is not None and \
                   self.global_step % self.config.eval_steps == 0:
                    self.evaluate(eval_step=self.global_step)

    @torch.no_grad()
    def evaluate(self, eval_step: Optional[int] = None):
        """Evaluate on validation set."""
        if eval_step is None:
            eval_step = self.global_step

        self.model.eval()

        eval_metrics = MetricTracker()

        for batch in tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with get_autocast_context(self.config):
                # Get features
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                vision_output = model_ref.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']
                fused_visual = model_ref.fuse_features(visual_tokens)
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Normalize features for metric computation
                image_features = F.normalize(image_pooled, dim=-1)
                text_features = F.normalize(text_pooled, dim=-1)
                
                # Compute enhanced metrics (top-5, MRR, similarity distribution)
                enhanced_metrics = compute_enhanced_contrastive_metrics(
                    image_features, text_features, temperature=0.07
                )
                eval_metrics.update(enhanced_metrics)
                
                # Also compute loss for backward compatibility
                _, loss_metrics = self.contrastive_loss(image_pooled, text_pooled)
                eval_metrics.update({k: v for k, v in loss_metrics.items() if 'loss' in k})

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            if self.wandb_logger is not None:
                self.wandb_logger.log(avg_metrics, step=eval_step)
            logger.info(f"Validation metrics: {avg_metrics}")

        self.model.train()
        return avg_metrics

    def save_checkpoint(self, custom_path: Optional[str] = None):
        """Save training checkpoint.

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
            checkpoints = sorted(
                Path(self.config.output_dir).glob('checkpoint-*'),
                key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0
            )

            while len(checkpoints) > self.config.max_checkpoints:
                oldest = checkpoints.pop(0)
                import shutil
                try:
                    if oldest.is_symlink():
                        # Remove symbolic link directly
                        oldest.unlink()
                    elif oldest.is_dir():
                        shutil.rmtree(oldest, ignore_errors=True)
                    else:
                        oldest.unlink()
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

                # Validation at end of epoch
                val_metrics = {}
                improved = False
                if self.val_dataloader is not None:
                    val_metrics = self.evaluate()
                    
                    # Track metrics history
                    self.metrics_history.append(val_metrics)
                    
                    # Compute monitoring metric: average of top-5 retrieval accuracies
                    monitor_metric = (
                        val_metrics.get('val_acc_i2t_top5', 0.0) +
                        val_metrics.get('val_acc_t2i_top5', 0.0)
                    ) / 2.0
                    
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

                # Save at end of epoch (regular checkpoint)
                if not improved:
                    self.save_checkpoint()

                # Push to HuggingFace Hub after each epoch
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'contrastive_loss': self.metric_tracker.get_average().get('contrastive_loss', 0.0),
                            'captioning_loss': self.metric_tracker.get_average().get('captioning_loss', 0.0),
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=epoch + 1,
                            stage="stage1",
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
                log_stage_summary("Stage 1: Visual-Language Alignment", self.metrics_history)
                logger.info(f"✅ Stage 1 training completed: {len(self.metrics_history)}/{num_epochs} epochs")
                logger.info(f"   Best checkpoint: {self.best_checkpoint_tracker.load_best_checkpoint()}")

                # Generate end-of-training visualizations
                logger.info("📊 Generating Stage 1 end-of-training visualizations...")
                try:
                    # Base visualizations (loss decomposition + convergence)
                    if self.wandb_logger is not None:
                        self.wandb_logger.generate_final_visualizations("stage1", self.global_step)

                    # Stage-specific: similarity matrix + t-SNE
                    if self.stage_visualizer is not None:
                        if self.last_image_embeds is not None and self.last_text_embeds is not None:
                            try:
                                _, sim_img = self.stage_visualizer.plot_similarity_matrix(
                                    self.last_image_embeds, self.last_text_embeds, self.global_step
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage1/similarity_matrix_final", sim_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final similarity matrix")
                            except Exception as e:
                                logger.warning(f"  Failed to generate similarity matrix: {e}")

                            try:
                                _, tsne_img = self.stage_visualizer.plot_embedding_tsne(
                                    self.last_image_embeds, self.last_text_embeds, self.global_step,
                                    n_samples=min(100, self.last_image_embeds.size(0))
                                )
                                if self.wandb_logger is not None:
                                    self.wandb_logger.log_image(
                                        "stage1/embedding_tsne_final", tsne_img, step=self.global_step
                                    )
                                logger.info("  ✓ Saved final t-SNE embedding")
                            except Exception as e:
                                logger.warning(f"  Failed to generate t-SNE: {e}")
                        else:
                            logger.info("  Skipping stage-specific viz: no cached embeddings")
                    logger.info("📊 Stage 1 visualizations complete")
                except Exception as e:
                    logger.warning(f"Failed to generate Stage 1 end-of-training visualizations: {e}")

                # Save metrics JSON for cross-stage collation
                try:
                    step_metrics = {}
                    if self.wandb_logger is not None and hasattr(self.wandb_logger, 'metrics_history'):
                        step_metrics = {k: v for k, v in self.wandb_logger.metrics_history.items()
                                       if isinstance(v, list) and v}
                    save_stage_metrics_json(
                        stage_name='stage1',
                        output_dir=self.config.output_dir,
                        metrics_history=self.metrics_history,
                        step_metrics=step_metrics,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save stage1 metrics JSON: {e}")

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


def run_stage1_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    num_epochs: int = 3,
    hub_repo_id: Optional[str] = None,
    vision_backbone: str = "repvit",
    language_backbone: str = "tinyllm",
):
    """
    Run Stage 1 training.

    Args:
        model: EmberVLM model
        config: Training configuration
        data_dir: Directory with training data
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
        hub_repo_id: HuggingFace Hub repo ID for epoch-level pushes
        vision_backbone: Vision backbone name
        language_backbone: Language backbone name
    """
    # Create data loaders
    train_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
        trial_mode=config.trial_mode,
    )

    val_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
        trial_mode=config.trial_mode,
    )

    # Create trainer
    trainer = Stage1Trainer(
        hub_repo_id=hub_repo_id,
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train(num_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    # Config
    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = EmberVLM()

    # Train
    run_stage1_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
    )

