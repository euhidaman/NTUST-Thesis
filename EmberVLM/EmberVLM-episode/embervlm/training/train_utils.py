"""
Training Utilities for EmberVLM

Includes distributed training setup, optimizer/scheduler creation,
checkpointing, and other training utilities.
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic
    seed: int = 42
    output_dir: str = "./outputs"
    trial_mode: bool = False  # If True, use reduced dataset for quick validation

    # Model architecture - CRITICAL for checkpoint loading
    vision_backbone: str = "repvit"  # Options: 'repvit', 'mobilevit_xs'
    language_backbone: str = "tinyllm"  # Options: 'tinyllm', 'smollm_135m'

    # Distributed
    distributed: bool = True
    backend: str = "nccl"
    find_unused_parameters: bool = False

    # Precision
    mixed_precision: str = "bf16"  # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool = True

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 1000  # Increased for better stability
    num_training_steps: int = 10000

    # Batch
    batch_size: int = 128
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.8  # Slightly lower for stability

    # Checkpointing
    save_steps: int = 500
    max_checkpoints: int = 1
    save_optimizer: bool = True

    # Logging
    log_steps: int = 50
    eval_steps: int = 500
    wandb_project: Optional[str] = None  # W&B project name (e.g., embervlm-tiny, embervlm-small)

    # HuggingFace
    push_to_hub: bool = False
    hub_model_id: str = "embervlm"
    hub_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def setup_distributed() -> Tuple[int, int, int]:
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1 and not dist.is_initialized():
        # Use a longer timeout for stability during evaluation phases
        from datetime import timedelta
        timeout = timedelta(minutes=30)  # 30 min timeout (default is 10 min)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def set_seed(seed: int, rank: int = 0):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    use_layer_wise_lr: bool = False,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model with optional layer-wise learning rates.

    Args:
        model: Model to optimize
        config: Training configuration
        use_layer_wise_lr: If True, use different learning rates for vision/projection/language

    Returns:
        Configured optimizer
    """
    no_decay_names = ['bias', 'LayerNorm', 'layer_norm', 'ln_']
    
    if use_layer_wise_lr:
        # Layer-wise learning rates: vision < language < projection
        vision_decay = []
        vision_no_decay = []
        projection_decay = []
        projection_no_decay = []
        language_decay = []
        language_no_decay = []
        other_decay = []
        other_no_decay = []
        
        vision_lr = config.learning_rate * 0.2  # 1e-5 if base is 5e-5
        language_lr = config.learning_rate  # 5e-5
        projection_lr = config.learning_rate * 4.0  # 2e-4
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Categorize by module
            if 'vision_encoder' in name:
                if any(nd in name for nd in no_decay_names):
                    vision_no_decay.append(param)
                else:
                    vision_decay.append(param)
            elif 'fusion' in name or 'projection' in name or 'adapter' in name:
                if any(nd in name for nd in no_decay_names):
                    projection_no_decay.append(param)
                else:
                    projection_decay.append(param)
            elif 'language_model' in name:
                if any(nd in name for nd in no_decay_names):
                    language_no_decay.append(param)
                else:
                    language_decay.append(param)
            else:
                if any(nd in name for nd in no_decay_names):
                    other_no_decay.append(param)
                else:
                    other_decay.append(param)
        
        optimizer_groups = [
            {'params': vision_decay, 'lr': vision_lr, 'weight_decay': config.weight_decay},
            {'params': vision_no_decay, 'lr': vision_lr, 'weight_decay': 0.0},
            {'params': projection_decay, 'lr': projection_lr, 'weight_decay': config.weight_decay},
            {'params': projection_no_decay, 'lr': projection_lr, 'weight_decay': 0.0},
            {'params': language_decay, 'lr': language_lr, 'weight_decay': config.weight_decay},
            {'params': language_no_decay, 'lr': language_lr, 'weight_decay': 0.0},
            {'params': other_decay, 'lr': language_lr, 'weight_decay': config.weight_decay},
            {'params': other_no_decay, 'lr': language_lr, 'weight_decay': 0.0},
        ]
        
        # Filter out empty groups
        optimizer_groups = [g for g in optimizer_groups if len(g['params']) > 0]
        
    else:
        # Standard uniform learning rate
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

    if config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate if not use_layer_wise_lr else language_lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=config.learning_rate if not use_layer_wise_lr else language_lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration

    Returns:
        Configured scheduler
    """
    if config.scheduler.lower() == 'cosine':
        # Cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))

            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.num_training_steps - config.warmup_steps)
            )

            min_lr_ratio = config.min_learning_rate / config.learning_rate
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler.lower() == 'linear':
        # Linear schedule with warmup
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))

            return max(
                config.min_learning_rate / config.learning_rate,
                float(config.num_training_steps - current_step) /
                float(max(1, config.num_training_steps - config.warmup_steps))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler.lower() == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    return scheduler


def get_grad_scaler(config: TrainingConfig) -> Optional[GradScaler]:
    """
    Create gradient scaler for mixed precision.

    Args:
        config: Training configuration

    Returns:
        GradScaler or None
    """
    if config.mixed_precision == 'fp16':
        return GradScaler()
    return None


def get_autocast_context(config: TrainingConfig):
    """
    Get autocast context for mixed precision.

    Args:
        config: Training configuration

    Returns:
        Autocast context
    """
    if config.mixed_precision == 'bf16':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    elif config.mixed_precision == 'fp16':
        return torch.amp.autocast('cuda', dtype=torch.float16)
    else:
        return torch.amp.autocast('cuda', enabled=False)


def wrap_model_ddp(
    model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        config: Training configuration
        device: Device to use

    Returns:
        Wrapped model
    """
    # Model should already be on correct device, but ensure it
    if not next(model.parameters()).is_cuda:
        model = model.to(device)

    if config.distributed and get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=config.find_unused_parameters,
        )

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DDP wrapper."""
    if isinstance(model, DDP):
        return model.module
    return model


def force_rebuild_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    device: torch.device = None,
    logger: logging.Logger = None,
) -> nn.Module:
    """
    Force complete reconstruction of embedding layers to fix CUDA memory issues.

    This function creates brand new embedding and lm_head layers with fresh CUDA
    memory allocation, rather than relying on resize_token_embeddings which may
    leave stale CUDA state.

    Args:
        model: The EmberVLM model (unwrapped from DDP)
        new_vocab_size: Target vocabulary size
        device: Target device (if None, uses CPU for safe reconstruction)
        logger: Logger for status messages

    Returns:
        Model with reconstructed embedding layers
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Move model to CPU for safe memory operations
    original_device = next(model.parameters()).device
    model = model.cpu()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"🔧 Force rebuilding embeddings to size {new_vocab_size}...")

    # Find the language model and its embedding layer
    lang_model = None
    embed_layer = None
    embed_attr_path = None

    if hasattr(model, 'language_model'):
        lang_model = model.language_model

        # Check for HuggingFace-style model (PretrainedTinyLLMBackbone)
        if hasattr(lang_model, 'model'):
            inner_model = lang_model.model

            # GPT-2 style: transformer.wte
            if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'wte'):
                embed_layer = inner_model.transformer.wte
                embed_attr_path = 'language_model.model.transformer.wte'
            # Generic: get_input_embeddings
            elif hasattr(inner_model, 'get_input_embeddings'):
                embed_layer = inner_model.get_input_embeddings()
                embed_attr_path = 'language_model.model (via get_input_embeddings)'

        # Check for custom TinyLLMBackbone
        elif hasattr(lang_model, 'model') and hasattr(lang_model.model, 'transformer'):
            if hasattr(lang_model.model.transformer, 'wte'):
                embed_layer = lang_model.model.transformer.wte
                embed_attr_path = 'language_model.model.transformer.wte'

        # Direct embedding access
        elif hasattr(lang_model, 'get_input_embeddings'):
            embed_layer = lang_model.get_input_embeddings()
            embed_attr_path = 'language_model (via get_input_embeddings)'

    if embed_layer is None:
        raise RuntimeError("Could not find embedding layer in model")

    logger.info(f"  Found embedding layer at: {embed_attr_path}")

    # Get current embedding properties
    old_vocab_size = embed_layer.weight.shape[0]
    embed_dim = embed_layer.weight.shape[1]
    old_weights = embed_layer.weight.data.clone()

    logger.info(f"  Current embedding: vocab_size={old_vocab_size}, embed_dim={embed_dim}")
    logger.info(f"  Target embedding: vocab_size={new_vocab_size}, embed_dim={embed_dim}")

    # Create brand new embedding layer
    new_embed = nn.Embedding(new_vocab_size, embed_dim)

    # Initialize with small random values
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)

    # Copy old weights
    copy_size = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_embed.weight[:copy_size] = old_weights[:copy_size]

    logger.info(f"  ✓ Created new embedding layer, copied {copy_size} token embeddings")

    # Replace the embedding layer
    if hasattr(model.language_model, 'model'):
        inner_model = model.language_model.model

        if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'wte'):
            inner_model.transformer.wte = new_embed
            logger.info(f"  ✓ Replaced transformer.wte")

        if hasattr(inner_model, 'set_input_embeddings'):
            inner_model.set_input_embeddings(new_embed)
            logger.info(f"  ✓ Called set_input_embeddings()")

        # Also rebuild lm_head if it exists and is tied
        if hasattr(inner_model, 'lm_head'):
            old_lm_head = inner_model.lm_head
            new_lm_head = nn.Linear(old_lm_head.in_features, new_vocab_size, bias=old_lm_head.bias is not None)

            # Initialize
            nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
            if new_lm_head.bias is not None:
                nn.init.zeros_(new_lm_head.bias)

            # Copy old weights
            with torch.no_grad():
                copy_size_lm = min(old_lm_head.weight.shape[0], new_vocab_size)
                new_lm_head.weight[:copy_size_lm] = old_lm_head.weight[:copy_size_lm]
                if old_lm_head.bias is not None and new_lm_head.bias is not None:
                    new_lm_head.bias[:copy_size_lm] = old_lm_head.bias[:copy_size_lm]

            inner_model.lm_head = new_lm_head
            logger.info(f"  ✓ Rebuilt lm_head to size {new_vocab_size}")

        # Update config
        if hasattr(inner_model, 'config'):
            inner_model.config.vocab_size = new_vocab_size
            logger.info(f"  ✓ Updated inner model config.vocab_size")

    elif hasattr(model.language_model, 'set_input_embeddings'):
        model.language_model.set_input_embeddings(new_embed)
        logger.info(f"  ✓ Called language_model.set_input_embeddings()")

    # Update language_model config
    if hasattr(model.language_model, 'config'):
        model.language_model.config.vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated language_model.config.vocab_size")

    if hasattr(model.language_model, 'hf_config'):
        model.language_model.hf_config.vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated language_model.hf_config.vocab_size")

    # Update main model config
    if hasattr(model, 'config'):
        model.config.language_vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated model.config.language_vocab_size")

    # Move to target device
    target_device = device if device is not None else original_device
    model = model.to(target_device)

    # Force CUDA synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Verify the rebuild worked
    verify_embed = None
    if hasattr(model.language_model, 'get_input_embeddings'):
        verify_embed = model.language_model.get_input_embeddings()
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'get_input_embeddings'):
            verify_embed = model.language_model.model.get_input_embeddings()

    if verify_embed is not None:
        actual_size = verify_embed.weight.shape[0]
        if actual_size != new_vocab_size:
            raise RuntimeError(
                f"Embedding rebuild verification failed! "
                f"Expected {new_vocab_size}, got {actual_size}"
            )
        logger.info(f"  ✓ Verified embedding size: {actual_size}")

    logger.info(f"✅ Embedding reconstruction complete on device {target_device}")

    return model


def validate_tensor_bounds(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    logger_instance: logging.Logger = None,
    clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and optionally clamp input_ids and labels to be within vocabulary bounds.

    This is a failsafe utility to prevent CUDA index out-of-bounds errors.
    Should be called before any embedding lookup operation.

    Args:
        input_ids: Token IDs tensor
        labels: Labels tensor (may contain -100 for ignore)
        vocab_size: Maximum valid token ID + 1
        logger_instance: Logger for warnings
        clamp: Whether to clamp out-of-bounds values (True) or raise error (False)

    Returns:
        Tuple of (validated_input_ids, validated_labels)
    """
    if logger_instance is None:
        logger_instance = logger

    modified = False

    # Validate input_ids
    if input_ids is not None:
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()

        if max_token_id >= vocab_size:
            msg = f"input_ids max={max_token_id} >= vocab_size={vocab_size}"
            if clamp:
                logger_instance.warning(f"⚠️ {msg} - Clamping to valid range")
                input_ids = torch.clamp(input_ids, max=vocab_size - 1)
                modified = True
            else:
                raise ValueError(f"❌ CRITICAL: {msg}")

        if min_token_id < 0:
            msg = f"input_ids min={min_token_id} < 0"
            if clamp:
                logger_instance.warning(f"⚠️ {msg} - Clamping to valid range")
                input_ids = torch.clamp(input_ids, min=0)
                modified = True
            else:
                raise ValueError(f"❌ CRITICAL: {msg}")

    # Validate labels (preserve -100 ignore index)
    if labels is not None:
        valid_labels_mask = labels != -100
        if valid_labels_mask.any():
            valid_labels = labels[valid_labels_mask]
            max_label = valid_labels.max().item()
            min_label = valid_labels.min().item()

            if max_label >= vocab_size or min_label < 0:
                if clamp:
                    if max_label >= vocab_size:
                        logger_instance.warning(
                            f"⚠️ labels max={max_label} >= vocab_size={vocab_size} - Clamping"
                        )
                    if min_label < 0:
                        logger_instance.warning(
                            f"⚠️ labels min={min_label} < 0 - Clamping"
                        )
                    labels = torch.where(
                        valid_labels_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )
                    modified = True
                else:
                    raise ValueError(
                        f"❌ CRITICAL: labels out of bounds "
                        f"(min={min_label}, max={max_label}, vocab_size={vocab_size})"
                    )

    if modified:
        logger_instance.debug(f"Tensors validated and clamped to range [0, {vocab_size - 1}]")

    return input_ids, labels


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    step: int,
    metrics: Dict[str, float],
    output_dir: str,
    scaler: Optional[GradScaler] = None,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        config: Training configuration
        step: Current training step
        metrics: Current metrics
        output_dir: Directory to save to
        scaler: Optional gradient scaler
    """
    if not is_main_process():
        return

    logger.info(f"=" * 80)
    logger.info(f"SAVE_CHECKPOINT called: step={step}, output_dir={output_dir}")
    logger.info(f"=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created checkpoint directory: {output_dir}")

    # Unwrap model
    model_to_save = unwrap_model(model)

    # Save model
    model_path = output_dir / 'pytorch_model.bin'
    torch.save(model_to_save.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    # Save config - CRITICAL for loading correct architecture
    config_saved = False

    # Try 1: Save from model.config
    if hasattr(model_to_save, 'config') and model_to_save.config is not None:
        try:
            config_path = output_dir / 'config.json'
            config_dict = model_to_save.config.to_dict()

            # CRITICAL FIX: Update vocab_size from actual model if embeddings were resized
            # Check multiple paths to find actual embedding size (handles TinyLLM, SmolLM, etc.)
            actual_vocab_size = None
            if hasattr(model_to_save, 'language_model') and hasattr(model_to_save.language_model, 'model'):
                lm_model = model_to_save.language_model.model

                # Path 1: SmolLM - model.model.embed_tokens
                if hasattr(lm_model, 'model') and hasattr(lm_model.model, 'embed_tokens'):
                    actual_vocab_size = lm_model.model.embed_tokens.weight.shape[0]
                    logger.info(f"  Detected vocab from model.model.embed_tokens: {actual_vocab_size}")
                # Path 2: Direct embed_tokens
                elif hasattr(lm_model, 'embed_tokens'):
                    actual_vocab_size = lm_model.embed_tokens.weight.shape[0]
                    logger.info(f"  Detected vocab from embed_tokens: {actual_vocab_size}")
                # Path 3: lm_head output size
                elif hasattr(lm_model, 'lm_head'):
                    actual_vocab_size = lm_model.lm_head.out_features
                    logger.info(f"  Detected vocab from lm_head: {actual_vocab_size}")
                # Path 4: TinyLLM transformer.wte
                elif hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'wte'):
                    actual_vocab_size = lm_model.transformer.wte.weight.shape[0]
                    logger.info(f"  Detected vocab from transformer.wte: {actual_vocab_size}")

            if actual_vocab_size is not None:
                config_vocab_size = config_dict.get('language_vocab_size') or config_dict.get('vocab_size', 0)
                if actual_vocab_size != config_vocab_size:
                    logger.warning(f"Vocab size mismatch: config has {config_vocab_size}, "
                                 f"but embeddings have {actual_vocab_size}. Using actual size.")
                config_dict['language_vocab_size'] = actual_vocab_size
                config_dict['vocab_size'] = actual_vocab_size

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"✓ Saved model config to {config_path} (from model.config)")
            logger.info(f"  vision_backbone: {config_dict.get('vision_backbone')}")
            logger.info(f"  language_backbone: {config_dict.get('language_backbone')}")
            logger.info(f"  vocab_size: {config_dict.get('vocab_size')}")
            config_saved = True
        except Exception as e:
            logger.warning(f"Failed to save model config from model.config: {e}", exc_info=True)

    # Try 2: Fallback - reconstruct from training config
    if not config_saved and hasattr(config, 'vision_backbone') and hasattr(config, 'language_backbone'):
        try:
            from embervlm.models.embervlm import EmberVLMConfig

            logger.info(f"[DEBUG] Attempting to save config from training config:")
            logger.info(f"  vision_backbone: {config.vision_backbone}")
            logger.info(f"  language_backbone: {config.language_backbone}")

            # Reconstruct model config from training config attributes
            model_config = EmberVLMConfig(
                vision_backbone=config.vision_backbone,
                language_backbone=config.language_backbone,
                pretrained_language_model=getattr(config, 'pretrained_language_model', None),
                image_size=getattr(config, 'image_size', 224),
                freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', True),
                freeze_language_model=getattr(config, 'freeze_language_model', False),
            )

            config_path = output_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(model_config.to_dict(), f, indent=2)
            logger.info(f"✓ Saved model config to {config_path} (from training config)")
            logger.info(f"  vision_backbone: {config.vision_backbone}")
            logger.info(f"  language_backbone: {config.language_backbone}")
            config_saved = True
        except Exception as e:
            logger.error(f"Failed to save model config from training config: {e}", exc_info=True)
    else:
        if not config_saved:
            logger.warning(f"[DEBUG] Cannot use fallback config save:")
            logger.warning(f"  hasattr(config, 'vision_backbone'): {hasattr(config, 'vision_backbone')}")
            logger.warning(f"  hasattr(config, 'language_backbone'): {hasattr(config, 'language_backbone')}")

    if not config_saved:
        logger.error(f"❌ Model config was NOT saved to {output_dir}!")
        logger.error(f"   Evaluation will load WRONG model architecture (defaults)!")
        logger.error(f"   This is a CRITICAL ERROR!")

        # Try one last desperate attempt - save raw config dict
        try:
            logger.error(f"   Attempting emergency config save...")
            config_path = output_dir / 'config.json'
            emergency_config = {
                'vision_backbone': getattr(config, 'vision_backbone', 'unknown'),
                'language_backbone': getattr(config, 'language_backbone', 'unknown'),
                'image_size': getattr(config, 'image_size', 224),
            }
            with open(config_path, 'w') as f:
                json.dump(emergency_config, f, indent=2)
            logger.error(f"   ✓ Emergency config saved: {emergency_config}")
        except Exception as e:
            logger.error(f"   Emergency config save also failed: {e}", exc_info=True)

    # Save training state
    training_state = {
        'step': step,
        'metrics': metrics,
        'training_config': config.to_dict(),
    }

    if config.save_optimizer:
        training_state['optimizer'] = optimizer.state_dict()
        training_state['scheduler'] = scheduler.state_dict()
        if scaler is not None:
            training_state['scaler'] = scaler.state_dict()

    state_path = output_dir / 'training_state.pt'
    torch.save(training_state, state_path)

    logger.info(f"Saved checkpoint at step {step} to {output_dir}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_dir: str,
    scaler: Optional[GradScaler] = None,
) -> int:
    """
    Load training checkpoint with robust handling of architecture changes.

    This function handles:
    - Shape mismatches between checkpoint and model (layers will be randomly initialized)
    - Missing keys (new layers added, will use model's initialization)
    - Extra keys in checkpoint (old layers removed, will be ignored)
    - Stage transitions where architecture changes (e.g., reasoning module changes)

    Args:
        model: Model to load into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        checkpoint_dir: Directory to load from
        scaler: Optional gradient scaler

    Returns:
        Training step from checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load model weights
    model_path = checkpoint_dir / 'pytorch_model.bin'
    if model_path.exists():
        model_to_load = unwrap_model(model)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Filter out mismatched keys (e.g., when model architecture changes between stages)
        model_state_dict = model_to_load.state_dict()
        filtered_state_dict = {}
        mismatched_keys = []
        skipped_keys = []

        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    mismatched_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
                    logger.warning(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
            else:
                # Key doesn't exist in current model (layer removed/renamed)
                skipped_keys.append(key)

        # Identify keys that will be missing (new layers in model not in checkpoint)
        new_keys = []
        for key in model_state_dict.keys():
            if key not in filtered_state_dict:
                new_keys.append(key)

        # Special handling for Stage transitions
        # Stage 2 -> Stage 3: reasoning module may have different output dimensions
        # Stage 3 -> Stage 4: reasoning module architecture may be refined
        reasoning_module_keys = [k for k in mismatched_keys if 'reasoning_module' in k or 'robot_head' in k]
        if reasoning_module_keys:
            logger.info(f"Detected reasoning module architecture change (likely Stage transition)")
            logger.info(f"Reasoning module layers will be randomly initialized: {len(reasoning_module_keys)} layer(s)")
            for key_info in reasoning_module_keys:
                logger.info(f"  - {key_info}")

        # Log summary
        logger.info(f"Checkpoint loading summary:")
        logger.info(f"  - Total checkpoint keys: {len(state_dict)}")
        logger.info(f"  - Compatible keys loaded: {len(filtered_state_dict)}")
        logger.info(f"  - Shape mismatches (skipped): {len(mismatched_keys)}")
        logger.info(f"  - Unknown checkpoint keys (skipped): {len(skipped_keys)}")
        logger.info(f"  - New model keys (random init): {len(new_keys)}")

        # Load filtered state dict with strict=False to allow missing keys
        # This MUST use filtered_state_dict (not original state_dict) to avoid shape errors
        load_result = model_to_load.load_state_dict(filtered_state_dict, strict=False)

        logger.info(f"Loaded model weights from {model_path}")
        if mismatched_keys:
            logger.warning(f"Skipped {len(mismatched_keys)} mismatched layer(s) - will be randomly initialized")
        if load_result.missing_keys:
            # Filter to show only significant missing keys (not the ones we know about)
            significant_missing = [k for k in load_result.missing_keys if k not in new_keys]
            if significant_missing:
                logger.info(f"Missing keys in checkpoint (using model's random init): {len(significant_missing)} keys")
        if load_result.unexpected_keys:
            logger.info(f"Unexpected keys in checkpoint (ignored): {len(load_result.unexpected_keys)} keys")

    # Load training state
    state_path = checkpoint_dir / 'training_state.pt'
    step = 0

    if state_path.exists():
        training_state = torch.load(state_path, map_location='cpu')
        step = training_state.get('step', 0)

        if optimizer is not None and 'optimizer' in training_state:
            optimizer.load_state_dict(training_state['optimizer'])
            logger.info("Loaded optimizer state")

        if scheduler is not None and 'scheduler' in training_state:
            scheduler.load_state_dict(training_state['scheduler'])
            logger.info("Loaded scheduler state")

        if scaler is not None and 'scaler' in training_state:
            scaler.load_state_dict(training_state['scaler'])
            logger.info("Loaded scaler state")

    return step


class MetricTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics[key] += value
            self.counts[key] += 1

    def get_average(self) -> Dict[str, float]:
        """Get averaged metrics."""
        return {
            key: self.metrics[key] / max(1, self.counts[key])
            for key in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score

        Returns:
            True if should stop
        """
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model: nn.Module):
    """Print trainable parameter info."""
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)

    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total parameters: {total:,}")


def get_parameter_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Get parameter groups by trainability."""
    trainable = []
    frozen = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    return {'trainable': trainable, 'frozen': frozen}


def enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing for memory efficiency.
    
    For HuggingFace models (e.g. LlamaForCausalLM), we must call
    gradient_checkpointing_enable() on the PreTrainedModel so that
    _gradient_checkpointing_func is set on each decoder layer.
    Simply setting layer.gradient_checkpointing = True is NOT enough
    for newer transformers versions.
    
    IMPORTANT: We pass use_reentrant=False to avoid conflicts with DDP.
    Reentrant checkpointing (the default) causes parameters to be marked
    ready multiple times during backward, crashing DDP with:
      "Expected to mark a variable ready only once"
    Non-reentrant checkpointing is both DDP-safe and the recommended
    default going forward in PyTorch.
    """
    enabled = False
    # Non-reentrant checkpointing is required for DDP compatibility.
    # Reentrant checkpointing triggers duplicate autograd hooks on the
    # same parameter, which DDP interprets as an error.
    gc_kwargs = {"use_reentrant": False}

    # 1. Try the model itself (works if model IS a PreTrainedModel)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        enabled = True
    # 2. Try model.model (SmolLMBackbone wraps LlamaForCausalLM in .model)
    elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        enabled = True
    # 3. Try model.model.model (in case of double wrapping)
    elif (hasattr(model, 'model') and hasattr(model.model, 'model')
          and hasattr(model.model.model, 'gradient_checkpointing_enable')):
        model.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        enabled = True
    # 4. Older API
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        enabled = True

    if not enabled:
        logger.warning(
            "Could not enable gradient checkpointing — no compatible method found. "
            "Training will proceed without it (higher memory usage)."
        )


def compute_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    """Compute effective batch size across all processes."""
    return batch_size * gradient_accumulation_steps * world_size


def push_checkpoint_to_hub(
    model: nn.Module,
    tokenizer: Any,
    repo_id: str,
    epoch: int,
    stage: str,
    metrics: Dict[str, Any],
    vision_backbone: str,
    language_backbone: str,
    carbon_emissions: Optional[float] = None,
    overwrite: bool = True,
    commit_message: Optional[str] = None,
):
    """
    Push model checkpoint to HuggingFace Hub after each epoch.
    
    Args:
        model: The EmberVLM model to push
        tokenizer: The tokenizer
        repo_id: HuggingFace repo ID (e.g., "username/embervlm-small")
        epoch: Current epoch number
        stage: Training stage (e.g., "stage1", "stage2", etc.)
        metrics: Dictionary of metrics to include in model card
        vision_backbone: Vision backbone name
        language_backbone: Language backbone name
        carbon_emissions: Total carbon emissions so far (kg CO2eq)
        overwrite: If True, always push to same repo (default)
        commit_message: Optional custom commit message
    """
    # Check if push is disabled
    if os.environ.get('DISABLE_HUB_PUSH', '').lower() in ('1', 'true', 'yes'):
        logger.info("Hub push disabled via DISABLE_HUB_PUSH environment variable")
        return
    
    # Check for HF token - try multiple sources
    hf_token = None
    token_source = None

    # 1. Check HF_TOKEN environment variable
    if os.environ.get('HF_TOKEN'):
        hf_token = os.environ.get('HF_TOKEN')
        token_source = "HF_TOKEN env var"
    # 2. Check HUGGING_FACE_HUB_TOKEN (alternative env var)
    elif os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
        token_source = "HUGGING_FACE_HUB_TOKEN env var"
    # 3. Check huggingface-cli login token
    else:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
            if hf_token:
                token_source = "huggingface-cli login"
        except ImportError:
            pass
        except Exception:
            pass

    # 4. Try to read from token file directly
    if not hf_token:
        token_paths = [
            Path.home() / ".huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        ]
        for token_path in token_paths:
            if token_path.exists():
                try:
                    hf_token = token_path.read_text().strip()
                    if hf_token:
                        token_source = f"token file ({token_path})"
                        break
                except Exception:
                    pass

    if not hf_token:
        logger.warning("HF token not found. Skipping hub push.")
        logger.warning("To enable, do one of the following:")
        logger.warning("  1. export HF_TOKEN=your_token")
        logger.warning("  2. huggingface-cli login")
        return
    
    logger.info(f"Using HF token from: {token_source}")

    try:
        from huggingface_hub import HfApi, create_repo
        from datetime import datetime
        
        # Unwrap DDP if needed
        model_to_save = unwrap_model(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model_to_save.parameters())
        trainable_params = sum(p.numel() for p in model_to_save.parameters() if p.requires_grad)
        
        # Determine model size
        if total_params < 100_000_000:
            size_category = "Tiny (~35M parameters)"
        else:
            size_category = "Small (~137M parameters)"
        
        # Create commit message
        if commit_message is None:
            commit_message = f"Update model - {stage.upper()} Epoch {epoch}"
            if 'loss' in metrics:
                commit_message += f" | Loss: {metrics['loss']:.4f}"
            if 'accuracy' in metrics:
                commit_message += f" | Acc: {metrics['accuracy']:.2%}"
        
        logger.info("="*60)
        logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
        logger.info(f"  Stage: {stage} | Epoch: {epoch}")
        logger.info(f"  Commit: {commit_message}")
        logger.info("="*60)
        
        # Create repo if it doesn't exist
        api = HfApi(token=hf_token)
        try:
            create_repo(repo_id, token=hf_token, exist_ok=True, private=False)
            logger.info(f"✓ Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.warning(f"Repo creation note: {e}")
        
        # Generate comprehensive model card
        model_card = _generate_model_card(
            vision_backbone=vision_backbone,
            language_backbone=language_backbone,
            total_params=total_params,
            trainable_params=trainable_params,
            size_category=size_category,
            stage=stage,
            epoch=epoch,
            metrics=metrics,
            carbon_emissions=carbon_emissions,
            repo_id=repo_id,
        )
        
        # Create temporary directory for upload
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Save model
            logger.info("Saving model...")
            model_to_save.save_pretrained(str(tmp_path))
            
            # Save tokenizer
            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(str(tmp_path))
            
            # Save model card
            logger.info("Saving model card...")
            with open(tmp_path / "README.md", "w", encoding="utf-8") as f:
                f.write(model_card)
            
            # Save training metadata
            training_info = {
                "stage": stage,
                "epoch": epoch,
                "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()},
                "carbon_emissions_kg": carbon_emissions,
                "timestamp": datetime.now().isoformat(),
                "vision_backbone": vision_backbone,
                "language_backbone": language_backbone,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
            with open(tmp_path / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            # Upload to hub
            logger.info("Uploading to HuggingFace Hub...")
            api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=hf_token,
            )
            
        logger.info(f"✅ Successfully pushed to https://huggingface.co/{repo_id}")
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _generate_model_card(
    vision_backbone: str,
    language_backbone: str,
    total_params: int,
    trainable_params: int,
    size_category: str,
    stage: str,
    epoch: int,
    metrics: Dict[str, Any],
    carbon_emissions: Optional[float],
    repo_id: str,
) -> str:
    """Generate comprehensive model card for HuggingFace Hub."""
    from datetime import datetime
    
    # Backbone descriptions
    vision_desc = {
        'repvit': 'RepViT-M0.9 (~5M params)',
        'mobilevit_xs': 'Apple MobileViT-XS (~2.3M params)'
    }.get(vision_backbone, vision_backbone)
    
    language_desc = {
        'tinyllm': 'TinyLLM-30M (30M params)',
        'smollm_135m': 'SmolLM-135M (135M params)'
    }.get(language_backbone, language_backbone)
    
    # Format metrics
    metrics_text = "\n".join([f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}" 
                              for k, v in sorted(metrics.items())])
    
    carbon_text = f"\n- **Carbon Emissions (so far)**: {carbon_emissions:.4f} kg CO2eq" if carbon_emissions else ""
    
    # Stage descriptions
    stage_desc = {
        'stage1': 'Visual-Language Alignment - Learning to ground vision and language',
        'stage2': 'Multimodal Instruction Tuning - Following complex instructions',
        'stage3': 'Robot Fleet Selection - Choosing optimal robots for tasks',
        'stage4': 'Chain-of-Thought Reasoning - Generating reasoning chains',
    }.get(stage, stage)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    card = f"""---
language:
- en
license: apache-2.0
tags:
- vision-language
- multimodal
- robotics
- edge-deployment
- tiny-vlm
- {vision_backbone}
- {language_backbone}
- {stage}
base_model:
- {language_backbone}
library_name: transformers
pipeline_tag: image-text-to-text
---

# EmberVLM: {size_category}

**🔥 Efficient Vision-Language Model for Edge Deployment & Robotic Applications**

This model is currently in training - **{stage.upper()} (Epoch {epoch})**.

## 📊 Current Training Status

- **Stage**: {stage_desc}
- **Epoch**: {epoch}
- **Last Updated**: {timestamp}

### Latest Metrics
{metrics_text}{carbon_text}

## 🏗️ Model Architecture

- **Size**: {size_category}
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)
- **Vision Encoder**: {vision_desc}
- **Language Model**: {language_desc}

## 🎯 Training Curriculum

EmberVLM follows a 4-stage training curriculum:

1. ✅ **Stage 1: Visual-Language Alignment** - Grounding vision and language
2. ✅ **Stage 2: Multimodal Instruction Tuning** - Following instructions
3. ✅ **Stage 3: Robot Fleet Selection** - Task-robot matching
4. ⏳ **Stage 4: Chain-of-Thought Reasoning** - Reasoning generation

**Current Stage**: {stage.upper()}

## 💻 Usage

```python
from transformers import AutoTokenizer
from embervlm import EmberVLM
from PIL import Image

# Load model and tokenizer
model = EmberVLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Load image
image = Image.open("scene.jpg")

# Generate response
prompt = "<image>Describe what you see and select the best robot for this task."
outputs = model.generate(
    image=image,
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=256
)

print(outputs)
```

## 🎓 Training Details

- **Vision Backbone**: {vision_backbone}
- **Language Backbone**: {language_backbone}
- **Optimization**: AdamW with cosine learning rate schedule
- **Mixed Precision**: bfloat16
- **Distributed Training**: Multi-GPU with DDP
- **Class Balancing**: Focal loss for robot selection (Stage 3)
- **Reasoning**: Chain-of-thought with reinforcement learning (Stage 4)

## 🌍 Environmental Impact

This model is designed for edge deployment to minimize energy consumption.{carbon_text}

## 🎯 Intended Use

- **Primary**: Edge deployment on resource-constrained devices
- **Applications**: 
  - Robotic vision-language understanding
  - Real-time multimodal reasoning
  - Robot fleet selection and task planning
  - Mobile/embedded AI systems

## ⚠️ Limitations

- Model is still in training - performance will improve as training progresses
- Optimized for efficiency over maximum accuracy
- Best suited for edge/mobile deployment scenarios
- Training focused on robot-centric scenarios

## 📚 Citation

```bibtex
@software{{embervlm_2026,
  title = {{EmberVLM: Efficient Vision-Language Model for Edge Deployment}},
  author = {{EmberVLM Team}},
  year = {{2026}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## 📝 License

Apache 2.0

---

**Note**: This is a checkpoint from {stage} training (epoch {epoch}). 
The model will be updated after each epoch with improved performance.
"""
    
    return card
