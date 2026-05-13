"""
FLOPs Counter for EmberVLM

Counts floating point operations for model analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class FLOPsCounter:
    """
    Count FLOPs for neural network models.

    Supports:
    - Linear layers
    - Convolutions
    - Attention mechanisms
    - Element-wise operations
    """

    def __init__(self):
        self.flops = 0
        self.params = 0
        self.hooks = []

    def reset(self):
        """Reset counters."""
        self.flops = 0
        self.params = 0

    @staticmethod
    def _linear_flops(module: nn.Linear, input: torch.Tensor) -> int:
        """Calculate FLOPs for linear layer."""
        batch_size = input.size(0) if input.dim() > 1 else 1
        seq_len = input.size(1) if input.dim() > 2 else 1

        # Multiply-accumulate: 2 * in_features * out_features
        flops = 2 * module.in_features * module.out_features

        # Add bias
        if module.bias is not None:
            flops += module.out_features

        return flops * batch_size * seq_len

    @staticmethod
    def _conv2d_flops(module: nn.Conv2d, input: torch.Tensor) -> int:
        """Calculate FLOPs for Conv2d layer."""
        batch_size = input.size(0)
        out_h = (input.size(2) + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
        out_w = (input.size(3) + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1

        # FLOPs per output element
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)

        # Total FLOPs
        flops = 2 * batch_size * module.out_channels * out_h * out_w * kernel_ops

        if module.bias is not None:
            flops += batch_size * module.out_channels * out_h * out_w

        return flops

    @staticmethod
    def _attention_flops(
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> int:
        """Calculate FLOPs for attention mechanism."""
        # QKV projection
        hidden_dim = num_heads * head_dim
        qkv_flops = 3 * 2 * seq_len * hidden_dim * hidden_dim

        # Attention scores: Q @ K^T
        attn_flops = 2 * seq_len * seq_len * hidden_dim

        # Softmax (approximate)
        softmax_flops = seq_len * seq_len * 5  # exp, sum, div operations

        # Attention @ V
        output_flops = 2 * seq_len * seq_len * hidden_dim

        # Output projection
        proj_flops = 2 * seq_len * hidden_dim * hidden_dim

        total = qkv_flops + attn_flops + softmax_flops + output_flops + proj_flops
        return batch_size * total

    @staticmethod
    def _layernorm_flops(
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> int:
        """Calculate FLOPs for LayerNorm."""
        # Mean, variance, normalization, scale, shift
        return batch_size * seq_len * hidden_dim * 5

    @staticmethod
    def _embedding_flops(
        batch_size: int,
        seq_len: int,
    ) -> int:
        """Calculate FLOPs for embedding lookup (minimal)."""
        return batch_size * seq_len

    def count_module(
        self,
        module: nn.Module,
        input: torch.Tensor,
    ) -> int:
        """Count FLOPs for a single module."""
        if isinstance(module, nn.Linear):
            return self._linear_flops(module, input)
        elif isinstance(module, nn.Conv2d):
            return self._conv2d_flops(module, input)
        elif isinstance(module, nn.LayerNorm):
            batch_size = input.size(0)
            seq_len = input.size(1) if input.dim() > 2 else 1
            hidden_dim = input.size(-1)
            return self._layernorm_flops(batch_size, seq_len, hidden_dim)
        elif isinstance(module, nn.Embedding):
            batch_size = input.size(0)
            seq_len = input.size(-1)
            return self._embedding_flops(batch_size, seq_len)
        return 0

    def _make_hook(self, module: nn.Module):
        """Create forward hook for counting FLOPs."""
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if input is not None:
                self.flops += self.count_module(module, input)
        return hook

    def attach_hooks(self, model: nn.Module):
        """Attach hooks to all modules."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Embedding)):
                hook = module.register_forward_hook(self._make_hook(module))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def count_params(self, model: nn.Module) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())

    def count_trainable_params(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_flops(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    seq_len: int = 512,
    image_size: int = 224,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Count FLOPs for EmberVLM model.

    Args:
        model: EmberVLM model
        input_ids: Optional input token IDs
        pixel_values: Optional input images
        batch_size: Batch size for counting
        seq_len: Sequence length
        image_size: Image size
        device: Device to run on

    Returns:
        Dictionary with FLOP counts
    """
    model = model.to(device)
    model.eval()

    # Create dummy inputs if not provided
    if input_ids is None:
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    if pixel_values is None:
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device)

    counter = FLOPsCounter()
    counter.attach_hooks(model)

    # Run forward pass
    with torch.no_grad():
        try:
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")

    total_flops = counter.flops
    counter.remove_hooks()

    # Get parameter counts
    total_params = counter.count_params(model)
    trainable_params = counter.count_trainable_params(model)

    # Format results
    def format_number(n):
        if n >= 1e12:
            return f"{n/1e12:.2f}T"
        elif n >= 1e9:
            return f"{n/1e9:.2f}G"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)

    return {
        'total_flops': total_flops,
        'total_flops_formatted': format_number(total_flops),
        'flops_per_sample': total_flops // batch_size,
        'total_params': total_params,
        'total_params_formatted': format_number(total_params),
        'trainable_params': trainable_params,
        'trainable_params_formatted': format_number(trainable_params),
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
    }


def estimate_training_flops(
    flops_per_sample: int,
    num_samples: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Estimate total training FLOPs.

    Args:
        flops_per_sample: FLOPs per forward pass
        num_samples: Total training samples
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Dictionary with FLOP estimates
    """
    # Forward pass FLOPs
    forward_flops = flops_per_sample * num_samples * num_epochs

    # Backward pass is approximately 2x forward
    backward_flops = 2 * forward_flops

    # Total training FLOPs
    total_flops = forward_flops + backward_flops

    # Format
    def format_flops(f):
        if f >= 1e18:
            return f"{f/1e18:.2f} EFLOP"
        elif f >= 1e15:
            return f"{f/1e15:.2f} PFLOP"
        elif f >= 1e12:
            return f"{f/1e12:.2f} TFLOP"
        return f"{f/1e9:.2f} GFLOP"

    return {
        'forward_flops': forward_flops,
        'backward_flops': backward_flops,
        'total_flops': total_flops,
        'total_flops_formatted': format_flops(total_flops),
        'num_samples': num_samples,
        'num_epochs': num_epochs,
    }


class ModelProfiler:
    """
    Profile model for performance analysis.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.flops_counter = FLOPsCounter()

    def profile(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Profile model performance.

        Args:
            input_ids: Input token IDs
            pixel_values: Optional input images
            num_warmup: Number of warmup runs
            num_runs: Number of profiling runs

        Returns:
            Profiling results
        """
        import time

        self.model.eval()
        device = next(self.model.parameters()).device

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_ids=input_ids, pixel_values=pixel_values)

        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()

            with torch.no_grad():
                _ = self.model(input_ids=input_ids, pixel_values=pixel_values)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        # Memory usage
        memory_stats = {}
        if device.type == 'cuda':
            memory_stats = {
                'allocated_mb': torch.cuda.memory_allocated(device) / 1e6,
                'reserved_mb': torch.cuda.memory_reserved(device) / 1e6,
                'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1e6,
            }

        return {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'throughput_samples_per_sec': input_ids.size(0) / avg_time,
            **memory_stats,
        }

