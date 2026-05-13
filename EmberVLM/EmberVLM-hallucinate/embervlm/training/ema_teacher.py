"""
Exponential Moving Average (EMA) Teacher for Self-Distillation

Implements momentum-based self-distillation where the model learns from
a moving average version of itself. This provides stable teaching signals
without requiring an external teacher model.

Key benefits:
- No architecture mismatch issues
- No extra VRAM for external teacher (EMA shares structure)
- Continuous improvement throughout training
- Used in Data2Vec, BYOL, and modern self-supervised learning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class EMATeacher(nn.Module):
    """
    Exponential Moving Average Teacher for self-distillation.
    
    Maintains a moving average of the student model's weights and uses it
    as a teacher. The EMA model is updated after each training step:
        ema_param = decay * ema_param + (1 - decay) * student_param
    
    Args:
        student_model: The model being trained
        decay: EMA decay rate (default: 0.9995)
            Higher = smoother, more stable
            Lower = follows student more closely
        update_after_step: Start EMA updates after N steps (default: 100)
            Allows student to learn basics before using as teacher
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        decay: float = 0.9995,
        update_after_step: int = 100,
    ):
        super().__init__()
        self.decay = decay
        self.update_after_step = update_after_step
        self.num_updates = 0
        
        # Create EMA model as a deep copy
        self.ema_model = copy.deepcopy(student_model)
        self.ema_model.eval()  # Always in eval mode
        
        # Freeze all EMA parameters (no gradient computation)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        logger.info(f"✓ EMA Teacher initialized (decay={decay}, warmup={update_after_step} steps)")
    
    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """
        Update EMA teacher weights using exponential moving average.
        
        Args:
            student_model: Current student model to learn from
        """
        # Wait for warmup period
        if self.num_updates < self.update_after_step:
            self.num_updates += 1
            # During warmup, just copy student weights
            for ema_param, student_param in zip(
                self.ema_model.parameters(),
                student_model.parameters()
            ):
                ema_param.data.copy_(student_param.data)
            return
        
        self.num_updates += 1
        
        # EMA update: ema = decay * ema + (1 - decay) * student
        for ema_param, student_param in zip(
            self.ema_model.parameters(),
            student_model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                student_param.data, alpha=1 - self.decay
            )
    
    @torch.no_grad()
    def get_teacher_outputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Get outputs from EMA teacher model.
        
        Args:
            pixel_values: Input images
            input_ids: Text tokens
            attention_mask: Attention mask
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with 'logits' and 'hidden_states'
        """
        # Ensure EMA model is in eval mode
        self.ema_model.eval()
        
        # Forward pass through EMA model
        outputs = self.ema_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        # Extract logits
        logits = outputs.get('logits')
        
        # Extract hidden states if needed
        hidden_states = None
        if output_hidden_states:
            # Try to get hidden states from the forward outputs
            try:
                hs = outputs.get('hidden_states')
                if hs is not None:
                    # hidden_states should be a tuple/list of tensors (one per layer)
                    # Return the full tuple/list to match external teacher format
                    if isinstance(hs, (tuple, list)) and len(hs) > 0:
                        hidden_states = hs
                    else:
                        logger.warning(f"⚠️ EMA hidden_states has unexpected type: {type(hs)}")
                        hidden_states = None
                else:
                    logger.warning("⚠️ EMA outputs['hidden_states'] is None")
            except Exception as e:
                logger.warning(f"⚠️ Could not extract hidden states from EMA teacher: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                hidden_states = None
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
        }
    
    def state_dict(self):
        """Return EMA model state dict for saving."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'num_updates': self.num_updates,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.num_updates = state_dict.get('num_updates', 0)
        self.decay = state_dict.get('decay', self.decay)
