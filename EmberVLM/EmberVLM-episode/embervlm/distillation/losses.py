"""
Distillation Losses for EmberVLM

Various loss functions for knowledge distillation from teacher models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class DistillationLoss(nn.Module):
    """
    Standard knowledge distillation loss.

    Combines soft (KL divergence) and hard (cross-entropy) losses.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: str = 'mean',
    ):
        """
        Args:
            temperature: Softmax temperature for soft targets
            alpha: Weight for soft loss (1-alpha for hard loss)
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction=reduction)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits [B, T, V]
            teacher_logits: Teacher model logits [B, T, V]
            labels: Optional hard labels [B, T]

        Returns:
            Dictionary with loss components
        """
        # Soft loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Hard loss (cross-entropy with labels)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            hard_loss = self.ce_loss(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
            )

        # Combined loss
        if labels is not None:
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return {
            'loss': total_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss,
            'temperature': self.temperature,
        }


class HiddenStateDistillationLoss(nn.Module):
    """
    Hidden state alignment loss for distillation.

    Aligns student hidden states with teacher hidden states.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        normalize: bool = True,
        loss_type: str = 'mse',
    ):
        """
        Args:
            student_dim: Student hidden dimension
            teacher_dim: Teacher hidden dimension
            normalize: Whether to L2-normalize before computing loss
            loss_type: 'mse', 'cosine', or 'l1'
        """
        super().__init__()

        self.normalize = normalize
        self.loss_type = loss_type

        # Projection if dimensions differ
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hidden state alignment loss.

        Args:
            student_hidden: Student hidden states [B, T, D_s]
            teacher_hidden: Teacher hidden states [B, T, D_t]
            attention_mask: Optional mask for valid positions

        Returns:
            Alignment loss
        """
        # Project student to teacher dimension
        student_proj = self.projection(student_hidden)

        # Normalize if specified
        if self.normalize:
            student_proj = F.normalize(student_proj, dim=-1)
            teacher_hidden = F.normalize(teacher_hidden, dim=-1)

        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_proj, teacher_hidden, reduction='none')
        elif self.loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(student_proj, teacher_hidden, dim=-1)
            loss = loss.unsqueeze(-1)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(student_proj, teacher_hidden, reduction='none')
        else:
            loss = F.mse_loss(student_proj, teacher_hidden, reduction='none')

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


class AttentionTransferLoss(nn.Module):
    """
    Attention map transfer loss.

    Aligns student attention patterns with teacher attention patterns.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.

        Args:
            student_attention: Student attention [B, H_s, T, T]
            teacher_attention: Teacher attention [B, H_t, T, T]

        Returns:
            Attention transfer loss
        """
        # Average over heads
        student_avg = student_attention.mean(dim=1)
        teacher_avg = teacher_attention.mean(dim=1)

        # Normalize attention distributions
        if self.normalize:
            student_avg = student_avg / student_avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            teacher_avg = teacher_avg / teacher_avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # KL divergence
        loss = F.kl_div(
            student_avg.log().clamp(min=-100),
            teacher_avg,
            reduction='batchmean',
        )

        return loss


class FeatureMapDistillationLoss(nn.Module):
    """
    Feature map distillation for vision encoder.

    Aligns intermediate feature maps between student and teacher.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
    ):
        super().__init__()

        if student_channels != teacher_channels:
            self.adapter = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.adapter = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature map distillation loss.

        Args:
            student_features: Student feature maps [B, C_s, H, W]
            teacher_features: Teacher feature maps [B, C_t, H, W]

        Returns:
            Feature distillation loss
        """
        # Adapt student features
        student_adapted = self.adapter(student_features)

        # Resize if needed
        if student_adapted.shape[-2:] != teacher_features.shape[-2:]:
            student_adapted = F.interpolate(
                student_adapted,
                size=teacher_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Normalize spatially
        student_norm = F.normalize(student_adapted.flatten(2), dim=-1)
        teacher_norm = F.normalize(teacher_features.flatten(2), dim=-1)

        # MSE loss
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss


class CombinedDistillationLoss(nn.Module):
    """
    Combined distillation loss with multiple components.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        temperature: float = 2.0,
        alpha: float = 0.5,
        hidden_weight: float = 0.1,
        attention_weight: float = 0.1,
    ):
        super().__init__()

        self.logit_loss = DistillationLoss(temperature, alpha)
        self.hidden_loss = HiddenStateDistillationLoss(student_dim, teacher_dim)
        self.attention_loss = AttentionTransferLoss()

        self.hidden_weight = hidden_weight
        self.attention_weight = attention_weight

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.

        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            labels: Optional hard labels

        Returns:
            Dictionary with all loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_outputs['logits'].device)

        # Logit distillation
        logit_losses = self.logit_loss(
            student_outputs['logits'],
            teacher_outputs['logits'],
            labels,
        )
        losses.update(logit_losses)
        total_loss = total_loss + logit_losses['loss']

        # Hidden state distillation
        if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
            if student_outputs['hidden_states'] is not None and teacher_outputs['hidden_states'] is not None:
                student_hidden = student_outputs['hidden_states'][-1]
                teacher_hidden = teacher_outputs['hidden_states'][-1]

                hidden_loss = self.hidden_loss(student_hidden, teacher_hidden)
                losses['hidden_loss'] = hidden_loss
                total_loss = total_loss + self.hidden_weight * hidden_loss

        # Attention distillation
        if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
            if student_outputs['attentions'] is not None and teacher_outputs['attentions'] is not None:
                student_attn = student_outputs['attentions'][-1]
                teacher_attn = teacher_outputs['attentions'][-1]

                attn_loss = self.attention_loss(student_attn, teacher_attn)
                losses['attention_loss'] = attn_loss
                total_loss = total_loss + self.attention_weight * attn_loss

        losses['total_loss'] = total_loss
        return losses

