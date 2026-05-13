"""
Reasoning Module for EmberVLM

Implements Chain-of-Thought reasoning capabilities inspired by DeepSeek-R1.
Includes reasoning heads for robot selection and action planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple


class ReasoningAttention(nn.Module):
    """
    Self-attention layer for reasoning heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Compute QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: [batch_size, num_heads, seq_len, seq_len]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            # Need to broadcast to [batch_size, num_heads, seq_len, seq_len]
            if attention_mask.dim() == 2:
                # Convert [B, seq_len] to [B, 1, 1, seq_len] then broadcast
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Create causal mask: [1, 1, seq_len, seq_len]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype) * float('-inf'),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # Combine masks: broadcast attention_mask and add causal mask
            mask = attention_mask + causal_mask
            attn = attn + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)

        return out


class ReasoningBlock(nn.Module):
    """
    Transformer block for reasoning module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = ReasoningAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ReasoningHead(nn.Module):
    """
    Reasoning head for generating chain-of-thought reasoning.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_layers: int = 2,
        num_heads: int = 4,
        num_reasoning_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_reasoning_steps = num_reasoning_steps

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Reasoning blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Step embeddings
        self.step_embeddings = nn.Embedding(num_reasoning_steps, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate reasoning for each step.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            step_indices: Which reasoning step [B] or None for all steps
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing reasoning outputs
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to reasoning space
        x = self.input_proj(hidden_states)

        # Add step embeddings if provided
        if step_indices is not None:
            step_emb = self.step_embeddings(step_indices)  # [B, hidden_dim]
            x = x + step_emb.unsqueeze(1)

        # Forward through reasoning blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Project back
        output = self.output_proj(x)
        output = self.norm(output)

        return {
            'reasoning_output': output,
            'reasoning_hidden': x,
        }


class RobotSelectionHead(nn.Module):
    """
    Head for selecting optimal robot(s) from fleet with top-N ranking.

    Supports:
    - Single robot selection (classification)
    - Multi-robot selection (multi-label)
    - Top-N robot ranking with scores
    - Reasoning-aware selection
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_robots: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_robots = num_robots

        # Feature extraction with more capacity
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Robot classification (single robot)
        self.classifier = nn.Linear(hidden_dim, num_robots)

        # Multi-robot selection (can select multiple robots)
        self.multi_robot_classifier = nn.Linear(hidden_dim, num_robots)

        # Confidence estimation (per-robot confidence)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_robots),
            nn.Sigmoid(),
        )

        # Robot-specific reasoning context
        self.robot_context = nn.Parameter(torch.randn(num_robots, hidden_dim) * 0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Select robot(s) from fleet with ranking.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            attention_mask: Optional attention mask
            top_k: Number of top robots to return in ranking

        Returns:
            Dictionary containing robot selection outputs with top-N ranking
        """
        # Pool across sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Extract features
        features = self.feature_net(pooled)

        # Robot logits (single selection)
        robot_logits = self.classifier(features)
        robot_probs = F.softmax(robot_logits, dim=-1)

        # Multi-robot logits (multi-label selection)
        multi_robot_logits = self.multi_robot_classifier(features)
        multi_robot_probs = torch.sigmoid(multi_robot_logits)

        # Per-robot confidence scores
        confidence = self.confidence(features)

        # Top-K robot selection with scores
        top_k = min(top_k, self.num_robots)
        top_scores, top_indices = torch.topk(robot_probs, top_k, dim=-1)

        # Compute robot-specific attention for reasoning
        # This helps explain WHY each robot was selected
        batch_size = features.size(0)
        robot_attention = torch.einsum('bh,rh->br', features, self.robot_context)
        robot_attention = F.softmax(robot_attention, dim=-1)

        return {
            'robot_logits': robot_logits,
            'robot_probs': robot_probs,
            'multi_robot_logits': multi_robot_logits,
            'multi_robot_probs': multi_robot_probs,
            'confidence': confidence,
            'top_k_indices': top_indices,  # [B, top_k]
            'top_k_scores': top_scores,    # [B, top_k]
            'robot_attention': robot_attention,  # [B, num_robots] - for interpretability
            'features': features,
        }


class ActionPlanningHead(nn.Module):
    """
    Head for generating action plans.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        max_plan_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_plan_steps = max_plan_steps

        # Plan encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Step-wise plan generation
        self.step_generator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Plan step projection
        self.step_proj = nn.Linear(hidden_dim, input_dim)

        # Plan coherence scoring
        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        robot_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate action plan.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            robot_features: Optional robot selection features [B, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing action plan outputs
        """
        batch_size = hidden_states.size(0)

        # Pool and encode
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        encoded = self.encoder(pooled)

        # Combine with robot features if provided
        if robot_features is not None:
            encoded = encoded + robot_features

        # Generate plan steps
        # Initialize GRU hidden state
        h0 = encoded.unsqueeze(0).repeat(2, 1, 1)

        # Create input sequence for autoregressive generation
        plan_input = encoded.unsqueeze(1).repeat(1, self.max_plan_steps, 1)

        plan_steps, _ = self.step_generator(plan_input, h0)
        plan_steps = self.step_proj(plan_steps)

        # Compute coherence score
        first_last = torch.cat([
            plan_steps[:, 0, :encoded.size(-1)],
            plan_steps[:, -1, :encoded.size(-1)]
        ], dim=-1)
        coherence = self.coherence_scorer(first_last)

        return {
            'plan_steps': plan_steps,
            'plan_hidden': encoded,
            'coherence_score': coherence,
        }


class ReasoningModule(nn.Module):
    """
    Complete reasoning module for EmberVLM.

    Combines chain-of-thought reasoning with robot selection
    and action planning capabilities.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_reasoning_layers: int = 2,
        num_reasoning_heads: int = 4,
        num_reasoning_steps: int = 4,
        num_robots: int = 5,
        max_plan_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_reasoning_steps = num_reasoning_steps

        # Reasoning head
        self.reasoning_head = ReasoningHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_reasoning_layers,
            num_heads=num_reasoning_heads,
            num_reasoning_steps=num_reasoning_steps,
            dropout=dropout,
        )

        # Robot selection head
        self.robot_head = RobotSelectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_robots=num_robots,
            dropout=dropout,
        )

        # Action planning head
        self.action_head = ActionPlanningHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_plan_steps=max_plan_steps,
            dropout=dropout,
        )

        # Special token embeddings
        self.special_tokens = nn.ParameterDict({
            'reasoning_start': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'reasoning_end': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'robot_selection': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'action_plan': nn.Parameter(torch.randn(1, input_dim) * 0.02),
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generate_reasoning: bool = True,
        select_robot: bool = True,
        plan_actions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete reasoning forward pass.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            attention_mask: Optional attention mask
            generate_reasoning: Whether to generate reasoning chain
            select_robot: Whether to perform robot selection
            plan_actions: Whether to generate action plan

        Returns:
            Dictionary containing all reasoning outputs
        """
        outputs = {}

        # Generate reasoning chain
        if generate_reasoning:
            reasoning_outputs = []
            current_hidden = hidden_states

            for step in range(self.num_reasoning_steps):
                step_idx = torch.full(
                    (hidden_states.size(0),),
                    step,
                    dtype=torch.long,
                    device=hidden_states.device
                )

                step_output = self.reasoning_head(
                    current_hidden,
                    step_indices=step_idx,
                    attention_mask=attention_mask,
                )
                reasoning_outputs.append(step_output['reasoning_output'])

                # Update hidden states with reasoning output
                current_hidden = current_hidden + 0.1 * step_output['reasoning_output']

            outputs['reasoning_chain'] = torch.stack(reasoning_outputs, dim=1)  # [B, steps, seq, dim]
            outputs['reasoning_hidden'] = current_hidden
        else:
            current_hidden = hidden_states

        # Robot selection
        if select_robot:
            robot_outputs = self.robot_head(current_hidden, attention_mask)
            outputs.update({
                'robot_logits': robot_outputs['robot_logits'],
                'robot_probs': robot_outputs['robot_probs'],
                'multi_robot_logits': robot_outputs['multi_robot_logits'],
                'multi_robot_probs': robot_outputs['multi_robot_probs'],
                'robot_confidence': robot_outputs['confidence'],
                'top_k_indices': robot_outputs['top_k_indices'],
                'top_k_scores': robot_outputs['top_k_scores'],
                'robot_attention': robot_outputs['robot_attention'],
            })
            robot_features = robot_outputs['features']
        else:
            robot_features = None

        # Action planning
        if plan_actions:
            action_outputs = self.action_head(
                current_hidden,
                robot_features=robot_features,
                attention_mask=attention_mask,
            )
            outputs.update({
                'plan_steps': action_outputs['plan_steps'],
                'plan_coherence': action_outputs['coherence_score'],
            })

        return outputs

    def get_special_token_embedding(self, token_name: str) -> torch.Tensor:
        """Get embedding for a special token."""
        return self.special_tokens[token_name]

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in reasoning module."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in robot selection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This helps prevent the model from collapsing to always predicting
    the majority class (e.g., Drone).
    """

    def __init__(
        self,
        num_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Class weights - can be set dynamically based on class distribution
        if alpha is None:
            # Default balanced weights
            self.register_buffer('alpha', torch.ones(num_classes))
        else:
            self.register_buffer('alpha', alpha)

    def set_class_weights(self, class_counts: torch.Tensor):
        """Set class weights inversely proportional to class frequency."""
        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (self.num_classes * class_counts.clamp(min=1))
        # Normalize so mean weight is 1
        weights = weights / weights.mean()
        self.alpha = weights.to(self.alpha.device)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [B, num_classes]
            targets: Class indices [B]
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probabilities for target class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Class weight
        alpha_t = self.alpha.gather(0, targets)

        # Combined loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ReasoningLoss(nn.Module):
    """
    Loss functions for reasoning module with focal loss for class balancing.
    """

    def __init__(
        self,
        num_robots: int = 5,
        reasoning_weight: float = 1.0,
        robot_weight: float = 1.0,
        action_weight: float = 1.0,
        consistency_weight: float = 0.5,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.reasoning_weight = reasoning_weight
        self.robot_weight = robot_weight
        self.action_weight = action_weight
        self.consistency_weight = consistency_weight
        self.use_focal_loss = use_focal_loss

        # Use focal loss for better class balancing
        if use_focal_loss:
            self.robot_criterion = FocalLoss(
                num_classes=num_robots,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            self.robot_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.action_criterion = nn.MSELoss()

        # Multi-robot loss (binary cross entropy for multi-hot targets)
        self.multi_robot_criterion = nn.BCEWithLogitsLoss()

    def set_class_weights(self, class_counts: torch.Tensor):
        """Set class weights for focal loss."""
        if self.use_focal_loss and hasattr(self.robot_criterion, 'set_class_weights'):
            self.robot_criterion.set_class_weights(class_counts)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reasoning losses.

        Args:
            outputs: Model outputs
            targets: Target values

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0

        # Robot selection loss (single robot)
        if 'robot_logits' in outputs and 'robot_target' in targets:
            robot_loss = self.robot_criterion(
                outputs['robot_logits'],
                targets['robot_target']
            )
            losses['robot_loss'] = robot_loss
            total_loss = total_loss + self.robot_weight * robot_loss

        # Multi-robot selection loss (multiple robots)
        if 'multi_robot_logits' in outputs and 'multi_robot_target' in targets:
            multi_loss = self.multi_robot_criterion(
                outputs['multi_robot_logits'],
                targets['multi_robot_target'].float()
            )
            losses['multi_robot_loss'] = multi_loss
            total_loss = total_loss + self.robot_weight * multi_loss

        # Action planning loss
        if 'plan_steps' in outputs and 'action_target' in targets:
            action_loss = self.action_criterion(
                outputs['plan_steps'],
                targets['action_target']
            )
            losses['action_loss'] = action_loss
            total_loss = total_loss + self.action_weight * action_loss

        # Reasoning consistency loss
        if 'reasoning_chain' in outputs:
            # Encourage smooth transitions between reasoning steps
            reasoning_chain = outputs['reasoning_chain']
            if reasoning_chain.size(1) > 1:
                step_diffs = reasoning_chain[:, 1:] - reasoning_chain[:, :-1]
                consistency_loss = torch.mean(step_diffs ** 2)
                losses['consistency_loss'] = consistency_loss
                total_loss = total_loss + self.consistency_weight * consistency_loss

        losses['total_loss'] = total_loss
        return losses

