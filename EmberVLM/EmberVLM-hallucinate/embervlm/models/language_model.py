"""
TinyLLM Language Model Backbone for EmberVLM

A lightweight GPT-2 style language model with ~30M parameters,
optimized for multimodal fusion and edge deployment.

Supports loading pretrained weights from HuggingFace:
- tinyllm/30M-0.4: 30M parameter GPT-2 model trained on FineWeb + SHL sensor data
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass

# HuggingFace imports for pretrained model loading
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoConfig = None

# Default pretrained model
PRETRAINED_TINYLLM_MODEL = "tinyllm/30M-0.4"

# SmolLM pretrained models
PRETRAINED_SMOLLM_135M = "HuggingFaceTB/SmolLM-135M"

# Language backbone type constants
BACKBONE_TINYLLM = "tinyllm"
BACKBONE_SMOLLM_135M = "smollm_135m"


@dataclass
class TinyLLMConfig:
    """Configuration for TinyLLM model."""

    vocab_size: int = 50257  # GPT-2 vocabulary
    hidden_size: int = 384  # tinyllm/30M-0.4 uses 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6  # tinyllm/30M-0.4 uses 6 heads
    intermediate_size: int = 1536  # 4 * hidden_size
    hidden_act: str = "gelu_new"  # GPT-2 style
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True

    # Special token IDs
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256

    # Additional config
    use_qk_norm: bool = True
    tie_word_embeddings: bool = True

    # Pretrained model settings
    use_pretrained: bool = True
    pretrained_model_name: str = "tinyllm/30M-0.4"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'max_position_embeddings': self.max_position_embeddings,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'use_cache': self.use_cache,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'use_qk_norm': self.use_qk_norm,
            'tie_word_embeddings': self.tie_word_embeddings,
            'use_pretrained': self.use_pretrained,
            'pretrained_model_name': self.pretrained_model_name,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TinyLLMConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class QKNorm(nn.Module):
    """Query-Key normalization for training stability."""

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply normalization to query and key tensors.

        Args:
            query: [B, num_heads, seq_len, head_dim]
            key: [B, num_heads, seq_len, head_dim]
        """
        query = self.q_norm(query)
        key = self.k_norm(key)
        return query, key


class TinyAttention(nn.Module):
    """Multi-head self-attention for TinyLLM."""

    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        assert self.hidden_size % self.num_heads == 0

        # QKV projection
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # QK Normalization
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = QKNorm(self.hidden_size, self.num_heads)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_position_embeddings,
                       config.max_position_embeddings))
            .view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
            persistent=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for attention layer.

        Args:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value
            use_cache: Whether to return cached key/value
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Compute QKV
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=2)

        # Reshape to [B, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads,
                       self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        # Apply QK normalization
        if self.use_qk_norm:
            query, key = self.qk_norm(query, key)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask
        query_length = query.size(2)
        key_length = key.size(2)
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(
            causal_mask.bool(), attn_weights, mask_value)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1,
                                 dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TinyMLP(nn.Module):
    """MLP block for TinyLLM."""

    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TinyLLMBlock(nn.Module):
    """Transformer block for TinyLLM."""

    def __init__(self, config: TinyLLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = TinyAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TinyMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:

        # Self-attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return (hidden_states,) + outputs


class TinyLLM(nn.Module):
    """
    TinyLLM - A lightweight GPT-2 style language model.

    ~30M parameters optimized for multimodal fusion.
    """

    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

        # Transformer blocks
        self.h = nn.ModuleList([
            TinyLLMBlock(config, i) for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range /
                                math.sqrt(2 * config.num_hidden_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0,
                            std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0,
                            std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TinyLLM.

        Args:
            input_ids: Token IDs [B, seq_len]
            inputs_embeds: Input embeddings [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, seq_len]
            position_ids: Position IDs [B, seq_len]
            past_key_values: Cached key/values for generation
            use_cache: Whether to return cached key/values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        if input_ids is not None:
            batch_size, seq_len = input_ids.size()

            # CRITICAL: Validate and clamp input_ids before embedding lookup
            vocab_size = self.wte.weight.shape[0]
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()

            if max_token_id >= vocab_size or min_token_id < 0:
                import logging
                logger = logging.getLogger(__name__)
                if max_token_id >= vocab_size:
                    logger.warning(
                        f"⚠️ TinyLLM.forward: Token ID {max_token_id} >= vocab_size {vocab_size}. Clamping."
                    )
                if min_token_id < 0:
                    logger.warning(
                        f"⚠️ TinyLLM.forward: Negative token ID {min_token_id}. Clamping."
                    )
                input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)

            inputs_embeds = self.wte(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.size()
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        # Handle past key values
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Position embeddings
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * \
                torch.finfo(hidden_states.dtype).min

        # Forward through transformer blocks
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if use_cache:
                presents += (outputs[1],)

            if output_attentions:
                all_attentions += (outputs[2]
                                   if len(outputs) > 2 else outputs[-1],)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }


class TinyLLMForCausalLM(nn.Module):
    """TinyLLM with causal language modeling head."""

    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config
        self.transformer = TinyLLM(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.transformer.wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional language modeling loss.

        Args:
            labels: Target token IDs for computing loss
        """
        outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs['last_hidden_state']
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # CRITICAL FIX: Validate labels before cross_entropy to prevent CUDA index out of bounds
            # This is the ACTUAL location where the gather operation occurs that causes the crash
            vocab_size = shift_logits.size(-1)

            # Flatten for validation
            flat_labels = shift_labels.view(-1)

            # Check for invalid labels (excluding -100 ignore index)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                valid_labels = flat_labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"❌ CRITICAL in TinyLLMForCausalLM: Labels out of bounds! "
                        f"max={max_label}, min={min_label}, vocab_size={vocab_size}. Clamping to prevent crash."
                    )
                    # Clamp the flattened shift_labels: preserve -100, clamp rest to valid range
                    flat_labels = torch.where(
                        valid_mask,
                        torch.clamp(flat_labels, min=0, max=vocab_size - 1),
                        flat_labels
                    )
                    shift_labels = flat_labels.view(shift_labels.shape)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs['past_key_values'],
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions'],
            'last_hidden_state': hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.

        Args:
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        # Handle inputs_embeds by running first forward pass
        if inputs_embeds is not None and input_ids is None:
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']

            # Get first token
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True)

            generated = next_token
            batch_size = inputs_embeds.size(0)
        else:
            generated = input_ids
            past_key_values = None
            batch_size = input_ids.size(0)

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            if past_key_values is not None:
                # Only use last token when using cache
                model_input = generated[:, -1:]
            else:
                model_input = generated

            outputs = self.forward(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = outputs['logits'][:, -1, :] / temperature
            past_key_values = outputs['past_key_values']

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated


class TinyLLMBackbone(nn.Module):
    """
    TinyLLM Backbone wrapper for EmberVLM.

    Provides interface for multimodal fusion with the language model.
    """

    def __init__(
        self,
        config: Optional[TinyLLMConfig] = None,
        freeze_base: bool = True,
        unfreeze_last_layer: bool = True,
    ):
        super().__init__()

        if config is None:
            config = TinyLLMConfig()

        self.config = config
        self.model = TinyLLMForCausalLM(config)

        # Freeze/unfreeze parameters
        if freeze_base:
            self._freeze_base()

        if unfreeze_last_layer:
            self._unfreeze_last_layer()

    def _freeze_base(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        """Unfreeze the last transformer layer."""
        # Count trainable params before unfreezing
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        num_layers = len(self.model.transformer.h)
        
        for param in self.model.transformer.h[-1].parameters():
            param.requires_grad = True
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        
        # Count trainable params after unfreezing
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        newly_unfrozen = trainable_after - trainable_before
        
        print(f"✓ Unfrozen TinyLLM components: last decoder layer (layer {num_layers-1}/{num_layers}), final layernorm, lm_head")
        print(f"  Newly trainable parameters: {newly_unfrozen:,}")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.model.set_input_embeddings(new_embeddings)

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Get token embeddings with validation to prevent index out of bounds."""
        vocab_size = self.model.transformer.wte.weight.shape[0]

        # CRITICAL: Create a clean copy and validate BEFORE any CUDA operations
        input_ids = input_ids.clone()

        # Check for out-of-bounds tokens
        invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)

        if invalid_mask.any():
            import logging
            logger = logging.getLogger(__name__)

            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()
            num_invalid = invalid_mask.sum().item()

            logger.warning(
                f"⚠️ TinyLLMBackbone.embed_tokens: {num_invalid} invalid tokens. "
                f"Range: [{min_token_id}, {max_token_id}], Valid: [0, {vocab_size - 1}]. Replacing with 0."
            )

            # Replace invalid tokens with 0
            input_ids = torch.where(
                invalid_mask, torch.zeros_like(input_ids), input_ids)

            # Force CUDA sync
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)

        return self.model.transformer.wte(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,  # Accept and ignore extra kwargs like return_dict
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the language model."""
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.LongTensor:
        """
        Generate text with safe token ID validation.

        Wraps the base model's generate with validation to prevent
        out-of-bounds token IDs that cause CUDA errors.
        """
        vocab_size = self.model.transformer.wte.weight.shape[0]

        # Call base generate
        generated = self.model.generate(**kwargs)

        # CRITICAL: Validate and clamp all generated token IDs
        if isinstance(generated, torch.Tensor):
            max_id = generated.max().item()
            min_id = generated.min().item()

            if max_id >= vocab_size or min_id < 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"⚠️ TinyLLM.generate produced invalid tokens: "
                    f"Range [{min_id}, {max_id}], Valid [0, {vocab_size-1}]. Clamping."
                )
                generated = torch.clamp(generated, min=0, max=vocab_size - 1)

        return generated

    def get_hidden_size(self) -> int:
        """Return hidden size."""
        return self.config.hidden_size

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.config.vocab_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class PretrainedTinyLLMBackbone(nn.Module):
    """
    Pretrained TinyLLM Backbone wrapper for EmberVLM.

    Loads pretrained weights from HuggingFace Hub (tinyllm/30M-0.4).
    This model is a 30M parameter GPT-2 style model trained on:
    - FineWeb dataset (web text)
    - SHL sensor dataset (human activity data)

    Model specs:
    - Hidden size: 384
    - Layers: 6
    - Attention heads: 6
    - Vocab size: 50257 (GPT-2)
    - Context length: 1024
    """

    def __init__(
        self,
        model_name: str = PRETRAINED_TINYLLM_MODEL,
        freeze_base: bool = True,
        unfreeze_last_layer: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for PretrainedTinyLLMBackbone. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.torch_dtype = torch_dtype

        # Load pretrained model from HuggingFace
        print(f"Loading pretrained TinyLLM from {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Get config from loaded model
        self.hf_config = self.model.config

        # Create compatible TinyLLMConfig
        self.config = TinyLLMConfig(
            vocab_size=self.hf_config.vocab_size,
            hidden_size=self.hf_config.n_embd,
            num_hidden_layers=self.hf_config.n_layer,
            num_attention_heads=self.hf_config.n_head,
            intermediate_size=self.hf_config.n_inner if self.hf_config.n_inner else 4 *
            self.hf_config.n_embd,
            hidden_act="gelu_new",
            max_position_embeddings=self.hf_config.n_positions,
            use_pretrained=True,
            pretrained_model_name=model_name,
        )

        print(f"Loaded model with hidden_size={self.config.hidden_size}, "
              f"layers={self.config.num_hidden_layers}, "
              f"heads={self.config.num_attention_heads}")

        # Freeze/unfreeze parameters
        if freeze_base:
            self._freeze_base()

        if unfreeze_last_layer:
            self._unfreeze_last_layer()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _freeze_base(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        """Unfreeze the last transformer layer and output head."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Count trainable params before unfreezing
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        unfrozen_components = []
        
        # Try GPT-2 style model structure first
        try:
            transformer = getattr(self.model, 'transformer', None)
            if transformer is not None:
                # Standard GPT-2 structure
                layers = getattr(transformer, 'h', None)
                if layers is not None and len(layers) > 0:
                    num_layers = len(layers)
                    for param in layers[-1].parameters():
                        param.requires_grad = True
                    unfrozen_components.append(f'last decoder layer (layer {num_layers-1}/{num_layers})')
                    
                final_norm = getattr(transformer, 'ln_f', None)
                if final_norm is not None:
                    for param in final_norm.parameters():
                        param.requires_grad = True
                    unfrozen_components.append('final layernorm')
        except Exception as e:
            logger.error(f"❌ Exception accessing transformer: {e}")
            
        # Try alternative structure (model.model.layers)
        if not unfrozen_components:
            try:
                base_model = getattr(self.model, 'model', None)
                if base_model is not None:
                    layers = getattr(base_model, 'layers', None)
                    if layers is not None and len(layers) > 0:
                        num_layers = len(layers)
                        for param in layers[-1].parameters():
                            param.requires_grad = True
                        unfrozen_components.append(f'last decoder layer (layer {num_layers-1}/{num_layers})')
                        
                    final_norm = getattr(base_model, 'norm', None)
                    if final_norm is not None:
                        for param in final_norm.parameters():
                            param.requires_grad = True
                        unfrozen_components.append('final norm')
            except Exception as e:
                logger.error(f"❌ Exception accessing model.model.layers: {e}")

        # Unfreeze LM head
        try:
            lm_head = getattr(self.model, 'lm_head', None)
            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad = True
                unfrozen_components.append('lm_head')
        except Exception as e:
            logger.error(f"❌ Exception unfreezing lm_head: {e}")
        
        # Count trainable params after unfreezing
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        newly_unfrozen = trainable_after - trainable_before
        
        if newly_unfrozen > 0:
            print(f"✓ Unfrozen pretrained LLM components: {', '.join(unfrozen_components)}")
            print(f"  Newly trainable parameters: {newly_unfrozen:,}")
        else:
            logger.error("❌ CRITICAL: Failed to unfreeze any components!")
            logger.error(f"   Model type: {type(self.model)}")
            logger.error(f"   Available attributes: {list(vars(self.model).keys())[:10]}")

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.model.set_input_embeddings(new_embeddings)

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Get token embeddings with validation to prevent index out of bounds."""
        embedding_layer = self.model.get_input_embeddings()
        vocab_size = embedding_layer.weight.shape[0]
        original_device = input_ids.device

        # CRITICAL FIX: Round-trip through CPU for guaranteed safety
        # This completely avoids CUDA async issues with tensor modifications
        input_ids_cpu = input_ids.detach().cpu()

        # Check and fix any invalid values on CPU (guaranteed synchronous)
        max_val = input_ids_cpu.max().item()
        min_val = input_ids_cpu.min().item()

        if max_val >= vocab_size or min_val < 0:
            import logging
            logger = logging.getLogger(__name__)
            num_invalid = ((input_ids_cpu >= vocab_size) |
                           (input_ids_cpu < 0)).sum().item()
            logger.warning(
                f"⚠️ embed_tokens: {num_invalid} invalid tokens. "
                f"Range: [{min_val}, {max_val}], Valid: [0, {vocab_size - 1}]. Clamping."
            )
            # Clamp on CPU - guaranteed synchronous
            input_ids_cpu = torch.clamp(
                input_ids_cpu, min=0, max=vocab_size - 1)

        # Move back to original device with blocking transfer
        input_ids_safe = input_ids_cpu.to(original_device, non_blocking=False)

        # Final sync before embedding lookup
        if original_device.type == 'cuda':
            torch.cuda.synchronize(original_device)

        return embedding_layer(input_ids_safe)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,  # Accept and ignore extra kwargs like return_dict
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the language model."""
        # CRITICAL: Validate and clamp input_ids and labels to prevent CUDA index out of bounds
        vocab_size = self.model.get_input_embeddings().weight.shape[0]

        if input_ids is not None:
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()

            if max_token_id >= vocab_size or min_token_id < 0:
                import logging
                logger = logging.getLogger(__name__)
                if max_token_id >= vocab_size:
                    logger.warning(
                        f"⚠️ PretrainedTinyLLMBackbone.forward: input_ids max {max_token_id} >= vocab_size {vocab_size}. Clamping."
                    )
                if min_token_id < 0:
                    logger.warning(
                        f"⚠️ PretrainedTinyLLMBackbone.forward: input_ids min {min_token_id} < 0. Clamping."
                    )
                input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)

        if labels is not None:
            # Validate labels (skip -100 which is ignore index)
            valid_mask = labels != -100
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    if max_label >= vocab_size:
                        logger.warning(
                            f"⚠️ PretrainedTinyLLMBackbone.forward: labels max {max_label} >= vocab_size {vocab_size}. Clamping."
                        )
                    if min_label < 0:
                        logger.warning(
                            f"⚠️ PretrainedTinyLLMBackbone.forward: labels min {min_label} < 0. Clamping."
                        )
                    # Clamp labels: preserve -100, clamp everything else
                    labels = torch.where(
                        valid_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )

        # CRITICAL FIX: Compute loss ourselves instead of letting HuggingFace do it
        # This ensures proper validation right before cross_entropy
        compute_loss_ourselves = labels is not None
        labels_for_loss = labels  # Save for later

        # Always request hidden states to get last_hidden_state
        # Pass labels=None to HuggingFace model, we'll compute loss ourselves
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,  # Don't let HF compute loss - we do it ourselves with validation
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states for last_hidden_state
            return_dict=True,
        )

        # Compute loss ourselves with proper validation
        loss = None
        if compute_loss_ourselves and labels_for_loss is not None:
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_for_loss[..., 1:].contiguous()

            # CRITICAL: Final validation right before cross_entropy
            actual_vocab_size = shift_logits.size(-1)
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100

            if valid_mask.any():
                valid_labels = flat_labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= actual_vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"❌ CRITICAL in PretrainedTinyLLMBackbone loss computation: "
                        f"max={max_label}, min={min_label}, logits_vocab={actual_vocab_size}. Clamping."
                    )
                    flat_labels = torch.where(
                        valid_mask,
                        torch.clamp(flat_labels, min=0,
                                    max=actual_vocab_size - 1),
                        flat_labels
                    )
                    shift_labels = flat_labels.view(shift_labels.shape)

            loss = F.cross_entropy(
                shift_logits.view(-1, actual_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Convert to dict format compatible with rest of EmberVLM
        return {
            'loss': loss,
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions,
            'last_hidden_state': outputs.hidden_states[-1] if outputs.hidden_states else None,
        }

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.LongTensor:
        """Generate text."""
        return self.model.generate(**kwargs)

    def get_hidden_size(self) -> int:
        """Return hidden size."""
        return self.config.hidden_size

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.config.vocab_size

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for special tokens."""
        self.model.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        # Also update hf_config if it exists
        if hasattr(self, 'hf_config'):
            self.hf_config.vocab_size = new_num_tokens

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class SmolLMBackbone(nn.Module):
    """
    SmolLM Backbone wrapper for EmberVLM.

    Loads pretrained weights from HuggingFace Hub (SmolLM-135M or SmolLM-360M).
    SmolLM is a family of small language models from HuggingFace.

    Supported models:
    - HuggingFaceTB/SmolLM-135M: 135M parameters
    - HuggingFaceTB/SmolLM-360M: 360M parameters

    Interface is compatible with TinyLLMBackbone and PretrainedTinyLLMBackbone.
    """

    def __init__(
        self,
        model_name: str = PRETRAINED_SMOLLM_135M,
        freeze_base: bool = True,
        unfreeze_last_layer: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for SmolLMBackbone. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.torch_dtype = torch_dtype

        # Load pretrained model from HuggingFace
        # Note: trust_remote_code=False for security - SmolLM doesn't require custom code
        print(f"Loading pretrained SmolLM from {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
        )

        # Get config from loaded model
        self.hf_config = self.model.config

        # Create compatible TinyLLMConfig for interface consistency
        self.config = TinyLLMConfig(
            vocab_size=self.hf_config.vocab_size,
            hidden_size=self.hf_config.hidden_size,
            num_hidden_layers=self.hf_config.num_hidden_layers,
            num_attention_heads=self.hf_config.num_attention_heads,
            intermediate_size=getattr(
                self.hf_config, 'intermediate_size', 4 * self.hf_config.hidden_size),
            hidden_act="silu",  # SmolLM uses SiLU activation
            max_position_embeddings=getattr(
                self.hf_config, 'max_position_embeddings', 2048),
            use_pretrained=True,
            pretrained_model_name=model_name,
        )

        print(f"Loaded SmolLM with hidden_size={self.config.hidden_size}, "
              f"layers={self.config.num_hidden_layers}, "
              f"heads={self.config.num_attention_heads}, "
              f"vocab_size={self.config.vocab_size}")

        # Freeze/unfreeze parameters
        if freeze_base:
            self._freeze_base()

        if unfreeze_last_layer:
            self._unfreeze_last_layer()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _freeze_base(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        """Unfreeze the last transformer layer and output head."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Count trainable params before unfreezing
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        unfrozen_components = []
        
        # SmolLM uses LlamaForCausalLM structure: model.model.layers
        try:
            # Direct access with verification
            base_model = getattr(self.model, 'model', None)
            if base_model is not None:
                layers = getattr(base_model, 'layers', None)
                if layers is not None and len(layers) > 0:
                    # Unfreeze last decoder layer
                    num_layers = len(layers)
                    last_layer = layers[-1]
                    unfrozen_count = 0
                    for param in last_layer.parameters():
                        param.requires_grad = True
                        unfrozen_count += 1
                    unfrozen_components.append(f'last decoder layer (layer {num_layers-1}/{num_layers}, {unfrozen_count} params)')
                    
                    # Unfreeze final norm
                    final_norm = getattr(base_model, 'norm', None)
                    if final_norm is not None:
                        for param in final_norm.parameters():
                            param.requires_grad = True
                        unfrozen_components.append('final norm')
                else:
                    logger.error(f"❌ SmolLM structure issue: layers={'exists' if layers is not None else 'None'}, len={len(layers) if layers else 0}")
            else:
                logger.error(f"❌ SmolLM missing 'model' attribute. Type: {type(self.model)}")
        except Exception as e:
            logger.error(f"❌ Exception while unfreezing SmolLM layers: {e}")
            import traceback
            traceback.print_exc()

        # Unfreeze LM head
        try:
            lm_head = getattr(self.model, 'lm_head', None)
            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad = True
                unfrozen_components.append('lm_head')
            else:
                logger.error("❌ SmolLM missing 'lm_head' attribute")
        except Exception as e:
            logger.error(f"❌ Exception while unfreezing lm_head: {e}")
        
        # Count trainable params after unfreezing
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        newly_unfrozen = trainable_after - trainable_before
        
        if newly_unfrozen > 0:
            print(f"✓ Unfrozen SmolLM components: {', '.join(unfrozen_components)}")
            print(f"  Newly trainable parameters: {newly_unfrozen:,}")
        else:
            logger.error("❌ CRITICAL: _unfreeze_last_layer did not unfreeze any parameters!")
            logger.error(f"   Model type: {type(self.model)}")
            logger.error(f"   Has 'model' attr: {hasattr(self.model, 'model')}")
            if hasattr(self.model, 'model'):
                logger.error(f"   model.model type: {type(self.model.model)}")
                logger.error(f"   Has 'layers': {hasattr(self.model.model, 'layers')}")
            logger.error(f"   Has 'lm_head': {hasattr(self.model, 'lm_head')}")

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.model.set_input_embeddings(new_embeddings)

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Get token embeddings with validation to prevent index out of bounds."""
        embedding_layer = self.model.get_input_embeddings()
        vocab_size = embedding_layer.weight.shape[0]
        original_device = input_ids.device

        # CRITICAL FIX: Round-trip through CPU for guaranteed safety
        input_ids_cpu = input_ids.detach().cpu()

        # Check and fix any invalid values on CPU (guaranteed synchronous)
        max_val = input_ids_cpu.max().item()
        min_val = input_ids_cpu.min().item()

        if max_val >= vocab_size or min_val < 0:
            import logging
            logger = logging.getLogger(__name__)
            num_invalid = ((input_ids_cpu >= vocab_size) |
                           (input_ids_cpu < 0)).sum().item()
            logger.warning(
                f"⚠️ SmolLMBackbone.embed_tokens: {num_invalid} invalid tokens. "
                f"Range: [{min_val}, {max_val}], Valid: [0, {vocab_size - 1}]. Clamping."
            )
            input_ids_cpu = torch.clamp(
                input_ids_cpu, min=0, max=vocab_size - 1)

        # Move back to original device with blocking transfer
        input_ids_safe = input_ids_cpu.to(original_device, non_blocking=False)

        # Final sync before embedding lookup
        if original_device.type == 'cuda':
            torch.cuda.synchronize(original_device)

        return embedding_layer(input_ids_safe)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the language model."""
        vocab_size = self.model.get_input_embeddings().weight.shape[0]

        # Validate input_ids
        if input_ids is not None:
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()

            if max_token_id >= vocab_size or min_token_id < 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"⚠️ SmolLMBackbone.forward: input_ids range [{min_token_id}, {max_token_id}] "
                    f"exceeds vocab_size {vocab_size}. Clamping."
                )
                input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)

        # Validate labels
        if labels is not None:
            valid_mask = labels != -100
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"⚠️ SmolLMBackbone.forward: labels range [{min_label}, {max_label}] "
                        f"exceeds vocab_size {vocab_size}. Clamping."
                    )
                    labels = torch.where(
                        valid_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )

        # Compute loss ourselves for validation
        compute_loss_ourselves = labels is not None
        labels_for_loss = labels

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,  # Compute loss ourselves
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states
            return_dict=True,
        )

        # Compute loss ourselves with validation
        loss = None
        if compute_loss_ourselves and labels_for_loss is not None:
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_for_loss[..., 1:].contiguous()

            actual_vocab_size = shift_logits.size(-1)
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100

            if valid_mask.any():
                valid_labels = flat_labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= actual_vocab_size or min_label < 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"❌ SmolLMBackbone loss: max={max_label}, min={min_label}, "
                        f"logits_vocab={actual_vocab_size}. Clamping."
                    )
                    flat_labels = torch.where(
                        valid_mask,
                        torch.clamp(flat_labels, min=0,
                                    max=actual_vocab_size - 1),
                        flat_labels
                    )
                    shift_labels = flat_labels.view(shift_labels.shape)

            loss = F.cross_entropy(
                shift_logits.view(-1, actual_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Return dict format compatible with EmberVLM
        return {
            'loss': loss,
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions,
            'last_hidden_state': outputs.hidden_states[-1] if outputs.hidden_states else None,
        }

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.LongTensor:
        """Generate text."""
        return self.model.generate(**kwargs)

    def get_hidden_size(self) -> int:
        """Return hidden size."""
        return self.config.hidden_size

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.config.vocab_size

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for special tokens."""
        self.model.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        if hasattr(self, 'hf_config'):
            self.hf_config.vocab_size = new_num_tokens

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def create_language_backbone(
    use_pretrained: bool = True,
    model_name: str = PRETRAINED_TINYLLM_MODEL,
    backbone_type: str = BACKBONE_TINYLLM,
    config: Optional[TinyLLMConfig] = None,
    freeze_base: bool = True,
    unfreeze_last_layer: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Union[PretrainedTinyLLMBackbone, TinyLLMBackbone, SmolLMBackbone]:
    """
    Factory function to create language backbone.

    Args:
        use_pretrained: If True, load pretrained weights from HuggingFace
        model_name: HuggingFace model name (overrides backbone_type if specified)
        backbone_type: Backbone type selector ('tinyllm', 'smollm_135m', 'smollm_360m')
        config: Custom TinyLLMConfig (only used if use_pretrained=False)
        freeze_base: Whether to freeze base model parameters
        unfreeze_last_layer: Whether to unfreeze last layer for fine-tuning
        torch_dtype: Data type for model weights

    Returns:
        Language model backbone (pretrained or from scratch)
    """
    # Determine model name from backbone_type if using default
    if model_name == PRETRAINED_TINYLLM_MODEL:
        if backbone_type == BACKBONE_SMOLLM_135M:
            model_name = PRETRAINED_SMOLLM_135M
        # else keep default PRETRAINED_TINYLLM_MODEL

    if use_pretrained and HF_AVAILABLE:
        # Select backbone class based on model name
        if 'SmolLM' in model_name or backbone_type == BACKBONE_SMOLLM_135M:
            return SmolLMBackbone(
                model_name=model_name,
                freeze_base=freeze_base,
                unfreeze_last_layer=unfreeze_last_layer,
                torch_dtype=torch_dtype,
            )
        else:
            return PretrainedTinyLLMBackbone(
                model_name=model_name,
                freeze_base=freeze_base,
                unfreeze_last_layer=unfreeze_last_layer,
                torch_dtype=torch_dtype,
            )
    else:
        if not HF_AVAILABLE and use_pretrained:
            print(
                "Warning: transformers not available, falling back to random initialization")
        return TinyLLMBackbone(
            config=config,
            freeze_base=freeze_base,
            unfreeze_last_layer=unfreeze_last_layer,
        )
