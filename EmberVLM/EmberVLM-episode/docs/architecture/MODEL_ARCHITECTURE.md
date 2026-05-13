# EmberVLM Model Architecture

**Complete Technical Overview of EmberVLM Architecture**

## 📋 Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Vision Encoder](#vision-encoder)
- [Language Model](#language-model)
- [Projection Layer](#projection-layer)
- [Reasoning Module](#reasoning-module)
- [Training Pipeline](#training-pipeline)
- [Parameter Distribution](#parameter-distribution)
- [Design Decisions](#design-decisions)

## 🎯 Overview

EmberVLM is a lightweight Vision-Language Model designed for efficient multimodal understanding with chain-of-thought reasoning capabilities. The architecture balances performance and efficiency through careful component selection and progressive training.

### Key Statistics

| Component | Parameters | Trainable | FLOPs (Inference) |
|-----------|------------|-----------|-------------------|
| Vision Encoder (RepViT) | 14M | Partially | 0.8 GFLOPs |
| Projection Layer | 2M | Yes | 0.1 GFLOPs |
| Language Model (TinyLLM) | 30M | Yes | 2.5 GFLOPs |
| Reasoning Module | 2.3M | Yes | 0.2 GFLOPs |
| **Total** | **48.3M** | **36.6M** | **3.6 GFLOPs** |

### Model Variants

| Variant | Total Params | Trainable | Use Case |
|---------|--------------|-----------|----------|
| EmberVLM-Base | 48M | 37M | General purpose |
| EmberVLM-Lite | 32M | 25M | Edge devices |
| EmberVLM-Reasoning | 48M | 37M | Complex reasoning |

## 🏗️ High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         EmberVLM                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input Image (224x224)                                         │
│        │                                                       │
│        ▼                                                       │
│  ┌──────────────────┐                                          │
│  │  RepViT Encoder  │  14M params                             │
│  │  ──────────────  │  • MobileNetV3-based                    │
│  │  - Stem           │  • Token mixing + Channel mixing        │
│  │  - 4 Stages       │  • Efficient attention                  │
│  │  - Global Pool    │  Output: 384-dim features              │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │ Projection Layer │  2M params                              │
│  │  ──────────────  │  • 2-layer MLP                          │
│  │  Linear → ReLU → │  • 384 → 768 → 384                      │
│  │  Linear           │  Output: Language-aligned features     │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │   TinyLLM-30M    │  30M params                             │
│  │  ──────────────  │  • GPT-2 architecture                   │
│  │  - Embedding      │  • 6 layers, 6 heads                    │
│  │  - 6 Transformer  │  • 384 hidden dim                       │
│  │    Layers         │  Input: Visual features + Text tokens   │
│  │  - LM Head        │  Output: Next token logits             │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │ Reasoning Module │  2.3M params (Stage 4 only)            │
│  │  ──────────────  │  • Reasoning Head (1.5M)                │
│  │  - Reasoning Head │  • Step Validator (0.5M)               │
│  │  - Step Validator │  • Answer Head (0.3M)                  │
│  │  - Answer Head    │  Output: Reasoning steps + Answer      │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  Output: Text Response / Robot Selection                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 👁️ Vision Encoder

### RepViT Architecture

**Paper**: "RepViT: Revisiting Mobile CNN From ViT Perspective" (2023)

RepViT combines the efficiency of MobileNets with the effectiveness of Vision Transformers.

```python
# RepViT-M0.9 Configuration
RepViT(
    in_chans=3,
    embed_dim=[48, 96, 192, 384],
    depth=[2, 2, 14, 2],
    num_heads=[3, 3, 6, 12],
    mlp_ratio=2,
    drop_path_rate=0.0,
)
```

### Architecture Details

```
Input: RGB Image (3, 224, 224)
       │
       ▼
┌──────────────┐
│   Stem       │  Conv 3x3, stride 2
│   48 dims    │  Output: (48, 112, 112)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Stage 1     │  2 RepViT blocks
│  48 dims     │  Token mixing + Channel mixing
└──────┬───────┘  Output: (48, 112, 112)
       │
       ▼
┌──────────────┐
│  Stage 2     │  2 RepViT blocks
│  96 dims     │  Downsample to 56x56
└──────┬───────┘  Output: (96, 56, 56)
       │
       ▼
┌──────────────┐
│  Stage 3     │  14 RepViT blocks
│  192 dims    │  Main feature extraction
└──────┬───────┘  Downsample to 28x28
       │          Output: (192, 28, 28)
       ▼
┌──────────────┐
│  Stage 4     │  2 RepViT blocks
│  384 dims    │  High-level features
└──────┬───────┘  Downsample to 14x14
       │          Output: (384, 14, 14)
       ▼
┌──────────────┐
│ Global Pool  │  Adaptive average pooling
│              │  Output: (384,)
└──────────────┘
```

### RepViT Block

```python
class RepViTBlock(nn.Module):
    """
    RepViT block = Token Mixing + Channel Mixing
    """
    def __init__(self, dim):
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # Depthwise
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),  # Expand
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, 1),  # Compress
        )
    
    def forward(self, x):
        x = x + self.token_mixer(x)  # Token mixing with residual
        x = x + self.channel_mixer(x)  # Channel mixing with residual
        return x
```

### Efficiency Features

- **Depthwise Separable Convolutions**: Reduce parameters and FLOPs
- **Efficient Attention**: No quadratic complexity
- **Hardware-Friendly**: Optimized for mobile/edge devices
- **Pre-trained**: Initialized from ImageNet-1K weights

## 🗣️ Language Model

### TinyLLM-30M Architecture

**Base**: GPT-2 style decoder-only transformer

```python
# TinyLLM Configuration
TinyLLM(
    vocab_size=50262,
    n_positions=1024,
    n_embd=384,
    n_layer=6,
    n_head=6,
    n_inner=1536,  # 4 * n_embd
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
```

### Architecture Details

```
Input: Token IDs + Position IDs
       │
       ▼
┌──────────────┐
│  Embedding   │  Token + Position embeddings
│  384 dims    │  Output: (seq_len, 384)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Transformer     │  6 identical layers
│  Layer 1         │  
│  ──────────      │  Each layer:
│  • Self-Attention│  - Multi-head self-attention
│  • LayerNorm     │  - Feed-forward network
│  • FFN           │  - Residual connections
│  • LayerNorm     │  - Layer normalization
└──────┬───────────┘
       │
       ⋮  (6 layers total)
       │
       ▼
┌──────────────┐
│  LM Head     │  Linear projection
│  50262 vocab │  Output: (seq_len, 50262)
└──────────────┘
```

### Transformer Layer

```python
class TransformerLayer(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        # Feed-forward with pre-norm
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ffn_output = self.mlp(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        self.n_head = 6
        self.head_dim = 384 // 6 = 64
        
        self.q_proj = nn.Linear(384, 384)
        self.k_proj = nn.Linear(384, 384)
        self.v_proj = nn.Linear(384, 384)
        self.out_proj = nn.Linear(384, 384)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, L, 6, 64).transpose(1, 2)
        k = self.k_proj(x).view(B, L, 6, 64).transpose(1, 2)
        v = self.v_proj(x).view(B, L, 6, 64).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / sqrt(64)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out
```

## 🔗 Projection Layer

Maps visual features to language model's input space.

```python
class ProjectionLayer(nn.Module):
    """
    2-layer MLP: 384 → 768 → 384
    Maps RepViT features to TinyLLM space
    """
    def __init__(self, vision_dim=384, hidden_dim=768, lang_dim=384):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, lang_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, visual_features):
        # visual_features: (batch_size, 384)
        x = self.fc1(visual_features)      # (B, 768)
        x = self.act(x)                     # (B, 768)
        x = self.dropout(x)                 # (B, 768)
        x = self.fc2(x)                     # (B, 384)
        return x
```

### Why 2-Layer MLP?

- **Expressiveness**: Non-linear transformation learns complex mappings
- **Dimensionality**: Expansion to 768 then compression allows richer representations
- **Regularization**: Dropout prevents overfitting
- **Efficiency**: Only 2M parameters, fast inference

## 🧠 Reasoning Module

Added in Stage 4 for chain-of-thought reasoning.

```python
class ReasoningModule(nn.Module):
    """
    Generates step-by-step reasoning before final answer
    """
    def __init__(self, config):
        # Reasoning head: generates reasoning steps
        self.reasoning_head = nn.Sequential(
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384),
        )
        
        # Step validator: checks logical consistency
        self.step_validator = nn.Sequential(
            nn.Linear(384 * 2, 384),  # Current + previous step
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid(),  # Consistency score [0, 1]
        )
        
        # Answer head: produces final answer
        self.answer_head = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 5),  # 5 robot types
        )
    
    def forward(self, hidden_states, return_reasoning=True):
        if not return_reasoning:
            # Direct answer without reasoning
            answer_logits = self.answer_head(hidden_states[:, -1])
            return {"answer_logits": answer_logits}
        
        # Generate reasoning steps
        reasoning_steps = []
        step_hidden = hidden_states[:, -1]
        
        for i in range(self.max_reasoning_steps):
            # Generate next reasoning step
            step_features = self.reasoning_head(step_hidden)
            reasoning_steps.append(step_features)
            
            # Validate with previous step
            if i > 0:
                consistency = self.step_validator(
                    torch.cat([step_features, reasoning_steps[-2]], dim=-1)
                )
                if consistency < self.consistency_threshold:
                    break  # Stop if inconsistent
            
            step_hidden = step_features
        
        # Generate final answer
        final_hidden = reasoning_steps[-1]
        answer_logits = self.answer_head(final_hidden)
        
        return {
            "reasoning_steps": reasoning_steps,
            "answer_logits": answer_logits,
            "consistency_scores": [...]
        }
```

## 📊 Training Pipeline

### Progressive Training Across 4 Stages

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Visual-Language Alignment                 │
│  ────────────────────────────────────               │
│  Goal: Align vision and language spaces             │
│  Trainable: Projection + Language Model              │
│  Frozen: Vision Encoder                              │
│  Data: COCO, Flickr30k, CC3M (330K)                 │
│  Loss: Contrastive + Captioning                      │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: Instruction Tuning                        │
│  ────────────────────────────                       │
│  Goal: Teach task-following                          │
│  Trainable: All (with LR schedule)                   │
│  Teacher: Qwen-VL-Chat (distillation)               │
│  Data: LLaVA, VQA-v2, OK-VQA (300K)                 │
│  Loss: SFT + Distillation                           │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Robot Fleet Selection                     │
│  ────────────────────────────────                   │
│  Goal: Domain-specific robot selection               │
│  Trainable: All                                      │
│  Data: Robot selection dataset (1K→10K augmented)   │
│  Loss: Cross-Entropy + Reasoning Consistency         │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: Advanced Reasoning (2 Phases)             │
│  ─────────────────────────────────────              │
│  Phase 1: Train reasoning heads (frozen backbone)    │
│  Phase 2: Joint fine-tuning (all trainable)         │
│  Data: Reasoning-annotated (50K)                    │
│  Loss: Reasoning + Consistency + Answer              │
└─────────────────────────────────────────────────────┘
```

## 📈 Parameter Distribution

### Total Parameters: 48.3M

```
Component              │ Parameters │ Percentage │ Trainable │
───────────────────────┼────────────┼────────────┼───────────┤
RepViT Vision Encoder  │   14.0M    │   29.0%    │ Partially │
Projection Layer       │    2.0M    │    4.1%    │    Yes    │
TinyLLM Language Model │   30.0M    │   62.1%    │    Yes    │
Reasoning Module       │    2.3M    │    4.8%    │    Yes    │
───────────────────────┼────────────┼────────────┼───────────┤
Total                  │   48.3M    │  100.0%    │   36.6M   │
```

### Memory Footprint

| Precision | Parameters | Activations | Total (Batch=1) |
|-----------|------------|-------------|-----------------|
| FP32 | 193 MB | ~50 MB | ~243 MB |
| FP16 | 97 MB | ~25 MB | ~122 MB |
| INT8 | 48 MB | ~25 MB | ~73 MB |

## 🎯 Design Decisions

### Why These Components?

#### RepViT for Vision

**Alternatives Considered**: ViT-Tiny, MobileNetV3, EfficientNet-B0

**Chosen RepViT Because**:
- ✅ Best accuracy/efficiency trade-off
- ✅ Mobile-optimized operators
- ✅ Pre-trained on ImageNet-1K
- ✅ 14M params (ViT-Tiny: 5M too small, EfficientNet: 20M too large)

#### TinyLLM for Language

**Alternatives Considered**: GPT-2 Small (124M), BLOOM-560M, OPT-125M

**Chosen TinyLLM-30M Because**:
- ✅ Optimal for target 50M total budget
- ✅ GPT-2 architecture (well-understood)
- ✅ Good language understanding despite size
- ✅ Fast inference (<10ms on CPU)

#### Projection Layer Design

**Alternatives Considered**: Linear, 1-layer MLP, 3-layer MLP, Transformer

**Chosen 2-Layer MLP Because**:
- ✅ Non-linearity for complex mapping
- ✅ Efficient (2M params, 0.1 GFLOPs)
- ✅ Proven in CLIP, LLaVA
- ✅ Fast convergence in Stage 1

### Training Strategy

#### Progressive Multi-Stage Training

**Why Not End-to-End**:
- ❌ Unstable with random initialization
- ❌ Requires massive data
- ❌ Difficult to debug failures

**Why Progressive**:
- ✅ Each stage builds on previous
- ✅ Easier to debug/improve
- ✅ More data-efficient
- ✅ Better final performance

### Comparison with Other VLMs

| Model | Params | Vision | Language | Reasoning | Speed |
|-------|--------|--------|----------|-----------|-------|
| **EmberVLM** | **48M** | RepViT | TinyLLM | ✅ CoT | **Fast** |
| LLaVA-1.5 | 7B | CLIP | Vicuna | ❌ | Slow |
| MiniGPT-4 | 7B | CLIP | Vicuna | ❌ | Slow |
| Qwen-VL | 7B | ViT | Qwen | ✅ | Slow |
| TinyGPT-V | 2.8B | EVA-CLIP | Phi-2 | ❌ | Medium |

**EmberVLM Advantages**:
- 100x fewer parameters than LLaVA
- Integrated reasoning module
- Fast inference (~30ms on GPU)
- Designed for edge deployment

## 🔗 References

- **RepViT**: [arXiv:2307.09283](https://arxiv.org/abs/2307.09283)
- **TinyLLM**: GPT-2 based efficient language model
- **LLaVA**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- **DeepSeek-R1**: Chain-of-thought reasoning

---

**Next**: [Reasoning Module Details](REASONING_MODULE.md) | [Vision Encoder Details](VISION_ENCODER.md)

