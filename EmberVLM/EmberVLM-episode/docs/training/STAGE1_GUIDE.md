# Stage 1: Visual-Language Alignment

**Complete Training Guide for Visual-Language Alignment**

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## 🎯 Overview

Stage 1 establishes the foundation for multimodal understanding by aligning the RepViT vision encoder's feature space with the TinyLLM language model's text space. This stage uses a combination of **contrastive learning** and **image captioning** to learn meaningful visual-language associations.

### Objectives

1. **Feature Alignment**: Map visual features to language embedding space
2. **Contrastive Learning**: Learn discriminative visual-text representations
3. **Caption Generation**: Teach basic image-to-text generation capabilities
4. **Foundation Building**: Create a strong base for subsequent stages

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Contrastive Accuracy | >75% | Image-text matching accuracy |
| CIDEr Score | >0.8 | Caption quality metric |
| BLEU-4 | >0.25 | N-gram overlap with references |
| Training Loss | <0.5 | Combined loss convergence |

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────┐
│                Stage 1 Model                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   RepViT     │      │   Projection    │ │
│  │  Encoder     │─────▶│     Layer       │ │
│  │  (Frozen)    │      │   (Trainable)   │ │
│  └──────────────┘      └────────┬────────┘ │
│                                 │          │
│                                 ▼          │
│                        ┌─────────────────┐ │
│                        │    TinyLLM      │ │
│                        │  (Trainable)    │ │
│                        └─────────────────┘ │
│                                             │
│  Training Objectives:                       │
│  • Contrastive Loss (50%)                   │
│  • Captioning Loss (50%)                    │
└─────────────────────────────────────────────┘
```

### Trainable Parameters

- **Vision Encoder (RepViT)**: **Frozen** - 14M parameters
- **Projection Layer**: **Trainable** - 2M parameters
- **Language Model (TinyLLM)**: **Trainable** - 30M parameters
- **Total Trainable**: ~32M parameters

### Loss Function

```python
total_loss = 0.5 * contrastive_loss + 0.5 * captioning_loss

# Contrastive Loss (InfoNCE)
contrastive_loss = -log(exp(sim(v, t+) / τ) / Σ exp(sim(v, t) / τ))

# Captioning Loss (Cross-Entropy)
captioning_loss = -Σ log P(w_t | w_<t, v)
```

Where:
- `v`: Visual features
- `t+`: Positive text
- `τ`: Temperature parameter (default: 0.07)
- `w_t`: Target token at position t

## 📊 Datasets

### Training Datasets

#### 1. COCO Captions (100K samples)
- **Source**: Microsoft COCO 2017
- **Format**: Images with 5 human-annotated captions each
- **Usage**: Primary dataset for both objectives

**Example**:
```json
{
  "image": "COCO_train2017_000000123456.jpg",
  "captions": [
    "A person riding a bike down the street",
    "Someone cycling on an urban road",
    "A cyclist traveling through the city",
    "A man on a bicycle in traffic",
    "Bicyclist riding on a city street"
  ]
}
```

#### 2. Flickr30k (30K samples)
- **Source**: Flickr30k Entities
- **Format**: Images with detailed captions
- **Usage**: Additional diversity for contrastive learning

#### 3. CC3M Subset (200K samples)
- **Source**: Conceptual Captions 3M
- **Format**: Web images with alt-text captions
- **Usage**: Large-scale contrastive learning

### Data Preprocessing

```python
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Text preprocessing
def preprocess_caption(caption):
    # Lowercase and remove special characters
    caption = caption.lower().strip()
    # Tokenize with TinyLLM tokenizer
    tokens = tokenizer(
        caption,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokens
```

### Dataset Statistics

| Dataset | Images | Captions | Avg Length | Vocabulary |
|---------|--------|----------|------------|------------|
| COCO | 100K | 500K | 10.5 words | 12K unique |
| Flickr30k | 30K | 150K | 12.3 words | 18K unique |
| CC3M | 200K | 200K | 8.7 words | 25K unique |
| **Total** | **330K** | **850K** | **10.1 words** | **35K unique** |

## 🚀 Training

### Training Configuration

From `configs/training_stages.yaml`:

```yaml
stage1:
  name: "visual_language_alignment"
  epochs: 3
  batch_size: 128
  learning_rate: 2.0e-4
  
  losses:
    contrastive_weight: 0.5
    captioning_weight: 0.5
  
  max_length: 256
```

### Training Command

#### Basic Training

```bash
# Single GPU
python scripts/train_all.py --stages 1 --output_dir outputs

# Multi-GPU (Distributed)
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 1 \
    --output_dir outputs \
    --wandb_project embervlm
```

#### Advanced Options

```bash
torchrun --nproc_per_node=4 scripts/train_all.py \
    --stages 1 \
    --output_dir outputs \
    --batch_size 128 \
    --learning_rate 2.0e-4 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 500 \
    --save_steps 1000 \
    --eval_steps 500 \
    --fp16 \
    --wandb_project embervlm \
    --wandb_run_name stage1_alignment
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 128 | Per-device batch size |
| `--learning_rate` | 2.0e-4 | Peak learning rate |
| `--warmup_steps` | 500 | Linear warmup steps |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--gradient_clip` | 1.0 | Gradient clipping norm |
| `--save_steps` | 1000 | Checkpoint save frequency |
| `--eval_steps` | 500 | Evaluation frequency |

### Learning Rate Schedule

```python
# Warmup + Cosine Decay
lr_schedule = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

**Schedule Visualization**:
```
LR
│
2e-4 ├─────╱────╲_____
     │    /      \     \___
     │   /        \        \___
     │  /          \           \___
0    └──┴─────┴─────┴─────┴─────┴──▶ Steps
     0  500   1000  2000  3000  Total
        Warmup  Peak   Decay
```

### Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total Time |
|----------|------------|----------------|------------|
| 1x V100 | 64 | ~4 hours | ~12 hours |
| 2x V100 | 128 | ~2.5 hours | ~7.5 hours |
| 4x A100 | 256 | ~1 hour | ~3 hours |
| 8x A100 | 512 | ~30 mins | ~1.5 hours |

### Memory Requirements

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| Batch size 64 | ~16 GB | Single V100 |
| Batch size 128 | ~24 GB | Single A100 |
| FP16 training | ~50% less | Use `--fp16` flag |
| Gradient checkpointing | ~30% less | Use `--gradient_checkpointing` |

## 📈 Evaluation

### Evaluation Metrics

#### 1. Contrastive Accuracy

Measures image-text matching:

```python
accuracy = correct_matches / total_samples

# Example output
Contrastive Accuracy: 78.3%
  Image-to-Text: 79.1%
  Text-to-Image: 77.5%
```

#### 2. CIDEr Score

Consensus-based caption quality:

```python
from pycocoevalcap.cider.cider import Cider

scorer = Cider()
score, scores = scorer.compute_score(references, candidates)

# Target: CIDEr > 0.8
```

#### 3. BLEU-4

N-gram overlap metric:

```python
from nltk.translate.bleu_score import sentence_bleu

bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

# Target: BLEU-4 > 0.25
```

### Evaluation Command

```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --stage 1 \
    --checkpoint outputs/stage1/checkpoint-final \
    --split val \
    --metrics all
```

### Monitoring with W&B

Stage 1 automatically logs to Weights & Biases:

```python
# Logged metrics
wandb.log({
    "stage1/contrastive_loss": contrastive_loss,
    "stage1/captioning_loss": captioning_loss,
    "stage1/total_loss": total_loss,
    "stage1/contrastive_accuracy": acc,
    "stage1/cider": cider_score,
    "stage1/bleu4": bleu4_score,
    "stage1/learning_rate": current_lr,
})
```

**Dashboard**: `https://wandb.ai/your-project/embervlm/runs/stage1`

## 💾 Checkpoints

### Checkpoint Structure

```
outputs/stage1/
├── checkpoint-1000/
│   ├── pytorch_model.bin      # Model weights
│   ├── optimizer.pt            # Optimizer state
│   ├── scheduler.pt            # LR scheduler state
│   ├── config.json             # Model configuration
│   └── training_args.json      # Training arguments
├── checkpoint-2000/
├── checkpoint-final/           # Final checkpoint
└── metrics.json                # Training metrics
```

### Saving Checkpoints

Checkpoints are automatically saved:

```python
# Periodic saves (every save_steps)
if global_step % save_steps == 0:
    save_checkpoint(model, optimizer, scheduler, output_dir, global_step)

# Final checkpoint
save_checkpoint(model, optimizer, scheduler, output_dir, "final")
```

### Loading Checkpoints

```python
from embervlm import EmberVLM

# Load from checkpoint
model = EmberVLM.from_pretrained("outputs/stage1/checkpoint-final")

# Or load for continued training
model = EmberVLM.from_pretrained("outputs/stage1/checkpoint-2000")
```

### Best Practices

1. **Save frequently**: Use `--save_steps 1000` to avoid data loss
2. **Keep multiple checkpoints**: Useful for choosing best model
3. **Monitor disk space**: Each checkpoint is ~150 MB
4. **Backup critical checkpoints**: Especially the final checkpoint

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
--batch_size 64  # Instead of 128

# Enable gradient accumulation
--gradient_accumulation_steps 2

# Use FP16 training
--fp16

# Enable gradient checkpointing
--gradient_checkpointing
```

#### 2. Slow Training

**Symptom**: Training slower than expected

**Solutions**:
- Check GPU utilization: `nvidia-smi`
- Use more GPUs: `--nproc_per_node=4`
- Increase batch size if memory allows
- Enable mixed precision: `--fp16`
- Use faster data loading: `--dataloader_num_workers 8`

#### 3. Poor Contrastive Accuracy

**Symptom**: Accuracy stuck below 70%

**Solutions**:
- Increase contrastive loss weight: `contrastive_weight: 0.7`
- Lower temperature: `temperature: 0.05`
- Use larger batch size for more negative samples
- Ensure vision encoder weights are properly loaded

#### 4. Low Caption Quality

**Symptom**: CIDEr score below 0.6

**Solutions**:
- Increase captioning loss weight: `captioning_weight: 0.7`
- Train for more epochs: `--epochs 5`
- Verify caption preprocessing is correct
- Check tokenizer configuration

### Debug Commands

```bash
# Check model loading
python -c "from embervlm import EmberVLM; EmberVLM()"

# Verify dataset
python scripts/verify_dataset.py --stage 1

# Test forward pass
python scripts/test_forward.py --stage 1 --batch_size 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🎓 Advanced Topics

### Custom Data Loading

Create custom dataset loader:

```python
from torch.utils.data import Dataset

class CustomVLDataset(Dataset):
    def __init__(self, image_paths, captions, transform):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        caption = self.captions[idx]
        return {"image": image, "text": caption}

# Use in training
from embervlm.data import create_dataloader
train_loader = create_dataloader(dataset, batch_size=128)
```

### Loss Weight Tuning

Experiment with different loss weights:

```yaml
# configs/training_stages.yaml
stage1:
  losses:
    contrastive_weight: 0.7  # Emphasize contrastive learning
    captioning_weight: 0.3
```

### Temperature Tuning

For contrastive learning:

```python
# Lower temperature = harder negatives
temperature = 0.05  # More discriminative
temperature = 0.10  # Balanced (default: 0.07)
```

### Ablation Studies

Test component importance:

```bash
# Only contrastive learning
--contrastive_weight 1.0 --captioning_weight 0.0

# Only captioning
--contrastive_weight 0.0 --captioning_weight 1.0

# Different ratios
--contrastive_weight 0.3 --captioning_weight 0.7
```

## 📚 References

1. **RepViT**: "RepViT: Revisiting Mobile CNN From ViT Perspective" (2023)
2. **TinyLLM**: Lightweight language model architecture
3. **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision" (2021)
4. **COCO**: "Microsoft COCO: Common Objects in Context" (2014)

## 🔗 Next Steps

After completing Stage 1, proceed to:

- **[Stage 2: Multimodal Instruction Tuning](STAGE2_GUIDE.md)**: Add task-following capabilities
- **[Model Architecture](../architecture/MODEL_ARCHITECTURE.md)**: Understand the full architecture
- **[Dataset Overview](../datasets/DATASET_OVERVIEW.md)**: Learn about all datasets

---

**Need Help?** Check our [FAQ](../FAQ.md) or open an [issue](https://github.com/yourusername/EmberVLM/issues).

