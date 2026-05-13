# Stage 4: Advanced Reasoning Integration

**Complete Training Guide for Chain-of-Thought Reasoning with DeepSeek-R1 Style**

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Two-Phase Training](#two-phase-training)
- [Reasoning Data Generation](#reasoning-data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## 🎯 Overview

Stage 4 integrates advanced chain-of-thought (CoT) reasoning capabilities inspired by DeepSeek-R1. This stage enables the model to generate step-by-step reasoning before producing final answers, significantly improving problem-solving and decision-making quality.

### Objectives

1. **Reasoning Generation**: Teach the model to break down complex problems
2. **Step-by-Step Thinking**: Enable explicit reasoning chains
3. **Consistency**: Ensure logical flow between reasoning steps
4. **Integration**: Seamlessly combine reasoning with multimodal understanding

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Reasoning Consistency | >0.85 | Logical flow between steps |
| Step Completeness | >0.90 | All necessary steps present |
| Logical Flow Score | >0.80 | Coherence of reasoning chain |
| Final Answer Accuracy | >75% | Correctness after reasoning |

## 🏗️ Architecture

### Reasoning Module Components

```
┌────────────────────────────────────────────────────────────┐
│                    Stage 4 Architecture                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input (Image + Question)                                  │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────┐                                      │
│  │  Vision Encoder  │  ──────────┐                        │
│  │    (Frozen)      │            │                        │
│  └──────────────────┘            │                        │
│                                   │                        │
│  ┌──────────────────┐            │                        │
│  │  Language Model  │◄───────────┘                        │
│  │   (Trainable)    │                                      │
│  └────────┬─────────┘                                      │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                      │
│  │ Reasoning Module │                                      │
│  │   (NEW - Stage 4)│                                      │
│  ├──────────────────┤                                      │
│  │ • Reasoning Head │  Generate reasoning steps           │
│  │ • Step Validator │  Validate logical flow              │
│  │ • Answer Head    │  Generate final answer              │
│  └────────┬─────────┘                                      │
│           │                                                │
│           ▼                                                │
│  Output: <reasoning> Step 1... Step 2... </reasoning>     │
│          <answer> Final Answer </answer>                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### New Components in Stage 4

#### 1. Reasoning Head
- **Purpose**: Generate intermediate reasoning steps
- **Architecture**: 2-layer transformer decoder
- **Parameters**: ~1.5M

#### 2. Step Validator
- **Purpose**: Ensure logical consistency between steps
- **Architecture**: Binary classifier
- **Parameters**: ~0.5M

#### 3. Answer Head
- **Purpose**: Produce final answer after reasoning
- **Architecture**: Linear projection
- **Parameters**: ~0.3M

**Total New Parameters**: ~2.3M (bringing total to ~39M)

### Training Phases

Stage 4 uses **two-phase training**:

```
Phase 1: Train Reasoning Heads (Epochs 1-5)
├── Freeze: Vision encoder, language model backbone
├── Train: Reasoning head, step validator, answer head
└── Goal: Learn reasoning pattern generation

Phase 2: Joint Fine-Tuning (Epochs 6-10)
├── Unfreeze: All components
├── Train: End-to-end with reduced learning rate
└── Goal: Integrate reasoning into full model
```

## 🎓 Two-Phase Training

### Phase 1: Reasoning Head Training (Epochs 1-5)

**Focus**: Train reasoning-specific components while keeping backbone frozen

```yaml
phase1:
  name: "train_reasoning_heads"
  epochs: 5
  learning_rate: 1.0e-4
  freeze_backbone: true
  
  trainable_components:
    - reasoning_head
    - step_validator
    - answer_head
  
  frozen_components:
    - vision_encoder
    - language_model_backbone
    - projection_layer
```

**Architecture**:
```
┌─────────────────────────────────────┐
│         Phase 1 Training            │
├─────────────────────────────────────┤
│  ❄️  Frozen Components              │
│  • Vision Encoder (14M params)      │
│  • Language Model (30M params)      │
│  • Projection Layer (2M params)     │
│                                     │
│  🔥 Trainable Components            │
│  • Reasoning Head (1.5M params)     │
│  • Step Validator (0.5M params)     │
│  • Answer Head (0.3M params)        │
│                                     │
│  Total Trainable: 2.3M (5.9%)       │
└─────────────────────────────────────┘
```

**Loss Function**:
```python
L_phase1 = (
    0.4 * L_reasoning +      # Reasoning step generation
    0.3 * L_consistency +    # Step-to-step consistency
    0.3 * L_answer           # Final answer accuracy
)
```

### Phase 2: Joint Fine-Tuning (Epochs 6-10)

**Focus**: End-to-end training with all components

```yaml
phase2:
  name: "joint_finetuning"
  epochs: 5
  learning_rate: 5.0e-5  # Reduced LR
  freeze_backbone: false
  
  trainable_components:
    - all  # Everything is trainable
  
  learning_rate_multipliers:
    vision_encoder: 0.1      # 5.0e-6
    language_model: 0.5      # 2.5e-5
    reasoning_module: 1.0    # 5.0e-5
```

**Architecture**:
```
┌─────────────────────────────────────┐
│         Phase 2 Training            │
├─────────────────────────────────────┤
│  🔥 All Components Trainable        │
│  • Vision Encoder (LR × 0.1)        │
│  • Language Model (LR × 0.5)        │
│  • Projection Layer (LR × 0.5)      │
│  • Reasoning Module (LR × 1.0)      │
│                                     │
│  Total Trainable: 39M (100%)        │
└─────────────────────────────────────┘
```

**Loss Function**:
```python
L_phase2 = (
    0.3 * L_reasoning +       # Reasoning quality
    0.2 * L_consistency +     # Logical flow
    0.3 * L_answer +          # Final answer
    0.2 * L_multimodal        # Visual grounding
)
```

### Why Two-Phase Training?

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Focus** | Learn reasoning pattern | Integrate with full model |
| **Speed** | Fast (few params) | Slower (all params) |
| **Stability** | Stable (frozen backbone) | Requires lower LR |
| **Convergence** | Quick | Gradual refinement |

## 📊 Reasoning Data Generation

### Data Sources

Stage 4 requires reasoning-annotated data. Multiple generation strategies:

#### 1. Teacher-Generated Reasoning (Recommended)

Use a large model to generate reasoning chains:

```bash
# Generate reasoning data
python scripts/generate_reasoning_data.py \
    --teacher_model deepseek-r1 \
    --input_dataset data/robot_selection.json \
    --output_file data/stage4/reasoning_data.json \
    --num_samples 50000
```

**Example Output**:
```json
{
  "image": "warehouse_scene.jpg",
  "question": "Which robot should pick boxes from high shelves?",
  "reasoning": [
    "<|reasoning_start|>",
    "Step 1: Analyze the task requirements",
    "- The task involves picking boxes from HIGH shelves",
    "- This requires significant reach capability",
    "- The robot must be able to lift heavy boxes",
    "",
    "Step 2: Evaluate robot capabilities",
    "- Humanoid robots have arms with good reach (2m+)",
    "- Drones cannot carry heavy boxes reliably",
    "- Wheeled robots have limited vertical reach",
    "",
    "Step 3: Consider stability and safety",
    "- Humanoid robots provide stable lifting platform",
    "- Can maintain balance while reaching high",
    "- Safety mechanisms prevent accidents",
    "",
    "Conclusion: Humanoid robot is the best choice",
    "<|reasoning_end|>"
  ],
  "answer": "Humanoid",
  "confidence": 0.95
}
```

#### 2. Human-Annotated Reasoning

High-quality manual annotations:

```json
{
  "image": "image.jpg",
  "question": "What robot combination works best?",
  "reasoning_steps": [
    "First, identify the task requirements...",
    "Second, analyze the environment constraints...",
    "Third, consider robot coordination..."
  ],
  "answer": ["Drone", "Humanoid"],
  "annotator_id": "expert_123"
}
```

#### 3. Self-Generated with Validation

Model generates reasoning, validated by expert:

```python
# Generate candidate reasoning
reasoning = model.generate_reasoning(image, question)

# Validate logical consistency
is_valid = reasoning_validator(reasoning)

# Keep only validated samples
if is_valid:
    save_to_dataset(reasoning)
```

### Data Augmentation

Augment reasoning data to reach 50K samples:

```python
augmentation_strategies = {
    "paraphrase": 0.3,      # Rephrase reasoning steps
    "reorder": 0.2,         # Change step order (if valid)
    "expand": 0.3,          # Add intermediate steps
    "simplify": 0.2         # Remove redundant steps
}
```

### Data Format

Unified format for Stage 4:

```python
{
    "image": "path/to/image.jpg",
    "instruction": "Select the best robot for...",
    "reasoning": "<|reasoning_start|>Step 1...<|reasoning_end|>",
    "answer": "Humanoid",
    "metadata": {
        "source": "teacher_generated",
        "complexity": "high",
        "num_steps": 5,
        "validated": true
    }
}
```

## 🚀 Training

### Training Configuration

From `configs/training_stages.yaml`:

```yaml
stage4:
  name: "cot_reasoning"
  
  phase1:
    name: "train_reasoning_heads"
    epochs: 5
    learning_rate: 1.0e-4
    freeze_backbone: true
  
  phase2:
    name: "joint_finetuning"
    epochs: 5
    learning_rate: 5.0e-5
    freeze_backbone: false
  
  batch_size: 32
  
  reasoning_augmentation:
    target_samples: 50000
    validate_reasoning: true
  
  scheduled_sampling:
    enabled: true
    initial_teacher_forcing: 1.0
    final_teacher_forcing: 0.5
```

### Training Commands

#### Phase 1: Reasoning Heads

```bash
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 4 \
    --phase 1 \
    --resume_from outputs/stage3/checkpoint-final \
    --output_dir outputs \
    --reasoning_data data/stage4/reasoning_data.json \
    --wandb_project embervlm \
    --wandb_run_name stage4_phase1
```

#### Phase 2: Joint Fine-Tuning

```bash
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 4 \
    --phase 2 \
    --resume_from outputs/stage4/phase1/checkpoint-final \
    --output_dir outputs \
    --reasoning_data data/stage4/reasoning_data.json \
    --wandb_project embervlm \
    --wandb_run_name stage4_phase2
```

#### Both Phases Sequential

```bash
# Train both phases automatically
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 4 \
    --resume_from outputs/stage3/checkpoint-final \
    --output_dir outputs \
    --reasoning_data data/stage4/reasoning_data.json \
    --run_both_phases \
    --wandb_project embervlm
```

### Training Parameters

| Parameter | Phase 1 | Phase 2 | Description |
|-----------|---------|---------|-------------|
| Epochs | 5 | 5 | Training epochs |
| Batch Size | 32 | 32 | Per-device batch size |
| Learning Rate | 1.0e-4 | 5.0e-5 | Peak learning rate |
| Warmup Steps | 500 | 200 | Linear warmup |
| Gradient Clip | 1.0 | 1.0 | Gradient clipping norm |
| Backbone Frozen | ✅ | ❌ | Freeze vision/LM |

### Scheduled Sampling

Gradually transition from teacher forcing to model generation:

```python
# Epoch 1: 100% teacher forcing (always use ground truth)
# Epoch 5: 50% teacher forcing (50% use model predictions)

teacher_forcing_ratio = max(
    0.5,  # Minimum 50%
    1.0 - (epoch / total_epochs) * 0.5
)

if random.random() < teacher_forcing_ratio:
    next_input = ground_truth_token
else:
    next_input = model_predicted_token
```

### Expected Training Time

| Hardware | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| 1x V100 | ~3 hours | ~6 hours | ~9 hours |
| 2x V100 | ~1.5 hours | ~3.5 hours | ~5 hours |
| 4x A100 | ~45 mins | ~1.5 hours | ~2.25 hours |
| 8x A100 | ~25 mins | ~45 mins | ~1.2 hours |

### Memory Requirements

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| Phase 1 (BS 32) | ~14 GB | Only reasoning head trainable |
| Phase 2 (BS 32) | ~20 GB | Full model trainable |
| FP16 training | ~40% less | Recommended |

## 📈 Evaluation

### Evaluation Metrics

#### 1. Reasoning Consistency

Measures logical flow between steps:

```python
def reasoning_consistency(steps):
    """
    Check if each step follows logically from previous
    """
    consistency_scores = []
    for i in range(1, len(steps)):
        score = semantic_similarity(steps[i-1], steps[i])
        consistency_scores.append(score)
    return np.mean(consistency_scores)

# Target: > 0.85
```

#### 2. Step Completeness

Checks if all necessary reasoning steps are present:

```python
def step_completeness(reasoning, expected_steps):
    """
    Expected steps: ['problem_analysis', 'constraint_checking', 
                     'solution_selection', 'conclusion']
    """
    present_steps = identify_step_types(reasoning)
    coverage = len(set(present_steps) & set(expected_steps)) / len(expected_steps)
    return coverage

# Target: > 0.90
```

#### 3. Logical Flow Score

End-to-end coherence:

```python
score = (
    0.4 * step_transitions_score +  # Smooth transitions
    0.3 * contradiction_check +     # No contradictions
    0.3 * conclusion_alignment      # Conclusion follows from steps
)

# Target: > 0.80
```

#### 4. Final Answer Accuracy

Correctness after reasoning:

```python
accuracy = (predicted_answer == ground_truth).mean()

# Target: > 75%
# Note: Should improve compared to no-reasoning baseline
```

### Evaluation Command

```bash
python scripts/evaluate.py \
    --stage 4 \
    --checkpoint outputs/stage4/phase2/checkpoint-final \
    --reasoning_metrics all \
    --split test \
    --output_file results/stage4_eval.json \
    --save_reasoning_samples
```

### Sample Inference

```python
from embervlm import EmberVLM

model = EmberVLM.from_pretrained("outputs/stage4/phase2/checkpoint-final")

# Generate with reasoning
result = model.generate_with_reasoning(
    image="warehouse.jpg",
    instruction="Which robots should sort packages by weight?",
    max_reasoning_steps=10,
    return_reasoning=True
)

print("=== Reasoning ===")
for i, step in enumerate(result['reasoning_steps'], 1):
    print(f"Step {i}: {step}")

print("\n=== Answer ===")
print(result['answer'])

print("\n=== Confidence ===")
print(f"{result['confidence']:.2%}")
```

**Example Output**:
```
=== Reasoning ===
Step 1: Analyze the task requirements
- Task: Sort packages by weight
- Requires: Weight sensors and manipulation
- Environment: Warehouse setting

Step 2: Evaluate robot capabilities
- Humanoid: Has manipulators, can integrate weight sensors
- Drone: Cannot handle heavy packages
- Wheeled: Good for ground transport, can add weight sensors

Step 3: Consider efficiency and coordination
- Multiple robots can work in parallel
- Humanoid for heavy packages (>10kg)
- Wheeled for light packages (<10kg)

Conclusion: Combination of Humanoid and Wheeled robots

=== Answer ===
Primary: Humanoid, Secondary: Robot with Wheels

=== Confidence ===
92.5%
```

### Visualizations

Stage 4 generates specialized visualizations:

```
outputs/stage4/visualizations/
├── reasoning_quality/
│   ├── step_distribution.png          # Steps per reasoning
│   ├── consistency_scores.png         # Consistency over time
│   └── completeness_heatmap.png       # Step coverage
├── attention_maps/
│   ├── reasoning_attention/           # Attention during reasoning
│   └── multimodal_grounding/          # Visual grounding
├── comparison/
│   ├── with_vs_without_reasoning.png  # Performance comparison
│   └── reasoning_impact.png           # Accuracy improvement
└── samples/
    ├── best_reasoning_examples.html   # Top quality samples
    └── failure_cases.html             # Cases needing improvement
```

## 💾 Checkpoints

### Checkpoint Structure

```
outputs/stage4/
├── phase1/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-final/              # End of Phase 1
│       ├── pytorch_model.bin
│       ├── reasoning_module.bin       # Reasoning-specific weights
│       └── config.json
├── phase2/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-final/              # Final Stage 4 model
├── visualizations/                    # Training visualizations
│   ├── reasoning_quality/
│   └── attention_maps/
└── reasoning_samples/                 # Generated reasoning examples
    ├── epoch_1_samples.json
    ├── epoch_5_samples.json
    └── epoch_10_samples.json
```

### Loading Checkpoints

```python
# Load Phase 1 checkpoint
model = EmberVLM.from_pretrained("outputs/stage4/phase1/checkpoint-final")

# Load final Stage 4 model
model = EmberVLM.from_pretrained("outputs/stage4/phase2/checkpoint-final")

# Continue Phase 2 training
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stages 4 \
    --phase 2 \
    --resume_from outputs/stage4/phase1/checkpoint-final
```

### Best Checkpoint Selection

```python
# Select best checkpoint based on reasoning quality
best_checkpoint = select_best_checkpoint(
    checkpoints_dir="outputs/stage4/phase2",
    metric="reasoning_consistency",
    higher_is_better=True
)

print(f"Best checkpoint: {best_checkpoint}")
# Best checkpoint: outputs/stage4/phase2/checkpoint-1500
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Poor Reasoning Quality

**Symptom**: Generated reasoning is repetitive or illogical

**Solutions**:
```bash
# Increase Phase 1 epochs
--phase1_epochs 7

# Improve reasoning data quality
python scripts/filter_reasoning_data.py \
    --input data/stage4/reasoning_data.json \
    --min_quality_score 0.8 \
    --output data/stage4/filtered_reasoning.json

# Adjust reasoning loss weight
# In config: reasoning_loss_weight: 0.5 (from 0.4)
```

#### 2. Phase 2 Instability

**Symptom**: Loss spikes or diverges in Phase 2

**Solutions**:
```bash
# Lower Phase 2 learning rate
--phase2_learning_rate 3.0e-5  # From 5.0e-5

# Increase warmup
--phase2_warmup_steps 500  # From 200

# Reduce learning rate for backbone
# vision_encoder: 0.05 (from 0.1)
# language_model: 0.3 (from 0.5)
```

#### 3. Reasoning Doesn't Ground in Image

**Symptom**: Reasoning ignores visual information

**Solutions**:
- Increase multimodal loss weight
- Use attention supervision
- Add visual grounding rewards
- Check image features are being used

```python
# Add visual grounding loss
L_grounding = cross_attention_loss(
    reasoning_tokens,
    visual_features
)

L_total += 0.2 * L_grounding
```

#### 4. Slow Phase 2 Training

**Symptom**: Phase 2 much slower than expected

**Solutions**:
```bash
# Use gradient checkpointing
--gradient_checkpointing

# Reduce reasoning sequence length
--max_reasoning_length 256  # From 512

# Increase batch size with accumulation
--batch_size 16 --gradient_accumulation_steps 4
```

### Debug Commands

```bash
# Validate reasoning data
python scripts/validate_reasoning_data.py \
    --data_file data/stage4/reasoning_data.json \
    --output_report data/stage4/validation_report.txt

# Test reasoning generation
python scripts/test_reasoning.py \
    --checkpoint outputs/stage4/phase1/checkpoint-500 \
    --num_samples 10 \
    --visualize

# Check Phase 1→2 transition
python scripts/check_phase_transition.py \
    --phase1_checkpoint outputs/stage4/phase1/checkpoint-final \
    --phase2_checkpoint outputs/stage4/phase2/checkpoint-500

# Analyze reasoning quality
python scripts/analyze_reasoning.py \
    --checkpoint outputs/stage4/phase2/checkpoint-final \
    --test_set data/stage4/test.json \
    --output_dir analysis/stage4
```

## 🎓 Advanced Topics

### Custom Reasoning Formats

Define custom reasoning templates:

```python
reasoning_templates = {
    "analysis": [
        "Problem Analysis:",
        "Constraints:",
        "Available Options:",
        "Selection Criteria:"
    ],
    "step_by_step": [
        "Step {i}:",
        "Rationale:",
        "Next Action:"
    ],
    "comparative": [
        "Option A:",
        "Option B:",
        "Comparison:",
        "Decision:"
    ]
}

# Use template
reasoning = model.generate_with_template(
    image=img,
    instruction=inst,
    template="analysis"
)
```

### Multi-Turn Reasoning

Enable interactive reasoning:

```python
# Initial reasoning
response1 = model.reason(image, "Which robot for heavy lifting?")

# Follow-up question building on reasoning
response2 = model.reason(
    image,
    "What if the item is also fragile?",
    previous_reasoning=response1['reasoning']
)
```

### Reasoning Verification

Add self-verification:

```python
# Generate reasoning
reasoning = model.generate_reasoning(image, question)

# Self-verify
is_consistent = model.verify_reasoning(reasoning)

# Regenerate if inconsistent
if not is_consistent:
    reasoning = model.generate_reasoning(
        image,
        question,
        constraint="ensure_logical_consistency"
    )
```

### Curriculum Learning

Progressively increase reasoning complexity:

```yaml
curriculum:
  stage1: # Epochs 1-3
    max_steps: 3
    complexity: "simple"
  stage2: # Epochs 4-7
    max_steps: 5
    complexity: "medium"
  stage3: # Epochs 8-10
    max_steps: 10
    complexity: "complex"
```

## 📚 References

1. **DeepSeek-R1**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs" (2024)
2. **Chain-of-Thought**: "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (Wei et al., 2022)
3. **Progressive Training**: "Curriculum Learning for Neural Networks" (Bengio et al., 2009)
4. **Reasoning Validation**: "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2023)

## 🔗 Next Steps

After completing Stage 4:

- **[Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md)**: Deploy your reasoning-capable model
- **[API Reference](../deployment/API_REFERENCE.md)**: Use the reasoning API
- **[Quantization Guide](../deployment/QUANTIZATION.md)**: Optimize for production

---

**Need Help?** Check our [FAQ](../FAQ.md) or open an [issue](https://github.com/yourusername/EmberVLM/issues).

**Congratulations!** You've completed all 4 training stages of EmberVLM! 🎉

