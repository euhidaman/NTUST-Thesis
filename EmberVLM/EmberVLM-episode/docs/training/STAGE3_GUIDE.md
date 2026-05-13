# Stage 3: Robot Fleet Selection Training Guide

## Overview

Stage 3 trains EmberVLM for intelligent robot fleet selection with explicit chain-of-thought reasoning. The model learns to select the most appropriate robot(s) from 5 types based on visual scenes and task requirements.

## Key Capabilities

### 1. **Top-N Robot Selection**
- Returns **top 5 robot candidates** ranked by confidence
- Each candidate includes:
  - Robot type (Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs)
  - Confidence score (0.0-1.0)
  - Selection reasoning (chain-of-thought explanation)

### 2. **Chain-of-Thought Reasoning**
The model generates 4-step reasoning for transparency:

```
Step 1: Task Analysis
- Identifies task requirements (inspect, transport, navigate, manipulate)
- Extracts key action verbs from task description

Step 2: Environment Assessment  
- Determines environment type (aerial, underwater, rough terrain, flat ground, indoor)
- Analyzes spatial constraints and accessibility

Step 3: Robot Capability Matching
- Matches robot strengths to task requirements
- Considers robot limitations and constraints

Step 4: Decision Justification
- Provides final selection rationale
- Explains why the chosen robot(s) are optimal
```

### 3. **Multi-Robot Coordination**
- Handles complex tasks requiring multiple robots
- Decomposes tasks into subtasks with execution order
- Assigns appropriate robots to each subtask

## Training Data

### Dataset Statistics
- **Total samples**: 8,138 scenarios
  - Single-robot: 1,252 samples
  - Multi-robot: 6,886 samples
- **Training split**: 5,696 → Augmented 3× → **17,088 samples**
- **Validation split**: 1,628 samples (no augmentation)
- **Test split**: 814 samples (no augmentation)

### Data Augmentation
The enhanced loader automatically applies 3× augmentation:
- Synonym replacement (inspect→examine, transport→carry)
- Task paraphrasing for better generalization
- Maintains semantic meaning while increasing diversity

### Scene Visualization
Auto-generates task-appropriate scene images:
- **Aerial tasks**: Light blue sky background
- **Underwater tasks**: Dark blue water environment
- **Indoor tasks**: Building interior with walls
- **Outdoor tasks**: Green ground/terrain

## Training Command

### Full Stage 3 Training (Recommended)

```bash
PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=2 scripts/train_all.py \
  --output_dir ./outputs \
  --stage 3 \
  --distributed \
  --mixed_precision bf16 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --gradient_accumulation 4 \
  --robot_data robot-selection-dataset \
  --stage3_robot_epochs 30 \
  --eval_steps 3 \
  --save_steps 500 \
  2>&1 | tee train_stage3.log
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--stage` | 3 | Run Stage 3 training only |
| `--stage3_robot_epochs` | 30 | Number of training epochs (default) |
| `--batch_size` | 32 | Per-GPU batch size |
| `--gradient_accumulation` | 4 | Effective batch = 32×2×4 = 256 |
| `--learning_rate` | 2e-4 | AdamW learning rate |
| `--eval_steps` | 3 | Evaluate every 3 batches (or use 500 for less frequent) |
| `--save_steps` | 500 | Save checkpoint every 500 steps |
| `--robot_data` | robot-selection-dataset | Path to robot selection data |

### Training Time

- **Hardware**: 2× NVIDIA A100 80GB GPUs
- **Duration**: ~2 hours for 30 epochs
  - 534 batches/epoch × 30 epochs = 16,020 training steps
  - ~0.4 seconds/batch with eval every 3 batches
- **Checkpoints**: Saved at steps 500, 1000, 1500, etc.

## Checkpoint Loading Fix

### Issue (Fixed in v1.1)
When transitioning from Stage 2 to Stage 3, the model architecture changed:
- **Stage 2**: `robot_head.confidence` outputs 1 value (binary confidence)
- **Stage 3**: `robot_head.confidence` outputs 5 values (top-N robot scores)

This caused a `RuntimeError` when loading Stage 2 checkpoints.

### Solution
The `load_checkpoint()` function now:
1. **Filters mismatched layers** before loading
2. **Logs warnings** for skipped layers
3. **Randomly initializes** mismatched layers for Stage 3 training
4. **Maintains compatibility** with all existing checkpoints

```python
# Automatic handling in train_utils.py
for key, value in state_dict.items():
    if key in model_state_dict:
        if model_state_dict[key].shape == value.shape:
            filtered_state_dict[key] = value  # Load matching layers
        else:
            mismatched_keys.append(key)  # Skip mismatched layers
```

### What Gets Loaded from Stage 2
✅ **Loaded layers** (transferred from Stage 2):
- Vision encoder (RepViT): All weights preserved
- Language model (TinyLLM): All weights preserved  
- Projection layers: All weights preserved
- Robot head features: Base feature extractors preserved

❌ **Skipped layers** (randomly initialized for Stage 3):
- `robot_head.confidence.2.weight`: Shape mismatch (1→5)
- `robot_head.confidence.2.bias`: Shape mismatch (1→5)

These 2 layers are small (~500 parameters) and will be trained from scratch in Stage 3.

## Training Output

### Expected Console Output

```
2026-01-07 11:17:59 INFO - Auto-detecting checkpoint from previous stage for Stage 3...
2026-01-07 11:17:59 INFO - Found latest checkpoint from Stage 2: outputs/stage2/checkpoint-789
2026-01-07 11:17:59 INFO - Loading model from checkpoint: outputs/stage2/checkpoint-789
2026-01-07 11:18:00 WARNING - Skipped 2 mismatched layer(s) - will be randomly initialized:
2026-01-07 11:18:00 WARNING -   - robot_head.confidence.2.weight: checkpoint torch.Size([1, 96]) vs model torch.Size([5, 96])
2026-01-07 11:18:00 WARNING -   - robot_head.confidence.2.bias: checkpoint torch.Size([1]) vs model torch.Size([5])
2026-01-07 11:18:00 INFO - Loaded model weights from outputs/stage2/checkpoint-789/pytorch_model.bin
2026-01-07 11:18:00 INFO - ✓ Loaded model weights from outputs/stage2/checkpoint-789

Robot Selection Epoch 0:   0%|          | 0/534 [00:00<?, ?it/s]
Robot Selection Epoch 0:   7%|▋         | 37/534 [00:15<03:19, 2.49it/s]
```

### Training Progress Metrics

Every evaluation (e.g., every 3 batches or 500 steps):

```
Evaluating: 100%|██████████| 51/51 [00:21<00:00, 2.40it/s]

Validation metrics (Epoch 0, Step 534):
  accuracy: 0.7214                     # Overall robot selection accuracy
  macro_f1: 0.6823                     # Macro-averaged F1 across all robots
  
  Per-Robot Performance:
    Drone_precision: 0.78              # Precision for Drone selection
    Drone_recall: 0.85                 # Recall for Drone selection
    Drone_f1: 0.81                     # F1 score for Drone
    
    Underwater_Robot_precision: 0.72
    Underwater_Robot_recall: 0.68
    Underwater_Robot_f1: 0.70
    
    Humanoid_precision: 0.82
    Humanoid_recall: 0.79
    Humanoid_f1: 0.80
    
    Robot_with_Wheels_precision: 0.65
    Robot_with_Wheels_recall: 0.71
    Robot_with_Wheels_f1: 0.68
    
    Robot_with_Legs_precision: 0.69
    Robot_with_Legs_recall: 0.73
    Robot_with_Legs_f1: 0.71
  
  expected_calibration_error: 0.12     # Confidence calibration (lower is better)
  multi_robot_iou: 0.76                # IoU for multi-robot task assignment
```

### Target Performance (End of Stage 3)

After 30 epochs, you should see:

| Metric | Target Value |
|--------|--------------|
| Overall Accuracy | 85-90% |
| Macro F1 | 80-85% |
| Per-Robot F1 | 75-90% (varies by robot) |
| Calibration Error (ECE) | <0.10 |
| Multi-Robot IoU | >0.80 |

## Visualizations

### Saved Artifacts

All visualizations are saved to `outputs/stage3/visualizations/`:

```
outputs/stage3/visualizations/
├── confusion_matrix_epoch{N}.png      # Per-epoch confusion matrices
├── robot_radar_epoch{N}.png           # Radar charts of robot performance
├── calibration_epoch{N}.png           # Confidence calibration curves
├── training_curves.png                # Loss and accuracy over time
└── final_evaluation_report.png        # Comprehensive final report
```

### 1. Confusion Matrix
Shows which robots are confused with each other:
- Diagonal: Correct predictions (high values = good)
- Off-diagonal: Misclassifications (low values = good)

**Example**: If `Drone` is often predicted as `Humanoid`, the confusion matrix highlights this pattern.

### 2. Robot Performance Radar Chart
Circular visualization showing per-robot metrics:
- Precision, Recall, F1 score for each robot type
- Larger area = Better overall performance
- Helps identify which robots are harder to select correctly

### 3. Calibration Curve
Shows how well confidence scores match actual accuracy:
- **Perfect calibration**: 45° diagonal line
- **Overconfident**: Curve below diagonal (high confidence, low accuracy)
- **Underconfident**: Curve above diagonal (low confidence, high accuracy)

**Expected Calibration Error (ECE)**: <0.10 is considered well-calibrated.

### 4. Training Curves
Shows learning progress over time:
- Training loss decreasing
- Validation accuracy increasing
- Stage 3 specific: Robot selection accuracy and F1 score trends

## Weights & Biases (W&B) Integration

### Automatic Logging

If W&B is configured, Stage 3 automatically logs:

**Metrics (every eval step):**
- `stage3/accuracy`: Overall robot selection accuracy
- `stage3/macro_f1`: Macro-averaged F1 score
- `stage3/{robot}_f1`: Per-robot F1 scores (5 robots)
- `stage3/expected_calibration_error`: Confidence calibration
- `stage3/multi_robot_iou`: Multi-robot task assignment IoU
- `stage3/loss`: Training loss
- `stage3/learning_rate`: Current learning rate

**Visualizations (every eval step):**
- `stage3/confusion_matrix`: Interactive confusion matrix
- `stage3/robot_radar`: Radar chart of robot performance
- `stage3/calibration`: Calibration reliability diagram

**System Metrics:**
- GPU memory usage
- Training throughput (samples/sec)
- Carbon emissions (via CodeCarbon)

### View Dashboard

1. During training, open: https://wandb.ai/your-username/embervlm
2. Navigate to run: `stage3_robot_selection`
3. Check tabs:
   - **Overview**: Training curves and key metrics
   - **Charts**: Interactive visualizations
   - **System**: GPU/memory usage
   - **Logs**: Console output

### Disable W&B (Optional)

```bash
export DISABLE_WANDB=1
torchrun --nproc_per_node=2 scripts/train_all.py --stage 3 ...
```

## Model Architecture Changes

### Stage 3 Enhancements

1. **Robot Selection Head**
   - Input: 384-dim language model hidden state
   - Output: 5 robot confidence scores (0.0-1.0)
   - Architecture: 3-layer MLP with dropout
     - Layer 1: 384 → 384 (ReLU + Dropout 0.1)
     - Layer 2: 384 → 96 (ReLU + Dropout 0.1)
     - Layer 3: 96 → 5 (Sigmoid activation)

2. **Reasoning Module**
   - Generates 4-step chain-of-thought reasoning
   - Integrated with language model decoder
   - Trained with reasoning consistency loss

3. **Loss Function**
   - **Cross-Entropy Loss** (60%): For robot classification
     - Uses focal loss with γ=2.0 for class imbalance
     - Label smoothing (0.1) to prevent overconfidence
   - **Reasoning Consistency Loss** (40%): For reasoning quality
     - Ensures reasoning aligns with robot selection
     - Penalizes contradictory reasoning steps

## Inference API

### Basic Usage

```python
from embervlm.models import EmberVLM
from transformers import AutoTokenizer
from PIL import Image

# Load trained model
model = EmberVLM.from_pretrained("outputs/stage3")
tokenizer = AutoTokenizer.from_pretrained("tinyllm/30M-0.4")

# Prepare input
image = Image.open("disaster_scene.jpg")
task = "Inspect damage and search for survivors in rubble"

# Get top-N robot recommendations
results = model.select_robots(
    image=image,
    task_description=task,
    top_k=3,  # Return top 3 robots
    return_reasoning=True
)

# Output format
for i, result in enumerate(results):
    print(f"\nRank {i+1}:")
    print(f"  Robot: {result['robot_type']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Reasoning:")
    print(f"    {result['reasoning']}")
```

### Example Output

```
Rank 1:
  Robot: Drone
  Confidence: 92.5%
  Reasoning:
    Step 1: Task requires aerial inspection and damage assessment
    Step 2: Environment is disaster area with potential aerial hazards
    Step 3: Drone excels at rapid aerial surveys and can access hard-to-reach areas
    Step 4: Selected Drone for efficient damage assessment from above

Rank 2:
  Robot: Robot with Legs
  Confidence: 78.3%
  Reasoning:
    Step 1: Task involves navigating through rubble and searching survivors
    Step 2: Environment has rough, uneven terrain with obstacles
    Step 3: Legged robot can traverse rubble better than wheeled alternatives
    Step 4: Selected Legged Robot for ground search operations

Rank 3:
  Robot: Humanoid
  Confidence: 65.1%
  Reasoning:
    Step 1: Task may require object manipulation for rescue operations
    Step 2: Environment requires dexterous manipulation capabilities
    Step 3: Humanoid has advanced manipulation for debris clearing
    Step 4: Selected Humanoid as backup for manipulation tasks
```

## Troubleshooting

### Issue 1: RuntimeError - Size Mismatch (FIXED)

**Error:**
```
RuntimeError: size mismatch for reasoning_module.robot_head.confidence.2.weight: 
copying a param with shape torch.Size([1, 96]) from checkpoint, 
the shape in current model is torch.Size([5, 96])
```

**Solution:** This is now handled automatically! Update to the latest version (v1.1+).

**Manual Fix (if needed):**
```bash
git pull  # Get latest code with fix
# or
# Delete outputs/stage2/checkpoint-* and retrain Stage 2
```

### Issue 2: Low Validation Accuracy (<50%)

**Possible causes:**
1. Insufficient training epochs (try 50-100 epochs)
2. Learning rate too high/low (try 1e-4 or 5e-4)
3. Class imbalance not addressed (check if focal loss is enabled)

**Solution:**
```bash
# Increase epochs
--stage3_robot_epochs 50

# Adjust learning rate
--learning_rate 1e-4

# Verify focal loss is enabled (default: ON)
# Check embervlm/training/stage3_incidents.py:
# self.reasoning_loss = ReasoningLoss(use_focal_loss=True, focal_gamma=2.0)
```

### Issue 3: High Calibration Error (ECE >0.15)

**Cause:** Model is overconfident or underconfident.

**Solution:**
- Increase label smoothing: `label_smoothing=0.15` in `ReasoningLoss`
- Add temperature scaling during inference
- Train longer for better calibration

### Issue 4: GPU Out of Memory

**Solution:**
```bash
# Reduce batch size and increase gradient accumulation
--batch_size 16 --gradient_accumulation 8  # Same effective batch (256)

# Enable gradient checkpointing (if implemented)
--gradient_checkpointing

# Use mixed precision (already enabled by default)
--mixed_precision bf16
```

### Issue 5: Poor Multi-Robot Performance

**Symptoms:** Low multi_robot_iou (<0.70)

**Solution:**
- Check that multi-robot samples are being loaded correctly
- Verify subtask assignment logic in model forward pass
- Increase `action_weight` in `ReasoningLoss` to emphasize multi-robot coordination

## Evaluation

### Internal Metrics (Automatic)

Stage 3 automatically computes these metrics every eval step:

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct robot selection | >85% |
| Macro F1 | Average F1 across all robots | >80% |
| Per-Robot F1 | F1 for each of 5 robot types | >75% |
| Calibration Error | ECE for confidence scores | <0.10 |
| Multi-Robot IoU | IoU for multi-robot assignment | >0.80 |

### External Benchmarks (Optional)

After Stage 3, you can evaluate on standard VLM benchmarks:

```bash
# Evaluate on MMBench, TextVQA, etc.
python scripts/evaluate_vlmevalkit.py \
    --model_path outputs/stage3 \
    --stage 3 \
    --output_dir eval_outputs/stage3 \
    --log_wandb
```

See main README for full evaluation guide.

## Next Steps

### Stage 4: Reasoning Integration (Optional)

After Stage 3, you can optionally run Stage 4 to further improve reasoning quality:

```bash
torchrun --nproc_per_node=2 scripts/train_all.py \
    --stage 4 \
    --distributed \
    --reasoning_data reasoning-augmented-dataset
```

Stage 4 adds:
- Enhanced reasoning quality metrics
- Reasoning chain verification
- Improved multi-step reasoning coherence

### Deployment

After training, deploy the model:

```bash
# Export to ONNX for production
python scripts/export_onnx.py --model_path outputs/stage3

# Quantize for edge devices
python scripts/quantize_model.py --model_path outputs/stage3 --bits 8

# Deploy with FastAPI
python scripts/serve_api.py --model_path outputs/stage3 --port 8000
```

## FAQ

**Q: Can I skip Stage 1 and 2?**  
A: No. Stage 3 requires vision-language alignment (Stage 1) and instruction tuning (Stage 2) for effective robot selection.

**Q: How many epochs should I train?**  
A: 30 epochs is recommended. You can train longer (50-100) if validation accuracy plateaus below 85%.

**Q: What if I only have 1 GPU?**  
A: Remove `--distributed` and `torchrun`, use `python scripts/train_all.py --stage 3` directly. Training will be ~2× slower.

**Q: Can I add custom robot types?**  
A: Yes! Modify `embervlm/models/reasoning_heads.py` and retrain from Stage 3. Update `num_robots` in model config.

**Q: Does Stage 3 do reasoning automatically?**  
A: Yes! The model generates chain-of-thought reasoning for every prediction. You can access it via `return_reasoning=True` in inference.

## Citation

If you use EmberVLM Stage 3 in your research:

```bibtex
@software{embervlm_stage3_2025,
  title={EmberVLM Stage 3: Robot Fleet Selection with Chain-of-Thought Reasoning},
  author={Your Name},
  year={2025},
  url={https://github.com/euhidaman/EmberVLM}
}
```

---

**Last Updated:** January 7, 2026  
**Version:** 1.1 (Checkpoint loading fix included)

