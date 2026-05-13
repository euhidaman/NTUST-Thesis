# EmberVLM Training and Evaluation Guide

## Overview

EmberVLM uses a **four-stage training pipeline** with optional evaluation. This guide explains how to:
1. Train without evaluation (faster, recommended for initial runs)
2. Evaluate trained models separately
3. Understand checkpoint structure

---

## Training Without Evaluation

To skip evaluation during training and run faster:

```bash
# Skip all benchmarks
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=1 --master_port=29500 \
scripts/train_all.py \
  --stage all \
  --size small \
  --batch_size 4 \
  --gradient_accumulation 4 \
  --learning_rate 5e-5 \
  --output_dir outputs/trial_run \
  --lmms_eval_repo /root/lmms-eval \
  --skip_benchmarks \
  --check trial
```

The `--skip_benchmarks` flag disables Stage 2.5 evaluation, saving ~1-2 hours.

---

## Checkpoint Structure

After training, checkpoints are saved at multiple points:

### Automatic Checkpoints (during training)
```
outputs/trial_run/mobilevit_xs_smollm_135m/
├── stage1/
│   ├── checkpoint-epoch-1/         # After epoch 1
│   ├── checkpoint-epoch-2/         # After epoch 2
│   ├── checkpoint-epoch-N/         # After epoch N
│   └── final/                      # ✓ FINAL Stage 1 checkpoint
├── stage2/
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-N/
│   └── final/                      # ✓ FINAL Stage 2 checkpoint
├── stage3/
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-N/
│   └── final/                      # ✓ FINAL Stage 3 checkpoint
├── stage4/
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-N/
│   └── final/                      # ✓ FINAL Stage 4 checkpoint
└── final/                          # ✓ FINAL overall checkpoint
```

### Key Checkpoints for Evaluation

- **`stage1/final/`** - After visual-language alignment
- **`stage2/final/`** - After instruction tuning (best for general VLM evaluation)
- **`stage3/final/`** - After robot selection training
- **`stage4/final/`** - After reasoning integration
- **`final/`** - Complete trained model (all stages)

---

## Evaluating Trained Models

Use the standalone evaluation script to benchmark any checkpoint:

### Basic Evaluation

```bash
# Evaluate Stage 2 final checkpoint (recommended)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final

# Evaluate final model
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/final
```

### Evaluation Presets

```bash
# Mini preset (~30 minutes, quick check)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final \
  --preset mini

# Standard preset (~1-2 hours, balanced)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final \
  --preset standard

# Full preset (~4-6 hours, comprehensive)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final \
  --preset full
```

### Controlling Evaluation Tiers

The evaluation system uses **four tiers** in order:

1. **VLMEvalKit** (PRIMARY) - Rule-based scoring, no LLM judge
2. **Lighteval** - NLP and multimodal benchmarks  
3. **UniBench** - Visual reasoning benchmarks
4. **lmms-eval** - Final fallback

You can disable specific tiers:

```bash
# Only VLMEvalKit + lmms-eval (skip Lighteval and UniBench)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final \
  --disable_lighteval \
  --disable_unibench

# Use all four tiers (default)
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final
```

### Specifying Repository Paths

```bash
python scripts/evaluate_model.py \
  --model_path outputs/trial_run/mobilevit_xs_smollm_135m/stage2/final \
  --lmms_eval_repo /root/lmms-eval \
  --vlmeval_repo /root/VLMEvalKit
```

---

## Evaluation Output

Results are saved to:
```
outputs/trial_run/mobilevit_xs_smollm_135m/stage2_5_evaluation/
├── fallback_evaluation_results.json    # Aggregate results
├── lmms_eval_mmmu_val/                 # lmms-eval results
├── lmms_eval_mathvista_testmini/
├── vlmeval_mmstar/                     # VLMEvalKit results
├── lighteval_*/                        # Lighteval results (if enabled)
├── unibench_*/                         # UniBench results (if enabled)
└── evaluation_summary.json             # Summary with all benchmarks
```

### Reading Results

```python
import json

# Load summary
with open("outputs/.../stage2_5_evaluation/evaluation_summary.json") as f:
    summary = json.load(f)

print(f"Aggregate score: {summary['aggregate_score']:.2f}%")
print(f"Benchmarks:")
for bench, score in summary['benchmarks'].items():
    print(f"  {bench}: {score:.2f}%")

# Check which framework was used
print(f"\nFramework usage:")
print(f"  VLMEvalKit: {summary['framework_info']['vlmeval_succeeded']}")
print(f"  Lighteval: {summary['framework_info']['lighteval_succeeded']}")
print(f"  UniBench: {summary['framework_info']['unibench_succeeded']}")
print(f"  lmms-eval: {summary['framework_info']['lmms_eval_succeeded']}")
```

---

## Complete Workflow Examples

### Example 1: Quick Training + Separate Evaluation

```bash
# 1. Train without evaluation (fast)
torchrun --nproc_per_node=1 scripts/train_all.py \
  --stage all \
  --skip_benchmarks \
  --output_dir outputs/my_run \
  --check trial

# 2. Evaluate Stage 2 (general VLM capabilities)
python scripts/evaluate_model.py \
  --model_path outputs/my_run/mobilevit_xs_smollm_135m/stage2/final \
  --preset mini

# 3. Evaluate final model (all capabilities)
python scripts/evaluate_model.py \
  --model_path outputs/my_run/mobilevit_xs_smollm_135m/final \
  --preset standard
```

### Example 2: Training with Integrated Evaluation

```bash
# Train with evaluation after Stage 2
torchrun --nproc_per_node=1 scripts/train_all.py \
  --stage all \
  --output_dir outputs/my_run \
  --benchmark_preset mini \
  --quality_threshold skip \
  --check trial
```

The `--quality_threshold skip` allows evaluation to run but doesn't block training if scores are low.

### Example 3: Evaluate Multiple Checkpoints

```bash
# Compare different stages
for stage in stage1 stage2 stage3 stage4 final; do
  echo "Evaluating $stage..."
  python scripts/evaluate_model.py \
    --model_path outputs/my_run/mobilevit_xs_smollm_135m/$stage/final \
    --preset mini \
    --output_dir outputs/my_run/eval_$stage
done
```

---

## FAQ

**Q: Which checkpoint should I evaluate?**
A: For general VLM evaluation, use `stage2/final`. For robot-specific tasks, use `stage3/final` or `final`.

**Q: Can I evaluate during training?**
A: Yes, remove `--skip_benchmarks`. Use `--quality_threshold skip` to prevent training from stopping if scores are low.

**Q: How long does evaluation take?**
A: 
- Mini preset: ~30 minutes
- Standard preset: ~1-2 hours  
- Full preset: ~4-6 hours

**Q: What if Lighteval or UniBench aren't installed?**
A: The system automatically skips them and uses VLMEvalKit + lmms-eval only.

**Q: How do I know which framework evaluated each benchmark?**
A: Check `framework_info` in `evaluation_summary.json` or look at the logs.

---

## Constraints

All evaluation is:
- ✅ **FREE** - No paid APIs (OpenAI, Anthropic, etc.)
- ✅ **OFFLINE** - Runs completely locally
- ✅ **RULE-BASED** - No LLM-as-a-judge scoring
- ✅ **DETERMINISTIC** - Same input = same output

---

## Troubleshooting

### "No results from evaluation"
- Check logs for errors
- Verify model path exists
- Try with `--preset mini` first

### "VLMEvalKit failed"
- Ensure `/root/VLMEvalKit` is cloned
- Check dependencies are installed
- System falls back to Lighteval/UniBench/lmms-eval

### "All tiers failed"
- Check CUDA is available
- Verify model checkpoint is valid
- Try loading model manually to debug

---

For more details, see:
- `scripts/train_all.py` - Main training script
- `scripts/evaluate_model.py` - Standalone evaluation script
- `embervlm/training/stage2_5_eval.py` - Evaluation implementation

