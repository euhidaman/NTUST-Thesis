# 🚀 Quick Start: EmberVLM Benchmarking

## ⚡ Standalone Evaluation (Already Trained Model)

If you've already completed training, run benchmarks separately:

```bash
# On Linux/Unix
python scripts/evaluate_vlmevalkit.py \
  --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10 \
  --preset standard

# On Windows PowerShell
python scripts/evaluate_vlmevalkit.py `
  --model_path outputs\mobilevit_xs_smollm_135m\stage2\checkpoint-epoch-10 `
  --preset standard
```

### Benchmark Presets

- **mini**: 2 benchmarks (~30 minutes) - Quick validation
- **standard**: 6 benchmarks (~90 minutes) - Recommended  
- **full**: 10+ benchmarks (~5 hours) - Comprehensive

### Custom Benchmarks

```bash
python scripts/evaluate_vlmevalkit.py \
  --model_path outputs/mobilevit_xs_smollm_135m/stage2/checkpoint-epoch-10 \
  --benchmarks MMBench_DEV_EN_V11 TextVQA_VAL ScienceQA_IMG \
  --output_dir ./my_benchmark_results
```

## 🔧 Setup VLMEvalKit (First Time Only)

```powershell
# 1. Run the setup script
python setup_vlmeval.py

# 2. Verify installation
python -c "import vlmeval; print('VLMEvalKit OK')"
```

## 🎓 Training With Integrated Benchmarking

## 🎓 Training With Integrated Benchmarking

```powershell
# 1. Set environment
$env:HF_TOKEN = "your_hf_token_here"

# 2. Start training (benchmarks run automatically after Stage 2)
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --stage all `
  --quality_threshold standard `
  --distributed `
  --mixed_precision bf16
```

**Note**: Use `--quality_threshold skip` to disable benchmarking during training.

## 📦 What Gets Installed?

1. **VLMEvalKit** - Benchmark evaluation framework
2. **Dependencies** - openpyxl, apted, colormath, distance, decord

**Total size**: ~500 MB (including benchmark data on first run)

## 🎯 Training Withourun benchmarks during training:

```powershell
# Use --quality_threshold skip to bypass Stage 2.5 evaluation
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --stage all `
  --quality_threshold skip `
  --distributed `
  --mixed_precision bf16
```
Evaluates

After Stage 2 (Instruction Tuning), your model is evaluated on:

### Standard Preset (6 benchmarks, ~90 min)
- **MMBench** - General VLM understanding
- **TextVQA** - Text recognition in images  
- **ScienceQA** - Scientific reasoning
- **AI2D** - Diagram understanding
- **ChartQA** - Chart interpretation
- **SEED_IMG** - Multi-modal comprehension

### Output

Results are saved to:
- `outputs/mobilevit_xs_smollm_135m/stage2_5_evaluation/evaluation_results.json`
- WandB dashboard (if initialized) with comparison tables

### Quality Thresholds

- **strict** (85%): Must achieve 85% of baseline scores
- **standard** (70%): Must achieve 70% of baseline scores  
- **permissive** (50%): Must achieve 50% of baseline scores
- **skip**: No quality check - benchmarks skipped entirely
- **ScienceQA** - Scientific reasoning
- **AI2D** - Diagram understanding
- **ChartQA** - Chart interpretation

Results are logged to WandB with comparison tables and visualizations.

## ⚙️ Benchmark Presets

- `--benchmark_preset quick` - 3 benchmarks, ~30 min ⚡
- `--benchmark_preset standard` - 5-6 benchmarks, ~1-2 hours (default) ✅
- `--benchmark_preset full` - 10+ benchmarks, ~4-6 hours 🔬

## 🛠️ Troubleshooting

### Issue: "VLMEvalKit not installed"
**Solution:** Run `python setup_vlmeval.py`

### Issue: Out of memory during benchmarking
**Solution:** Use `--benchmark_preset quick`

### Issue: Benchmarks taking too long
**Solution:** Run training overnight, or use `--skip_benchmarks`

## 📚 Full Documentation

See [VLMEVAL_SETUP.md](VLMEVAL_SETUP.md) for complete setup guide and advanced options.

## ✨ New Features Added

### Enhanced VLM Training:
- ✅ **Perplexity** metric for Stage 2 (language quality)
- ✅ **Top-5 accuracy & MRR** for Stage 1 (retrieval quality)  
- ✅ **Early stopping** with best checkpoint tracking
- ✅ **Increased epochs**: Stage 1: 3→7, Stage 2: 3→10

### Stage 2.5: Benchmark Evaluation:
- ✅ **Automated VLM benchmarking** using VLMEvalKit
- ✅ **Quality gating** - stops training if VLM is too weak
- ✅ **WandB logging** with comparison tables and charts
- ✅ **Graceful fallback** if VLMEvalKit not installed

### Training Visualizations:
- ✅ **visualize_training.py** script for plot generation
- ✅ **Stage summaries** with improvement statistics
- ✅ **Convergence tracking** for all stages

## 🎓 Updated Training Command

### With Benchmarking (Recommended):
```powershell
$env:HF_TOKEN = "hf_..."
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --run_benchmarks `
  --benchmark_preset standard `
  --quality_threshold auto `
  --stage all `
  --distributed `
  --mixed_precision bf16 `
  --batch_size 18 `
  --gradient_accumulation 16 `
  --stage1_epochs 7 `
  --stage2_epochs 10 `
  --stage3_robot_epochs 30 `
  2>&1 | Tee-Object -FilePath train.log
```

### Without Benchmarking (Faster):
```powershell
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --stage all `
  --distributed `
  --mixed_precision bf16 `
  --batch_size 18 `
  --gradient_accumulation 16 `
  2>&1 | Tee-Object -FilePath train.log
```

## 📈 Monitor Training

- **WandB Dashboard**: Real-time metrics, tables, and visualizations
- **Training Log**: `train.log` file with detailed progress
- **Checkpoints**: Saved in `outputs/` with best models marked

---

**Ready to train?** Run `python setup_vlmeval.py` and start benchmarking! 🎉
