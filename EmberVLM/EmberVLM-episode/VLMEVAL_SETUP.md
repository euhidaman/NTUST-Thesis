# VLMEvalKit Setup for EmberVLM

## Quick Setup (Recommended)

Run the automated setup script:

```powershell
python setup_vlmeval.py
```

This will:
1. ✅ Install VLMEvalKit in development mode
2. ✅ Install all required dependencies
3. ✅ Verify the installation

## Manual Setup

If you prefer to install manually:

### 1. Install VLMEvalKit

```powershell
cd ..\VLMEvalKit
pip install -e .
cd ..\EmberVLM
```

### 2. Install Additional Dependencies

```powershell
pip install -r requirements.txt
```

The key packages needed for benchmarking:
- `openpyxl` - Excel file handling
- `apted` - Tree edit distance
- `colormath` - Color processing
- `decord` - Video processing (optional)
- `distance` - String distance metrics

### 3. Verify Installation

```python
python -c "import vlmeval; print('✅ VLMEvalKit installed successfully!')"
```

## Usage

### Training with Benchmarking

To enable benchmarking after Stage 2:

```powershell
python scripts/train_all.py --run_benchmarks --benchmark_preset quick
```

### Training without Benchmarking

If VLMEvalKit is not installed or you want to skip benchmarking:

```powershell
python scripts/train_all.py --skip_benchmarks
# or simply don't use --run_benchmarks (benchmarking is opt-in)
```

### Benchmark Presets

- **quick** (~30 minutes): 3 benchmarks, 500 samples each
  - Good for rapid testing

- **standard** (~1-2 hours): 5-6 benchmarks, full test sets
  - Recommended for validation
  - Default preset

- **full** (~4-6 hours): 10+ benchmarks
  - Comprehensive evaluation

## Troubleshooting

### Import Error: vlmeval module not found

**Solution:** Run `python setup_vlmeval.py`

### Benchmark data not downloading

**Solution:** Check internet connection. VLMEvalKit downloads benchmark data automatically on first run. The data is cached in `~/.cache/huggingface/`

### Out of Memory during evaluation

**Solution:** Use smaller benchmark preset:
```powershell
python scripts/train_all.py --run_benchmarks --benchmark_preset quick
```

### Slow benchmark execution

**Solution:** Benchmarking is compute-intensive. Consider:
- Using GPU for inference
- Reducing batch size in VLMEvalKit config
- Running overnight for full evaluation

## What Gets Downloaded?

When you run benchmarks for the first time, VLMEvalKit will download:

1. **Benchmark datasets** (~2-5 GB):
   - MMBench, TextVQA, ScienceQA, etc.
   - Cached in `~/.cache/huggingface/datasets/`

2. **Evaluation scripts**:
   - Already included in VLMEvalKit

3. **No models** are downloaded (uses your trained EmberVLM)

## Supported Benchmarks

EmberVLM supports evaluation on:
- **Visual QA**: TextVQA, VQAv2, ScienceQA
- **Visual Reasoning**: MMBench, AI2D
- **OCR & Documents**: OCRBench, ChartQA
- **General Understanding**: SEED-Bench

Full list in `embervlm/evaluation/vlm_benchmarks.py`

## Advanced Configuration

### Custom VLMEvalKit Path

If VLMEvalKit is in a different location:

```powershell
python scripts/train_all.py --run_benchmarks --vlmeval_repo "path/to/VLMEvalKit"
```

### Quality Thresholds

Control when to stop training if VLM quality is insufficient:

```powershell
# Strict (85% of baseline) - stops if model is weak
python scripts/train_all.py --run_benchmarks --quality_threshold strict

# Permissive (50% of baseline) - allows weaker models
python scripts/train_all.py --run_benchmarks --quality_threshold permissive

# Skip gating (no quality check, always continue)
python scripts/train_all.py --run_benchmarks --quality_threshold skip
```

## Need Help?

- VLMEvalKit Issues: https://github.com/open-compass/VLMEvalKit/issues
- EmberVLM Issues: Check your repository issues
