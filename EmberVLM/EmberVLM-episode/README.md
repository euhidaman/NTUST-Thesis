# EmberVLM: Tiny Vision-Language Model for Robot Fleet Selection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.36+](https://img.shields.io/badge/Transformers-4.36%2B-blueviolet.svg)](https://huggingface.co/docs/transformers/)
[![DINOv2-Small](https://img.shields.io/badge/Vision-DINOv2--Small-green.svg)](https://huggingface.co/facebook/dinov2-small)
[![SmolLM-135M](https://img.shields.io/badge/Language-SmolLM--135M-orange.svg)](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

EmberVLM is a lightweight multimodal Vision-Language Model that pairs a frozen DINOv2-Small vision encoder with an efficient SmolLM-135M language backbone for robot fleet selection with chain-of-thought reasoning. The architecture is deliberately compact (~164M total parameters, ~40M trainable) so that training and inference fit on a single consumer GPU while still achieving visually-grounded instruction following.

A distinguishing feature is the **Visual-Absence (VA) Refiner**, an inference-time hallucination suppression module. It hooks into the middle feed-forward layers of SmolLM, extracts neuron-level activation features, and classifies each generated token as visually grounded or visually absent. Tokens that score above a type-aware threshold are penalised or blocked, reducing hallucinated visual content without retraining the backbone.

Training follows a four-stage progressive curriculum — vision-language alignment, instruction tuning with teacher distillation, robot fleet selection, and chain-of-thought reasoning — plus an automated evaluation gate (Stage 2.5) that benchmarks the checkpoint before proceeding to downstream tasks. The full pipeline is launched with a single `torchrun` command and supports distributed data-parallel training across multiple GPUs in `bf16` mixed precision. Carbon emissions are tracked via CodeCarbon, and trained checkpoints can be pushed to HuggingFace Hub automatically.

---

## Architecture

```text
                                  EmberVLM Architecture
       ┌────────────────┐                          ┌───────────────────────────┐
       │     Image      │                          │        Instruction        │
       └───────┬────────┘                          └─────────────┬─────────────┘
               │                                                 │
       ┌───────▼────────┐                          ┌─────────────▼─────────────┐
       │  DINOv2-Small  │ (Frozen)                 │        SmolLM-135M        │
       │  (258 x 384d)  │                          │    (30 Decoder Layers)    │
       └───────┬────────┘                          └─────────────▲─────────────┘
               │                                                 │
               │         ┌────────────────────────┐              │
               └────────▶│  Gated Fusion Module   ├──────────────┘
                         │ (Adaptive Projection)  │
                         └───────────┬────────────┘
                                     │   α · sigmoid(fusion_gate)
                                     │
                         ┌───────────▼────────────┐
                         │    Reasoning Module    │
                         │ ┌────────────────────┐ │
                         │ │   ReasoningHead    │ │ (4-step CoT)
                         │ ├────────────────────┤ │
                         │ │ RobotSelectionHead │ │ (5-way Classifier)
                         │ ├────────────────────┤ │
                         │ │ ActionPlanningHead │ │ (GRU-based steps)
                         │ └────────────────────┘ │
                         └────────────────────────┘
```

The model architecture is built on a frozen **DINOv2-Small** vision encoder and an efficient **SmolLM-135M** language backbone. A learnable **Fusion Module** bridges the visual features into the language model's embedding space, enabling instruction-following and robot-selection capabilities. 

- **Vision Encoder (DINOv2-Small)**: Processes 224x224 RGB images into 258 visual tokens (384-dim). The parameters (22M) remain frozen during all training stages to preserve powerful pretrained visual representations.
- **Fusion Module**: A gated bottleneck adapter ($sigmoid(\alpha)$) with visual self-attention and an image summary token. It projects the 384-dim vision features to the 576-dim language space.
- **Language Backbone (SmolLM-135M)**: A 30-layer Llama-style decoder. During training, only the final layer and the LM head are typically unfrozen to focus on downstream tasks while maintaining general knowledge.
- **Reasoning Module**: Adds specialized heads for chain-of-thought (ReasoningHead), 5-way robot classification (RobotSelectionHead), and structured plan generation (ActionPlanningHead).
- **Forward Paths**:
  - `forward`: Combines multimodal inputs for general text generation and training.
  - `forward_vision_only`: Bypasses text embedding for efficient Stage 3 robot selection training.

### VA Refiner — Hallucination Suppression

The Visual-Absence (VA) Refiner is an inference-time plugin that identifies and penalizes visually ungrounded tokens using hidden states from the language model's middle layers.

```text
                           VA Refiner Mechanism (Inference-time)
 ┌─────────────┐
 │ SmolLM Hidden States (Layers 10-14, MLP)
 └──────┬──────┘
        │   Extract Neuron Activations (Top-128 / Layer)
        ▼
 ┌─────────────┐             ┌─────────────┐             ┌─────────────┐
 │ VAFeature   │             │ VA Classifier│             │  p_VA Score │
 │ Extractor   ├────────────▶│    (MLP)     ├────────────▶│  (Probability)
 └─────────────┘             └─────────────┘             └──────┬──────┘
                                                                │
        ┌───────────────────────────────────────────────────────┘
        │   Dual-View Filtering (Grounding Discrepancy)
        ▼
 ┌─────────────┐                                         ┌─────────────┐
 │ Next Token  │      Logit Penalty (α · p_VA)           │  Final Token│
 │ Raw Logits  ├────────────────┬──────────────────────▶│  Prediction │
 └─────────────┘                │                        └─────────────┘
                                ▼
                       Keyword Thresholds
                    (Visual: 0.6, Non-visual: 0.8)
```

- **Mechanism**: Hooks into the middle feed-forward layers of SmolLM and extracts neuron activations. A lightweight MLP (VAClassifier) predicts the probability of a token being "visually absent."
- **Visual Grounding**: Uses a dual-view approach (logits with vs. without image) to assess grounding. High discrepancy indicates a token is strongly tied to the image.
- **Dynamic Thresholding**: Applies stricter thresholds (0.6) for visual keywords (colors, objects) and looser ones (0.8) for non-visual tokens.
- **Burst Mitigation**: Monitors a sliding window of VA scores. If hallucination is detected, it enters a "burst" mode, applying hard penalties to prevent degenerate cascades.

### Generation Loop

The generation process uses an enhanced version of the autoregressive loop in `_safe_autoregressive_generation`:
- **Safe Sampling**: Validates token IDs to prevent crashes and supports `top-k`, `top-p`, and temperature scaling.
- **Repetition Control**: Employs repetition penalties and n-gram blocking (default trigram) to ensure diversity.
- **VA Integration**: When enabled, the VA Refiner adjusts the token distribution before sampling at each step.
- **Overflow Protection**: Automatically stops if the generation reaches the 2,048-token limit or emits an EOS token.


### Episodic Memory — Multimodal Knowledge Updates  *(episode branch)*

The episodic memory module enables **training-free knowledge injection** after
deployment.  New image–text episodes are written into a fixed-size memory matrix
and retrieved at inference time to condition generation.

```
  Multimodal Input  ──►  Fusion Module  ──►  LM Hidden States
       (Image + Text)        │                        │
                             ▼                        │
                      Scope Detector  ── Y ──►  Memory Read  ── z_m
                        (in scope?)               │           │
                             │                    ▼           ▼
                             └──── N ──►    logits + z_m  (residual)
                                                  │
                                                  ▼
                                           Final Prediction
```

- **Memory Matrix M** (K × C): K = 512 slots of C = 576 fused embeddings.
  Gaussian or pseudoinverse addressing computes softmax-normalised weights.
- **Scope Detector**: 2-layer MLP with GELU; outputs p(use memory) ∈ [0, 1].
  Trained in Stage 1.5 on memory-populated data.
- **Novelty Gating**: σ(z|M) = min_k ‖z − M_k‖².  Only episodes with
  σ > threshold_novel are written; near-duplicates are skipped.
- **LRU + LFU Replacement**: When all slots are occupied, the least-recently
  and least-frequently used slot is overwritten.
- **Consolidation** (Stage 5): Frozen memory serves as teacher targets for
  an MSE alignment loss that bakes retrieved knowledge into model weights.

Key public APIs on `EmberVLM`:

| Method                          | Description                                |
| ------------------------------- | ------------------------------------------ |
| `update_memory(z)`              | Smart-write with novelty gating            |
| `forget_memory(slot_index=...)` | Selective erase via negative-alpha write    |
| `save_memory_state(path)`       | Serialise M, cov, metadata to disk         |
| `load_memory_state(path)`       | Restore memory from checkpoint             |

Training stages added:

| Stage | Name             | Purpose                                     |
| ----- | ---------------- | ------------------------------------------- |
| 1.5   | memory_init      | Populate memory matrix, train scope detector|
| 5     | consolidation    | Distil memory content into model weights    |

Config: `configs/episode_config.yaml`.  All W&B / HF runs are isolated under
the project/repo name **EmberVLM-Episode** so they do not pollute the baseline
`hallucinate` branch results.


---

## Model Configuration

| Component        | Identifier                     | Params    | Trainable | Output Shape        |
| ---------------- | ------------------------------ | --------- | --------- | ------------------- |
| Vision Encoder   | `facebook/dinov2-small`        | 22.1M     | 0         | 258 × 384           |
| Fusion Module    | AdapterBlock + gate            | ~2.5M     | ~2.5M     | 258 × 576           |
| Language Model   | `HuggingFaceTB/SmolLM-135M`    | 134.5M    | ~32M      | seq × 49 152 logits |
| Reasoning Module | ReasoningHead + Robot + Action | ~5M       | ~5M       | 5-way + plan steps  |
| VA Refiner       | FeatureExtractor + Classifier  | < 0.1M    | 0 (infer) | per-token p_VA      |
| **Total**        |                                | **~164M** | **~40M**  |                     |

Nine special tokens are added to the vocabulary during initialisation:

| Token                                           | Purpose                                             |
| ----------------------------------------------- | --------------------------------------------------- |
| `<image>` / `</image>`                          | Delimit the visual token span in the input sequence |
| `<question>` / `</question>`                    | Wrap the user query                                 |
| `<answer>`                                      | Marks the start of the model's generated response   |
| `<\|reasoning_start\|>` / `<\|reasoning_end\|>` | Wrap the chain-of-thought reasoning block           |
| `<\|robot_selection\|>`                         | Precedes the robot selection output                 |
| `<\|action_plan\|>`                             | Precedes the step-by-step action plan               |

---

## Training Pipeline

The training pipeline is a four-stage progressive curriculum with an automated evaluation gate (Stage 2.5) to ensure model quality before downstream robot mapping.

```text
                               Progessive Curriculum Training
 ┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
 │   Stage 1     │─────▶│   Stage 2     │─────▶│   Stage 2.5   │─────▶│   Stage 3     │
 │  Alignment    │      │   Instruct    │      │   Eval Gate   │      │   Robot       │
 └───────┬───────┘      └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
         │                      │                      │                      │
   Vision/Language        Multimodal SFT         MBench/TextVQA         Fleet Selection
   (Frozen Backbones)      (Distillation)        (Pass/Fail Gate)       (Head-only SFT)
         │                      │                      │                      │
         └──────────────────────┴───▶ Stage 4: CoT Reasoning & Fine-Tuning ───┘
                                      (Full joint training)
```

| Stage   | Name      | Key Focus              | Trainable Components            |
| ------- | --------- | ---------------------- | ------------------------------- |
| **1**   | Alignment | Vision-Language bridge | Fusion Module (Adapter + Gate)  |
| **2**   | Instruct  | Follow instructions    | Fusion + Final SmolLM Layer     |
| **2.5** | Eval Gate | Quality benchmark      | (None - Evaluation only)        |
| **3**   | Robot     | Fleet selection        | Reasoning Module (RobotHead)    |
| **4**   | Reasoning | Chain-of-Thought       | All trainable heads + Backbones |

### Stage-by-Stage Breakdown

- **Stage 1 (Alignment)**: Aligns DINOv2 visual tokens with SmolLM embeddings using contrastive and captioning losses.
  - *Datasets*: CC3M (lazy-loaded), GQA, RefCOCO.
- **Stage 2 (Instruction Tuning)**: Conversational fine-tuning, optionally with teacher distillation from larger models (e.g., Qwen2-VL).
  - *Datasets*: LLaVA-Instruct-150K, VQAv2, OK-VQA.
- **Stage 2.5 (Evaluation Gate)**: Automated quality check using `lmms-eval` benchmarks before proceeding to downstream tasks.
- **Stage 3 (Robot Fleet Selection)**: Trains the `RobotSelectionHead` on mission-specific scenarios using Focal Loss ($\gamma=2.0$).
  - *Fleet*: Drone (Aerial), Underwater, Humanoid, Wheels, Legs.
- **Stage 4 (CoT Reasoning)**: Generates 4-step chain-of-thought and action plans via scheduled sampling.

### Hyperparameter Defaults

| Parameter  | Alignment (S1) | Instruct (S2) | Robot (S3) | Reasoning (S4) |
| ---------- | -------------- | ------------- | ---------- | -------------- |
| Max Steps  | 15,000         | 20,000        | 20,000     | 10,000         |
| Learn Rate | 2e-4           | 2e-4          | 1e-4       | 5e-5           |
| Batch Size | 32/GPU         | 32/GPU        | 32/GPU     | 16/GPU         |

### Visualisation and Monitoring

Diagnostic tools generate curves and plots at each stage:
- **Embedding Quality**: t-SNE / PCA clustering of visual tokens.
- **Classification**: Confusion matrices and calibration error (ECE) for robot selection.
- **Hallucination**: Per-token VA score heatmaps over generated responses.
- **Monitoring**: Carbon emissions tracked via CodeCarbon.


---

## Quick Start

### Install

```bash
git clone https://github.com/euhidaman/EmberVLM.git
cd EmberVLM
git checkout hallucinate
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Train

```bash
torchrun --nproc_per_node=2 scripts/train_all.py \
    --size medium --distributed --mixed_precision bf16 \
    --batch_size 64 --gradient_accumulation 4 \
    --stage1_data data/base_vlm --stage2_data data/base_vlm/llava \
    --robot_data robot-selection-dataset --check main
```
- `--size medium`: Selects DINOv2-Small + SmolLM-135M.
- `--check main`: Runs full training (use `trial` for fast validation).

### Inference

```python
from embervlm.models import EmberVLM, EmberVLMConfig
from transformers import AutoTokenizer

config = EmberVLMConfig(vision_backbone="dinov2_small", language_backbone="smollm_135m")
model = EmberVLM(config).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
model.sync_tokenizer_and_embeddings(tokenizer)

# Prepare multimodal input
pixel_values = model.image_preprocessor(image).unsqueeze(0).cuda()
prompt = "<image></image><question>Which robot should respond?</question><answer>"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# Generate with VA Refiner
generated = model.generate(input_ids=input_ids, pixel_values=pixel_values, 
                           use_va_refiner=True, tokenizer=tokenizer)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

- **Robot Selection**: Call `model.forward_vision_only(pixels)` for high-speed fleet mapping.
- **Incident Analysis**: Use `model.analyze_incident(pixels, text)` for end-to-end reasoning and ranking.
- **Ranking**: `model.select_robots_topn(...)` provides a multi-robot fallback list.


---

## Datasets

| Stage | Source                  | Purpose                       | Loading    |
| ----- | ----------------------- | ----------------------------- | ---------- |
| 1     | CC3M / GQA / RefCOCO    | Vision-Language Alignment     | Lazy/Eager |
| 2     | LLaVA / VQAv2 / OK-VQA  | Multi-turn Instruction Tuning | Eager      |
| 3     | Robot selection (5 cls) | Fleet mission mapping         | Eager      |
| 4     | Reasoning-augmented     | Chain-of-Thought generation   | Eager      |

*Note: Memory-safe caps (e.g., 4M for Stage 1) are enforced automatically to prevent OOM on shared machines.*


---

## Evaluation

### VLM Benchmark Setup

Run `python setup_vlmeval.py` to auto-clone and install `lmms-eval`.

### Benchmark Presets

The Stage 2.5 gate and `scripts/evaluate_vlmevalkit.py` support three presets:
- `mini`: MMBench-EN-dev, TextVQA-val (~30 min).
- `standard`: Adds GQA, ScienceQA, POPE, AI2D (~2 hr).
- `full`: Adds MMMU, MathVista, RealWorldQA (~6 hr).

```bash
python scripts/evaluate_vlmevalkit.py --preset standard --model_path path/to/ckpt
```

### Robot-Selection Evaluation

Computed by `scripts/evaluate.py`:
- **Metrics**: Per-class F1, macro F1, and confusion matrices.
- **Calibrartion**: Expected Calibration Error (ECE) for mission confidence.
- **Ranking**: Top-N accuracy via `select_robots_topn()`.

---

## Troubleshooting

- **CUDA OOM**: Reduce `--batch_size` or `--gradient_accumulation`. Cache is cleared automatically every 100 steps.
- **`lmms-eval` missing**: Run `python setup_vlmeval.py`.
- **Size Mismatch**: Ensure `model.sync_tokenizer_and_embeddings(tokenizer)` is called.
- **NCCL Timeout**: Set `NCCL_TIMEOUT=1800` for distributed runs.
- **Latency**: Disable VA Refiner with `use_va_refiner=False` if speed is priority.


---

## Project Structure

```text
EmberVLM/
├── embervlm/
│   ├── models/
│   │   ├── embervlm.py        # Main model & config
│   │   ├── vision_encoder.py  # DINOv2-Small (frozen)
│   │   ├── language_model.py  # SmolLM (Llama-base)
│   │   ├── fusion_module.py   # Gated Adapter & Vision-SA
│   │   ├── hallucination.py   # VA Refiner & Feature Extractor
│   │   └── reasoning_heads.py # Robot Selection & Plan heads
│   └── data/
│       ├── loaders.py         # S1/S2 data loading
│       └── robot_loader.py    # S3/S4 robot data
├── scripts/
│   ├── train_all.py           # Training orchestrator
│   ├── evaluate.py            # Robot metrics
│   └── sample_inference.py    # Quick demo
├── configs/                   # Hyperparameter yaml/json
└── robot-selection-dataset/   # Fleet scenario data
```


---

## Citation

```bibtex
@misc{embervlm2026,
  title   = {EmberVLM: Tiny Vision-Language Model for Robot Fleet Selection},
  author  = {euhidaman},
  year    = {2026},
  url     = {https://github.com/euhidaman/EmberVLM}
}
```

## License

MIT — see [LICENSE](LICENSE).

