"""
EmberVLM Master Training Script

Orchestrates all four training stages:
1. Visual-Language Alignment
2. Multimodal Instruction Tuning
3. Robot Fleet Selection Training
4. Chain-of-Thought Reasoning Integration

MEMORY SAFE: Implements safeguards for distributed training on shared servers.
"""

import os
import argparse
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText

from embervlm.models import (
    EmberVLM,
    EmberVLMConfig,
    BACKBONE_TINYLLM,
    BACKBONE_SMOLLM_135M,
    VISION_BACKBONE_REPVIT,
    VISION_BACKBONE_MOBILEVIT_XS,
    EpisodicMemoryController,
    ScopeDetector,
)
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    set_seed,
)
from embervlm.training.stage1_align import run_stage1_training
from embervlm.training.stage2_instruct import run_stage2_training
from embervlm.training.stage2_5_eval import run_stage2_5_evaluation
from embervlm.training.stage3_incidents import run_stage3_training
from embervlm.training.stage4_reasoning import run_stage4_training
from embervlm.monitoring.paper_eval_visualizations import generate_robot_topn_visualizations
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

# Paper figure generation (runs automatically after training on rank 0)
_PAPER_FIGURES_AVAILABLE = False
_PAPER_FIGURES_IMPORT_ERROR = None
try:
    from scripts.generate_paper_figures import generate_suite as _generate_paper_figures
    _PAPER_FIGURES_AVAILABLE = True
except ImportError:
    try:
        # Fallback: direct import when running from repo root
        import importlib.util, sys as _sys
        _spec = importlib.util.spec_from_file_location(
            "generate_paper_figures",
            str(Path(__file__).resolve().parent / "generate_paper_figures.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _generate_paper_figures = _mod.generate_suite
        _PAPER_FIGURES_AVAILABLE = True
    except Exception as _import_err:
        _PAPER_FIGURES_AVAILABLE = False
        _PAPER_FIGURES_IMPORT_ERROR = str(_import_err)
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"Paper figure module could not be imported: {_import_err}"
        )

# Import sample inference for visual output checking
SAMPLE_INFERENCE_AVAILABLE = False
SAMPLE_INFERENCE_IMPORT_ERROR = None
try:
    # Try importing from same directory (scripts/)
    import sys
    from pathlib import Path as _Path
    _script_dir = _Path(__file__).parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    from sample_inference import run_sample_inference
    SAMPLE_INFERENCE_AVAILABLE = True
except ImportError as e:
    SAMPLE_INFERENCE_IMPORT_ERROR = str(e)

import torch.nn as nn

# Set environment variables for memory safety BEFORE any CUDA operations
# Prevent OpenMP thread explosion
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')  # Prevent MKL thread explosion
# Prevent tokenizer warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# NCCL configuration for distributed training stability
os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 min timeout (up from 10 min default)
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Enable async error handling
os.environ.setdefault('NCCL_DEBUG_SUBSYS', 'WARN')  # Only show warnings
# CUDA memory configuration (prefer the new env var name)
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # Backward compat

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not SAMPLE_INFERENCE_AVAILABLE and SAMPLE_INFERENCE_IMPORT_ERROR:
    logger.warning(f"sample_inference not available - skipping visual checks: {SAMPLE_INFERENCE_IMPORT_ERROR}")


# Default pretrained language model
PRETRAINED_LANGUAGE_MODEL = "tinyllm/30M-0.4"


def find_latest_checkpoint(stage_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a stage directory.

    Checkpoints are named like 'checkpoint-{step}' where step is an integer.
    Returns the checkpoint with the highest step number.

    Args:
        stage_dir: Path to the stage directory (e.g., outputs/stage2)

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not stage_dir.exists():
        return None

    checkpoints = []
    for item in stage_dir.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            try:
                step = int(item.name.split('-')[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                # Handle checkpoint-epoch-N format
                if 'epoch' in item.name:
                    try:
                        epoch_num = int(item.name.split('-')[-1])
                        # Use negative step numbers for epoch-based checkpoints
                        # so numbered step checkpoints take priority
                        checkpoints.append((-1000 + epoch_num, item))
                    except (ValueError, IndexError):
                        continue
                continue

    if checkpoints:
        # Sort by step number and return the latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    # Fall back to 'final' directory if no numbered checkpoints exist
    final_dir = stage_dir / 'final'
    if final_dir.is_dir():
        return final_dir

    return None


def get_previous_stage_checkpoint(output_dir: Path, current_stage: str) -> Optional[Path]:
    """
    Get the latest checkpoint from the previous training stage.

    Args:
        output_dir: Base output directory (e.g., ./outputs)
        current_stage: Current stage number as string ('2', '3', '4')

    Returns:
        Path to the latest checkpoint from the previous stage, or None
    """
    stage_num = int(current_stage)
    if stage_num <= 1:
        return None  # Stage 1 has no previous stage

    prev_stage = stage_num - 1
    prev_stage_dir = output_dir / f'stage{prev_stage}'

    checkpoint = find_latest_checkpoint(prev_stage_dir)
    if checkpoint:
        logger.info(
            f"Found latest checkpoint from Stage {prev_stage}: {checkpoint}")
    else:
        logger.warning(f"No checkpoint found in {prev_stage_dir}")

    return checkpoint


def create_model(config: Optional[EmberVLMConfig] = None) -> EmberVLM:
    """Create and initialize EmberVLM model."""
    if config is None:
        config = EmberVLMConfig()

    model = EmberVLM(config)

    # Print parameter info
    param_counts = model.count_parameters()
    logger.info(
        f"Model created with {param_counts['total']:,} total parameters")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Vision backbone: {config.vision_backbone}")
    logger.info(f"Language backbone: {config.language_backbone}")

    return model


def log_pre_training_dataset_summary(
    args,
    model: EmberVLM,
    tokenizer,
    training_config: TrainingConfig,
):
    """
    Log a comprehensive pre-training dataset summary with token counts and
    research-paper-relevant statistics.

    Called once on rank 0 before training begins. Lightweight: scans data
    directories and samples a few items to estimate token statistics without
    loading the full dataset into memory.
    """
    import json
    import time
    from pathlib import Path

    start_time = time.time()

    # ── Model & tokenizer info ──────────────────────────────────────────
    model_ref = model.module if hasattr(model, 'module') else model
    config = model_ref.config
    num_visual_tokens = getattr(config, 'num_visual_tokens', 258)
    image_size = getattr(config, 'image_size', 224)
    vision_dim = getattr(config, 'vision_output_dim', 384)
    language_dim = getattr(config, 'language_hidden_size', 576)
    vocab_size = len(tokenizer)
    max_length = 512  # default sequence length used by datasets
    is_trial = (args.check == 'trial')

    # ── Collect per-stage dataset statistics ─────────────────────────────
    stage_stats = {}

    def _count_json_samples(data_dir: Path, max_files: int = 50) -> dict:
        """Count samples across JSON files in a directory without loading images."""
        stats = {'total_samples': 0, 'json_files': 0, 'sources': {}}
        if not data_dir or not data_dir.exists():
            return stats

        json_files = sorted(data_dir.rglob('*.json'))[:max_files]
        skip_patterns = [
            'download_summary', 'dataset_info', 'instances_', 'person_keypoints_',
            '__MACOSX', '.lock', 'dataset.json',
        ]

        for jf in json_files:
            name_lower = jf.name.lower()
            if any(p in name_lower for p in skip_patterns):
                continue
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    # Some datasets wrap items under a key
                    for key in ['data', 'annotations', 'samples', 'questions']:
                        if key in data and isinstance(data[key], list):
                            count = len(data[key])
                            break
                    else:
                        count = 1
                else:
                    count = 0

                stats['total_samples'] += count
                stats['json_files'] += 1
                source_name = jf.parent.name if jf.parent != data_dir else jf.stem
                stats['sources'][source_name] = stats['sources'].get(source_name, 0) + count
            except Exception:
                continue
        return stats

    def _check_cc3m(data_dir: Path) -> int:
        """Check if CC3M dataset exists and estimate sample count."""
        if not data_dir:
            return 0
        cc3m_dir = data_dir / 'cc3m'
        if not cc3m_dir.exists():
            return 0
        # Check for arrow files (HuggingFace datasets cache)
        arrow_files = list(cc3m_dir.rglob('*.arrow'))
        if arrow_files:
            # Estimate from file sizes: ~1KB per sample in arrow format
            total_bytes = sum(f.stat().st_size for f in arrow_files)
            estimated = total_bytes // 1024
            return min(estimated, 3_300_000)  # Cap at CC3M max
        return 0

    def _check_refcoco(data_dir: Path) -> int:
        """Check for RefCOCO arrow data."""
        if not data_dir:
            return 0
        refcoco_count = 0
        for variant in ['refcoco', 'refcoco+', 'refcocog']:
            arrow_dir = data_dir / variant
            if arrow_dir.exists():
                arrow_files = list(arrow_dir.rglob('*.arrow'))
                for af in arrow_files:
                    refcoco_count += af.stat().st_size // 2048  # Rough estimate
        return refcoco_count

    def _estimate_text_tokens_from_sample(data_dir: Path, tokenizer, num_samples: int = 20) -> dict:
        """
        Sample a few JSON entries and tokenize them to estimate average text
        token count per sample (excluding padding).
        """
        result = {'avg_tokens': 0, 'min_tokens': 0, 'max_tokens': 0, 'sampled': 0}
        if not data_dir or not data_dir.exists():
            return result

        texts = []
        json_files = sorted(data_dir.rglob('*.json'))[:10]
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data[:num_samples]:
                    # Gather text from various fields
                    text_parts = []
                    for key in ['caption', 'text', 'instruction', 'response', 'question', 'answer', 'input', 'output']:
                        if key in item and isinstance(item[key], str):
                            text_parts.append(item[key])
                    if 'conversations' in item:
                        for conv in item['conversations']:
                            if 'value' in conv:
                                text_parts.append(conv['value'])
                    if text_parts:
                        texts.append(' '.join(text_parts))
                    if len(texts) >= num_samples:
                        break
            except Exception:
                continue
            if len(texts) >= num_samples:
                break

        if not texts:
            return result

        token_lengths = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
            non_pad = (enc['attention_mask'].sum()).item()
            token_lengths.append(int(non_pad))

        result['avg_tokens'] = int(sum(token_lengths) / len(token_lengths))
        result['min_tokens'] = min(token_lengths)
        result['max_tokens'] = min(max(token_lengths), max_length)
        result['sampled'] = len(token_lengths)
        return result

    # Stage 1: Visual-Language Alignment
    stage1_data_path = Path(args.stage1_data) if args.stage1_data else None
    if stage1_data_path and stage1_data_path.exists() and args.stage in ['all', '1']:
        json_stats = _count_json_samples(stage1_data_path)
        cc3m_count = _check_cc3m(stage1_data_path)
        refcoco_count = _check_refcoco(stage1_data_path)
        token_stats = _estimate_text_tokens_from_sample(stage1_data_path, tokenizer)

        total_samples = json_stats['total_samples'] + cc3m_count + refcoco_count
        if is_trial:
            from embervlm.data.loaders import MAX_STAGE1_TOTAL_SAMPLES_TRIAL
            total_samples = min(total_samples, MAX_STAGE1_TOTAL_SAMPLES_TRIAL)
        else:
            from embervlm.data.loaders import MAX_STAGE1_TOTAL_SAMPLES
            total_samples = min(total_samples, MAX_STAGE1_TOTAL_SAMPLES)

        sources = dict(json_stats['sources'])
        if cc3m_count > 0:
            sources['CC3M (lazy)'] = cc3m_count
        if refcoco_count > 0:
            sources['RefCOCO'] = refcoco_count

        stage_stats['Stage 1: Visual-Language Alignment'] = {
            'samples': total_samples,
            'sources': sources,
            'epochs': args.stage1_epochs,
            'steps_per_epoch': total_samples // max(args.batch_size, 1),
            'max_steps': args.stage1_steps or (total_samples // max(args.batch_size, 1) * args.stage1_epochs),
            'token_stats': token_stats,
            'has_images': True,
            'data_path': str(stage1_data_path),
        }

    # Stage 2: Instruction Tuning
    stage2_data_path = Path(args.stage2_data) if args.stage2_data else None
    if stage2_data_path and stage2_data_path.exists() and args.stage in ['all', '2']:
        json_stats = _count_json_samples(stage2_data_path)
        token_stats = _estimate_text_tokens_from_sample(stage2_data_path, tokenizer)

        total_samples = json_stats['total_samples']
        if is_trial:
            from embervlm.data.loaders import MAX_STAGE2_TOTAL_SAMPLES_TRIAL
            total_samples = min(total_samples, MAX_STAGE2_TOTAL_SAMPLES_TRIAL)
        else:
            from embervlm.data.loaders import MAX_STAGE2_TOTAL_SAMPLES
            total_samples = min(total_samples, MAX_STAGE2_TOTAL_SAMPLES)

        stage_stats['Stage 2: Instruction Tuning'] = {
            'samples': total_samples,
            'sources': json_stats['sources'],
            'epochs': args.stage2_epochs,
            'steps_per_epoch': total_samples // max(args.batch_size, 1),
            'max_steps': args.stage2_steps or (total_samples // max(args.batch_size, 1) * args.stage2_epochs),
            'token_stats': token_stats,
            'has_images': True,
            'data_path': str(stage2_data_path),
        }

    # Stage 3: Robot Fleet Selection
    robot_dir = Path(args.robot_data) if args.robot_data else None
    if robot_dir is None:
        robot_dir = Path(args.stage1_data).parent / 'robot_selection_data' if args.stage1_data else None
    if robot_dir and robot_dir.exists() and args.stage in ['all', '3']:
        json_stats = _count_json_samples(robot_dir)
        token_stats = _estimate_text_tokens_from_sample(robot_dir, tokenizer)

        stage_stats['Stage 3: Robot Fleet Selection'] = {
            'samples': json_stats['total_samples'],
            'sources': json_stats['sources'],
            'epochs': args.stage3_robot_epochs,
            'steps_per_epoch': json_stats['total_samples'] // max(args.batch_size, 1),
            'max_steps': args.stage3_robot_epochs * (json_stats['total_samples'] // max(args.batch_size, 1)),
            'token_stats': token_stats,
            'has_images': True,  # Scene images are generated procedurally
            'num_classes': 5,
            'class_names': ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs'],
            'data_path': str(robot_dir),
        }

    # Stage 4: CoT Reasoning
    reasoning_dir = None
    if args.stage in ['all', '4']:
        # Same logic as in training: check reasoning/, stage4_data/, robot_data/
        candidates = []
        if args.stage1_data:
            base = Path(args.stage1_data).parent
            candidates.extend([
                base / 'reasoning_data',
                base / 'robot_selection_data',
            ])
        if args.robot_data:
            candidates.append(Path(args.robot_data))

        for c in candidates:
            if c and c.exists():
                reasoning_dir = c
                break

        if reasoning_dir:
            json_stats = _count_json_samples(reasoning_dir)
            token_stats = _estimate_text_tokens_from_sample(reasoning_dir, tokenizer)

            total_phase_epochs = getattr(args, 'stage4_phase1_epochs', 5) + getattr(args, 'stage4_phase2_epochs', 5)
            stage_stats['Stage 4: CoT Reasoning'] = {
                'samples': json_stats['total_samples'],
                'sources': json_stats['sources'],
                'epochs': total_phase_epochs,
                'steps_per_epoch': json_stats['total_samples'] // max(args.batch_size, 1),
                'max_steps': total_phase_epochs * (json_stats['total_samples'] // max(args.batch_size, 1)),
                'token_stats': token_stats,
                'has_images': True,
                'data_path': str(reasoning_dir),
            }

    # ── Compute aggregates ───────────────────────────────────────────────
    grand_total_samples = sum(s['samples'] for s in stage_stats.values())
    grand_total_steps = sum(s['max_steps'] for s in stage_stats.values())

    # Estimate total tokens:
    # Text tokens = samples × avg_tokens_per_sample (from sampling)
    # Visual tokens = samples_with_images × num_visual_tokens
    total_text_tokens = 0
    total_visual_tokens = 0
    total_image_samples = 0
    for sname, sdata in stage_stats.items():
        avg_tok = sdata['token_stats'].get('avg_tokens', max_length // 2)
        if avg_tok == 0:
            avg_tok = max_length // 2  # fallback
        text_tok = sdata['samples'] * avg_tok
        total_text_tokens += text_tok
        sdata['est_text_tokens'] = text_tok

        if sdata['has_images']:
            vis_tok = sdata['samples'] * num_visual_tokens
            total_visual_tokens += vis_tok
            total_image_samples += sdata['samples']
            sdata['est_visual_tokens'] = vis_tok
        else:
            sdata['est_visual_tokens'] = 0

    total_tokens = total_text_tokens + total_visual_tokens

    # ── Format and log ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    separator = "=" * 78

    lines = [
        "",
        separator,
        "  PRE-TRAINING DATASET & TOKEN SUMMARY  (for research paper reference)",
        separator,
        "",
        "  Model Configuration",
        f"    Vision backbone     : {config.vision_backbone}",
        f"    Language backbone    : {config.language_backbone}",
        f"    Image resolution    : {image_size} x {image_size}",
        f"    Visual tokens/image : {num_visual_tokens}",
        f"    Vision hidden dim   : {vision_dim}",
        f"    Language hidden dim  : {language_dim}",
        f"    Max sequence length  : {max_length}",
        f"    Vocabulary size      : {vocab_size:,}",
        f"    Mode                : {'TRIAL' if is_trial else 'FULL'}",
        "",
        "  Tokenizer",
        f"    Name                : {tokenizer.name_or_path}",
        f"    Vocab size          : {vocab_size:,}",
        f"    Pad token           : {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})",
        f"    EOS token           : {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})",
        f"    BOS token           : {repr(tokenizer.bos_token)} (id={getattr(tokenizer, 'bos_token_id', None)})",
        "",
    ]

    # Per-stage breakdown
    lines.append("  Per-Stage Dataset Statistics")
    lines.append("  " + "-" * 74)

    for stage_name, sdata in stage_stats.items():
        ts = sdata['token_stats']
        lines.extend([
            f"",
            f"  {stage_name}",
            f"    Data path           : {sdata['data_path']}",
            f"    Total samples       : {sdata['samples']:,}",
            f"    Training epochs     : {sdata['epochs']}",
            f"    Steps/epoch         : ~{sdata['steps_per_epoch']:,}",
            f"    Total steps         : ~{sdata['max_steps']:,}",
            f"    Batch size          : {args.batch_size}",
        ])

        if ts.get('sampled', 0) > 0:
            lines.extend([
                f"    Avg text tokens/sample : {ts['avg_tokens']} (sampled {ts['sampled']} items)",
                f"    Text token range    : [{ts['min_tokens']}, {ts['max_tokens']}]",
            ])

        lines.append(f"    Est. text tokens    : {sdata['est_text_tokens']:,}")
        if sdata['has_images']:
            lines.append(f"    Est. visual tokens  : {sdata['est_visual_tokens']:,}  ({num_visual_tokens}/image x {sdata['samples']:,})")
        lines.append(f"    Est. total tokens   : {sdata['est_text_tokens'] + sdata['est_visual_tokens']:,}")

        if 'num_classes' in sdata:
            lines.append(f"    Classification      : {sdata['num_classes']} classes: {sdata['class_names']}")

        # Data sources breakdown
        if sdata['sources']:
            lines.append(f"    Data sources:")
            for src, cnt in sorted(sdata['sources'].items(), key=lambda x: -x[1]):
                pct = cnt / max(sdata['samples'], 1) * 100
                lines.append(f"      {src:30s} : {cnt:>10,} ({pct:5.1f}%)")

    # Grand totals
    lines.extend([
        "",
        "  " + "-" * 74,
        "  Grand Totals (All Stages)",
        "  " + "-" * 74,
        f"    Total training samples   : {grand_total_samples:,}",
        f"    Total image samples      : {total_image_samples:,}",
        f"    Total training steps     : ~{grand_total_steps:,}",
        f"    Estimated text tokens    : {total_text_tokens:,}",
        f"    Estimated visual tokens  : {total_visual_tokens:,}",
        f"    Estimated TOTAL tokens   : {total_tokens:,}",
        "",
    ])

    # Token budget breakdown as percentages
    if total_tokens > 0:
        text_pct = total_text_tokens / total_tokens * 100
        vis_pct = total_visual_tokens / total_tokens * 100
        lines.extend([
            "  Token Modality Split",
            f"    Text tokens          : {text_pct:5.1f}%  ({total_text_tokens:,})",
            f"    Visual tokens        : {vis_pct:5.1f}%  ({total_visual_tokens:,})",
            "",
        ])

    # Training hyperparameters
    lines.extend([
        "  Training Hyperparameters",
        f"    Learning rate       : {args.learning_rate}",
        f"    Batch size          : {args.batch_size}",
        f"    Gradient accum.     : {args.gradient_accumulation}",
        f"    Effective batch     : {args.batch_size * args.gradient_accumulation}",
        f"    Mixed precision     : {args.mixed_precision}",
        f"    Seed                : {args.seed}",
        "",
        f"  (Summary computed in {elapsed:.1f}s)",
        separator,
        "",
    ])

    for line in lines:
        logger.info(line)


def get_tokenizer_model_name(language_backbone: str) -> str:
    """Get the appropriate tokenizer model name based on language backbone."""
    if language_backbone == BACKBONE_SMOLLM_135M:
        return "HuggingFaceTB/SmolLM-135M"
    else:
        return PRETRAINED_LANGUAGE_MODEL


def create_tokenizer(model_name: str = PRETRAINED_LANGUAGE_MODEL) -> AutoTokenizer:
    """
    Create and configure tokenizer.

    Uses the tokenizer from the pretrained language model.
    Supports TinyLLM (GPT-2 based) and SmolLM tokenizers.
    """
    try:
        # Try to load tokenizer from pretrained model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {model_name}")
    except Exception as e:
        # Fallback to GPT-2 tokenizer (compatible with tinyllm/30M-0.4)
        logger.warning(f"Could not load tokenizer from {model_name}: {e}")
        logger.info("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for EmberVLM
    # The <image>/</ image> pair wraps injected vision tokens,
    # <question>/</ question> wraps the user query, <answer> starts generation.
    special_tokens = {
        'additional_special_tokens': [
            '<|reasoning_start|>',
            '<|reasoning_end|>',
            '<|robot_selection|>',
            '<|action_plan|>',
            '<image>',
            '</image>',
            '<question>',
            '</question>',
            '<answer>',
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens to tokenizer")

    return tokenizer


def run_sample_inference_preview(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    rank: int,
) -> None:
    """Run sample inference preview and print outputs to logs (rank 0 only)."""
    if rank != 0 or args.stage not in ['all', '2']:
        return

    if not SAMPLE_INFERENCE_AVAILABLE:
        logger.warning("sample_inference module unavailable; skipping sample output preview")
        return

    try:
        logger.info("="*60)
        logger.info("Running sample inference to display model outputs...")
        logger.info("="*60)

        stage2_checkpoint = output_dir / 'stage2' / 'checkpoint-epoch-1'
        if not stage2_checkpoint.exists():
            stage2_checkpoint = output_dir / 'stage2' / 'final'

        sample_dir = args.robot_topn_image_dir if hasattr(args, 'robot_topn_image_dir') else 'sample-checks'
        run_type = args.check if args.check in ['trial', 'main'] else 'trial'
        output_name = f"sample-checks-{run_type}.png"
        output_path = output_dir / output_name

        success = run_sample_inference(
            checkpoint_path=str(stage2_checkpoint),
            sample_dir=sample_dir,
            output_path=str(output_path),
            run_type=run_type,
            device=str(device),
        )

        if success:
            logger.info(f"✓ Sample inference completed: {output_path}")
        else:
            logger.warning("⚠️ Sample inference encountered errors")
    except Exception as e:
        logger.warning(f"Sample inference failed: {e}")
        import traceback
        traceback.print_exc()


# Teacher model candidates ranked by quality (best to worst)
TEACHER_MODEL_CANDIDATES = [
    {
        'id': 'Qwen/Qwen2-VL-7B-Instruct',
        'name': 'Qwen2-VL-7B',
        'vram_gb': 16,
        'quality': 'highest'
    },
    {
        'id': 'microsoft/Phi-3.5-vision-instruct',
        'name': 'Phi-3.5-Vision',
        'vram_gb': 12,
        'quality': 'high'
    },
    {
        'id': 'Qwen/Qwen2-VL-2B-Instruct',
        'name': 'Qwen2-VL-2B',
        'vram_gb': 8,
        'quality': 'excellent'
    },
    {
        'id': 'HuggingFaceTB/SmolVLM-500M-Instruct',
        'name': 'SmolVLM-500M',
        'vram_gb': 4,
        'quality': 'good'
    },
]


def load_teacher_model_and_tokenizer(
    model_id: Optional[str] = None,
    device: str = 'cuda',
    auto_fallback: bool = True,
):
    """
    Load an external teacher model + tokenizer + processor for vocab-agnostic hidden-state distillation.
    
    Args:
        model_id: Specific model ID to load. If None and auto_fallback=True, tries candidates in order.
        device: Device to load model on
        auto_fallback: If True, automatically tries teacher models from best to worst
    
    Returns:
        (teacher_model, teacher_tokenizer, teacher_processor) or (None, None, None) if all fail
    """
    if model_id is not None:
        # User specified a model, try only that one
        candidates = [{'id': model_id, 'name': model_id.split('/')[-1], 'quality': 'user-specified'}]
    elif auto_fallback:
        # Try models in order from best to worst
        candidates = TEACHER_MODEL_CANDIDATES
        logger.info("="*60)
        logger.info("Auto-selecting teacher model (trying best → worst):")
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"  {i}. {candidate['name']} ({candidate['quality']} quality, ~{candidate['vram_gb']}GB VRAM)")
        logger.info("="*60)
    else:
        # No model specified and auto_fallback disabled
        return None, None, None
    
    for candidate in candidates:
        model_id = candidate['id']
        model_name = candidate['name']
        
        logger.info("="*60)
        logger.info(f"Attempting to load: {model_name}")
        logger.info(f"Model ID: {model_id}")
        logger.info("="*60)
        
        try:
            from transformers import AutoProcessor
            
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            
            # Load processor for image preprocessing (critical for Qwen2-VL)
            teacher_processor = None
            try:
                teacher_processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                logger.info(f"✓ Loaded processor for {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load processor for {model_name}: {e}")
                logger.warning("   Will attempt to use tokenizer-only approach")

            teacher_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

            logger.info("="*60)
            logger.info(f"✅ SUCCESS: {model_name} loaded (frozen)")
            logger.info(f"   Quality: {candidate.get('quality', 'unknown')}")
            logger.info(f"   Processor: {'✓ loaded' if teacher_processor else '✗ not available'}")
            logger.info("="*60)
            return teacher_model, teacher_tokenizer, teacher_processor
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                logger.warning(f"⚠️  OOM Error loading {model_name} (needs ~{candidate.get('vram_gb', '?')}GB VRAM)")
                logger.warning(f"   Trying next smaller model...")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"❌ Error loading {model_name}: {e}")
                if len(candidates) > 1:
                    logger.warning("   Trying next model...")
                    continue
                else:
                    return None, None, None
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            if len(candidates) > 1:
                logger.warning("   Trying next model...")
                continue
            else:
                return None, None, None
    
    # All candidates failed
    logger.error("="*60)
    logger.error("❌ All teacher models failed to load")
    logger.error("   Training will proceed without external teacher")
    logger.error("="*60)
    return None, None, None





def generate_model_card(
    vision_backbone: str,
    language_backbone: str,
    total_params: int,
    trainable_params: int,
    carbon_emissions: float = None,
) -> str:
    """Generate model card for HuggingFace Hub."""

    # Model size category
    if total_params < 100_000_000:
        size_category = "Tiny (~35M parameters)"
        variant_name = "embervlm-tiny"
    else:
        size_category = "Small (~137M parameters)"
        variant_name = "embervlm-small"

    # Backbone descriptions
    vision_desc = {
        'repvit': 'RepViT-M0.9 (~5M params)',
        'mobilevit_xs': 'Apple MobileViT-XS (~2.3M params)'
    }.get(vision_backbone, vision_backbone)

    language_desc = {
        'tinyllm': 'TinyLLM-30M (30M params)',
        'smollm_135m': 'SmolLM-135M (135M params)'
    }.get(language_backbone, language_backbone)

    carbon_info = f"\n- **Carbon Emissions**: {carbon_emissions:.4f} kg CO2eq" if carbon_emissions else ""

    card = f"""---
language:
- en
license: apache-2.0
tags:
- vision-language
- multimodal
- robotics
- edge-deployment
- {vision_backbone}
- {language_backbone}
---

# {variant_name.upper()}: {size_category}

EmberVLM is an efficient vision-language model optimized for edge deployment and robotic applications.

## Model Details

- **Model Type**: Vision-Language Model (VLM)
- **Size**: {size_category}
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}{carbon_info}

### Architecture

- **Vision Encoder**: {vision_desc}
- **Language Model**: {language_desc}
- **Training Stages**: 4-stage curriculum
  1. Visual-Language Alignment
  2. Multimodal Instruction Tuning
  3. Robot Fleet Selection
  4. Chain-of-Thought Reasoning

## Usage

```python
from embervlm import EmberVLM
from transformers import AutoTokenizer
from PIL import Image

# Load model and tokenizer
model = EmberVLM.from_pretrained("{variant_name}")
tokenizer = AutoTokenizer.from_pretrained("{variant_name}")

# Prepare input
image = Image.open("robot_scene.jpg")
prompt = "<image>What is happening in this scene?"

# Generate response
outputs = model.generate(image=image, prompt=prompt, tokenizer=tokenizer)
print(outputs)
```

## Training Configuration

- **Vision Backbone**: {vision_backbone}
- **Language Backbone**: {language_backbone}
- **Optimization**: AdamW with cosine learning rate schedule
- **Mixed Precision**: bfloat16
- **Stages Completed**: 1-4 (Full curriculum)

## Intended Use

- Edge deployment on resource-constrained devices
- Robotic vision-language understanding
- Real-time multimodal reasoning
- Robot fleet selection and task planning

## Limitations

- Optimized for efficiency over maximum accuracy
- Best suited for edge/mobile deployment scenarios
- Training focused on robot-centric scenarios

## Citation

```bibtex
@software{{embervlm_{variant_name.replace('-', '_')},
  title = {{EmberVLM-{size_category.split()[0]}}},
  author = {{EmberVLM Team}},
  year = {{2026}},
  url = {{https://huggingface.co/{variant_name}}}
}}
```
"""
    return card


def count_model_parameters(model) -> tuple:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def push_to_hub(
    model,
    tokenizer,
    vision_backbone: str,
    language_backbone: str,
    hub_username: str,
    carbon_emissions: float = None,
    is_trial: bool = False,
    repo_name_override: str = None,
):
    """Push model to HuggingFace Hub with automatic repo selection."""
    # Check if push is disabled
    if os.environ.get('DISABLE_HUB_PUSH', '').lower() in ('1', 'true', 'yes'):
        logger.info(
            "Hub push disabled via DISABLE_HUB_PUSH environment variable")
        return

    # Check for HF token - try multiple sources in order of priority
    hf_token = None
    token_source = None

    # 1. Check HF_TOKEN environment variable (highest priority)
    if os.environ.get('HF_TOKEN'):
        hf_token = os.environ.get('HF_TOKEN')
        token_source = "HF_TOKEN environment variable"

    # 2. Check HUGGING_FACE_HUB_TOKEN (alternative env var)
    if not hf_token and os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
        token_source = "HUGGING_FACE_HUB_TOKEN environment variable"

    # 3. Try huggingface_hub's HfFolder (used by huggingface-cli login)
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
            if hf_token:
                token_source = "huggingface-cli login (HfFolder)"
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get token from HfFolder: {e}")

    # 4. Try huggingface_hub's get_token() function
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
            if hf_token:
                token_source = "huggingface_hub.get_token()"
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get token from get_token(): {e}")

    # 5. Try to read from token files directly (multiple possible locations)
    if not hf_token:
        from pathlib import Path as PathLib
        token_paths = [
            PathLib.home() / '.huggingface' / 'token',      # Newer location
            PathLib.home() / '.cache' / 'huggingface' / 'token',  # Older location
            PathLib('/root/.huggingface/token'),  # Root user on Linux
            PathLib('/root/.cache/huggingface/token'),  # Root user alternate
        ]
        for token_path in token_paths:
            if token_path.exists():
                try:
                    hf_token = token_path.read_text().strip()
                    if hf_token:
                        token_source = f"token file ({token_path})"
                        break
                except Exception as e:
                    logger.debug(f"Could not read token from {token_path}: {e}")

    if not hf_token:
        logger.warning("="*60)
        logger.warning("HuggingFace token not found. Skipping hub push.")
        logger.warning("To enable hub push, use one of:")
        logger.warning("  1. Set environment variable: export HF_TOKEN='your_token'")
        logger.warning("  2. Login via CLI: huggingface-cli login")
        logger.warning("  3. Use: export HF_TOKEN=$(cat ~/.huggingface/token)")
        logger.warning("="*60)
        return
    
    logger.info(f"✓ Using HuggingFace token from: {token_source}")

    # Determine repo name based on backbone
    total_params, trainable_params = count_model_parameters(model)

    if repo_name_override:
        repo_name = repo_name_override
    elif total_params < 100_000_000:
        repo_name = "embervlm-tiny"
    else:
        repo_name = "embervlm-small"
    
    # Add -trial suffix for trial runs
    if is_trial:
        repo_name += "-trial"

    repo_id = f"{hub_username}/{repo_name}"

    logger.info("="*60)
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
    logger.info(f"  Vision: {vision_backbone}")
    logger.info(f"  Language: {language_backbone}")
    logger.info(f"  Total params: {total_params:,}")
    logger.info("="*60)

    try:
        from huggingface_hub import HfApi, create_repo

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=False,
                exist_ok=True,
            )
            logger.info(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")

        # Generate model card
        model_card = generate_model_card(
            vision_backbone=vision_backbone,
            language_backbone=language_backbone,
            total_params=total_params,
            trainable_params=trainable_params,
            carbon_emissions=carbon_emissions,
        )

        # Save model locally first (EmberVLM doesn't have push_to_hub, we use save_pretrained + upload)
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            logger.info(f"Saving model to temporary directory: {tmpdir}")

            # Save model and tokenizer
            try:
                model.save_pretrained(str(tmpdir))
                logger.info("✓ Model weights saved")
            except Exception as e:
                logger.error(f"Failed to save model weights: {e}")
                raise

            try:
                tokenizer.save_pretrained(str(tmpdir))
                logger.info("✓ Tokenizer saved")
            except Exception as e:
                logger.error(f"Failed to save tokenizer: {e}")
                raise

            # Save model card
            try:
                (tmpdir / "README.md").write_text(model_card, encoding='utf-8')
                logger.info("✓ Model card saved")
            except Exception as e:
                logger.error(f"Failed to save model card: {e}")
                raise

            logger.info("Uploading files to Hub...")
            api = HfApi()

            # Upload entire directory
            api.upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Upload {repo_name} ({vision_backbone}+{language_backbone})",
            )

        logger.info("✓ Model, tokenizer, and model card uploaded")

        logger.info("="*60)
        logger.info(
            f"✅ Successfully pushed to: https://huggingface.co/{repo_id}")
        logger.info("="*60)

    except ImportError:
        logger.warning(
            "huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        logger.warning("Training completed successfully, but hub push failed.")


def run_all_stages(args: argparse.Namespace):
    """Run all training stages."""
    
    # ========================================================================
    # TRIAL MODE: Override steps for quick end-to-end validation
    # ========================================================================
    TRIAL_STEPS = 150
    TRIAL_EPOCHS = 1
    
    if args.check == 'trial':
        logger.info("="*60)
        logger.info("🧪 TRIAL MODE ENABLED")
        logger.info("="*60)
        logger.info(f"  Purpose: Quick end-to-end pipeline validation")
        logger.info(f"  Steps per stage: {TRIAL_STEPS}")
        logger.info(f"  Epochs per stage: {TRIAL_EPOCHS}")
        logger.info("  All stages, models, and code paths remain the same")
        logger.info("  Batch sizes: SAME AS MAIN MODE")
        logger.info("  Benchmarks: FULL (same as main mode)")
        logger.info("  Quality threshold: SAME AS MAIN MODE")
        logger.info("="*60)
        
        # Override ONLY step/epoch counts - nothing else
        args.stage1_steps = TRIAL_STEPS
        args.stage1_epochs = TRIAL_EPOCHS
        args.stage2_steps = TRIAL_STEPS
        args.stage2_epochs = TRIAL_EPOCHS
        args.stage3_steps = TRIAL_STEPS
        args.stage3_robot_epochs = TRIAL_EPOCHS
        args.stage4_steps = TRIAL_STEPS
        args.stage4_phase1_epochs = TRIAL_EPOCHS
        args.stage4_phase2_epochs = TRIAL_EPOCHS
        
        # Episodic memory trial overrides
        if getattr(args, 'use_episodic_memory', False):
            args.memory_init_steps = TRIAL_STEPS
            args.consolidation_steps = TRIAL_STEPS

        # NOTE: Benchmarks, quality thresholds, and batch sizes are NOT modified
        # This ensures trial mode catches any errors in the full pipeline

        logger.info("  Hub push: ENABLED (to -trial repo)")
        logger.info("")

    # Map size to backbone configuration
    if args.size == 'tiny':
        vision_backbone = 'repvit'
        language_backbone = 'tinyllm'
        wandb_project = 'embervlm-tiny'
    elif args.size == 'small':
        vision_backbone = 'mobilevit_xs'
        language_backbone = 'smollm_135m'
        wandb_project = 'embervlm-small'
    else:  # medium
        vision_backbone = 'dinov2_small'
        language_backbone = 'smollm_135m'
        wandb_project = 'embervlm-medium'
    
    # Allow backbone overrides via command-line arguments
    if args.vision_backbone:
        vision_backbone = args.vision_backbone
        logger.info(f"Vision backbone overridden to: {vision_backbone}")
    if args.language_backbone:
        language_backbone = args.language_backbone
        logger.info(f"Language backbone overridden to: {language_backbone}")
    
    # Append trial suffix to wandb project if in trial mode
    if args.check == 'trial':
        wandb_project = f"{wandb_project}-trial"

    # --- Episodic memory isolation: force W&B / HF project names ---
    use_episodic_memory = getattr(args, 'use_episodic_memory', False)
    if use_episodic_memory:
        wandb_project = 'EmberVLM-Episode'
        if args.check == 'trial':
            wandb_project = 'EmberVLM-Episode-trial'
        logger.info("Episodic memory ENABLED — W&B project forced to: %s", wandb_project)

    logger.info(f"Model size: {args.size}")
    logger.info(f"  Vision backbone: {vision_backbone}")
    logger.info(f"  Language backbone: {language_backbone}")
    logger.info(f"  W&B project: {wandb_project}")

    # Setup distributed training if enabled
    if args.distributed:
        rank, local_rank, world_size = setup_distributed()
        logger.info(
            f"Distributed training initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Single process training on device: {device}")

    # Setup
    set_seed(args.seed)

    # Create backbone-specific output directory
    # This prevents different configurations from overwriting each other
    base_output_dir = Path(args.output_dir)
    backbone_suffix = f"{vision_backbone}_{language_backbone}"
    output_dir = base_output_dir / backbone_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Vision backbone: {vision_backbone}")
    logger.info(f"Language backbone: {language_backbone}")
    logger.info("="*60)

    # Create tokenizer based on language backbone
    logger.info("Creating tokenizer...")
    tokenizer_model_name = get_tokenizer_model_name(language_backbone)
    logger.info(f"Using tokenizer from: {tokenizer_model_name}")
    tokenizer = create_tokenizer(tokenizer_model_name)
    tokenizer.save_pretrained(output_dir / 'tokenizer')

    # Create model config with backbone selection
    logger.info("="*60)
    logger.info("Creating model config...")
    logger.info(f"  Vision backbone: {vision_backbone}")
    logger.info(f"  Language backbone: {language_backbone}")
    logger.info("="*60)
    config = EmberVLMConfig(
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
        use_episodic_memory=use_episodic_memory,
        memory_slots=getattr(args, 'memory_slots', 512),
        memory_addressing=getattr(args, 'memory_addressing', 'gaussian'),
        memory_alpha=getattr(args, 'memory_alpha', 1.0),
        memory_temperature=getattr(args, 'memory_temperature', 0.1),
        memory_variance=getattr(args, 'memory_variance', 1.0),
        novelty_threshold_novel=getattr(args, 'novelty_threshold_novel', 0.7),
        novelty_threshold_similar=getattr(args, 'novelty_threshold_similar', 0.2),
        scope_detection_method=getattr(args, 'scope_detection_method', 'internal'),
        enable_memory_consolidation=getattr(args, 'enable_memory_consolidation', False),
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Resize embeddings for special tokens - MUST happen before any DDP wrapping
    # This ensures all ranks have the same embedding size
    target_vocab_size = len(tokenizer)
    logger.info(
        f"Target vocabulary size (with special tokens): {target_vocab_size}")

    # Resize embeddings
    if hasattr(model.language_model, 'resize_token_embeddings'):
        model.language_model.resize_token_embeddings(target_vocab_size)
        logger.info(f"Resized token embeddings to {target_vocab_size}")
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'resize_token_embeddings'):
            model.language_model.model.resize_token_embeddings(
                target_vocab_size)
            logger.info(f"Resized token embeddings to {target_vocab_size}")

    # CRITICAL: Also update the model's internal config to reflect new vocab size
    # This ensures consistency throughout training
    if hasattr(model.language_model, 'config'):
        model.language_model.config.vocab_size = target_vocab_size
        logger.info(
            f"Updated language_model.config.vocab_size to {target_vocab_size}")
    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'config'):
        model.language_model.model.config.vocab_size = target_vocab_size
        logger.info(
            f"Updated language_model.model.config.vocab_size to {target_vocab_size}")

    # Verify embedding size after resize
    actual_vocab_size = None
    if hasattr(model.language_model, 'get_input_embeddings'):
        actual_vocab_size = model.language_model.get_input_embeddings(
        ).weight.shape[0]
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'get_input_embeddings'):
            actual_vocab_size = model.language_model.model.get_input_embeddings(
            ).weight.shape[0]

    if actual_vocab_size is not None:
        if actual_vocab_size != target_vocab_size:
            raise RuntimeError(
                f"❌ Embedding resize failed! Actual: {actual_vocab_size}, Expected: {target_vocab_size}"
            )
        logger.info(f"✓ Verified embedding size: {actual_vocab_size}")

    # Load from checkpoint - either explicitly provided or auto-detected from previous stage
    checkpoint_path = None

    if args.resume_from_checkpoint:
        # User explicitly provided a checkpoint path
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.warning(
                f"Specified checkpoint {checkpoint_path} does not exist")
            checkpoint_path = None
    elif args.stage != 'all' and args.stage != '1':
        # Running a specific stage (not 'all' or stage 1) - auto-detect previous stage checkpoint
        logger.info(
            f"Auto-detecting checkpoint from previous stage for Stage {args.stage}...")
        checkpoint_path = get_previous_stage_checkpoint(output_dir, args.stage)

    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        from embervlm.training.train_utils import load_checkpoint
        load_checkpoint(model, None, None, str(checkpoint_path))
        logger.info(f"✓ Loaded model weights from {checkpoint_path}")
    elif args.stage != 'all' and args.stage != '1':
        logger.warning(
            f"No checkpoint found for Stage {args.stage}. Starting from base model weights.")
        logger.warning(
            f"For best results, run previous stages first or provide --resume_from_checkpoint")

    # Synchronize all ranks after model initialization
    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized after model initialization")

    # Training configuration
    training_config = TrainingConfig(
        seed=args.seed,
        output_dir=str(output_dir),
        distributed=args.distributed,
        mixed_precision=args.mixed_precision,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        push_to_hub=False,  # We handle hub push manually after training
        hub_model_id=None,
        wandb_project=wandb_project,  # Size-specific project name
        trial_mode=(args.check == 'trial'),  # Enable trial mode for reduced datasets
        # CRITICAL: Include backbone info for proper checkpoint saving
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
    )

    # Carbon tracking - ONLY on rank 0 to prevent duplicate tracking
    carbon_tracker = None
    total_emissions = None
    if rank == 0:
        try:
            carbon_tracker = CarbonTracker(
                output_dir=str(output_dir / 'emissions'),
                project_name="EmberVLM",
            )
            carbon_tracker.start()
            logger.info("Carbon tracking started (rank 0 only)")
        except Exception as e:
            logger.warning(f"Failed to initialize carbon tracker: {e}")

    # Synchronize before training starts
    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # ── Pre-training dataset & token summary (rank 0 only) ──
    if rank == 0:
        try:
            log_pre_training_dataset_summary(args, model, tokenizer, training_config)
        except Exception as e:
            logger.warning(f"Failed to log dataset summary: {e}")

    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    try:
        # Stage 1: Visual-Language Alignment
        if args.stage in ['all', '1']:
            logger.info("="*60)
            logger.info("Stage 1: Visual-Language Alignment")
            logger.info("="*60)

            stage1_dict = training_config.to_dict()
            # Use smaller batch size for Stage 1 (image-text alignment needs less memory)
            stage1_batch_size = args.batch_size
            stage1_dict.update({
                'output_dir': str(output_dir / 'stage1'),
                'batch_size': stage1_batch_size,
                'num_training_steps': args.stage1_steps,
                # Stage 1 needs generous warmup for alignment to settle
                # before cosine decay kicks in.  ~12 % of steps keeps the
                # LR at peak while the contrastive heads bootstrap.
                'warmup_steps': max(3000, args.stage1_steps // 8),
            })
            stage1_config = TrainingConfig(**stage1_dict)

            stage1_data_path = Path(
                args.stage1_data) if args.stage1_data else None
            if stage1_data_path and stage1_data_path.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    if use_episodic_memory:
                        _ep_repo = getattr(args, 'hf_repo', None) or 'embervlm-episode'
                        if args.check == 'trial':
                            _ep_repo += '-trial'
                        repo_name = _ep_repo
                    else:
                        repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage1_training(
                    model=model,
                    config=stage1_config,
                    data_dir=str(stage1_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage1_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)

                # Save final Stage 1 checkpoint
                if rank == 0:
                    stage1_final = output_dir / 'stage1' / 'final'
                    stage1_final.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(stage1_final))
                    tokenizer.save_pretrained(str(stage1_final))
                    logger.info(f"✓ Stage 1 final checkpoint saved to: {stage1_final}")
            else:
                logger.warning(f"Stage 1 data not provided, skipping...")

        # =================================================================
        # Stage 1.5: Memory Initialisation (episode branch only)
        # =================================================================
        if use_episodic_memory and args.stage in ['all', '1']:
            logger.info("="*60)
            logger.info("Stage 1.5: Episodic Memory Initialisation")
            logger.info("="*60)

            from embervlm.training.train_utils import unwrap_model as _unwrap
            _model_ref = _unwrap(model)

            if _model_ref.episodic_memory is not None:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    torch.cuda.empty_cache()

                mem_init_steps = getattr(args, 'memory_init_steps', 5000)
                logger.info(f"  Memory init steps: {mem_init_steps}")
                logger.info(f"  Memory slots: {config.memory_slots}")
                logger.info(f"  Addressing: {config.memory_addressing}")

                # Initialise episodic memory tracker for visualisations
                try:
                    from embervlm.visualization.episodic_memory_plots import EpisodicMemoryTracker
                    mem_tracker = EpisodicMemoryTracker()
                except Exception:
                    mem_tracker = None

                # Use Stage 1 data to populate memory with multimodal episodes
                stage1_data_path = Path(args.stage1_data) if args.stage1_data else None
                if stage1_data_path and stage1_data_path.exists():
                    try:
                        from embervlm.data.loaders import AlignmentDataset
                        from torch.utils.data import DataLoader

                        mem_dataset = AlignmentDataset(
                            data_dir=str(stage1_data_path),
                            tokenizer=tokenizer,
                            split='train',
                            max_length=512,
                            image_size=config.image_size,
                            trial_mode=(args.check == 'trial'),
                        )
                        mem_loader = DataLoader(
                            mem_dataset,
                            batch_size=min(getattr(args, 'batch_size', 32), 64),
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                        )

                        _model_ref.eval()
                        step = 0
                        scope_optimizer = torch.optim.AdamW(
                            _model_ref.scope_detector.parameters(), lr=1e-4, weight_decay=0.01
                        ) if _model_ref.scope_detector is not None else None

                        # Decide when to capture novelty distribution snapshots
                        _snapshot_steps = {0, mem_init_steps // 4, mem_init_steps // 2,
                                           3 * mem_init_steps // 4, mem_init_steps - 1}

                        for batch in mem_loader:
                            if step >= mem_init_steps:
                                break

                            pixel_values = batch.get('pixel_values')
                            input_ids = batch.get('input_ids')
                            if pixel_values is None or input_ids is None:
                                continue
                            pixel_values = pixel_values.to(device)
                            input_ids = input_ids.to(device)

                            with torch.no_grad():
                                inputs_embeds, _ = _model_ref.prepare_inputs_embeds(
                                    input_ids, pixel_values
                                )
                                lm_out = _model_ref.language_model(
                                    inputs_embeds=inputs_embeds, output_hidden_states=False
                                )
                                last_h = lm_out.get('last_hidden_state', lm_out.get('logits'))
                                if last_h is not None and last_h.dim() == 3:
                                    fused_repr = last_h[:, -1, :]
                                else:
                                    continue

                            # Write to memory
                            write_stats = _model_ref.episodic_memory.smart_write(fused_repr)

                            # Train scope detector (positive = close to memory, negative = far)
                            _scope_loss_val = None
                            _scope_preds = None
                            if scope_optimizer is not None:
                                scope_optimizer.zero_grad()
                                sigma = _model_ref.episodic_memory.novelty_score(fused_repr.detach())
                                # Positive labels: sigma < novelty_threshold_novel
                                labels_scope = (sigma < config.novelty_threshold_novel).float()
                                _scope_preds = _model_ref.scope_detector(fused_repr.detach())
                                scope_loss = F.binary_cross_entropy(_scope_preds, labels_scope)
                                scope_loss.backward()
                                scope_optimizer.step()
                                _scope_loss_val = scope_loss.item()

                                # Capture novelty snapshot for histogram plot
                                if mem_tracker is not None and step in _snapshot_steps:
                                    mem_tracker.capture_novelty_snapshot(step, sigma)
                            else:
                                sigma = None

                            # Track metrics for visualisations
                            if mem_tracker is not None and rank == 0:
                                mem_tracker.log_init_step(
                                    step=step,
                                    write_stats=write_stats,
                                    scope_loss_val=_scope_loss_val,
                                    scope_preds=_scope_preds,
                                    memory_controller=_model_ref.episodic_memory,
                                )

                            if step % 100 == 0 and rank == 0:
                                logger.info(
                                    f"  [MemInit] step={step}/{mem_init_steps} "
                                    f"written={write_stats.get('num_written', 0)} "
                                    f"sigma_mean={write_stats.get('sigma_mean', 0):.4f}"
                                )

                            step += 1

                        if rank == 0:
                            mem_state_path = str(output_dir / 'stage1.5_memory_state.pt')
                            _model_ref.save_memory_state(mem_state_path)
                            logger.info(f"✓ Stage 1.5 memory state saved to: {mem_state_path}")

                            # Generate Stage 1.5 visualisations
                            if mem_tracker is not None:
                                try:
                                    mem_tracker.capture_final_memory(_model_ref.episodic_memory)
                                    vis_dir = str(output_dir / 'stage1.5_visualizations')
                                    mem_tracker.generate_plots(vis_dir)
                                    logger.info(f"📊 Stage 1.5 episodic memory plots saved to: {vis_dir}")
                                except Exception as ve:
                                    logger.warning(f"Stage 1.5 visualisation failed (non-fatal): {ve}")

                    except Exception as e:
                        logger.warning(f"Stage 1.5 memory init failed (non-fatal): {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning("Stage 1.5: No Stage 1 data found, skipping memory init.")
            else:
                logger.warning("Stage 1.5: episodic_memory not initialised on model, skipping.")

        # Clean up and sync before Stage 2
        if args.stage in ['all', '2']:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()  # Synchronize all ranks
                torch.cuda.empty_cache()  # Clear CUDA cache

        # Stage 2: Instruction Tuning
        if args.stage in ['all', '2']:
            logger.info("="*60)
            logger.info("Stage 2: Multimodal Instruction Tuning")
            logger.info("="*60)

            # Automatic teacher model selection (tries best → worst)
            teacher_model = None
            teacher_tokenizer = None
            teacher_processor = None
            teacher_device = f'cuda:{local_rank}' if args.distributed else (
                f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
            )

            # NOTE: Both trial and main mode use the same teacher model for consistency
            # Memory issues are handled via automatic batch size reduction based on teacher size
            # Try to load teacher (auto-fallback from best to worst if not specified)
            teacher_model, teacher_tokenizer, teacher_processor = load_teacher_model_and_tokenizer(
                model_id=args.teacher_model_id,  # None = auto-select best available
                device=teacher_device,
                auto_fallback=True,
            )

            stage2_dict = training_config.to_dict()

            # Determine batch size for Stage 2 based on teacher model size
            # Large teacher models (7B+) need reduced batch size to avoid OOM
            # Stage 2 also has longer sequences (multi-turn conversations) than Stage 1
            stage2_batch_size = args.batch_size
            if teacher_model is not None:
                # Check teacher model size and reduce batch size accordingly
                teacher_params = sum(p.numel() for p in teacher_model.parameters())
                teacher_params_b = teacher_params / 1e9
                logger.info(f"Teacher model size: {teacher_params_b:.2f}B parameters")

                if teacher_params_b > 5:  # 5B+ params (e.g., Qwen2-VL-7B)
                    # Reduce batch size by 8x for large teachers
                    stage2_batch_size = max(8, args.batch_size // 8)
                    logger.info(f"⚠️ Large teacher ({teacher_params_b:.1f}B params) detected")
                    logger.info(f"   Reducing Stage 2 batch size: {args.batch_size} → {stage2_batch_size}")
                elif teacher_params_b > 2:  # 2B+ params (e.g., Qwen2-VL-2B)
                    # Reduce batch size by 4x for medium teachers
                    stage2_batch_size = max(12, args.batch_size // 4)
                    logger.info(f"⚠️ Medium teacher ({teacher_params_b:.1f}B params) detected")
                    logger.info(f"   Reducing Stage 2 batch size: {args.batch_size} → {stage2_batch_size}")
                elif teacher_params_b > 0.3:  # 300M+ params (e.g., SmolVLM-500M)
                    # Reduce batch size by 2x for small teachers
                    stage2_batch_size = max(24, args.batch_size // 2)
                    logger.info(f"ℹ️ Small teacher ({teacher_params_b:.2f}B params) detected")
                    logger.info(f"   Reducing Stage 2 batch size: {args.batch_size} → {stage2_batch_size}")
                else:
                    # Very small teacher - keep batch size as is
                    logger.info(f"ℹ️ Tiny teacher ({teacher_params_b:.3f}B params) - no batch size reduction needed")

            # Use lower learning rate for Stage 2 to prevent repetitive outputs
            stage2_lr = args.learning_rate * 0.6  # 60% of base LR for fine-tuning
            stage2_dict.update({
                'output_dir': str(output_dir / 'stage2'),
                'batch_size': stage2_batch_size,
                'learning_rate': stage2_lr,
                'num_training_steps': args.stage2_steps,
                'find_unused_parameters': True,  # Reasoning heads not used in instruction tuning
                'warmup_steps': 1500,  # More warmup for Stage 2 stability
            })
            logger.info(f"Stage 2 learning rate: {stage2_lr:.2e} (60% of {args.learning_rate:.2e})")
            stage2_config = TrainingConfig(**stage2_dict)

            stage2_data_path = Path(
                args.stage2_data) if args.stage2_data else None
            if stage2_data_path and stage2_data_path.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    if use_episodic_memory:
                        _ep_repo = getattr(args, 'hf_repo', None) or 'embervlm-episode'
                        if args.check == 'trial':
                            _ep_repo += '-trial'
                        repo_name = _ep_repo
                    else:
                        repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage2_training(
                    model=model,
                    config=stage2_config,
                    data_dir=str(stage2_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage2_epochs,
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tokenizer,
                    teacher_processor=teacher_processor,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                    use_ema_teacher=args.use_ema_teacher,
                    use_multi_scale=args.use_multi_scale,
                    use_layer_wise_lr=args.use_layer_wise_lr,
                )
                
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)

                # Save final Stage 2 checkpoint
                if rank == 0:
                    stage2_final = output_dir / 'stage2' / 'final'
                    stage2_final.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(stage2_final))
                    tokenizer.save_pretrained(str(stage2_final))
                    logger.info(f"✓ Stage 2 final checkpoint saved to: {stage2_final}")

                # Show sample inference outputs immediately after Stage 2
                run_sample_inference_preview(
                    args=args,
                    output_dir=output_dir,
                    device=device,
                    rank=rank,
                )
            else:
                logger.warning(
                    f"Stage 2 data not found at {stage2_data_path}, skipping...")

        # Stage 2.5: VLM Evaluation (Lighteval + Coherence Checks)
        # Uses ONLY Lighteval and coherence checks - NO lmms-eval, NO VLMEvalKit
        # Runs by default unless --skip_benchmarks is explicitly set
        # IMPORTANT: Only run on rank 0 to avoid duplicate evaluations
        coherence_failed = False  # Initialize for all ranks
        if args.stage in ['all', '2'] and not args.skip_benchmarks and rank == 0:
            logger.info("="*60)
            logger.info("STAGE 2.5: VLM EVALUATION (LIGHTEVAL + COHERENCE CHECKS)")
            logger.info("="*60)
            logger.info(f"  Method: Coherence checks + Lighteval (if available)")
            logger.info(f"  Quality threshold mode: {args.quality_threshold}")
            logger.info(f"  NO lmms-eval, NO VLMEvalKit dependencies")
            logger.info("="*60)
            
            # Get best checkpoint from Stage 2
            stage2_dir = output_dir / 'stage2'
            best_checkpoint_marker = stage2_dir / 'best_checkpoint_path.txt'
            
            if best_checkpoint_marker.exists():
                with open(best_checkpoint_marker, 'r') as f:
                    checkpoint_path = f.read().strip()
                logger.info(f"Using best Stage 2 checkpoint: {checkpoint_path}")
            else:
                # Fall back to latest checkpoint
                checkpoint_path = find_latest_checkpoint(stage2_dir)
                if checkpoint_path:
                    logger.info(f"Using latest Stage 2 checkpoint: {checkpoint_path}")
                else:
                    logger.warning("No Stage 2 checkpoint found, using current model")
                    checkpoint_path = None
            
            # Run evaluation (Lighteval + coherence checks)
            try:
                passed, results_summary = run_stage2_5_evaluation(
                    model_path=str(checkpoint_path) if checkpoint_path else str(stage2_dir),
                    output_dir=str(output_dir),
                    preset=args.benchmark_preset,
                    threshold_mode=args.quality_threshold,
                    enable_lighteval=True,
                    openvlm_baselines_path=args.openvlm_baselines_path,
                    trial_mode=(args.check == 'trial'),  # Pass trial mode flag
                )
                
                # Log results to WandB if available
                if rank == 0:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'stage2_5/coherence_score': results_summary.get('aggregate_score', 0),
                                'stage2_5/quality_passed': passed,
                            })
                            logger.info("✓ Evaluation results logged to W&B")
                    except Exception as e:
                        logger.warning(f"Failed to log to WandB: {e}")
                
                # Store result for later synchronization
                coherence_failed = (args.quality_threshold != 'skip' and not passed)

                # Check quality threshold (ONLY if not in 'skip' mode)
                if coherence_failed:
                    logger.error("="*80)
                    logger.error("❌ VLM COHERENCE CHECK FAILED")
                    logger.error("="*80)
                    logger.error("The model did not produce coherent outputs.")
                    logger.error("Recommendations:")
                    logger.error("  1. Increase Stage 1 epochs (--stage1_epochs)")
                    logger.error("  2. Increase Stage 2 epochs (--stage2_epochs)")
                    logger.error("  3. Check training data quality")
                    logger.error("  4. Adjust quality threshold (--quality_threshold permissive)")
                    logger.error("  5. Use --skip_benchmarks to skip evaluation entirely")
                    logger.error("="*80)
                else:
                    logger.info("="*80)
                    logger.info("✅ VLM COHERENCE CHECK PASSED")
                    logger.info("="*80)
                    logger.info("Model produces coherent outputs.")
                    logger.info("Proceeding to robot-specific training stages...")
                    
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                coherence_failed = False  # Don't fail on evaluation errors
                if args.quality_threshold != 'skip':
                    logger.warning("Proceeding with training despite evaluation failure")

        # Barrier: Wait for evaluation to complete on rank 0 before continuing
        # CRITICAL: All ranks must reach this barrier to avoid NCCL timeout
        coherence_failed_global = False
        if args.stage in ['all', '2'] and not args.skip_benchmarks:
            import torch.distributed as dist
            if dist.is_initialized():
                logger.info("[Sync] Waiting for all ranks to synchronize after evaluation...")
                dist.barrier()
                logger.info("[Sync] All ranks synchronized")

                # Broadcast coherence failure result from rank 0 to all ranks
                if rank == 0:
                    coherence_failed_tensor = torch.tensor([1 if coherence_failed else 0], device=device)
                else:
                    coherence_failed_tensor = torch.tensor([0], device=device)
                dist.broadcast(coherence_failed_tensor, src=0)
                coherence_failed_global = coherence_failed_tensor.item() == 1
            else:
                # Non-distributed: use local result
                if rank == 0:
                    coherence_failed_global = coherence_failed
        elif rank == 0:
            # Non-distributed case
            coherence_failed_global = coherence_failed if 'coherence_failed' in dir() else False

        # All ranks check if training should stop due to coherence failure
        if coherence_failed_global and args.stop_on_stage2_5_fail:
            logger.info("Stopping training due to coherence check failure (--stop_on_stage2_5_fail enabled)")
            return model
        elif coherence_failed_global:
            logger.warning("Continuing training despite Stage 2.5 coherence failure")

        # Clean up and sync before Stage 3
        if args.stage in ['all', '3']:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()  # Synchronize all ranks
                torch.cuda.empty_cache()  # Clear CUDA cache

        # Stage 3: Robot Selection Training
        if args.stage in ['all', '3']:
            logger.info("="*60)
            logger.info("Stage 3: Robot Fleet Selection Training")
            logger.info("="*60)

            # CRITICAL FIX: Verify embedding layer size matches tokenizer BEFORE Stage 3
            # This prevents index out of bounds errors when special tokens are used
            from embervlm.training.train_utils import unwrap_model
            model_unwrapped = unwrap_model(model)

            # Move model to CPU temporarily to safely resize embeddings
            device_before = next(model_unwrapped.parameters()).device
            model_unwrapped = model_unwrapped.cpu()
            torch.cuda.empty_cache()

            # Check embedding size vs tokenizer size
            current_vocab_size = None
            if hasattr(model_unwrapped.language_model, 'get_input_embeddings'):
                current_vocab_size = model_unwrapped.language_model.get_input_embeddings(
                ).weight.shape[0]
            elif hasattr(model_unwrapped.language_model, 'model'):
                if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                    current_vocab_size = model_unwrapped.language_model.model.get_input_embeddings(
                    ).weight.shape[0]

            required_vocab_size = len(tokenizer)

            if current_vocab_size is not None and current_vocab_size != required_vocab_size:
                logger.warning(
                    f"⚠️ CRITICAL: Embedding size mismatch detected!")
                logger.warning(
                    f"   Current embedding size: {current_vocab_size}")
                logger.warning(
                    f"   Required (tokenizer size): {required_vocab_size}")
                logger.warning(
                    f"   Special tokens in use: {tokenizer.additional_special_tokens}")
                logger.info(f"🔧 Resizing embeddings to match tokenizer...")

                # Resize embeddings to match tokenizer
                try:
                    if hasattr(model_unwrapped.language_model, 'resize_token_embeddings'):
                        model_unwrapped.language_model.resize_token_embeddings(
                            required_vocab_size)
                        logger.info(
                            f"✓ Resized embeddings via resize_token_embeddings()")
                    elif hasattr(model_unwrapped.language_model, 'model'):
                        if hasattr(model_unwrapped.language_model.model, 'resize_token_embeddings'):
                            model_unwrapped.language_model.model.resize_token_embeddings(
                                required_vocab_size)
                            logger.info(
                                f"✓ Resized embeddings via model.resize_token_embeddings()")

                    # Also resize LM head if it exists
                    if hasattr(model_unwrapped.language_model, 'lm_head'):
                        old_lm_head = model_unwrapped.language_model.lm_head
                        new_lm_head = torch.nn.Linear(
                            old_lm_head.in_features,
                            required_vocab_size,
                            bias=old_lm_head.bias is not None
                        )
                        # Copy old weights
                        with torch.no_grad():
                            new_lm_head.weight[:current_vocab_size] = old_lm_head.weight
                            if old_lm_head.bias is not None:
                                new_lm_head.bias[:current_vocab_size] = old_lm_head.bias
                        model_unwrapped.language_model.lm_head = new_lm_head
                        logger.info(
                            f"✓ Resized LM head to {required_vocab_size}")
                    elif hasattr(model_unwrapped.language_model, 'model'):
                        if hasattr(model_unwrapped.language_model.model, 'lm_head'):
                            old_lm_head = model_unwrapped.language_model.model.lm_head
                            new_lm_head = torch.nn.Linear(
                                old_lm_head.in_features,
                                required_vocab_size,
                                bias=old_lm_head.bias is not None
                            )
                            # Copy old weights
                            with torch.no_grad():
                                new_lm_head.weight[:current_vocab_size] = old_lm_head.weight
                                if old_lm_head.bias is not None:
                                    new_lm_head.bias[:current_vocab_size] = old_lm_head.bias
                            model_unwrapped.language_model.model.lm_head = new_lm_head
                            logger.info(
                                f"✓ Resized LM head to {required_vocab_size}")

                except Exception as e:
                    logger.error(f"❌ Failed to resize embeddings: {e}")
                    raise RuntimeError(
                        f"Failed to resize embeddings to match tokenizer vocabulary: {e}")

                # Verify resize worked
                new_vocab_size = None
                if hasattr(model_unwrapped.language_model, 'get_input_embeddings'):
                    new_vocab_size = model_unwrapped.language_model.get_input_embeddings(
                    ).weight.shape[0]
                elif hasattr(model_unwrapped.language_model, 'model'):
                    if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                        new_vocab_size = model_unwrapped.language_model.model.get_input_embeddings(
                        ).weight.shape[0]

                if new_vocab_size == required_vocab_size:
                    logger.info(
                        f"✅ Embedding resize successful: {new_vocab_size} tokens")
                else:
                    error_msg = f"❌ Embedding resize FAILED! Size is {new_vocab_size}, expected {required_vocab_size}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                logger.info(
                    f"✓ Embedding size matches tokenizer: {required_vocab_size} tokens")

            # Move model back to original device
            model_unwrapped = model_unwrapped.to(device_before)
            torch.cuda.synchronize()

            # Use the unwrapped model for Stage 3
            model = model_unwrapped

            stage3_dict = training_config.to_dict()
            # Use smaller batch size for Stage 3 (robot selection has more complex loss)
            stage3_batch_size = args.batch_size // 2
            stage3_dict.update({
                'output_dir': str(output_dir / 'stage3'),
                'batch_size': stage3_batch_size,
                'num_training_steps': args.stage3_steps,
                'find_unused_parameters': True,  # Reasoning heads may not be used yet
            })
            stage3_config = TrainingConfig(**stage3_dict)

            robot_dir = Path(args.robot_data) if args.robot_data else Path(
                Path(__file__).parent.parent / 'robot-selection-dataset')

            if robot_dir.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    if use_episodic_memory:
                        _ep_repo = getattr(args, 'hf_repo', None) or 'embervlm-episode'
                        if args.check == 'trial':
                            _ep_repo += '-trial'
                        repo_name = _ep_repo
                    else:
                        repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage3_training(
                    model=model,
                    config=stage3_config,
                    robot_data_dir=str(robot_dir),
                    tokenizer=tokenizer,
                    robot_epochs=args.stage3_robot_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)

                # Save final Stage 3 checkpoint
                if rank == 0:
                    stage3_final = output_dir / 'stage3' / 'final'
                    stage3_final.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(stage3_final))
                    tokenizer.save_pretrained(str(stage3_final))
                    logger.info(f"✓ Stage 3 final checkpoint saved to: {stage3_final}")
            else:
                logger.warning(
                    f"Stage 3 robot data not found at {robot_dir}, skipping...")

        # Stage 4: Reasoning Integration
        if args.stage in ['all', '4']:
            logger.info("="*60)
            logger.info("Stage 4: Chain-of-Thought Reasoning Integration")
            logger.info("="*60)

            stage4_dict = training_config.to_dict()
            # In trial mode, use smaller batch size
            stage4_batch_size = args.batch_size // 2 if args.check == 'trial' else args.batch_size // 2
            stage4_dict.update({
                'output_dir': str(output_dir / 'stage4'),
                'batch_size': stage4_batch_size,
                'num_training_steps': args.stage4_steps,
                'find_unused_parameters': True,  # May have conditional parameter usage
            })
            stage4_config = TrainingConfig(**stage4_dict)

            # Priority order for reasoning data:
            # 1. Explicitly provided reasoning_data path
            # 2. outputs/reasoning-data directory
            # 3. Fall back to robot_data (auto-generates reasoning chains)
            reasoning_dir = args.reasoning_data or str(
                output_dir / 'reasoning-data')

            data_dir_to_use = None
            data_source_type = None

            if Path(reasoning_dir).exists():
                data_dir_to_use = reasoning_dir
                data_source_type = "explicit reasoning data"
            elif args.robot_data and Path(args.robot_data).exists():
                # Fall back to robot selection data - ReasoningDataset will auto-generate chains
                data_dir_to_use = args.robot_data
                data_source_type = "robot selection data (auto-generating reasoning chains)"
                logger.info(
                    "No dedicated reasoning data found. Using robot selection data with auto-generated reasoning chains.")
            else:
                # Check default robot data location
                default_robot_dir = str(
                    output_dir.parent / 'robot-selection-dataset')
                if Path(default_robot_dir).exists():
                    data_dir_to_use = default_robot_dir
                    data_source_type = "robot selection data (auto-generating reasoning chains)"
                    logger.info(
                        "No dedicated reasoning data found. Using robot selection data with auto-generated reasoning chains.")

            if data_dir_to_use is not None:
                logger.info(f"Stage 4 data source: {data_source_type}")
                logger.info(f"Stage 4 data directory: {data_dir_to_use}")
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    if use_episodic_memory:
                        _ep_repo = getattr(args, 'hf_repo', None) or 'embervlm-episode'
                        if args.check == 'trial':
                            _ep_repo += '-trial'
                        repo_name = _ep_repo
                    else:
                        repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage4_training(
                    model=model,
                    config=stage4_config,
                    data_dir=data_dir_to_use,
                    tokenizer=tokenizer,
                    phase1_epochs=args.stage4_phase1_epochs,
                    phase2_epochs=args.stage4_phase2_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)

                # Save final Stage 4 checkpoint
                if rank == 0:
                    stage4_final = output_dir / 'stage4' / 'final'
                    stage4_final.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(stage4_final))
                    tokenizer.save_pretrained(str(stage4_final))
                    logger.info(f"✓ Stage 4 final checkpoint saved to: {stage4_final}")
            else:
                logger.warning(
                    "Stage 4: No reasoning or robot selection data found, skipping...")
                logger.warning("  To run Stage 4, provide either:")
                logger.warning(
                    "    --reasoning_data <path>  (explicit reasoning chains)")
                logger.warning(
                    "    --robot_data <path>      (will auto-generate reasoning chains)")

        # =================================================================
        # Stage 5: Memory Consolidation (optional, episode branch only)
        # =================================================================
        if (use_episodic_memory
                and getattr(args, 'enable_memory_consolidation', False)
                and args.stage in ['all', '4']):
            logger.info("="*60)
            logger.info("Stage 5: Memory Consolidation")
            logger.info("="*60)

            from embervlm.training.train_utils import unwrap_model as _unwrap5
            _model5 = _unwrap5(model)

            if _model5.episodic_memory is not None:
                consol_steps = getattr(args, 'consolidation_steps', 3000)
                consol_lr = 5e-5
                logger.info(f"  Consolidation steps: {consol_steps}")
                logger.info(f"  Consolidation LR: {consol_lr}")

                # Initialise tracker for Stage 5 visualisations
                try:
                    from embervlm.visualization.episodic_memory_plots import EpisodicMemoryTracker
                    consol_tracker = EpisodicMemoryTracker()
                except Exception:
                    consol_tracker = None

                # Freeze memory matrices
                _model5.episodic_memory.M.requires_grad_(False)
                _model5.episodic_memory.cov.requires_grad_(False)

                # Collect trainable parameters (fusion + reasoning heads + top LM layer)
                consol_params = []
                consol_params += list(_model5.fusion_module.parameters())
                if _model5.config.reasoning_enabled:
                    consol_params += list(_model5.reasoning_module.parameters())
                if _model5.scope_detector is not None:
                    consol_params += list(_model5.scope_detector.parameters())
                # Optionally unfreeze top LM layer
                if hasattr(_model5.language_model, 'model'):
                    lm_inner = _model5.language_model.model
                    if hasattr(lm_inner, 'model') and hasattr(lm_inner.model, 'layers'):
                        last_layer = list(lm_inner.model.layers)[-1]
                        consol_params += list(last_layer.parameters())

                consol_optimizer = torch.optim.AdamW(consol_params, lr=consol_lr, weight_decay=0.01)

                # Use Stage 1 data (or robot data) for consolidation
                data_path = Path(args.stage1_data) if args.stage1_data else None
                if data_path and data_path.exists():
                    try:
                        from embervlm.data.loaders import AlignmentDataset
                        from torch.utils.data import DataLoader

                        consol_dataset = AlignmentDataset(
                            data_dir=str(data_path),
                            tokenizer=tokenizer,
                            split='train',
                            max_length=512,
                            image_size=config.image_size,
                            trial_mode=(args.check == 'trial'),
                        )
                        consol_loader = DataLoader(
                            consol_dataset,
                            batch_size=min(getattr(args, 'batch_size', 16), 16),
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                        )

                        _model5.train()
                        step = 0
                        lam_align = 0.1
                        for batch in consol_loader:
                            if step >= consol_steps:
                                break

                            pixel_values = batch.get('pixel_values')
                            input_ids = batch.get('input_ids')
                            labels = batch.get('labels')
                            if pixel_values is None or input_ids is None:
                                continue

                            pixel_values = pixel_values.to(device)
                            input_ids = input_ids.to(device)
                            if labels is not None:
                                labels = labels.to(device)

                            consol_optimizer.zero_grad()

                            # Forward pass
                            outputs_c = _model5(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=labels,
                            )
                            sup_loss = outputs_c.get('loss', torch.tensor(0.0, device=device))

                            # Representation alignment loss
                            _last_h = outputs_c.get('hidden_states')
                            if _last_h is not None and isinstance(_last_h, (list, tuple)):
                                _last_h = _last_h[-1]
                            if _last_h is None:
                                # Re-run to get hidden states
                                with torch.no_grad():
                                    ie, _ = _model5.prepare_inputs_embeds(input_ids, pixel_values)
                                    _lm_out = _model5.language_model(inputs_embeds=ie, output_hidden_states=True)
                                    _last_h = _lm_out.get('last_hidden_state')

                            align_loss = torch.tensor(0.0, device=device)
                            _cosine_sim_val = None
                            if _last_h is not None and _last_h.dim() == 3:
                                z_model = _last_h[:, -1, :]  # (B, C)
                                # Read from memory (frozen)
                                with torch.no_grad():
                                    z_mem = _model5.episodic_memory.read(z_model.detach())
                                align_loss = lam_align * F.mse_loss(z_model, z_mem)
                                # Compute cosine similarity for tracking
                                with torch.no_grad():
                                    _cos = F.cosine_similarity(z_model, z_mem, dim=-1).mean().item()
                                    _cosine_sim_val = _cos

                            total_loss = sup_loss + align_loss
                            if total_loss.requires_grad:
                                total_loss.backward()
                                consol_optimizer.step()

                            # Track consolidation metrics
                            if consol_tracker is not None and rank == 0:
                                consol_tracker.log_consolidation_step(
                                    step=step,
                                    sup_loss_val=sup_loss.item(),
                                    align_loss_val=align_loss.item(),
                                    cosine_sim_val=_cosine_sim_val,
                                )

                            if step % 100 == 0 and rank == 0:
                                logger.info(
                                    f"  [Consolidation] step={step}/{consol_steps} "
                                    f"sup={sup_loss.item():.4f} align={align_loss.item():.4f}"
                                    + (f" cos_sim={_cosine_sim_val:.4f}" if _cosine_sim_val is not None else "")
                                )
                            step += 1

                        if rank == 0:
                            consol_dir = output_dir / 'stage5' / 'final'
                            consol_dir.mkdir(parents=True, exist_ok=True)
                            _model5.save_pretrained(str(consol_dir))
                            tokenizer.save_pretrained(str(consol_dir))
                            logger.info(f"✓ Stage 5 checkpoint saved to: {consol_dir}")

                            # Generate Stage 5 visualisations
                            if consol_tracker is not None:
                                try:
                                    consol_tracker.capture_final_memory(_model5.episodic_memory)
                                    vis_dir = str(output_dir / 'stage5_visualizations')
                                    consol_tracker.generate_plots(vis_dir)
                                    logger.info(f"📊 Stage 5 consolidation plots saved to: {vis_dir}")
                                except Exception as ve:
                                    logger.warning(f"Stage 5 visualisation failed (non-fatal): {ve}")

                    except Exception as e:
                        logger.warning(f"Stage 5 consolidation failed (non-fatal): {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning("Stage 5: No data found for consolidation, skipping.")

                model = _model5

        # Unwrap model before saving (in case it's still wrapped)
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # Save final model
        logger.info("="*60)
        logger.info("Saving final model...")
        logger.info("="*60)

        final_output = output_dir / 'final'
        model.save_pretrained(str(final_output))
        tokenizer.save_pretrained(str(final_output))

        logger.info(f"Final model saved to {final_output}")

        # Generate robot top-N score matrix from sample images (rank 0 only)
        if rank == 0 and not args.skip_robot_topn_viz:
            try:
                robot_topn_model = output_dir / 'stage3' / 'final'
                if not robot_topn_model.exists():
                    robot_topn_model = final_output

                robot_figures = generate_robot_topn_visualizations(
                    model_path=str(robot_topn_model),
                    image_dir=args.robot_topn_image_dir,
                    output_dir=str(output_dir / 'paper_figures' / 'robot_topn'),
                    top_n=args.robot_topn_k,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                if robot_figures:
                    logger.info("✓ Robot top-N visualizations generated:")
                    for key, path in robot_figures.items():
                        logger.info(f"  - {key}: {path}")
            except Exception as e:
                logger.warning(f"Robot top-N visualization generation failed: {e}")

        # ── Generate collated cross-stage visualizations (rank 0 only) ────
        if rank == 0:
            try:
                from embervlm.monitoring.stage_visualizations import CollatedVisualizer
                import json as _json

                logger.info("="*60)
                logger.info("Generating collated cross-stage visualizations...")
                logger.info("="*60)

                collate_dir = str(output_dir / 'collate_visuals')
                collated_viz = CollatedVisualizer(output_dir=collate_dir)

                # Load per-stage metrics from saved JSON files
                stage_metrics = {}
                for stage_key in ['stage1', 'stage2', 'stage3', 'stage4']:
                    metrics_path = output_dir / stage_key / 'training_metrics.json'
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, 'r') as f:
                                data = _json.load(f)
                            stage_metrics[stage_key] = data.get('step_metrics', {})
                            logger.info(f"  Loaded {stage_key} metrics: {list(stage_metrics[stage_key].keys())[:5]}...")
                        except Exception as e:
                            logger.warning(f"  Failed to load {stage_key} metrics: {e}")

                # Load Stage 2.5 evaluation summary
                stage2_5_results = None
                eval_summary_path = output_dir / 'stage2_5_evaluation' / 'evaluation_summary.json'
                if eval_summary_path.exists():
                    try:
                        with open(eval_summary_path, 'r') as f:
                            stage2_5_results = _json.load(f)
                        logger.info("  Loaded Stage 2.5 evaluation results")
                    except Exception as e:
                        logger.warning(f"  Failed to load Stage 2.5 results: {e}")

                if stage_metrics:
                    saved_figs = collated_viz.generate_all(
                        stage_metrics=stage_metrics,
                        stage2_5_results=stage2_5_results,
                    )
                    if saved_figs:
                        logger.info(f"✓ Collated visualizations: {len(saved_figs)} figures saved to {collate_dir}")
                else:
                    logger.warning("No stage metrics found, skipping collated visualizations")

            except Exception as e:
                logger.warning(f"Collated visualization generation failed (non-fatal): {e}")

        # ── Generate paper-quality figures (rank 0 only) ────────────
        if rank == 0:
            paper_fig_dir = str(output_dir / 'paper_figures' / 'suite')
            if _PAPER_FIGURES_AVAILABLE:
                try:
                    logger.info("="*60)
                    logger.info("Generating paper-quality figures...")
                    logger.info("="*60)
                    _generate_paper_figures(paper_fig_dir)
                    logger.info(f"Paper figures saved to {paper_fig_dir}")
                except Exception as e:
                    logger.warning(f"Paper figure generation failed (non-fatal): {e}")
            else:
                logger.info("Paper figure generation skipped (import not available: %s)",
                            _PAPER_FIGURES_IMPORT_ERROR or 'unknown')

        # Stop carbon tracking and get emissions
        total_emissions = None
        if carbon_tracker is not None:
            try:
                total_emissions = carbon_tracker.stop()
                logger.info(
                    f"Total training emissions: {total_emissions:.4f} kg CO2eq")
            except Exception as e:
                logger.warning(f"Error stopping carbon tracker: {e}")

        # Push to HuggingFace Hub (only on rank 0)
        if rank == 0 and args.hub_username:
            _repo_override = None
            if use_episodic_memory:
                _repo_override = getattr(args, 'hf_repo', None) or 'embervlm-episode'
                logger.info(f"Episodic memory: HF repo forced to {args.hub_username}/{_repo_override}")

            push_to_hub(
                model=model,
                tokenizer=tokenizer,
                vision_backbone=vision_backbone,
                language_backbone=language_backbone,
                hub_username=args.hub_username,
                carbon_emissions=total_emissions,
                is_trial=(args.check == 'trial'),
                repo_name_override=_repo_override,
            )

            # Save episodic memory state alongside model
            if use_episodic_memory:
                try:
                    from embervlm.training.train_utils import unwrap_model as _uw_hub
                    _m_hub = _uw_hub(model)
                    if _m_hub.episodic_memory is not None:
                        mem_path = str(output_dir / 'final' / 'episodic_memory_state.pt')
                        _m_hub.save_memory_state(mem_path)
                        logger.info(f"✓ Episodic memory state saved to: {mem_path}")

                        # Generate final memory snapshot visualisation
                        try:
                            from embervlm.visualization.episodic_memory_plots import EpisodicMemoryTracker
                            final_tracker = EpisodicMemoryTracker()
                            final_tracker.capture_final_memory(_m_hub.episodic_memory)
                            vis_dir = str(output_dir / 'final' / 'episodic_memory_plots')
                            final_tracker.generate_plots(vis_dir)
                            logger.info(f"📊 Final episodic memory visualisations saved to: {vis_dir}")
                        except Exception as ve:
                            logger.warning(f"Final memory visualisation failed (non-fatal): {ve}")
                except Exception as e:
                    logger.warning(f"Failed to save episodic memory state: {e}")

    finally:
        # Final carbon tracking cleanup (in case push_to_hub wasn't reached)
        if carbon_tracker is not None and total_emissions is None:
            try:
                total_emissions = carbon_tracker.stop()
                logger.info(
                    f"Total training emissions: {total_emissions:.4f} kg CO2eq")
            except Exception as e:
                logger.warning(f"Error stopping carbon tracker: {e}")

        # Cleanup distributed training
        if args.distributed:
            cleanup_distributed()
            logger.info("Distributed training cleanup completed")

    return model


def main():
    parser = argparse.ArgumentParser(description="EmberVLM Training")

    # General arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', '1', '2', '3', '4'],
                        help='Training stage to run')

    # Model size selection
    parser.add_argument('--size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size: tiny (~35M params, repvit+tinyllm), small (~137M params, mobilevit_xs+smollm_135m), or medium (~160M params, dinov2_small+smollm_135m)')
    parser.add_argument('--vision_backbone', type=str, default=None,
                        choices=['repvit', 'mobilevit_xs', 'dinov2_small'],
                        help='Override vision backbone (overrides --size selection)')
    parser.add_argument('--language_backbone', type=str, default=None,
                        choices=['tinyllm', 'smollm_135m'],
                        help='Override language backbone (overrides --size selection)')

    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Mixed precision training')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=50,
                        help='Log metrics every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')

    # Data paths
    parser.add_argument('--stage1_data', type=str, default='data/base_vlm',
                        help='Path to Stage 1 alignment data')
    parser.add_argument('--stage2_data', type=str, default='data/base_vlm/llava',
                        help='Path to Stage 2 instruction data')
    parser.add_argument('--robot_data', type=str, default='robot-selection-dataset',
                        help='Path to robot selection data')
    parser.add_argument('--reasoning_data', type=str, default=None,
                        help='Path to reasoning augmented data')

    # Stage-specific epochs
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage1_steps', type=int, default=25000)
    parser.add_argument('--stage2_epochs', type=int, default=15)
    parser.add_argument('--stage2_steps', type=int, default=20000)
    parser.add_argument('--teacher_model_id', type=str, default=None,
                        help='External teacher for hidden-state distillation (vocab-agnostic). '
                        'If not specified, automatically tries models from best to worst: '
                        '1) Qwen2-VL-7B (highest quality, 16GB VRAM), '
                        '2) Phi-3.5-Vision (high quality, 12GB VRAM), '
                        '3) Qwen2-VL-2B (excellent quality, 8GB VRAM), '
                        '4) SmolVLM-500M (good quality, 4GB VRAM). '
                        'Falls back to smaller models on OOM. '
                        'Specify a model ID to override auto-selection.')
    parser.add_argument('--use_ema_teacher', type=lambda x: x.lower() != 'false', default=False,
                        help='Enable EMA teacher for self-distillation (DISABLED BY DEFAULT). '
                        'WARNING: Only enable after model shows basic coherence (>60%% text accuracy). '
                        'Early self-distillation can amplify garbage outputs.')
    parser.add_argument('--use_multi_scale', action='store_true',
                        help='Enable multi-scale vision feature extraction')
    parser.add_argument('--use_layer_wise_lr', action='store_true',
                        help='Enable layer-wise learning rates (vision=1e-5, projection=2e-4, language=5e-5)')
    parser.add_argument('--stage3_robot_epochs', type=int, default=20)
    parser.add_argument('--stage3_steps', type=int, default=20000)
    parser.add_argument('--stage4_phase1_epochs', type=int, default=5)
    parser.add_argument('--stage4_phase2_epochs', type=int, default=5)
    parser.add_argument('--stage4_steps', type=int, default=10000)
    
    # Trial/Check Mode - for quick validation of entire pipeline
    parser.add_argument('--check', type=str, default='main',
                        choices=['main', 'trial'],
                        help='Validation mode: main (full training, default), '
                        'trial (150 steps per stage for quick end-to-end validation)')

    # HuggingFace Hub (automatic push after training)
    parser.add_argument('--hub_username', type=str, default='euhidaman',
                        help='HuggingFace username/org for automatic model push (default: euhidaman). '
                        'Trial mode pushes to {username}/embervlm-small-trial. '
                        'Main mode pushes to {username}/embervlm-small. '
                        'Requires HF_TOKEN environment variable.')

    # VLM Benchmark Evaluation (Stage 2.5)
    parser.add_argument('--run_benchmarks', action='store_true', default=True,
                        help='Run VLM benchmarks after Stage 2 (default: True)')
    parser.add_argument('--skip_benchmarks', action='store_true', default=False,
                        help='Explicitly skip VLM benchmark evaluation')
    parser.add_argument('--benchmark_preset', type=str, default='standard',
                        choices=['mini', 'standard', 'full'],
                        help='Benchmark suite: mini (~30min), standard (~1-2hr), full (~4-6hr)')
    parser.add_argument('--quality_threshold', type=str, default='auto',
                        choices=['strict', 'standard', 'permissive', 'auto', 'skip'],
                        help='Quality threshold for VLM: strict (85%%), standard (70%%), '
                        'permissive (50%%), auto (65%%), skip (no gating)')
    parser.add_argument('--stop_on_stage2_5_fail', action='store_true', default=False,
                        help='Stop training if Stage 2.5 quality check fails (default: continue training)')
    parser.add_argument('--openvlm_baselines_path', type=str, default='configs/open_vlm_baselines.json',
                        help='Optional JSON file for OpenVLM-style baseline comparison plots')
    parser.add_argument('--skip_robot_topn_viz', action='store_true', default=False,
                        help='Skip robot top-N matrix/JSON figure generation')
    parser.add_argument('--robot_topn_image_dir', type=str, default='sample-checks',
                        help='Image directory for robot top-N analysis figures')
    parser.add_argument('--robot_topn_k', type=int, default=3,
                        help='Top-N robots to include in robot selection analysis')
    parser.add_argument('--lmms_eval_repo', type=str, default=None,
                        help='Path to lmms-eval repository (auto-detected if not specified)')

    # Resume from checkpoint
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/stage2/checkpoint-789)')

    # ── Episodic memory (episode branch) ──
    parser.add_argument('--use_episodic_memory', action='store_true', default=False,
                        help='Enable multimodal episodic memory (episode branch feature)')
    parser.add_argument('--memory_slots', type=int, default=512,
                        help='Number of episodic memory slots K')
    parser.add_argument('--memory_addressing', type=str, default='gaussian',
                        choices=['gaussian', 'pseudoinverse'],
                        help='Episodic memory addressing mode')
    parser.add_argument('--memory_alpha', type=float, default=1.0,
                        help='Episodic memory update strength alpha')
    parser.add_argument('--memory_temperature', type=float, default=0.1,
                        help='Episodic memory addressing temperature')
    parser.add_argument('--memory_variance', type=float, default=1.0,
                        help='Episodic memory Gaussian variance')
    parser.add_argument('--novelty_threshold_novel', type=float, default=0.7,
                        help='Novelty threshold above which episodes are written')
    parser.add_argument('--novelty_threshold_similar', type=float, default=0.2,
                        help='Novelty threshold below which episodes are skipped')
    parser.add_argument('--scope_detection_method', type=str, default='internal',
                        help='Scope detector method')
    parser.add_argument('--enable_memory_consolidation', action='store_true', default=False,
                        help='Enable Stage 5 memory consolidation')
    parser.add_argument('--memory_init_steps', type=int, default=5000,
                        help='Training steps for Stage 1.5 memory initialisation')
    parser.add_argument('--consolidation_steps', type=int, default=3000,
                        help='Training steps for Stage 5 memory consolidation')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Override W&B project name (auto-set for episodic memory)')
    parser.add_argument('--hf_repo', type=str, default=None,
                        help='Override HF repo name (auto-set for episodic memory)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., configs/episode_config.yaml)')

    args = parser.parse_args()

    # --- Load YAML config and merge into args ---
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        model_cfg = yaml_cfg.get('model', {})
        if model_cfg.get('use_episodic_memory'):
            args.use_episodic_memory = True
        for key in ['memory_slots', 'memory_addressing', 'memory_alpha',
                     'memory_temperature', 'memory_variance',
                     'novelty_threshold_novel', 'novelty_threshold_similar',
                     'scope_detection_method', 'enable_memory_consolidation']:
            if key in model_cfg and getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, model_cfg[key])
        log_cfg = yaml_cfg.get('logging', {})
        wandb_cfg = log_cfg.get('wandb', {})
        if wandb_cfg.get('project') and args.wandb_project is None:
            args.wandb_project = wandb_cfg['project']
        hf_cfg = log_cfg.get('huggingface', {})
        if hf_cfg.get('repo_name') and args.hf_repo is None:
            args.hf_repo = hf_cfg['repo_name']

    # Run training
    model = run_all_stages(args)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
