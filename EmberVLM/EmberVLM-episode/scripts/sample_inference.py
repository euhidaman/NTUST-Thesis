"""
Sample Inference Script for EmberVLM

Runs inference on sample images in sample-checks/ directory and produces
visual results showing input images and model outputs.

Produces:
- sample-checks-trial.png (after trial run)
- sample-checks-main.png (after main run)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer
import re
import inspect

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embervlm.models import EmberVLM, EmberVLMConfig
from embervlm.models.vision_encoder import ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[EmberVLM, AutoTokenizer]:
    """Load model and tokenizer from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    logger.info(f"Loading model from: {checkpoint_path}")

    # Try to load config
    config_path = checkpoint_path / 'config.json'
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = EmberVLMConfig.from_dict(config_dict)
    else:
        # Use default small config (DINOv2 + SmolLM)
        config = EmberVLMConfig(
            vision_backbone='dinov2_small',
            language_backbone='smollm_135m',
        )

    # Create model
    model = EmberVLM(config)

    # Load weights
    model_path = checkpoint_path / 'pytorch_model.bin'
    if not model_path.exists():
        model_path = checkpoint_path / 'model.safetensors'
    if not model_path.exists():
        # Try loading from directory directly
        model = EmberVLM.from_pretrained(str(checkpoint_path))
    else:
        state_dict = torch.load(model_path, map_location='cpu')

        # Detect vocab size from checkpoint and resize embeddings if needed
        embed_key = 'language_model.model.model.embed_tokens.weight'
        if embed_key in state_dict:
            ckpt_vocab = state_dict[embed_key].shape[0]
            current_vocab = None
            try:
                current_vocab = model.language_model.get_input_embeddings().weight.shape[0]
            except Exception:
                try:
                    current_vocab = model.language_model.model.get_input_embeddings().weight.shape[0]
                except Exception:
                    pass
            if current_vocab and current_vocab != ckpt_vocab:
                logger.info(f"Resizing embeddings from {current_vocab} to {ckpt_vocab} to match checkpoint")
                try:
                    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'resize_token_embeddings'):
                        model.language_model.model.resize_token_embeddings(ckpt_vocab)
                    elif hasattr(model.language_model, 'resize_token_embeddings'):
                        model.language_model.resize_token_embeddings(ckpt_vocab)
                except Exception as e:
                    logger.warning(f"Embedding resize failed: {e}")

        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep tokenizer special tokens and embedding size aligned for inference
    if hasattr(model, 'sync_tokenizer_and_embeddings'):
        try:
            model.sync_tokenizer_and_embeddings(
                tokenizer,
                add_special_tokens=True,
                force_resize=False,
                logger=logger,
            )
        except TypeError:
            # Backward compatibility for older EmberVLM versions without `logger` arg
            model.sync_tokenizer_and_embeddings(
                tokenizer,
                add_special_tokens=True,
                force_resize=False,
            )

    return model, tokenizer


def load_and_preprocess_images(
    image_paths: List[str],
    image_size: int = 224,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """Load and preprocess images for model input."""
    preprocessor = ImagePreprocessor(image_size=image_size)

    pil_images = []
    tensors = []

    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            pil_images.append(img)

            # Preprocess image - returns a tensor
            tensor = preprocessor(img)

            # Ensure it's a 3D tensor [C, H, W]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            tensors.append(tensor)
            logger.debug(f"Preprocessed {path}: tensor shape {tensor.shape}")
        except Exception as e:
            logger.error(f"Error loading/preprocessing {path}: {e}")
            raise

    # Stack into batch [B, C, H, W] and convert to model dtype
    pixel_values = torch.stack(tensors).to(device=device, dtype=dtype)
    logger.debug(f"Stacked pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

    return pixel_values, pil_images


def _sanitize_output(text: str, reject_unreliable: bool = True) -> str:
    """Clean up model output to avoid displaying obvious gibberish."""
    if not text:
        return "[No output]"

    # Remove control chars and collapse whitespace
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Strip any accidental prompt echoes
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:")[-1].strip()
    if "User:" in cleaned:
        cleaned = cleaned.split("User:")[-1].strip()

    if reject_unreliable:
        # Reject likely code outputs (common failure mode on tiny/trial checkpoints)
        code_patterns = [
            r'\bdef\s+\w+\s*\(',
            r'\bimport\s+\w+',
            r'\bclass\s+\w+',
            r'self\.\w+',
            r'cv2\.',
            r'torch\.',
            r'return\s+',
        ]
        code_hits = sum(1 for pattern in code_patterns if re.search(pattern, cleaned))
        if code_hits >= 2:
            return "[Unreliable output: model generated code-like text]"

        # Heuristic: reject outputs dominated by digits/symbols
        if cleaned:
            alpha = sum(c.isalpha() for c in cleaned)
            digits = sum(c.isdigit() for c in cleaned)
            total = max(len(cleaned), 1)
            if alpha / total < 0.2 or digits / total > 0.4:
                return "[Unreliable output: model response not coherent]"

    # Keep it concise for visualization
    if len(cleaned) > 400:
        cleaned = cleaned[:400].rstrip() + "..."

    return cleaned or "[No output]"


def _truncate_text(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _format_output(raw_text: str, cleaned_text: str) -> str:
    raw_text = _truncate_text(raw_text.strip()) if raw_text else "[No output]"
    cleaned_text = _truncate_text(cleaned_text.strip()) if cleaned_text else "[No output]"
    return f"Raw: {raw_text}\nClean: {cleaned_text}"


def _filter_generate_kwargs(model: EmberVLM, gen_kwargs: dict) -> dict:
    """Filter generate kwargs to the model's supported signature."""
    try:
        sig = inspect.signature(model.generate)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in gen_kwargs.items() if k in allowed}
    except (ValueError, TypeError):
        # If signature is not introspectable, pass through as-is
        return gen_kwargs


def run_inference(
    model: EmberVLM,
    tokenizer: AutoTokenizer,
    pixel_values: torch.Tensor,
    prompts: List[str],
    max_new_tokens: int = 64,
    run_type: str = "trial",
) -> List[dict]:
    """Run inference on images with prompts."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    use_autocast = device.type == 'cuda' and model_dtype == torch.bfloat16
    outputs = []

    reject_unreliable = run_type.lower() != "trial"

    for i, prompt in enumerate(prompts):
        # Tokenize prompt - CRITICAL: add special tokens for proper generation
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,  # CRITICAL: Add BOS/EOS tokens
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Get single image (already in correct dtype from preprocessing)
        single_image = pixel_values[i:i+1]

        try:
            # Generate
            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": input_ids,
                    "pixel_values": single_image,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "repetition_penalty": 1.15,
                    "no_repeat_ngram_size": 3,
                    # NOTE: EmberVLM.generate() does not accept pad_token_id/eos_token_id
                }

                gen_kwargs = _filter_generate_kwargs(model, gen_kwargs)

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        generated_ids = model.generate(**gen_kwargs)
                else:
                    generated_ids = model.generate(**gen_kwargs)

            # Decode
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Remove prompt from output if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            cleaned = _sanitize_output(generated_text, reject_unreliable=reject_unreliable)
            outputs.append({"raw": generated_text, "clean": cleaned})

        except Exception as e:
            logger.error(f"Error generating for image {i}: {e}")
            outputs.append({"raw": "", "clean": f"[Error: {str(e)[:100]}]"})

    return outputs


def create_visual_results(
    pil_images: List[Image.Image],
    prompts: List[str],
    outputs: List[str],
    output_path: str,
    run_type: str = "trial",
):
    """Create visual results image showing inputs and outputs."""
    n_images = len(pil_images)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 5 * n_images))
    gs = gridspec.GridSpec(n_images, 2, width_ratios=[1, 2], hspace=0.3, wspace=0.1)

    # Title
    fig.suptitle(
        f'EmberVLM Sample Inference Results ({run_type.upper()} Run)',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    for i in range(n_images):
        # Image subplot
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(pil_images[i])
        ax_img.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Text subplot
        ax_txt = fig.add_subplot(gs[i, 1])
        ax_txt.axis('off')

        # Format text
        prompt_text = f"Prompt: {prompts[i]}"
        output_text = f"Output: {outputs[i]}"

        # Wrap text
        import textwrap
        prompt_wrapped = textwrap.fill(prompt_text, width=80)
        output_wrapped = textwrap.fill(output_text, width=80)

        full_text = f"{prompt_wrapped}\n\n{output_wrapped}"

        ax_txt.text(
            0.02, 0.95, full_text,
            transform=ax_txt.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"✓ Visual results saved to: {output_path}")


def run_sample_inference(
    checkpoint_path: str,
    sample_dir: str,
    output_path: str,
    run_type: str = "trial",
    device: str = 'cuda',
):
    """Main function to run sample inference and create visual results."""

    # Check if checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    # Find sample images
    sample_dir = Path(sample_dir)
    image_paths = sorted(sample_dir.glob('*.jpg')) + sorted(sample_dir.glob('*.png'))

    # Exclude generated output images from previous runs
    image_paths = [
        p for p in image_paths
        if not p.name.startswith('sample-checks-')
    ]

    if not image_paths:
        logger.error(f"No images found in: {sample_dir}")
        return False

    logger.info(f"Found {len(image_paths)} sample images")

    # Load model
    try:
        model, tokenizer = load_model_from_checkpoint(str(checkpoint_path), device)

        # On CUDA, prefer bf16 when supported; otherwise use fp32
        if device.startswith('cuda') and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

        model = model.to(dtype=target_dtype)

        logger.info(
            f"Model loaded. bf16 supported: {device.startswith('cuda') and torch.cuda.is_bf16_supported()}, "
            f"target dtype: {target_dtype}"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Load and preprocess images
    try:
        pixel_values, pil_images = load_and_preprocess_images(
            [str(p) for p in image_paths],
            image_size=224,
            device=device,
            dtype=target_dtype,
        )

        logger.info(
            f"Preprocessed {len(pil_images)} images, pixel_values shape: {pixel_values.shape}, "
            f"dtype: {pixel_values.dtype}"
        )
    except Exception as e:
        logger.error(f"Failed to preprocess images: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Use the same comprehensive visual description prompt for all images
    # This prompt is designed to elicit detailed, grounded descriptions
    visual_description_prompt = (
        "<image>\n"
        "User: Provide a complete and accurate visual description of the image. "
        "Enumerate all visible objects, their attributes (color, shape, size), "
        "spatial relationships, and any observable actions. "
        "Do not include assumptions or information not directly visible.\n"
        "Assistant:"
    )

    # Use the same prompt for all images
    prompts = [visual_description_prompt] * len(pil_images)

    # Run inference
    logger.info(f"Running inference (run_type={run_type}, reject_unreliable_outputs={run_type.lower() != 'trial'})...")
    try:
        outputs = run_inference(
            model,
            tokenizer,
            pixel_values,
            prompts,
            max_new_tokens=64,
            run_type=run_type,
        )
    except Exception as e:
        logger.error(f"Failed during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print results
    logger.info("="*60)
    logger.info(f"INFERENCE RESULTS ({run_type.upper()} RUN)")
    logger.info("="*60)
    for i, (path, prompt, output) in enumerate(zip(image_paths, prompts, outputs)):
        logger.info(f"\n--- Image {i+1}: {path.name} ---")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Raw Output: {_truncate_text(output.get('raw', '').strip())}")
        logger.info(f"Clean Output: {_truncate_text(output.get('clean', '').strip())}")
    logger.info("="*60)

    # Create visual results
    try:
        formatted_outputs = [
            _format_output(item.get("raw", ""), item.get("clean", ""))
            for item in outputs
        ]
        create_visual_results(pil_images, prompts, formatted_outputs, output_path, run_type)
    except Exception as e:
        logger.error(f"Failed to create visual results: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run sample inference on EmberVLM")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sample_dir', type=str, default='sample-checks',
                        help='Directory containing sample images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for visual results (e.g., sample-checks-trial.png)')
    parser.add_argument('--run_type', type=str, default='trial',
                        choices=['trial', 'main'],
                        help='Type of run (for labeling)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    success = run_sample_inference(
        checkpoint_path=args.checkpoint,
        sample_dir=args.sample_dir,
        output_path=args.output,
        run_type=args.run_type,
        device=args.device,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

