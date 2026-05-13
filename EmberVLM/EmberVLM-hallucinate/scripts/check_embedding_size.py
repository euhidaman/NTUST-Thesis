#!/usr/bin/env python3
"""
Diagnostic Script for Token Embedding Size Issues

This script helps diagnose and fix token embedding size mismatches
that cause CUDA index out of bounds errors during training.

Usage:
    python check_embedding_size.py --checkpoint outputs/stage2/checkpoint-789
    python check_embedding_size.py --model_path outputs/final --tokenizer_path outputs/tokenizer
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_model_from_checkpoint(checkpoint_path: str, tokenizer_path: str = None):
    """Check embedding size from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path / "model.pt", map_location='cpu')

    # Extract embedding layer size
    embedding_keys = [k for k in checkpoint.keys() if 'embed' in k.lower() and 'weight' in k]

    if not embedding_keys:
        logger.warning("No embedding layers found in checkpoint")
        return False

    embedding_key = embedding_keys[0]
    embedding_weight = checkpoint[embedding_key]
    vocab_size = embedding_weight.shape[0]
    embed_dim = embedding_weight.shape[1]

    logger.info(f"✓ Found embedding layer: {embedding_key}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Embedding dimension: {embed_dim}")

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = checkpoint_path.parent.parent / "tokenizer"

    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        logger.warning(f"Tokenizer not found at {tokenizer_path}")
        logger.warning("Cannot verify tokenizer vocabulary size")
        return True

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"✓ Tokenizer vocabulary size: {tokenizer_vocab_size}")

    # Check for special tokens
    special_tokens = tokenizer.additional_special_tokens
    if special_tokens:
        logger.info(f"✓ Special tokens ({len(special_tokens)}):")
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            status = "✓ OK" if token_id < vocab_size else "❌ OUT OF BOUNDS"
            logger.info(f"    {token} → ID {token_id} {status}")

    # Compare sizes
    if vocab_size == tokenizer_vocab_size:
        logger.info("✅ PASS: Embedding size matches tokenizer vocabulary size")
        return True
    else:
        logger.error(f"❌ FAIL: Embedding size mismatch!")
        logger.error(f"    Model embedding size: {vocab_size}")
        logger.error(f"    Tokenizer vocab size: {tokenizer_vocab_size}")
        logger.error(f"    Difference: {tokenizer_vocab_size - vocab_size}")

        if tokenizer_vocab_size > vocab_size:
            logger.error(f"\n⚠️  TOKEN IDs {vocab_size} to {tokenizer_vocab_size-1} will cause CUDA errors!")
            logger.error(f"\nSOLUTION:")
            logger.error(f"  1. The model embeddings need to be resized")
            logger.error(f"  2. This should be done in train_all.py before Stage 3")
            logger.error(f"  3. Check the FIX_STAGE3_CRASH.md document for details")

        return False


def check_model_object(model, tokenizer):
    """Check embedding size of a model object."""
    logger.info("Checking model object...")

    # Get vocabulary size
    current_vocab_size = None

    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'get_input_embeddings'):
            current_vocab_size = lm.get_input_embeddings().weight.shape[0]
        elif hasattr(lm, 'model'):
            if hasattr(lm.model, 'get_input_embeddings'):
                current_vocab_size = lm.model.get_input_embeddings().weight.shape[0]

    if current_vocab_size is None:
        logger.error("❌ Could not find embedding layer in model")
        return False

    logger.info(f"✓ Model embedding size: {current_vocab_size}")

    required_vocab_size = len(tokenizer)
    logger.info(f"✓ Tokenizer vocabulary size: {required_vocab_size}")

    # Check special tokens
    special_tokens = tokenizer.additional_special_tokens
    if special_tokens:
        logger.info(f"✓ Special tokens ({len(special_tokens)}):")
        max_special_id = 0
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            max_special_id = max(max_special_id, token_id)
            status = "✓ OK" if token_id < current_vocab_size else "❌ OUT OF BOUNDS"
            logger.info(f"    {token} → ID {token_id} {status}")

        if max_special_id >= current_vocab_size:
            logger.error(f"\n❌ CRITICAL: Special tokens will cause CUDA index out of bounds!")
            logger.error(f"    Max special token ID: {max_special_id}")
            logger.error(f"    Embedding layer size: {current_vocab_size}")
            return False

    # Compare
    if current_vocab_size == required_vocab_size:
        logger.info("✅ PASS: Embedding size matches tokenizer")
        return True
    else:
        logger.error(f"❌ FAIL: Embedding size mismatch!")
        logger.error(f"    Model: {current_vocab_size}, Tokenizer: {required_vocab_size}")
        logger.error(f"    Difference: {required_vocab_size - current_vocab_size}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check token embedding size compatibility"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        help='Path to tokenizer (default: auto-detect from checkpoint)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to saved model'
    )

    args = parser.parse_args()

    if args.checkpoint:
        success = check_model_from_checkpoint(args.checkpoint, args.tokenizer_path)
    elif args.model_path:
        logger.info(f"Loading model from: {args.model_path}")
        from embervlm.models import EmberVLM
        model = EmberVLM.from_pretrained(args.model_path)

        tokenizer_path = args.tokenizer_path or Path(args.model_path) / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        success = check_model_object(model, tokenizer)
    else:
        logger.error("Please provide --checkpoint or --model_path")
        return 1

    if success:
        logger.info("\n✅ All checks passed!")
        return 0
    else:
        logger.error("\n❌ Checks failed - embedding size mismatch detected")
        logger.error("See FIX_STAGE3_CRASH.md for resolution steps")
        return 1


if __name__ == "__main__":
    exit(main())

