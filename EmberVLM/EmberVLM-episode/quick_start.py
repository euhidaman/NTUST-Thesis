"""
EmberVLM Quick Start Script

Simple script to get started with EmberVLM.
"""

import os
import sys

# Add embervlm to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer


def create_model():
    """Create EmberVLM model."""
    from embervlm.models import EmberVLM, EmberVLMConfig

    config = EmberVLMConfig(
        vision_model="repvit_m0_9",  # Using timm model
        vision_pretrained=True,  # Now works with timm models
        freeze_vision=True,
        num_visual_tokens=8,
        vision_output_dim=384,
        language_hidden_size=768,
        language_num_layers=6,
        language_num_heads=12,
        language_vocab_size=50257,
        reasoning_enabled=True,
    )

    model = EmberVLM(config)

    # Print model info
    params = model.count_parameters()
    print(f"Model created!")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Vision encoder: {params['vision_encoder']:,}")
    print(f"  Language model: {params['language_model']:,}")
    print(f"  Fusion module: {params['fusion_module']:,}")

    if 'reasoning_module' in params:
        print(f"  Reasoning module: {params['reasoning_module']:,}")

    return model


def create_tokenizer():
    """Create tokenizer with special tokens."""
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '<|reasoning_start|>',
            '<|reasoning_end|>',
            '<|robot_selection|>',
            '<|action_plan|>',
            '<|image|>',
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    print(f"Tokenizer created with {len(tokenizer)} tokens")

    return tokenizer


def test_forward_pass(model, tokenizer):
    """Test model forward pass."""
    print("\nTesting forward pass...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    seq_len = 64

    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=device)
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_reasoning=True,
        )

    print(f"Forward pass successful!")
    print(f"  Logits shape: {outputs['logits'].shape}")

    if 'robot_logits' in outputs and outputs['robot_logits'] is not None:
        print(f"  Robot logits shape: {outputs['robot_logits'].shape}")

    if 'robot_probs' in outputs and outputs['robot_probs'] is not None:
        print(f"  Robot probabilities: {outputs['robot_probs'][0].tolist()}")

    return True


def test_generation(model, tokenizer):
    """Test text generation."""
    print("\nTesting generation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Create inputs
    prompt = "Analyze this incident scene and select a robot."
    tokens = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = tokens['input_ids'].to(device)

    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text[:100]}...")

    return True


def test_robot_selection(model, tokenizer):
    """Test robot selection functionality."""
    print("\nTesting robot selection...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    # Use analyze_incident
    with torch.no_grad():
        result = model.analyze_incident(
            pixel_values=pixel_values,
            instruction="Survey damage from aerial perspective after earthquake",
            tokenizer=tokenizer,
        )

    print(f"Robot selection result:")
    print(f"  Selected robot: {result['selected_robot']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Robot probabilities: {result['robot_probabilities']}")

    return True


def main():
    """Main quick start demo."""
    print("=" * 60)
    print("EmberVLM Quick Start")
    print("=" * 60)

    # Create model and tokenizer
    model = create_model()
    tokenizer = create_tokenizer()

    # Resize embeddings for special tokens
    model.language_model.model.resize_token_embeddings(len(tokenizer))

    # Run tests
    try:
        test_forward_pass(model, tokenizer)
        test_generation(model, tokenizer)
        test_robot_selection(model, tokenizer)

        print("\n" + "=" * 60)
        print("All tests passed! EmberVLM is ready to use.")
        print("=" * 60)

        print("\nNext steps:")
        print("1. Prepare training data in the expected format")
        print("2. Run training: python scripts/train_all.py --output_dir ./outputs")
        print("3. Evaluate: python scripts/evaluate.py --model_path ./outputs/final")
        print("4. Deploy: python scripts/deploy.py package --model_path ./outputs/final")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

