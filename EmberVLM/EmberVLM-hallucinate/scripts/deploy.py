"""
EmberVLM Deployment Script

Quantize and package model for edge deployment.
"""

import os
import argparse
import json
from pathlib import Path
import logging

import torch

from embervlm.models import EmberVLM
from embervlm.quantization import quantize_model, QuantizationConfig
from embervlm.deployment import EmberVLMEdge
from embervlm.deployment.pi_runtime import create_edge_model_package

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_for_edge(
    model_path: str,
    output_path: str,
    bits: int = 8,
    calibration_data: str = None,
) -> str:
    """
    Quantize model for edge deployment.

    Args:
        model_path: Path to trained model
        output_path: Output path for quantized model
        bits: Quantization bits (4 or 8)
        calibration_data: Optional calibration data path

    Returns:
        Path to quantized model
    """
    logger.info(f"Loading model from {model_path}")
    model = EmberVLM.from_pretrained(model_path)

    # Configure quantization
    config = QuantizationConfig(
        method="dynamic" if bits == 8 else "awq",
        bits=bits,
        output_format="pytorch",
    )

    logger.info(f"Quantizing to {bits}-bit...")
    quantized_model = quantize_model(model, config, output_path=output_path)

    # Compute size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

    logger.info(f"Original size: {original_size / 1e6:.1f} MB")
    logger.info(f"Quantized size: {quantized_size / 1e6:.1f} MB")
    logger.info(f"Compression: {original_size / quantized_size:.1f}x")

    return output_path


def create_deployment_package(
    model_path: str,
    output_dir: str,
    quantize: bool = True,
    bits: int = 8,
):
    """
    Create complete deployment package.

    Args:
        model_path: Path to trained model
        output_dir: Output directory for package
        quantize: Whether to quantize the model
        bits: Quantization bits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quantize if requested
    if quantize:
        quantized_path = output_dir / 'model_quantized.pt'
        quantize_for_edge(model_path, str(quantized_path), bits)
        model_path = str(quantized_path)

    # Create package
    logger.info("Creating deployment package...")
    create_edge_model_package(model_path, str(output_dir))

    # Create requirements file
    requirements = """# EmberVLM Edge Dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.36.0
pillow>=10.0.0
numpy>=1.24.0
"""

    with open(output_dir / 'requirements.txt', 'w') as f:
        f.write(requirements)

    # Create deployment info
    deployment_info = {
        'model_name': 'EmberVLM',
        'quantization': f'{bits}-bit' if quantize else 'none',
        'target_device': 'Raspberry Pi Zero',
        'max_memory_mb': 85,
        'target_latency_s': 5.0,
        'robot_fleet': ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"],
    }

    with open(output_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)

    logger.info(f"Deployment package created at {output_dir}")
    print(f"\nDeployment package contents:")
    for item in output_dir.iterdir():
        print(f"  - {item.name}")


def validate_deployment(package_dir: str, test_image: str = None):
    """
    Validate deployment package.

    Args:
        package_dir: Path to deployment package
        test_image: Optional test image
    """
    package_dir = Path(package_dir)

    logger.info(f"Validating deployment package at {package_dir}")

    # Check required files
    required_files = ['config.json', 'run_inference.py']
    for filename in required_files:
        if not (package_dir / filename).exists():
            raise ValueError(f"Missing required file: {filename}")

    # Try loading model
    model_dir = package_dir / 'model'
    if model_dir.exists():
        try:
            edge_model = EmberVLMEdge(model_path=str(model_dir))
            edge_model.load()
            logger.info("Model loaded successfully")

            # Memory check
            stats = edge_model.get_stats()
            logger.info(f"Memory usage: {stats.get('memory_mb', 'N/A')} MB")

            # Run inference if test image provided
            if test_image and os.path.exists(test_image):
                logger.info(f"Testing with image: {test_image}")
                result = edge_model.analyze_incident(test_image)
                logger.info(f"Result: {result['selected_robot']} "
                          f"(confidence: {result['confidence']:.2%})")
                logger.info(f"Latency: {result['latency_ms']:.1f} ms")

        except Exception as e:
            logger.error(f"Failed to validate model: {e}")
            raise

    logger.info("Deployment package validation passed!")


def main():
    parser = argparse.ArgumentParser(description="EmberVLM Deployment")

    subparsers = parser.add_subparsers(dest='command')

    # Quantize command
    quant_parser = subparsers.add_parser('quantize', help='Quantize model')
    quant_parser.add_argument('--model_path', type=str, required=True)
    quant_parser.add_argument('--output_path', type=str, required=True)
    quant_parser.add_argument('--bits', type=int, default=8, choices=[4, 8])
    quant_parser.add_argument('--calibration_data', type=str, default=None)

    # Package command
    pkg_parser = subparsers.add_parser('package', help='Create deployment package')
    pkg_parser.add_argument('--model_path', type=str, required=True)
    pkg_parser.add_argument('--output_dir', type=str, required=True)
    pkg_parser.add_argument('--no_quantize', action='store_true')
    pkg_parser.add_argument('--bits', type=int, default=8, choices=[4, 8])

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate deployment package')
    val_parser.add_argument('--package_dir', type=str, required=True)
    val_parser.add_argument('--test_image', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'quantize':
        quantize_for_edge(
            args.model_path,
            args.output_path,
            args.bits,
            args.calibration_data,
        )

    elif args.command == 'package':
        create_deployment_package(
            args.model_path,
            args.output_dir,
            quantize=not args.no_quantize,
            bits=args.bits,
        )

    elif args.command == 'validate':
        validate_deployment(args.package_dir, args.test_image)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

