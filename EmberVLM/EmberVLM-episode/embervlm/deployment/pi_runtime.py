"""
Raspberry Pi Runtime for EmberVLM

Optimized inference runtime for edge deployment on
Raspberry Pi Zero and similar constrained devices.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Robot fleet for reference
ROBOT_FLEET = ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"]


class EmberVLMEdge:
    """
    Edge deployment runtime for EmberVLM.

    Optimized for:
    - Maximum RAM usage: 85MB
    - Inference latency: <5 seconds for 50 tokens
    - No GPU acceleration (CPU-only)
    """

    def __init__(
        self,
        model_path: str,
        max_memory_mb: int = 85,
        max_tokens: int = 50,
        target_latency_s: float = 5.0,
    ):
        """
        Initialize edge runtime.

        Args:
            model_path: Path to quantized model
            max_memory_mb: Maximum memory usage
            max_tokens: Maximum tokens to generate
            target_latency_s: Target latency in seconds
        """
        self.model_path = Path(model_path)
        self.max_memory_mb = max_memory_mb
        self.max_tokens = max_tokens
        self.target_latency_s = target_latency_s

        self.model = None
        self.tokenizer = None
        self.image_processor = None

        self._last_reasoning_chain = []
        self._inference_stats = {
            'total_inferences': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
        }

    def load(self):
        """Load model and prepare for inference."""
        import torch

        logger.info(f"Loading EmberVLM from {self.model_path}")

        # Check memory before loading
        self._check_memory()

        try:
            # Load tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            from embervlm.models import EmberVLM

            if self.model_path.is_dir():
                self.model = EmberVLM.from_pretrained(str(self.model_path))
            else:
                # Load from state dict
                self.model = EmberVLM()
                state_dict = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)

            # Set to eval mode
            self.model.eval()

            # Optimize for inference
            self.model = torch.jit.script(self.model) if hasattr(torch.jit, 'script') else self.model

            logger.info("Model loaded successfully")
            self._check_memory()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _check_memory(self):
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1e6

            if memory_mb > self.max_memory_mb:
                logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
            else:
                logger.info(f"Memory usage: {memory_mb:.1f}MB")

            return memory_mb
        except ImportError:
            return 0

    def preprocess_image(self, image_path: str) -> Any:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        import torch
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        image = Image.open(image_path).convert('RGB')
        pixel_values = transform(image).unsqueeze(0)

        return pixel_values

    def analyze_incident(
        self,
        image_path: str,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze incident image and select appropriate robot.

        Args:
            image_path: Path to incident image
            task_description: Optional task description

        Returns:
            Dictionary with:
                - selected_robot: Best robot for the task
                - confidence: Selection confidence
                - action_plan: Recommended actions
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Preprocess image
        pixel_values = self.preprocess_image(image_path)

        # Prepare text input
        if task_description:
            prompt = f"Task: {task_description}\nAnalyze this incident and select the best robot."
        else:
            prompt = "Analyze this incident scene and select the most appropriate robot from the fleet."

        tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens['input_ids'],
                pixel_values=pixel_values,
                attention_mask=tokens['attention_mask'],
                return_reasoning=True,
            )

        # Parse outputs
        result = self._parse_outputs(outputs)

        # Track statistics
        elapsed = time.perf_counter() - start_time
        self._update_stats(elapsed * 1000)

        result['latency_ms'] = elapsed * 1000

        return result

    def _parse_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model outputs."""
        import torch

        result = {}

        # Robot selection
        if 'robot_logits' in outputs:
            logits = outputs['robot_logits']
            probs = torch.softmax(logits, dim=-1)[0]
            robot_idx = probs.argmax().item()

            result['selected_robot'] = ROBOT_FLEET[robot_idx]
            result['robot_index'] = robot_idx
            result['robot_probabilities'] = {
                name: probs[i].item()
                for i, name in enumerate(ROBOT_FLEET)
            }

        # Confidence
        if 'robot_confidence' in outputs:
            result['confidence'] = outputs['robot_confidence'].item()
        else:
            result['confidence'] = max(result.get('robot_probabilities', {}).values(), default=0)

        # Action plan coherence
        if 'plan_coherence' in outputs:
            result['plan_coherence'] = outputs['plan_coherence'].item()

        # Generate action plan description
        result['action_plan'] = self._generate_action_plan(
            result.get('selected_robot', 'Drone'),
            result.get('confidence', 0.5),
        )

        return result

    def _generate_action_plan(self, robot: str, confidence: float) -> str:
        """Generate action plan based on selected robot."""
        plans = {
            'Drone': "1. Deploy drone for aerial reconnaissance\n"
                    "2. Survey affected area from above\n"
                    "3. Identify access points and hazards\n"
                    "4. Report findings to command center",

            'Humanoid': "1. Deploy humanoid robot to site\n"
                       "2. Navigate through human-scale passages\n"
                       "3. Perform manipulation tasks as needed\n"
                       "4. Assist with search and rescue",

            'Wheeled': "1. Load supplies onto wheeled robot\n"
                      "2. Navigate via accessible routes\n"
                      "3. Deliver payload to destination\n"
                      "4. Return for additional loads if needed",

            'Legged': "1. Deploy legged robot to rough terrain\n"
                     "2. Navigate obstacles and stairs\n"
                     "3. Reach inaccessible areas\n"
                     "4. Provide ground-level assessment",

            'Underwater': "1. Deploy underwater robot to water body\n"
                         "2. Survey underwater conditions\n"
                         "3. Inspect submerged infrastructure\n"
                         "4. Document findings with onboard camera",
        }

        return plans.get(robot, "Generic response plan")

    def _update_stats(self, latency_ms: float):
        """Update inference statistics."""
        n = self._inference_stats['total_inferences']
        avg = self._inference_stats['avg_latency_ms']

        self._inference_stats['total_inferences'] = n + 1
        self._inference_stats['avg_latency_ms'] = (avg * n + latency_ms) / (n + 1)
        self._inference_stats['max_latency_ms'] = max(
            self._inference_stats['max_latency_ms'],
            latency_ms
        )

    def get_reasoning_chain(self) -> List[str]:
        """Get the last reasoning chain generated."""
        return self._last_reasoning_chain

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            **self._inference_stats,
            'memory_mb': self._check_memory(),
        }

    def benchmark(
        self,
        image_path: str,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Run benchmark on single image.

        Args:
            image_path: Path to test image
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        latencies = []

        for _ in range(num_runs):
            result = self.analyze_incident(image_path)
            latencies.append(result['latency_ms'])

        import numpy as np

        return {
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'num_runs': num_runs,
        }


def create_edge_model_package(
    model_path: str,
    output_dir: str,
    include_tokenizer: bool = True,
):
    """
    Create deployment package for edge device.

    Args:
        model_path: Path to trained model
        output_dir: Output directory for package
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    model_output = output_dir / 'model'
    if Path(model_path).is_dir():
        shutil.copytree(model_path, model_output, dirs_exist_ok=True)
    else:
        model_output.mkdir(exist_ok=True)
        shutil.copy(model_path, model_output / 'pytorch_model.bin')

    # Create config
    config = {
        'model_name': 'EmberVLM',
        'version': '1.0.0',
        'max_memory_mb': 85,
        'target_latency_s': 5.0,
        'max_tokens': 50,
        'robot_fleet': ROBOT_FLEET,
    }

    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create run script
    run_script = '''#!/usr/bin/env python3
"""EmberVLM Edge Inference Script"""

import sys
from embervlm.deployment.pi_runtime import EmberVLMEdge

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <image_path> [task_description]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    task_description = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load model
    model = EmberVLMEdge(model_path='./model')
    model.load()
    
    # Run inference
    result = model.analyze_incident(image_path, task_description)
    
    print(f"Selected Robot: {result['selected_robot']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    print(f"\\nAction Plan:\\n{result['action_plan']}")

if __name__ == "__main__":
    main()
'''

    with open(output_dir / 'run_inference.py', 'w') as f:
        f.write(run_script)

    logger.info(f"Created edge deployment package at {output_dir}")

