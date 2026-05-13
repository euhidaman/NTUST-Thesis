"""
Synthetic Data Generator for EmberVLM

Generates synthetic training data using teacher models
for reasoning chain augmentation.
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic training data using teacher models.

    Creates:
    - Reasoning chains for existing samples
    - Augmented instruction variants
    - Validated reasoning with rule-based checker
    """

    def __init__(
        self,
        teacher=None,
        output_dir: str = "./synthetic_data",
    ):
        self.teacher = teacher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Robot fleet for validation
        self.robot_fleet = ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"]

        # Reasoning templates
        self.reasoning_templates = [
            "Step 1: Identify key incident characteristics - {incident_type}",
            "Step 2: Match robot capabilities to requirements - {capability_match}",
            "Step 3: Consider environmental constraints - {constraints}",
            "Step 4: Select optimal robot - {selection_reason}",
        ]

    def generate_reasoning_chain(
        self,
        image_path: str,
        instruction: str,
        selected_robot: str,
    ) -> List[str]:
        """
        Generate reasoning chain for a sample.

        Args:
            image_path: Path to incident image
            instruction: Task instruction
            selected_robot: Target robot selection

        Returns:
            List of reasoning steps
        """
        # If teacher available, use it
        if self.teacher is not None:
            try:
                prompt = f"""Given this incident scene, explain step-by-step why {selected_robot} is the best robot choice.

Task: {instruction}

Provide reasoning in 4 steps:
1. Identify key incident characteristics
2. Match robot capabilities to requirements  
3. Consider environmental constraints
4. Explain why {selected_robot} is optimal

Format each step as "Step N: <reasoning>"
"""
                response = self.teacher.generate_response(image_path, prompt)

                # Parse response into steps
                steps = self._parse_reasoning_response(response)
                if steps:
                    return steps
            except Exception as e:
                logger.warning(f"Teacher generation failed: {e}")

        # Fallback to template-based generation
        return self._generate_template_reasoning(instruction, selected_robot)

    def _parse_reasoning_response(self, response: str) -> List[str]:
        """Parse teacher response into reasoning steps."""
        steps = []

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Step') and ':' in line:
                steps.append(line)

        return steps if len(steps) >= 3 else []

    def _generate_template_reasoning(
        self,
        instruction: str,
        selected_robot: str,
    ) -> List[str]:
        """Generate reasoning using templates."""

        # Robot-specific reasoning
        robot_reasons = {
            "Drone": {
                "incident_type": "aerial view needed, rapid assessment required",
                "capability_match": "need aerial reconnaissance, wide coverage",
                "constraints": "limited ground access, need bird's eye view",
                "selection_reason": "Drone provides fast aerial survey with broad coverage",
            },
            "Humanoid": {
                "incident_type": "manipulation task, human-scale environment",
                "capability_match": "need fine motor control, tool operation",
                "constraints": "human-designed space, requires dexterity",
                "selection_reason": "Humanoid can manipulate objects and use human tools",
            },
            "Wheeled": {
                "incident_type": "transport task, road-accessible area",
                "capability_match": "high payload capacity, long range needed",
                "constraints": "flat terrain available, need efficient transport",
                "selection_reason": "Wheeled robot has maximum payload and range",
            },
            "Legged": {
                "incident_type": "rough terrain navigation, obstacle traversal",
                "capability_match": "need terrain adaptability, climbing ability",
                "constraints": "uneven surfaces, debris, stairs present",
                "selection_reason": "Legged robot can navigate challenging terrain",
            },
            "Underwater": {
                "incident_type": "aquatic environment, submerged target",
                "capability_match": "need underwater operation capability",
                "constraints": "flooded area, underwater infrastructure",
                "selection_reason": "Underwater robot is designed for aquatic operations",
            },
        }

        reasons = robot_reasons.get(selected_robot, robot_reasons["Drone"])

        steps = [
            template.format(**reasons)
            for template in self.reasoning_templates
        ]

        return steps

    def validate_reasoning(
        self,
        reasoning_chain: List[str],
        selected_robot: str,
    ) -> Dict[str, Any]:
        """
        Validate reasoning chain quality.

        Args:
            reasoning_chain: List of reasoning steps
            selected_robot: Target robot

        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'score': 1.0,
        }

        # Check minimum steps
        if len(reasoning_chain) < 3:
            validation['valid'] = False
            validation['issues'].append("Too few reasoning steps")
            validation['score'] -= 0.3

        # Check step structure
        for i, step in enumerate(reasoning_chain):
            if not step.lower().startswith('step'):
                validation['issues'].append(f"Step {i+1} missing 'Step' prefix")
                validation['score'] -= 0.1

        # Check robot mentioned in final step
        if reasoning_chain:
            final_step = reasoning_chain[-1].lower()
            if selected_robot.lower() not in final_step:
                validation['issues'].append("Selected robot not mentioned in conclusion")
                validation['score'] -= 0.2

        # Check for key reasoning components
        full_text = ' '.join(reasoning_chain).lower()

        key_terms = ['identify', 'capability', 'consider', 'select']
        found_terms = sum(1 for term in key_terms if term in full_text)

        if found_terms < 2:
            validation['issues'].append("Missing key reasoning terms")
            validation['score'] -= 0.2

        validation['score'] = max(0, validation['score'])
        validation['valid'] = validation['score'] >= 0.5

        return validation

    def augment_dataset(
        self,
        input_file: str,
        output_file: str,
        num_augmented: int = 50000,
    ):
        """
        Augment dataset with synthetic reasoning chains.

        Args:
            input_file: Path to input dataset JSON
            output_file: Path to output augmented dataset
            num_augmented: Target number of augmented samples
        """
        logger.info(f"Loading dataset from {input_file}")

        with open(input_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            samples = data.get('samples', [data])
        else:
            samples = data

        augmented_samples = []

        for sample in tqdm(samples, desc="Augmenting"):
            # Original sample
            augmented_samples.append(sample)

            # Generate variations
            num_variations = num_augmented // len(samples)

            for _ in range(num_variations):
                new_sample = self._create_variation(sample)

                # Validate
                if 'reasoning_chain' in new_sample:
                    validation = self.validate_reasoning(
                        new_sample['reasoning_chain'],
                        new_sample.get('selected_robot', ''),
                    )

                    if validation['valid']:
                        new_sample['reasoning_validation'] = validation
                        augmented_samples.append(new_sample)

        # Save augmented dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(augmented_samples, f, indent=2)

        logger.info(f"Created {len(augmented_samples)} samples, saved to {output_path}")

        return str(output_path)

    def _create_variation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create variation of a sample."""
        new_sample = sample.copy()

        # Vary instruction
        instruction = new_sample.get('instruction', '')
        instruction = self._vary_instruction(instruction)
        new_sample['instruction'] = instruction

        # Generate new reasoning chain
        if 'selected_robot' in new_sample:
            new_sample['reasoning_chain'] = self.generate_reasoning_chain(
                new_sample.get('image', ''),
                instruction,
                new_sample['selected_robot'],
            )

        return new_sample

    def _vary_instruction(self, instruction: str) -> str:
        """Create variation of instruction text."""
        # Word substitutions
        substitutions = {
            'analyze': ['examine', 'assess', 'evaluate', 'investigate'],
            'select': ['choose', 'pick', 'identify', 'determine'],
            'incident': ['scene', 'situation', 'event', 'scenario'],
            'appropriate': ['suitable', 'optimal', 'best', 'ideal'],
            'robot': ['unit', 'robot unit', 'robotic system'],
        }

        result = instruction.lower()

        for word, alternatives in substitutions.items():
            if word in result and random.random() > 0.5:
                result = result.replace(word, random.choice(alternatives), 1)

        # Capitalize first letter
        result = result[0].upper() + result[1:] if result else result

        return result


def create_reasoning_augmented_dataset(
    base_dataset_path: str,
    output_dir: str,
    teacher_model_name: Optional[str] = None,
    num_samples: int = 50000,
):
    """
    Create reasoning-augmented dataset.

    Args:
        base_dataset_path: Path to base dataset
        output_dir: Output directory
        teacher_model_name: Optional teacher model for generation
        num_samples: Target number of samples
    """
    # Initialize teacher if specified
    teacher = None
    if teacher_model_name:
        try:
            from embervlm.distillation.teacher import TeacherWrapper
            teacher = TeacherWrapper(teacher_model_name)
        except Exception as e:
            logger.warning(f"Could not load teacher: {e}")

    # Create generator
    generator = SyntheticDataGenerator(
        teacher=teacher,
        output_dir=output_dir,
    )

    # Augment dataset
    output_file = "reasoning_augmented.json"
    generator.augment_dataset(
        base_dataset_path,
        output_file,
        num_augmented=num_samples,
    )

    return str(Path(output_dir) / output_file)

