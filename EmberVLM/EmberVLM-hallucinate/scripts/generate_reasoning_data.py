"""
Generate Reasoning Data for Stage 4 Training

This script generates chain-of-thought reasoning data from robot selection datasets.
The generated data includes step-by-step reasoning chains that explain the robot selection.

Inspired by tiny-r1 (DeepSeek-R1 style), uses structured XML format:
<reasoning>
...step by step analysis...
</reasoning>
<answer>
...final robot selection...
</answer>
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# System prompt for structured reasoning (DeepSeek-R1 style)
SYSTEM_PROMPT = """You are a robot fleet selection expert. When given a task, analyze it step-by-step and select the most suitable robot(s).

Respond in the following format:
<reasoning>
Step 1: [Analyze the task environment and requirements]
Step 2: [Evaluate relevant robot capabilities]
Step 3: [Compare suitable candidates]
Step 4: [Make final selection with justification]
</reasoning>
<answer>
[Selected Robot Name]
</answer>
"""

# Robot types mapping for validation
ROBOT_TYPES = ["Drone", "Humanoid", "Robot with Wheels", "Robot with Legs", "Underwater Robot"]


# Robot mapping with detailed attributes for reasoning
ROBOT_ATTRIBUTES = {
    "Drone": {
        "full_name": "Drone",
        "strengths": [
            "aerial navigation and surveillance",
            "fast deployment over large areas",
            "accessing hard-to-reach locations",
            "aerial inspection without physical contact",
            "lightweight transport capabilities"
        ],
        "limitations": [
            "limited payload capacity",
            "weather dependent operations",
            "limited battery life",
            "noise during operation"
        ],
        "ideal_environments": [
            "outdoor spaces",
            "large indoor facilities",
            "high-altitude locations",
            "areas requiring aerial perspective"
        ]
    },
    "Humanoid": {
        "full_name": "Humanoid",
        "strengths": [
            "human-like manipulation and dexterity",
            "tool use and complex task execution",
            "natural human interaction",
            "navigating human-designed spaces",
            "bipedal locomotion on stairs"
        ],
        "limitations": [
            "slow movement speed",
            "balance challenges on uneven terrain",
            "high power consumption"
        ],
        "ideal_environments": [
            "indoor environments",
            "human-designed spaces",
            "areas with stairs and doorways",
            "complex terrain requiring manipulation"
        ]
    },
    "Robot with Wheels": {
        "full_name": "Robot with Wheels",
        "strengths": [
            "fast and efficient movement",
            "good payload capacity",
            "stable platform for equipment",
            "energy efficient locomotion"
        ],
        "limitations": [
            "restricted to flat surfaces",
            "limited climbing ability",
            "obstacle navigation challenges"
        ],
        "ideal_environments": [
            "warehouses and factories",
            "flat outdoor areas",
            "roads and paved surfaces",
            "indoor corridors"
        ]
    },
    "Robot with Legs": {
        "full_name": "Robot with Legs",
        "strengths": [
            "rough terrain navigation",
            "high stability on uneven ground",
            "good load carrying capacity",
            "stair climbing ability"
        ],
        "limitations": [
            "limited manipulation capabilities",
            "height restrictions in some environments"
        ],
        "ideal_environments": [
            "outdoor rough terrain",
            "construction sites",
            "stairs and multi-level structures",
            "search and rescue scenarios"
        ]
    },
    "Underwater Robot": {
        "full_name": "Underwater Robot",
        "strengths": [
            "underwater navigation and exploration",
            "deep sea operations",
            "marine inspection capabilities",
            "underwater manipulation"
        ],
        "limitations": [
            "restricted to aquatic environments",
            "communication limitations underwater",
            "pressure constraints at depth"
        ],
        "ideal_environments": [
            "oceans and seas",
            "pools and tanks",
            "underwater pipelines",
            "marine research areas"
        ]
    }
}

# Reasoning templates - Now generates XML-structured output (DeepSeek-R1 style)
REASONING_TEMPLATES = [
    {
        "pattern": "environment_analysis",
        "template": [
            "Step 1: Analyzing the task environment - {environment_analysis}",
            "Step 2: Identifying required capabilities - {required_capabilities}",
            "Step 3: Comparing robot options - {comparison}",
            "Step 4: Final selection - {robot} is chosen because {justification}"
        ]
    },
    {
        "pattern": "capability_matching",
        "template": [
            "Step 1: The task requires - {task_requirements}",
            "Step 2: Evaluating each robot's capabilities for this task",
            "Step 3: {capability_analysis}",
            "Step 4: {robot} is the optimal choice because {justification}"
        ]
    },
    {
        "pattern": "elimination",
        "template": [
            "Step 1: Understanding the task - {task_description}",
            "Step 2: Eliminating unsuitable robots - {eliminations}",
            "Step 3: Comparing remaining options ({remaining_robots}) - {comparison}",
            "Step 4: {robot} is selected because {justification}"
        ]
    },
    {
        "pattern": "constraint_based",
        "template": [
            "Step 1: Identifying key constraints - {constraints}",
            "Step 2: Checking which robots meet constraints - {constraint_check}",
            "Step 3: Comparing suitable candidates - {comparison}",
            "Step 4: Final selection is {robot} because {justification}"
        ]
    }
]


def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted response (tiny-r1 style)."""
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        return answer.strip()
    return text.strip()


def extract_xml_reasoning(text: str) -> str:
    """Extract the reasoning from XML-formatted response."""
    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[-1].split("</reasoning>")[0]
        return reasoning.strip()
    return ""


def validate_xml_format(text: str) -> Tuple[bool, float]:
    """
    Validate the XML format and return (is_valid, format_score).
    Based on tiny-r1's reward functions.
    """
    score = 0.0

    # Check for reasoning tags
    if text.count("<reasoning>") == 1:
        score += 0.125
    if text.count("</reasoning>") == 1:
        score += 0.125

    # Check for answer tags
    if text.count("<answer>") == 1:
        score += 0.125
    if text.count("</answer>") == 1:
        score += 0.125

    # Check proper order
    if "<reasoning>" in text and "</reasoning>" in text and "<answer>" in text and "</answer>" in text:
        reasoning_end = text.find("</reasoning>")
        answer_start = text.find("<answer>")
        if reasoning_end < answer_start:
            score += 0.25

    # Penalize content after closing answer tag
    if "</answer>" in text:
        after_answer = text.split("</answer>")[-1]
        score -= len(after_answer.strip()) * 0.001

    is_valid = score >= 0.5
    return is_valid, max(0.0, score)


def extract_task_from_input(input_text: str) -> str:
    """Extract the task description from the input text."""
    if "Task:" in input_text:
        return input_text.split("Task:")[-1].strip()
    return input_text.strip()


def normalize_robot_name(robot_name: str) -> str:
    """Normalize robot name to match ROBOT_ATTRIBUTES keys."""
    robot_name = robot_name.strip()

    # Handle variations
    name_map = {
        "drone": "Drone",
        "humanoid": "Humanoid",
        "robot with wheels": "Robot with Wheels",
        "wheeled": "Robot with Wheels",
        "wheeled robot": "Robot with Wheels",
        "robot with legs": "Robot with Legs",
        "legged": "Robot with Legs",
        "legged robot": "Robot with Legs",
        "underwater robot": "Underwater Robot",
        "underwater": "Underwater Robot"
    }

    normalized = name_map.get(robot_name.lower(), robot_name)
    return normalized


def generate_reasoning_chain(task: str, selected_robot: str) -> str:
    """
    Generate a reasoning chain for robot selection in XML format (DeepSeek-R1 style).

    Returns a complete XML-formatted response with <reasoning> and <answer> tags.
    """
    robot_name = normalize_robot_name(selected_robot)

    if robot_name not in ROBOT_ATTRIBUTES:
        # Handle multi-robot selection
        robots = [normalize_robot_name(r.strip()) for r in selected_robot.split(",")]
        robot_name = robots[0] if robots else "Drone"

    robot_info = ROBOT_ATTRIBUTES.get(robot_name, ROBOT_ATTRIBUTES["Drone"])

    # Generate reasoning steps
    reasoning_steps = []

    # Step 1: Task Analysis
    reasoning_steps.append(f"Step 1: Analyzing the task - '{task}'")

    # Step 2: Environment/Requirement Analysis
    if "underwater" in task.lower() or "marine" in task.lower() or "sea" in task.lower() or "ocean" in task.lower():
        env_analysis = "This task involves an aquatic/underwater environment requiring specialized underwater capabilities"
    elif "aerial" in task.lower() or "fly" in task.lower() or "high" in task.lower() or "above" in task.lower() or "surveillance" in task.lower():
        env_analysis = "This task requires aerial capabilities, flight, or high-altitude access"
    elif "stair" in task.lower() or "terrain" in task.lower() or "rough" in task.lower() or "uneven" in task.lower():
        env_analysis = "This task involves navigating difficult terrain, stairs, or uneven surfaces"
    elif "warehouse" in task.lower() or "flat" in task.lower() or "factory" in task.lower() or "floor" in task.lower():
        env_analysis = "This task is in a structured indoor environment with flat, smooth surfaces"
    elif "human" in task.lower() or "interact" in task.lower() or "manipulat" in task.lower() or "tool" in task.lower():
        env_analysis = "This task requires human-like interaction, manipulation, or tool use capabilities"
    else:
        env_analysis = "Evaluating the general task requirements and operational environment"

    reasoning_steps.append(f"Step 2: Environment analysis - {env_analysis}")

    # Step 3: Capability Matching
    strength = random.choice(robot_info["strengths"])
    ideal_env = random.choice(robot_info["ideal_environments"])
    reasoning_steps.append(f"Step 3: {robot_name} excels at {strength} and is ideal for {ideal_env}")

    # Step 4: Comparison with alternatives
    other_robots = [r for r in ROBOT_ATTRIBUTES.keys() if r != robot_name]
    comparison_robot = random.choice(other_robots)
    comparison_info = ROBOT_ATTRIBUTES[comparison_robot]
    limitation = random.choice(comparison_info["limitations"])
    reasoning_steps.append(f"Step 4: Compared to {comparison_robot} (limited by {limitation}), {robot_name} is better suited for this task")

    # Build XML-formatted response
    reasoning_text = "\n".join(reasoning_steps)

    xml_response = f"""<reasoning>
{reasoning_text}
</reasoning>
<answer>
{robot_name}
</answer>"""

    return xml_response


def generate_reasoning_chain_list(task: str, selected_robot: str) -> List[str]:
    """
    Generate a list of reasoning steps (for backward compatibility).
    """
    robot_name = normalize_robot_name(selected_robot)

    if robot_name not in ROBOT_ATTRIBUTES:
        robots = [normalize_robot_name(r.strip()) for r in selected_robot.split(",")]
        robot_name = robots[0] if robots else "Drone"

    robot_info = ROBOT_ATTRIBUTES.get(robot_name, ROBOT_ATTRIBUTES["Drone"])

    reasoning_chain = []

    reasoning_chain.append(f"Analyzing the task: '{task}'")

    if "underwater" in task.lower() or "marine" in task.lower() or "sea" in task.lower():
        reasoning_chain.append("The task involves an aquatic/underwater environment")
    elif "aerial" in task.lower() or "fly" in task.lower() or "high" in task.lower():
        reasoning_chain.append("The task requires aerial capabilities or high-altitude access")
    elif "stair" in task.lower() or "terrain" in task.lower() or "rough" in task.lower():
        reasoning_chain.append("The task involves navigating difficult terrain or stairs")
    elif "warehouse" in task.lower() or "flat" in task.lower() or "indoor" in task.lower():
        reasoning_chain.append("The task is in a structured indoor environment with flat surfaces")
    elif "human" in task.lower() or "interact" in task.lower() or "manipulat" in task.lower():
        reasoning_chain.append("The task requires human-like interaction or manipulation capabilities")
    else:
        reasoning_chain.append("Evaluating the task requirements and operational environment")

    strength = random.choice(robot_info["strengths"])
    reasoning_chain.append(f"Key capability needed: {strength}")

    other_robots = [r for r in ROBOT_ATTRIBUTES.keys() if r != robot_name]
    comparison_robot = random.choice(other_robots)
    comparison_info = ROBOT_ATTRIBUTES[comparison_robot]
    limitation = random.choice(comparison_info["limitations"])
    reasoning_chain.append(f"Comparing with {comparison_robot}: limited by {limitation}")

    ideal_env = random.choice(robot_info["ideal_environments"])
    reasoning_chain.append(f"{robot_name} is ideal for {ideal_env}, making it the best choice")

    return reasoning_chain


def process_robot_selection_data(
    input_file: str,
    output_dir: str,
    num_samples: int = None,
    include_images: bool = False
) -> List[Dict[str, Any]]:
    """Process robot selection data and add reasoning chains with XML format."""

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    reasoning_samples = []

    for idx, item in enumerate(data):
        task = item.get('input', '')
        if not task:
            task = extract_task_from_input(item.get('instruction', ''))

        task = extract_task_from_input(task)
        selected_robot = item.get('output', '')

        if not task or not selected_robot:
            continue

        # Generate XML-formatted reasoning (DeepSeek-R1 style)
        xml_response = generate_reasoning_chain(task, selected_robot)

        # Also generate list-format for backward compatibility
        reasoning_chain_list = generate_reasoning_chain_list(task, selected_robot)

        # Normalize robot name
        robot_target = normalize_robot_name(selected_robot.split(",")[0].strip())

        # Create reasoning sample with both formats
        sample = {
            # Chat format (DeepSeek-R1 style)
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task}"}
            ],
            # Full XML response for training
            "response": xml_response,
            # Extracted answer for evaluation
            "answer": robot_target,
            # Legacy format for backward compatibility
            "instruction": f"Select the most suitable robot for the following task and explain your reasoning step by step.\n\nTask: {task}",
            "reasoning_chain": reasoning_chain_list,
            "robot_target": robot_target,
            "task": task,
            "full_instruction": item.get('instruction', ''),
            # Metadata
            "format": "xml",  # Mark as XML-formatted
        }

        # Add placeholder image path if images are expected
        if include_images:
            sample["image"] = f"images/robot_task_{idx:04d}.jpg"

        reasoning_samples.append(sample)

    return reasoning_samples


def augment_reasoning_data(
    samples: List[Dict[str, Any]],
    augmentation_factor: int = 3
) -> List[Dict[str, Any]]:
    """Augment reasoning data with variations (maintaining XML format)."""

    augmented = []

    for sample in samples:
        # Add original
        augmented.append(sample)

        # Create variations
        for _ in range(augmentation_factor - 1):
            new_sample = sample.copy()

            # Re-generate reasoning with different template
            task = sample.get("task", "")
            robot = sample.get("robot_target", "")

            if task and robot:
                # Generate new XML response
                new_sample["response"] = generate_reasoning_chain(task, robot)
                new_sample["reasoning_chain"] = generate_reasoning_chain_list(task, robot)

            # Vary instruction slightly while maintaining XML output expectation
            instruction_variations = [
                f"Which robot should be deployed for this task? Provide your reasoning in the specified format.\n\nTask: {task}",
                f"Analyze and select the optimal robot for: {task}",
                f"Given the task below, determine the best robot with step-by-step reasoning.\n\nTask: {task}",
                f"Select the most suitable robot for this scenario and explain why:\n\nTask: {task}",
            ]
            new_sample["instruction"] = random.choice(instruction_variations)

            # Update prompt with same variation
            new_sample["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task}"}
            ]

            augmented.append(new_sample)

    return augmented


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning data for Stage 4 training")
    parser.add_argument("--input_dir", type=str, default="robot-selection-dataset",
                       help="Input directory with robot selection data")
    parser.add_argument("--output_dir", type=str, default="outputs/reasoning-data",
                       help="Output directory for reasoning data")
    parser.add_argument("--augmentation_factor", type=int, default=3,
                       help="Augmentation factor for reasoning data")
    parser.add_argument("--include_images", action="store_true",
                       help="Include image placeholders in the data")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Process single robot selection data
    input_dir = Path(args.input_dir)
    single_robot_file = input_dir / "single_robot_selection.json"
    if single_robot_file.exists():
        logger.info(f"Processing {single_robot_file}")
        samples = process_robot_selection_data(
            str(single_robot_file),
            str(output_dir),
            include_images=args.include_images
        )
        all_samples.extend(samples)
        logger.info(f"Generated {len(samples)} samples from single robot selection")

    # Process multi robot selection data
    multi_robot_file = input_dir / "multi_robot_selection_dataset.json"
    if multi_robot_file.exists():
        logger.info(f"Processing {multi_robot_file}")
        samples = process_robot_selection_data(
            str(multi_robot_file),
            str(output_dir),
            include_images=args.include_images
        )
        all_samples.extend(samples)
        logger.info(f"Generated {len(samples)} samples from multi robot selection")

    # Augment data
    logger.info(f"Augmenting data with factor {args.augmentation_factor}")
    augmented_samples = augment_reasoning_data(all_samples, args.augmentation_factor)

    # Shuffle
    random.shuffle(augmented_samples)

    # Split into train/val
    split_idx = int(len(augmented_samples) * 0.9)
    train_samples = augmented_samples[:split_idx]
    val_samples = augmented_samples[split_idx:]

    # Save
    train_file = output_dir / "reasoning_train.json"
    val_file = output_dir / "reasoning_val.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(train_samples)} training samples to {train_file}")

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(val_samples)} validation samples to {val_file}")

    # Also save all samples in one file for compatibility
    all_file = output_dir / "reasoning_data.json"
    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved all {len(augmented_samples)} samples to {all_file}")

    logger.info("Reasoning data generation complete!")
    logger.info(f"Total samples: {len(augmented_samples)}")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")


if __name__ == "__main__":
    main()

