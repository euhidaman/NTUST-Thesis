"""
Enhanced Robot Selection Dataset Loader for EmberVLM

Comprehensive implementation with:
- Single-robot and multi-robot dataset loading
- Chain-of-thought reasoning generation
- Subtask decomposition for multi-robot scenarios
- Data augmentation (10K+ samples)
- Scene graph to image generation
- Proper train/val/test splits
"""

import os
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# Robot fleet definitions matching the dataset format
ROBOT_MAPPING = {
    "Drone": 0,
    "Underwater Robot": 1,
    "Humanoid": 2,
    "Robot with Wheels": 3,
    "Robot with Legs": 4,
}

ROBOT_IDX_TO_NAME = {idx: name for name, idx in ROBOT_MAPPING.items()}
NUM_ROBOTS = len(ROBOT_MAPPING)

# Robot capabilities extracted from dataset instructions
ROBOT_CAPABILITIES = {
    "Drone": {
        "strengths": ["fastest", "aerial navigation", "surveillance", "lightweight transport", "aerial inspection"],
        "weaknesses": ["limited payload", "loud", "weather dependent", "battery life"],
        "environments": ["outdoor", "large indoor spaces", "hard to reach areas"],
        "keywords": ["aerial", "fly", "height", "above", "survey from air", "roof", "exterior", "building"],
    },
    "Underwater Robot": {
        "strengths": ["underwater navigation", "deep sea exploration", "marine inspection", "underwater manipulation"],
        "weaknesses": ["water environments only", "communication limitations", "pressure constraints"],
        "environments": ["underwater", "marine", "pools", "pipes", "ocean exploration"],
        "keywords": ["underwater", "ocean", "sea", "marine", "aquatic", "diving", "submerged", "pipe", "leak"],
    },
    "Humanoid": {
        "strengths": ["manipulation", "walking", "human interaction", "complex tasks", "tool use"],
        "weaknesses": ["slow movement", "balance issues", "high power consumption"],
        "environments": ["indoor", "human environments", "stairs", "complex terrain"],
        "keywords": ["manipulate", "tool", "door", "human", "interact", "stairs", "indoor", "delicate"],
    },
    "Robot with Wheels": {
        "strengths": ["fast movement", "good payload", "stable platform", "efficient"],
        "weaknesses": ["flat surfaces only", "limited climbing", "obstacle avoidance"],
        "environments": ["indoor", "warehouse", "flat outdoor areas", "roads"],
        "keywords": ["transport", "warehouse", "road", "flat", "cargo", "delivery", "fast", "indoor"],
    },
    "Robot with Legs": {
        "strengths": ["rough terrain navigation", "stability", "load carrying", "inspection"],
        "weaknesses": ["limited manipulation", "height restrictions"],
        "environments": ["outdoor", "uneven terrain", "stairs", "industrial sites", "search and rescue"],
        "keywords": ["rough terrain", "climb", "mountain", "rocky", "stairs", "uneven", "rubble", "rescue"],
    },
}


class EnhancedRobotSelectionDataset(Dataset):
    """
    Enhanced robot selection dataset with reasoning chains.

    Supports:
    - Single-robot selection (1252 samples)
    - Multi-robot selection with subtasks (6886 samples)
    - Chain-of-thought reasoning generation
    - Data augmentation to 10K+ samples
    - Scene visualization
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        split: str = 'train',
        max_length: int = 1024,
        image_size: int = 224,
        include_reasoning: bool = True,
        include_multi_robot: bool = True,
        augment_data: bool = True,
        augmentation_factor: int = 3,
        curriculum_level: Optional[str] = None,  # 'easy', 'medium', 'hard', 'expert'
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.include_reasoning = include_reasoning
        self.include_multi_robot = include_multi_robot
        self.augment_data = augment_data and split == 'train'
        self.augmentation_factor = augmentation_factor
        self.curriculum_level = curriculum_level

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Load all data
        self.samples = self._load_all_data()

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"  Single-robot: {sum(1 for s in self.samples if s['type'] == 'single')}")
        logger.info(f"  Multi-robot: {sum(1 for s in self.samples if s['type'] == 'multi')}")

    def _load_all_data(self) -> List[Dict[str, Any]]:
        """Load both single and multi-robot datasets."""
        all_samples = []

        # Load single-robot dataset
        single_path = self.data_dir / 'single_robot_selection.json'
        if single_path.exists():
            with open(single_path, 'r', encoding='utf-8') as f:
                single_data = json.load(f)
            for item in single_data:
                sample = self._parse_single_robot(item)
                if sample:
                    all_samples.append(sample)
            logger.info(f"Loaded {len(single_data)} single-robot samples from {single_path}")
        else:
            logger.warning(f"Single-robot dataset not found at {single_path}")

        # Load multi-robot dataset if enabled
        if self.include_multi_robot:
            multi_path = self.data_dir / 'multi_robot_selection_dataset.json'
            if multi_path.exists():
                with open(multi_path, 'r', encoding='utf-8') as f:
                    multi_data = json.load(f)
                for item in multi_data:
                    sample = self._parse_multi_robot(item)
                    if sample:
                        all_samples.append(sample)
                logger.info(f"Loaded {len(multi_data)} multi-robot samples from {multi_path}")
            else:
                logger.warning(f"Multi-robot dataset not found at {multi_path}")

        if not all_samples:
            logger.warning("No data found! Creating synthetic samples...")
            all_samples = self._create_synthetic_samples()

        # Apply curriculum filtering if specified
        if self.curriculum_level:
            all_samples = self._filter_by_curriculum(all_samples)

        # Create deterministic train/val/test split
        all_samples = self._create_splits(all_samples)

        # Augment training data
        if self.augment_data and self.split == 'train':
            original_count = len(all_samples)
            all_samples = self._augment_samples(all_samples)
            logger.info(f"Augmented training data from {original_count} to {len(all_samples)} samples")

        return all_samples

    def _parse_single_robot(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse single-robot selection sample."""
        try:
            task = item['input'].replace('Task: ', '').strip()
            robot_output = item['output']

            # Parse robot names (can be comma-separated)
            robot_names = [r.strip() for r in robot_output.split(',')]
            primary_robot = robot_names[0]

            # Generate reasoning chain
            reasoning = self._generate_reasoning(task, primary_robot)

            return {
                'type': 'single',
                'task': task,
                'instruction': item['instruction'],
                'primary_robot': primary_robot,
                'all_robots': robot_names,
                'reasoning': reasoning,
                'subtasks': None,
                'difficulty': self._assess_difficulty(task, len(robot_names)),
            }
        except Exception as e:
            logger.warning(f"Failed to parse single-robot sample: {e}")
            return None

    def _parse_multi_robot(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse multi-robot selection with subtasks."""
        try:
            task = item['input'].replace('Task: ', '').strip()
            subtasks = item['subtasks']

            # Extract robot assignments and execution order
            robot_assignments = {}
            for subtask in subtasks:
                robot = subtask['assigned_robot']
                order = subtask['execution_order']
                if robot not in robot_assignments:
                    robot_assignments[robot] = []
                robot_assignments[robot].append({
                    'subtask': subtask['subtask'],
                    'order': order
                })

            # Primary robot is the one with most subtasks or earliest order
            primary_robot = max(robot_assignments.keys(),
                              key=lambda r: (len(robot_assignments[r]), -min(s['order'] for s in robot_assignments[r])))

            # Generate reasoning for multi-robot coordination
            reasoning = self._generate_multi_robot_reasoning(task, robot_assignments, subtasks)

            return {
                'type': 'multi',
                'task': task,
                'instruction': item['instruction'],
                'primary_robot': primary_robot,
                'all_robots': list(robot_assignments.keys()),
                'reasoning': reasoning,
                'subtasks': subtasks,
                'robot_assignments': robot_assignments,
                'difficulty': self._assess_difficulty(task, len(robot_assignments)),
            }
        except Exception as e:
            logger.warning(f"Failed to parse multi-robot sample: {e}")
            return None

    def _generate_reasoning(self, task: str, robot: str) -> str:
        """Generate chain-of-thought reasoning for single robot selection in XML format (DeepSeek-R1 style)."""
        reasoning_parts = []

        # Step 1: Analyze task requirements
        task_lower = task.lower()
        requirements = []
        for keyword in ['inspect', 'survey', 'transport', 'navigate', 'search', 'deliver', 'climb', 'underwater', 'rescue', 'monitor']:
            if keyword in task_lower:
                requirements.append(keyword)

        reasoning_parts.append(f"Step 1: Task Analysis")
        reasoning_parts.append(f"The task requires: {', '.join(requirements) if requirements else 'general navigation and operation'}.")

        # Step 2: Environment analysis
        environments = []
        env_keywords = {
            'outdoor': ['outdoor', 'field', 'desert', 'mountain', 'forest', 'external'],
            'indoor': ['indoor', 'building', 'warehouse', 'room', 'inside', 'facility'],
            'aerial': ['air', 'roof', 'above', 'height', 'exterior', 'high', 'sky', 'flying'],
            'underwater': ['underwater', 'ocean', 'sea', 'water', 'pipe', 'marine', 'submerged'],
            'rough terrain': ['rough', 'rocky', 'rubble', 'stairs', 'climb', 'uneven', 'mountain'],
        }
        for env_type, keywords in env_keywords.items():
            if any(kw in task_lower for kw in keywords):
                environments.append(env_type)

        reasoning_parts.append(f"Step 2: Environment Assessment")
        reasoning_parts.append(f"Identified environment type(s): {', '.join(environments) if environments else 'standard environment'}.")

        # Step 3: Robot capability matching
        reasoning_parts.append(f"Step 3: Robot Capability Matching")
        if robot in ROBOT_CAPABILITIES:
            caps = ROBOT_CAPABILITIES[robot]
            strengths = caps['strengths'][:3]
            weaknesses = caps['weaknesses'][:2]
            reasoning_parts.append(f"{robot} has these key strengths: {', '.join(strengths)}.")
            reasoning_parts.append(f"Considerations: {', '.join(weaknesses)}.")
        else:
            reasoning_parts.append(f"{robot} is being evaluated for this task.")

        # Step 4: Final justification
        reasoning_parts.append(f"Step 4: Decision")
        reasoning_parts.append(f"Based on the task requirements and environmental analysis, {robot} is the optimal choice because its capabilities align well with the mission objectives.")

        # Return in XML format (DeepSeek-R1 / tiny-r1 style)
        reasoning_text = "\n".join(reasoning_parts)
        return reasoning_text

    def _generate_reasoning_list(self, task: str, robot: str) -> List[str]:
        """Generate reasoning as a list of steps (legacy format)."""
        reasoning_text = self._generate_reasoning(task, robot)
        # Split by "Step X:" to get list format
        import re
        steps = re.split(r'Step \d+:', reasoning_text)
        return [s.strip() for s in steps if s.strip()]

    def _generate_multi_robot_reasoning(self, task: str, robot_assignments: Dict, subtasks: List) -> str:
        """Generate reasoning for multi-robot coordination in XML format (DeepSeek-R1 style)."""
        reasoning_parts = []

        # Step 1: Task decomposition
        reasoning_parts.append(f"Step 1: Task Decomposition")
        reasoning_parts.append(f"This is a complex task requiring coordination of {len(robot_assignments)} different robot types.")
        reasoning_parts.append(f"Total subtasks identified: {len(subtasks)}.")

        # Step 2: Subtask analysis
        reasoning_parts.append(f"Step 2: Subtask Analysis")
        for i, subtask in enumerate(subtasks[:4], 1):  # Show first 4 subtasks
            reasoning_parts.append(f"  - Subtask {i}: {subtask['subtask'][:60]}...")

        # Step 3: Robot assignment strategy
        reasoning_parts.append(f"Step 3: Robot Assignment Strategy")
        for robot, tasks in robot_assignments.items():
            if robot in ROBOT_CAPABILITIES:
                strengths = ROBOT_CAPABILITIES[robot]['strengths'][:2]
                reasoning_parts.append(f"  - {robot}: Assigned {len(tasks)} subtask(s). Key capabilities: {', '.join(strengths)}.")
            else:
                reasoning_parts.append(f"  - {robot}: Assigned {len(tasks)} subtask(s).")

        # Step 4: Execution coordination
        execution_orders = sorted(set(s['execution_order'] for s in subtasks))
        reasoning_parts.append(f"Step 4: Execution Plan")
        reasoning_parts.append(f"Mission will be executed in {len(execution_orders)} phases with coordinated robot deployment.")
        reasoning_parts.append(f"Primary robot takes lead based on task complexity and mission criticality.")

        # Return in XML format (DeepSeek-R1 / tiny-r1 style)
        reasoning_text = "\n".join(reasoning_parts)
        return reasoning_text

    def _generate_multi_robot_reasoning_list(self, task: str, robot_assignments: Dict, subtasks: List) -> List[str]:
        """Generate multi-robot reasoning as a list of steps (legacy format)."""
        reasoning_text = self._generate_multi_robot_reasoning(task, robot_assignments, subtasks)
        import re
        steps = re.split(r'Step \d+:', reasoning_text)
        return [s.strip() for s in steps if s.strip()]

    def _assess_difficulty(self, task: str, num_robots: int) -> str:
        """Assess task difficulty for curriculum learning."""
        task_lower = task.lower()

        # Easy: Single robot, obvious match
        if num_robots == 1:
            for robot, caps in ROBOT_CAPABILITIES.items():
                if any(kw in task_lower for kw in caps['keywords'][:2]):
                    return 'easy'

        # Medium: Single robot but requires reasoning
        if num_robots == 1:
            return 'medium'

        # Hard: Multiple robots (2-3)
        if num_robots <= 3:
            return 'hard'

        # Expert: Complex multi-robot (4+)
        return 'expert'

    def _filter_by_curriculum(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples by curriculum difficulty."""
        filtered = [s for s in samples if s['difficulty'] == self.curriculum_level]
        if not filtered:
            logger.warning(f"No samples found for curriculum level {self.curriculum_level}, using all samples")
            return samples
        return filtered

    def _create_splits(self, samples: List[Dict]) -> List[Dict]:
        """Create deterministic train/val/test split (70/20/10)."""
        # Use hash of task for deterministic splitting
        def get_split(task: str) -> str:
            hash_val = int(hashlib.md5(task.encode()).hexdigest(), 16)
            if hash_val % 100 < 70:
                return 'train'
            elif hash_val % 100 < 90:
                return 'val'
            else:
                return 'test'

        # Filter by split
        filtered = [s for s in samples if get_split(s['task']) == self.split]

        # Shuffle within split (with fixed seed for reproducibility)
        random.seed(42)
        random.shuffle(filtered)

        return filtered

    def _augment_samples(self, samples: List[Dict]) -> List[Dict]:
        """Augment training samples with variations."""
        augmented = []

        for sample in samples:
            # Add original
            augmented.append(sample)

            # Create variations
            for aug_idx in range(self.augmentation_factor - 1):
                aug_sample = self._create_augmentation(sample, aug_idx)
                augmented.append(aug_sample)

        return augmented

    def _create_augmentation(self, sample: Dict, aug_idx: int) -> Dict:
        """Create an augmented version of a sample."""
        aug_sample = sample.copy()

        # Paraphrase task description
        task = sample['task']
        task = self._paraphrase_task(task, aug_idx)
        aug_sample['task'] = task

        # Add slight variation to reasoning
        if sample['reasoning']:
            # Handle both string and list reasoning formats
            if isinstance(sample['reasoning'], str):
                # For string reasoning, just copy it (no reordering possible)
                aug_sample['reasoning'] = sample['reasoning']
            elif isinstance(sample['reasoning'], list):
                aug_sample['reasoning'] = sample['reasoning'].copy()
                # Reorder some reasoning steps randomly
                if len(aug_sample['reasoning']) > 2 and random.random() > 0.5:
                    # Swap two middle steps
                    mid = len(aug_sample['reasoning']) // 2
                    aug_sample['reasoning'][mid-1], aug_sample['reasoning'][mid] = \
                        aug_sample['reasoning'][mid], aug_sample['reasoning'][mid-1]
            else:
                # For any other type, just copy the reference
                aug_sample['reasoning'] = sample['reasoning']

        return aug_sample

    def _paraphrase_task(self, task: str, aug_idx: int) -> str:
        """Paraphrase task description."""
        synonyms = {
            'inspect': ['examine', 'check', 'assess', 'evaluate'],
            'survey': ['scan', 'map', 'assess', 'overview'],
            'transport': ['carry', 'deliver', 'move', 'convey'],
            'navigate': ['traverse', 'cross', 'travel through', 'pass through'],
            'search': ['look for', 'find', 'locate', 'seek'],
            'monitor': ['watch', 'observe', 'track', 'supervise'],
            'deliver': ['bring', 'transport', 'carry', 'convey'],
            'explore': ['investigate', 'examine', 'survey', 'scout'],
        }

        words = task.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in synonyms and random.random() > 0.5:
                replacement = synonyms[word_lower][aug_idx % len(synonyms[word_lower])]
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement

        return ' '.join(words)

    def _create_synthetic_samples(self) -> List[Dict]:
        """Create synthetic samples if no data available."""
        logger.warning("Creating 100 synthetic samples...")
        samples = []

        scenarios = [
            ("Survey agricultural fields from above", "Drone", 'easy'),
            ("Inspect underwater pipeline for damage", "Underwater Robot", 'easy'),
            ("Navigate rough mountain terrain", "Robot with Legs", 'medium'),
            ("Transport heavy cargo in warehouse", "Robot with Wheels", 'easy'),
            ("Manipulate tools in indoor environment", "Humanoid", 'medium'),
        ]

        for task, robot, difficulty in scenarios * 20:  # Repeat to get 100 samples
            reasoning = self._generate_reasoning(task, robot)
            samples.append({
                'type': 'single',
                'task': task,
                'instruction': "Select the most appropriate robot for this task.",
                'primary_robot': robot,
                'all_robots': [robot],
                'reasoning': reasoning,
                'subtasks': None,
                'difficulty': difficulty,
            })

        return samples

    def _generate_scene_image(self, task: str, robot: str) -> Image.Image:
        """Generate a simple scene visualization."""
        # Create blank image
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)

        # Try to load a default font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Draw task type indicator
        task_lower = task.lower()
        if 'aerial' in task_lower or 'above' in task_lower or 'roof' in task_lower:
            # Draw sky
            draw.rectangle([0, 0, self.image_size, self.image_size//3], fill='lightblue')
            draw.text((10, 10), "Aerial Task", fill='black', font=font)
        elif 'underwater' in task_lower or 'ocean' in task_lower:
            # Draw water
            draw.rectangle([0, 0, self.image_size, self.image_size], fill='darkblue')
            draw.text((10, 10), "Underwater Task", fill='white', font=font)
        elif 'indoor' in task_lower or 'building' in task_lower:
            # Draw building
            draw.rectangle([20, 40, self.image_size-20, self.image_size-20], fill='lightgray', outline='black')
            draw.text((10, 10), "Indoor Task", fill='black', font=font)
        else:
            # Draw outdoor ground
            draw.rectangle([0, self.image_size//2, self.image_size, self.image_size], fill='lightgreen')
            draw.text((10, 10), "Outdoor Task", fill='black', font=font)

        # Draw robot indicator
        draw.text((10, self.image_size - 30), f"Robot: {robot}", fill='red', font=small_font)

        return img

    def _format_prompt(self, sample: Dict) -> str:
        """Format training prompt with reasoning chain.

        Uses standard text markers instead of special tokens to avoid
        tokenizer/embedding size mismatches that cause CUDA index errors.
        """
        parts = []

        # Instruction
        parts.append(f"Instruction: {sample['instruction'][:200]}")

        # Task
        parts.append(f"\nTask: {sample['task']}")

        # For multi-robot, include subtask information
        if sample['type'] == 'multi' and sample['subtasks']:
            parts.append(f"\nSubtasks: {len(sample['subtasks'])} identified")
            for i, subtask in enumerate(sample['subtasks'][:3]):  # Show first 3
                parts.append(f"  {i+1}. {subtask['subtask']} (Order: {subtask['execution_order']})")

        # Reasoning chain - use XML-style tags that tokenize as regular subwords
        # This avoids special token ID mismatches that cause CUDA index errors
        if self.include_reasoning and sample['reasoning']:
            parts.append("\n[REASONING]")
            for i, step in enumerate(sample['reasoning'], 1):
                parts.append(f"Step {i}: {step}")
            parts.append("[/REASONING]")

        # Robot selection - use simple text format
        parts.append(f"\n[ROBOT] {sample['primary_robot']} [/ROBOT]")

        # For multi-robot, list all assigned robots
        if sample['type'] == 'multi' and len(sample['all_robots']) > 1:
            parts.append(f"Additional robots: {', '.join(sample['all_robots'][1:])}")

        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Generate scene image
        img = self._generate_scene_image(sample['task'], sample['primary_robot'])
        pixel_values = self.transform(img)

        # Create prompt
        text = self._format_prompt(sample)

        # Tokenize with explicit handling to prevent out-of-bounds issues
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        except Exception as e:
            logger.error(f"Tokenization failed for sample {idx}: {e}")
            # Return a safe dummy sample
            dummy_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            dummy_input_ids[0] = self.tokenizer.bos_token_id or 0
            return {
                'pixel_values': pixel_values,
                'input_ids': dummy_input_ids,
                'attention_mask': torch.ones(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                'robot_target': torch.tensor(0, dtype=torch.long),
                'multi_robot_target': torch.zeros(NUM_ROBOTS),
                'is_multi_robot': torch.tensor(0.0),
                'difficulty': 'easy',
                'task_description': sample['task'],
            }

        # Clone tensors immediately to avoid CUDA async issues
        input_ids = encoding['input_ids'].squeeze(0).clone()
        attention_mask = encoding['attention_mask'].squeeze(0).clone()

        # CRITICAL FIX: Robust validation and fixing of token IDs
        # Get the actual vocabulary size from tokenizer
        vocab_size = self.tokenizer.vocab_size
        # Also check the full length including special tokens
        full_vocab_size = len(self.tokenizer)

        # Use the larger of the two to be safe
        effective_vocab_size = max(vocab_size, full_vocab_size)

        # Check for invalid tokens (out of bounds or negative)
        invalid_mask = (input_ids >= effective_vocab_size) | (input_ids < 0)

        if invalid_mask.any():
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()
            num_invalid = invalid_mask.sum().item()

            # Log detailed information for debugging
            logger.warning(
                f"⚠️ {num_invalid} invalid token IDs! Range: [{min_token_id}, {max_token_id}], "
                f"Valid: [0, {effective_vocab_size - 1}]. Sample idx: {idx}. Replacing with 0."
            )

            # Find which positions have invalid tokens (for debugging)
            invalid_positions = invalid_mask.nonzero(as_tuple=True)[0].tolist()[:5]
            invalid_values = input_ids[invalid_mask][:5].tolist()
            logger.warning(f"   Invalid positions (first 5): {invalid_positions}")
            logger.warning(f"   Invalid values (first 5): {invalid_values}")

            # Replace invalid tokens with 0 (safer and more predictable)
            input_ids = torch.where(invalid_mask, torch.zeros_like(input_ids), input_ids)

        # Labels for language modeling - clone again for safety
        labels = input_ids.clone()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Final validation: ensure labels are within vocab bounds (excluding -100)
        valid_labels_mask = labels != -100
        if valid_labels_mask.any():
            invalid_labels = valid_labels_mask & ((labels >= effective_vocab_size) | (labels < 0))
            if invalid_labels.any():
                logger.warning(
                    f"⚠️ Invalid label IDs detected. Replacing with 0. Sample idx: {idx}"
                )
                labels = torch.where(invalid_labels, torch.zeros_like(labels), labels)

        # Robot target (primary robot index)
        robot_target = ROBOT_MAPPING.get(sample['primary_robot'], 0)

        # Multi-robot target (multi-hot encoding)
        multi_robot_target = torch.zeros(NUM_ROBOTS)
        for robot_name in sample['all_robots']:
            if robot_name in ROBOT_MAPPING:
                multi_robot_target[ROBOT_MAPPING[robot_name]] = 1.0

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'robot_target': torch.tensor(robot_target, dtype=torch.long),
            'multi_robot_target': multi_robot_target,
            'is_multi_robot': torch.tensor(1.0 if sample['type'] == 'multi' else 0.0),
            'difficulty': sample['difficulty'],
            'task_description': sample['task'],
        }


def get_robot_selection_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create enhanced dataloader for robot selection."""
    dataset = EnhancedRobotSelectionDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    logger.info(f"Created {split} dataset with {len(dataset)} samples")

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=None,  # Use default collate
    )

