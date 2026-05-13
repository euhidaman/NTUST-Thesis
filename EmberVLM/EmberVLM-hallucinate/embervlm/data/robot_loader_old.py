"""
Robot Selection Dataset Loader for EmberVLM

Loads and processes robot fleet selection data with
reasoning chains for training robot selection capabilities.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from PIL import Image
import torchvision.transforms as transforms


# Robot fleet definition
ROBOT_FLEET = [
    {
        "name": "Drone",
        "capabilities": {
            "aerial_survey": True,
            "payload_kg": 2,
            "speed_kmh": 60,
            "battery_hours": 0.5,
            "terrain": ["air"],
            "strengths": "Rapid aerial reconnaissance, hard-to-reach areas",
            "weaknesses": "Limited payload, short battery life, affected by weather"
        }
    },
    {
        "name": "Humanoid",
        "capabilities": {
            "manipulation": True,
            "payload_kg": 20,
            "speed_kmh": 5,
            "battery_hours": 4,
            "terrain": ["indoor", "outdoor_flat"],
            "strengths": "Human-like manipulation, can use human tools",
            "weaknesses": "Slow, unstable on rough terrain"
        }
    },
    {
        "name": "Wheeled",
        "capabilities": {
            "heavy_transport": True,
            "payload_kg": 100,
            "speed_kmh": 30,
            "battery_hours": 8,
            "terrain": ["road", "indoor"],
            "strengths": "High payload, long range, stable platform",
            "weaknesses": "Limited to flat surfaces, cannot navigate stairs"
        }
    },
    {
        "name": "Legged",
        "capabilities": {
            "rough_terrain": True,
            "payload_kg": 30,
            "speed_kmh": 15,
            "battery_hours": 2,
            "terrain": ["rocky", "stairs", "slopes"],
            "strengths": "Versatile terrain navigation, can climb stairs",
            "weaknesses": "Moderate payload, complex control"
        }
    },
    {
        "name": "Underwater",
        "capabilities": {
            "aquatic_ops": True,
            "payload_kg": 10,
            "speed_kmh": 10,
            "battery_hours": 3,
            "terrain": ["water"],
            "strengths": "Underwater operations, diving capability",
            "weaknesses": "Limited to aquatic environments"
        }
    }
]

ROBOT_NAMES = [r["name"] for r in ROBOT_FLEET]
ROBOT_NAME_TO_IDX = {name: idx for idx, name in enumerate(ROBOT_NAMES)}


class RobotSelectionDataset(Dataset):
    """
    Dataset for robot fleet selection training.

    Each sample contains:
    - Image of an incident/task scenario
    - Task description
    - Reasoning chain for robot selection
    - Selected robot(s)
    - Action plan
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        split: str = 'train',
        max_length: int = 768,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        augment_data: bool = True,
        augmentation_factor: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.augment_data = augment_data and split == 'train'
        self.augmentation_factor = augmentation_factor

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        self.samples = self._load_data()

        # Augment if training
        if self.augment_data:
            self.samples = self._augment_samples(self.samples)

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load robot selection data."""
        samples = []

        # Look for JSON files
        json_files = list(self.data_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    sample = self._parse_sample(item)
                    if sample:
                        samples.append(sample)
            elif isinstance(data, dict):
                # Single sample or keyed data
                if 'samples' in data:
                    for item in data['samples']:
                        sample = self._parse_sample(item)
                        if sample:
                            samples.append(sample)
                else:
                    sample = self._parse_sample(data)
                    if sample:
                        samples.append(sample)

        # If no data found, create synthetic samples
        if not samples:
            samples = self._create_synthetic_samples()

        # Split
        random.seed(42)
        random.shuffle(samples)

        split_idx = int(len(samples) * 0.8)
        if self.split == 'train':
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]

        return samples

    def _parse_sample(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single sample."""
        try:
            return {
                'image': item.get('image', ''),
                'instruction': item.get('instruction', item.get('task', '')),
                'robots': item.get('robots', ROBOT_NAMES),
                'capabilities': item.get('capabilities', {}),
                'reasoning_chain': item.get('reasoning_chain', []),
                'selected_robot': item.get('selected_robot', item.get('robot', '')),
                'action_plan': item.get('action_plan', ''),
                'confidence_score': item.get('confidence_score', 0.9),
            }
        except Exception:
            return None

    def _create_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Create synthetic training samples."""
        scenarios = [
            {
                "task": "Survey damage from aerial perspective after earthquake",
                "robot": "Drone",
                "reasoning": [
                    "Identify key requirements: aerial view, quick assessment, wide coverage",
                    "Evaluate terrain: urban area with collapsed buildings",
                    "Consider safety: ground access may be dangerous",
                    "Match capabilities: Drone provides aerial survey capability"
                ],
                "action": "Deploy drone for systematic aerial grid survey of affected area"
            },
            {
                "task": "Navigate through rubble to search for survivors",
                "robot": "Legged",
                "reasoning": [
                    "Identify key requirements: traverse uneven terrain, navigate obstacles",
                    "Evaluate terrain: collapsed structures, debris, unstable surfaces",
                    "Consider safety: wheeled robots cannot access this terrain",
                    "Match capabilities: Legged robot can climb over obstacles"
                ],
                "action": "Deploy legged robot to systematically search rubble piles"
            },
            {
                "task": "Transport heavy medical supplies to disaster zone",
                "robot": "Wheeled",
                "reasoning": [
                    "Identify key requirements: heavy payload, long range transport",
                    "Evaluate terrain: roads available but congested",
                    "Consider efficiency: need maximum cargo capacity",
                    "Match capabilities: Wheeled robot has 100kg payload, 8hr battery"
                ],
                "action": "Load wheeled robot with supplies and navigate via clear routes"
            },
            {
                "task": "Inspect underwater pipeline for damage after flood",
                "robot": "Underwater",
                "reasoning": [
                    "Identify key requirements: underwater operation, visual inspection",
                    "Evaluate environment: flooded area with submerged infrastructure",
                    "Consider access: only aquatic robot can reach target",
                    "Match capabilities: Underwater robot designed for aquatic operations"
                ],
                "action": "Deploy underwater robot to follow pipeline route and document damage"
            },
            {
                "task": "Open doors and manipulate tools in damaged building",
                "robot": "Humanoid",
                "reasoning": [
                    "Identify key requirements: fine manipulation, tool use, door operation",
                    "Evaluate environment: indoor building with human-scale obstacles",
                    "Consider capabilities: requires human-like manipulation",
                    "Match capabilities: Humanoid robot can operate human tools and doors"
                ],
                "action": "Deploy humanoid to navigate building and perform manipulation tasks"
            },
        ]

        samples = []
        for scenario in scenarios:
            samples.append({
                'image': '',  # Will use placeholder
                'instruction': scenario['task'],
                'robots': ROBOT_NAMES,
                'capabilities': {r['name']: r['capabilities'] for r in ROBOT_FLEET},
                'reasoning_chain': scenario['reasoning'],
                'selected_robot': scenario['robot'],
                'action_plan': scenario['action'],
                'confidence_score': 0.9,
            })

        return samples

    def _augment_samples(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Augment samples with variations."""
        augmented = []

        # Task variations
        task_synonyms = {
            "survey": ["inspect", "examine", "assess", "evaluate"],
            "navigate": ["traverse", "move through", "cross", "travel across"],
            "transport": ["carry", "deliver", "move", "bring"],
            "search": ["look for", "find", "locate", "seek"],
        }

        for sample in samples:
            augmented.append(sample)

            for _ in range(self.augmentation_factor - 1):
                new_sample = sample.copy()

                # Vary instruction
                instruction = new_sample['instruction']
                for word, synonyms in task_synonyms.items():
                    if word in instruction.lower():
                        replacement = random.choice(synonyms)
                        instruction = instruction.lower().replace(word, replacement)
                        instruction = instruction[0].upper() + instruction[1:]
                        break

                new_sample['instruction'] = instruction
                augmented.append(new_sample)

        return augmented

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image or create placeholder."""
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                return self.transform(image)
            except Exception:
                pass

        # If path doesn't exist, try relative paths
        if image_path:
            possible_paths = [
                self.data_dir / 'images' / image_path,
                self.data_dir / image_path,
            ]
            for path in possible_paths:
                if path.exists():
                    try:
                        image = Image.open(path).convert('RGB')
                        return self.transform(image)
                    except Exception:
                        pass

        # Return placeholder (noise image for training)
        return torch.randn(3, self.image_size, self.image_size) * 0.1

    def _create_prompt(self, sample: Dict[str, Any]) -> str:
        """Create training prompt with reasoning chain."""
        instruction = sample['instruction']

        # Add robot fleet info
        fleet_info = "Available robots: " + ", ".join(ROBOT_NAMES)

        # Reasoning chain
        reasoning_text = ""
        if sample['reasoning_chain']:
            steps = "\n".join([
                f"Step {i+1}: {step}"
                for i, step in enumerate(sample['reasoning_chain'])
            ])
            reasoning_text = f"\n<|reasoning_start|>\n{steps}\n<|reasoning_end|>\n"

        # Robot selection
        robot_text = f"<|robot_selection|>{sample['selected_robot']}"

        # Action plan
        action_text = ""
        if sample['action_plan']:
            action_text = f"\n<|action_plan|>{sample['action_plan']}"

        full_text = (
            f"Task: {instruction}\n"
            f"{fleet_info}\n"
            f"Analyze this incident scene and select the most appropriate robot from the fleet."
            f"{reasoning_text}"
            f"{robot_text}"
            f"{action_text}"
        )

        return full_text

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        pixel_values = self._load_image(sample['image'])

        # Create prompt
        text = self._create_prompt(sample)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Robot target
        robot_name = sample['selected_robot']
        robot_target = ROBOT_NAME_TO_IDX.get(robot_name, 0)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'robot_target': torch.tensor(robot_target, dtype=torch.long),
            'confidence': torch.tensor(sample['confidence_score'], dtype=torch.float),
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
    """Create dataloader for robot selection data."""
    dataset = RobotSelectionDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

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
    )


def create_robot_selection_dataset(
    output_dir: str,
    num_samples: int = 100,
):
    """Create a sample robot selection dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Create sample data
    dataset = RobotSelectionDataset(
        data_dir=output_dir,
        tokenizer=None,
        split='train',
        augment_data=False,
    )

    samples = dataset._create_synthetic_samples()

    # Save to JSON
    output_file = Path(output_dir) / 'multi-robot-selection.json'
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Created {len(samples)} samples at {output_file}")
    return str(output_file)

