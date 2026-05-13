"""
Data Augmentation for EmberVLM

Image and text augmentation strategies for robust training.
"""

import random
from typing import Optional, List, Dict, Any, Callable, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms


class ImageAugmentation:
    """
    Image augmentation strategies for vision-language training.
    """

    def __init__(
        self,
        image_size: int = 224,
        use_color_jitter: bool = True,
        use_random_crop: bool = True,
        use_horizontal_flip: bool = True,
        use_rotation: bool = False,
        use_gaussian_blur: bool = False,
    ):
        self.image_size = image_size

        # Build augmentation pipeline
        aug_list = []

        if use_random_crop:
            aug_list.append(transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ))
        else:
            aug_list.append(transforms.Resize((image_size, image_size)))

        if use_horizontal_flip:
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if use_rotation:
            aug_list.append(transforms.RandomRotation(15))

        if use_color_jitter:
            aug_list.append(transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ))

        if use_gaussian_blur:
            aug_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

        aug_list.append(transforms.ToTensor())
        aug_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

        self.transform = transforms.Compose(aug_list)

    def __call__(self, image):
        return self.transform(image)


class TextAugmentation:
    """
    Text augmentation strategies for instruction tuning.
    """

    def __init__(
        self,
        use_synonym_replacement: bool = True,
        use_random_insertion: bool = False,
        use_random_deletion: bool = False,
        use_random_swap: bool = False,
        aug_probability: float = 0.3,
    ):
        self.use_synonym_replacement = use_synonym_replacement
        self.use_random_insertion = use_random_insertion
        self.use_random_deletion = use_random_deletion
        self.use_random_swap = use_random_swap
        self.aug_probability = aug_probability

        # Common synonyms for task-related words
        self.synonyms = {
            'analyze': ['examine', 'assess', 'evaluate', 'study', 'investigate'],
            'select': ['choose', 'pick', 'identify', 'determine', 'designate'],
            'incident': ['scene', 'situation', 'event', 'scenario', 'occurrence'],
            'robot': ['unit', 'system', 'machine', 'device'],
            'deploy': ['send', 'dispatch', 'launch', 'activate'],
            'survey': ['scan', 'examine', 'inspect', 'assess'],
            'navigate': ['traverse', 'cross', 'move through', 'travel'],
            'transport': ['carry', 'move', 'deliver', 'bring'],
            'search': ['look for', 'find', 'locate', 'seek'],
            'inspect': ['examine', 'check', 'review', 'assess'],
            'damage': ['harm', 'destruction', 'impact', 'injury'],
            'rescue': ['save', 'recover', 'retrieve', 'aid'],
            'emergency': ['crisis', 'urgent situation', 'disaster'],
            'area': ['zone', 'region', 'sector', 'location'],
        }

    def __call__(self, text: str) -> str:
        """Apply text augmentation."""
        if random.random() > self.aug_probability:
            return text

        augmented = text

        if self.use_synonym_replacement:
            augmented = self._synonym_replacement(augmented)

        if self.use_random_deletion:
            augmented = self._random_deletion(augmented)

        if self.use_random_swap:
            augmented = self._random_swap(augmented)

        return augmented

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()

        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')

            if word_lower in self.synonyms and random.random() < 0.3:
                replacement = random.choice(self.synonyms[word_lower])

                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()

                # Preserve punctuation
                if word[-1] in '.,!?':
                    replacement += word[-1]

                words[i] = replacement

        return ' '.join(words)

    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words."""
        words = text.split()

        if len(words) <= 3:
            return text

        kept = [w for w in words if random.random() > p]

        return ' '.join(kept) if kept else text

    def _random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap adjacent words."""
        words = text.split()

        if len(words) < 2:
            return text

        for _ in range(n):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return ' '.join(words)


class ReasoningAugmentation:
    """
    Augmentation for reasoning chains.
    """

    def __init__(self):
        self.step_variations = {
            'identify': ['recognize', 'determine', 'establish', 'pinpoint'],
            'analyze': ['examine', 'evaluate', 'assess', 'study'],
            'consider': ['account for', 'factor in', 'take into account', 'weigh'],
            'select': ['choose', 'opt for', 'determine', 'pick'],
            'match': ['align', 'pair', 'correlate', 'connect'],
        }

    def augment_chain(self, chain: List[str]) -> List[str]:
        """Augment reasoning chain."""
        augmented = []

        for step in chain:
            # Sometimes vary the step wording
            if random.random() < 0.3:
                step = self._vary_step(step)

            augmented.append(step)

        # Sometimes add elaboration
        if random.random() < 0.2 and len(augmented) > 0:
            idx = random.randint(0, len(augmented) - 1)
            augmented[idx] = self._add_elaboration(augmented[idx])

        return augmented

    def _vary_step(self, step: str) -> str:
        """Vary wording of a reasoning step."""
        for word, alternatives in self.step_variations.items():
            if word in step.lower():
                replacement = random.choice(alternatives)
                step = step.replace(word, replacement, 1)
                step = step.replace(word.capitalize(), replacement.capitalize(), 1)
                break

        return step

    def _add_elaboration(self, step: str) -> str:
        """Add elaboration to a step."""
        elaborations = [
            " This is crucial for effective response.",
            " This directly impacts robot performance.",
            " Safety considerations support this.",
            " Efficiency gains are significant here.",
        ]

        return step + random.choice(elaborations)


class MixupAugmentation:
    """
    Mixup augmentation for images and labels.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.

        Args:
            images1: First batch of images [B, C, H, W]
            images2: Second batch of images [B, C, H, W]
            labels1: First batch of labels [B] or [B, num_classes]
            labels2: Second batch of labels [B] or [B, num_classes]

        Returns:
            Mixed images and labels
        """
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1

        mixed_images = lam * images1 + (1 - lam) * images2
        mixed_labels = lam * labels1 + (1 - lam) * labels2

        return mixed_images, mixed_labels


class CutoutAugmentation:
    """
    Cutout (random erasing) augmentation for images.
    """

    def __init__(
        self,
        probability: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        self.transform = transforms.RandomErasing(
            p=probability,
            scale=scale,
            ratio=ratio,
            value='random',
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


def create_train_augmentation(
    image_size: int = 224,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Create augmentation transforms for training.

    Args:
        image_size: Target image size
        is_training: Whether in training mode

    Returns:
        Dictionary with augmentation transforms
    """
    if is_training:
        image_aug = ImageAugmentation(
            image_size=image_size,
            use_color_jitter=True,
            use_random_crop=True,
            use_horizontal_flip=True,
        )
        text_aug = TextAugmentation(
            use_synonym_replacement=True,
            aug_probability=0.3,
        )
        reasoning_aug = ReasoningAugmentation()
    else:
        # Minimal augmentation for validation
        image_aug = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        text_aug = None
        reasoning_aug = None

    return {
        'image': image_aug,
        'text': text_aug,
        'reasoning': reasoning_aug,
    }

