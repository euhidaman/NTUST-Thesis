"""
Reward Functions for Stage 4 Reasoning Training

Inspired by tiny-r1 (DeepSeek-R1 style), these reward functions encourage:
1. Proper XML format (<reasoning>...</reasoning><answer>...</answer>)
2. Correct robot selection
3. Step-by-step reasoning quality
"""

import re
from typing import List, Dict, Any, Optional
import torch


# Robot type mapping for validation
VALID_ROBOTS = {
    "drone": "Drone",
    "humanoid": "Humanoid",
    "robot with wheels": "Robot with Wheels",
    "wheeled robot": "Robot with Wheels",
    "robot with legs": "Robot with Legs",
    "legged robot": "Robot with Legs",
    "underwater robot": "Underwater Robot",
}


def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted response."""
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


def normalize_robot_name(name: str) -> str:
    """Normalize robot name to standard format."""
    name_lower = name.lower().strip()
    return VALID_ROBOTS.get(name_lower, name.strip())


def count_xml_tags(text: str) -> float:
    """
    Count and score XML tags in the text (tiny-r1 style).
    Returns a score between 0 and 1.
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

    # Check for proper newlines (optional, adds polish)
    if "<reasoning>\n" in text:
        score += 0.05
    if "\n</reasoning>" in text:
        score += 0.05
    if "\n<answer>" in text or "</reasoning>\n<answer>" in text:
        score += 0.05
    if "\n</answer>" in text:
        score += 0.05

    # Penalize content after closing answer tag
    if "</answer>" in text:
        after_answer = text.split("</answer>")[-1]
        penalty = len(after_answer.strip()) * 0.001
        score -= min(penalty, 0.1)  # Cap penalty

    return max(0.0, min(1.0, score))


def correctness_reward(completions: List[str], targets: List[str]) -> List[float]:
    """
    Reward function that checks if the robot selection is correct.
    Returns 2.0 for correct, 0.0 for incorrect.
    """
    rewards = []
    for completion, target in zip(completions, targets):
        extracted = extract_xml_answer(completion)
        extracted_normalized = normalize_robot_name(extracted)
        target_normalized = normalize_robot_name(target)

        if extracted_normalized.lower() == target_normalized.lower():
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards


def valid_robot_reward(completions: List[str]) -> List[float]:
    """
    Reward function that checks if the answer is a valid robot type.
    Returns 0.5 for valid robot, 0.0 for invalid.
    """
    valid_robot_names = set([
        "drone", "humanoid", "robot with wheels", "robot with legs",
        "underwater robot", "wheeled robot", "legged robot"
    ])

    rewards = []
    for completion in completions:
        extracted = extract_xml_answer(completion).lower()
        if extracted in valid_robot_names or normalize_robot_name(extracted).lower() in valid_robot_names:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


def strict_format_reward(completions: List[str]) -> List[float]:
    """
    Reward function that checks for strict XML format.
    Pattern: <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$"

    rewards = []
    for completion in completions:
        if re.match(pattern, completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


def soft_format_reward(completions: List[str]) -> List[float]:
    """
    Reward function that checks for soft XML format (more lenient).
    Pattern: <reasoning>...</reasoning>...<answer>...</answer>
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

    rewards = []
    for completion in completions:
        if re.search(pattern, completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


def xml_count_reward(completions: List[str]) -> List[float]:
    """
    Reward function based on XML tag counting (tiny-r1 style).
    """
    return [count_xml_tags(c) for c in completions]


def reasoning_step_reward(completions: List[str]) -> List[float]:
    """
    Reward function that checks for step-by-step reasoning.
    Looks for "Step 1:", "Step 2:", etc. in the reasoning section.
    """
    rewards = []
    for completion in completions:
        reasoning = extract_xml_reasoning(completion)

        step_count = 0
        for i in range(1, 6):  # Check for Steps 1-5
            if f"Step {i}" in reasoning or f"step {i}" in reasoning.lower():
                step_count += 1

        # Reward based on number of steps found (normalized to 0-0.5)
        reward = min(step_count / 4.0, 1.0) * 0.5
        rewards.append(reward)

    return rewards


def reasoning_length_reward(completions: List[str], min_length: int = 100, max_length: int = 500) -> List[float]:
    """
    Reward function that encourages appropriate reasoning length.
    Not too short (low effort) or too long (verbose).
    """
    rewards = []
    for completion in completions:
        reasoning = extract_xml_reasoning(completion)
        length = len(reasoning)

        if length < min_length:
            # Too short - linear penalty
            reward = (length / min_length) * 0.3
        elif length > max_length:
            # Too long - gentle penalty
            excess = length - max_length
            reward = max(0.3 - (excess / 1000) * 0.1, 0.1)
        else:
            # Good length
            reward = 0.3

        rewards.append(reward)

    return rewards


def compute_total_reward(
    completions: List[str],
    targets: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[float]:
    """
    Compute total reward from all reward functions.

    Default weights (sum to reasonable total):
    - correctness: 2.0 (if targets provided)
    - valid_robot: 0.5
    - strict_format: 0.5
    - soft_format: 0.5
    - xml_count: 0.5
    - reasoning_steps: 0.5
    - reasoning_length: 0.3

    Total possible: ~5.0 (with correctness) or ~2.8 (without)
    """
    if weights is None:
        weights = {
            "correctness": 1.0,
            "valid_robot": 1.0,
            "strict_format": 1.0,
            "soft_format": 1.0,
            "xml_count": 1.0,
            "reasoning_steps": 1.0,
            "reasoning_length": 1.0,
        }

    total_rewards = [0.0] * len(completions)

    # Correctness (if targets provided)
    if targets is not None and weights.get("correctness", 0) > 0:
        correctness = correctness_reward(completions, targets)
        for i, r in enumerate(correctness):
            total_rewards[i] += r * weights["correctness"]

    # Valid robot
    if weights.get("valid_robot", 0) > 0:
        valid_robot = valid_robot_reward(completions)
        for i, r in enumerate(valid_robot):
            total_rewards[i] += r * weights["valid_robot"]

    # Format rewards
    if weights.get("strict_format", 0) > 0:
        strict = strict_format_reward(completions)
        for i, r in enumerate(strict):
            total_rewards[i] += r * weights["strict_format"]

    if weights.get("soft_format", 0) > 0:
        soft = soft_format_reward(completions)
        for i, r in enumerate(soft):
            total_rewards[i] += r * weights["soft_format"]

    if weights.get("xml_count", 0) > 0:
        xml_count = xml_count_reward(completions)
        for i, r in enumerate(xml_count):
            total_rewards[i] += r * weights["xml_count"]

    # Reasoning quality
    if weights.get("reasoning_steps", 0) > 0:
        steps = reasoning_step_reward(completions)
        for i, r in enumerate(steps):
            total_rewards[i] += r * weights["reasoning_steps"]

    if weights.get("reasoning_length", 0) > 0:
        length = reasoning_length_reward(completions)
        for i, r in enumerate(length):
            total_rewards[i] += r * weights["reasoning_length"]

    return total_rewards


class ReasoningRewardModel:
    """
    Reward model for reasoning training.
    Can be used as a callable for GRPO-style training.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "correctness": 1.0,
            "valid_robot": 1.0,
            "strict_format": 0.8,
            "soft_format": 0.5,
            "xml_count": 0.5,
            "reasoning_steps": 0.7,
            "reasoning_length": 0.3,
        }

    def __call__(
        self,
        completions: List[str],
        targets: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """Compute rewards for completions."""
        return compute_total_reward(completions, targets, self.weights)

    def get_detailed_rewards(
        self,
        completions: List[str],
        targets: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Get detailed breakdown of all reward components."""
        detailed = {
            "valid_robot": valid_robot_reward(completions),
            "strict_format": strict_format_reward(completions),
            "soft_format": soft_format_reward(completions),
            "xml_count": xml_count_reward(completions),
            "reasoning_steps": reasoning_step_reward(completions),
            "reasoning_length": reasoning_length_reward(completions),
        }

        if targets is not None:
            detailed["correctness"] = correctness_reward(completions, targets)

        detailed["total"] = self(completions, targets)

        return detailed


# Loss function that incorporates reward signals
class ReasoningRewardLoss(torch.nn.Module):
    """
    Loss function that incorporates reasoning reward signals.
    Can be combined with standard language model loss.
    """

    def __init__(self, reward_model: Optional[ReasoningRewardModel] = None):
        super().__init__()
        self.reward_model = reward_model or ReasoningRewardModel()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        generated_texts: Optional[List[str]] = None,
        target_robots: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute loss with reward signal.

        If generated_texts are provided, incorporates reward as a weighting factor.
        Otherwise, just computes standard cross-entropy.
        """
        # Standard cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # If we have generated texts, incorporate reward signal
        if generated_texts is not None:
            rewards = self.reward_model(generated_texts, target_robots)
            # Convert to tensor
            reward_tensor = torch.tensor(rewards, device=logits.device, dtype=logits.dtype)
            # Normalize rewards to 0-1 range
            max_reward = max(rewards) if max(rewards) > 0 else 1.0
            normalized_rewards = reward_tensor / max_reward
            # Use reward as inverse weight (lower reward = higher loss contribution)
            reward_weight = 2.0 - normalized_rewards.mean()
            ce_loss = ce_loss * reward_weight

        return ce_loss

