"""
Behavioral Metrics Analyzer for EmberVLM

Extracts qualitative behavioral signals from model outputs to validate
that the model has truly learned the intended behaviors, not just
optimized numerical metrics.

Implements checks for:
- Caption quality and visual grounding
- Attention distribution analysis
- Reasoning coherence
- Robustness to perturbations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """Analyzes model behavior beyond numerical metrics."""

    def __init__(
        self,
        tokenizer: Any,
        robot_names: List[str] = None,
    ):
        self.tokenizer = tokenizer
        self.robot_names = robot_names or ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"]

        # Visual grounding keywords (objects, actions, spatial relations)
        self.visual_keywords = set([
            'see', 'visible', 'shown', 'image', 'picture', 'photo',
            'person', 'people', 'object', 'building', 'car', 'tree',
            'left', 'right', 'center', 'top', 'bottom', 'front', 'back',
            'color', 'red', 'blue', 'green', 'white', 'black',
            'large', 'small', 'big', 'tall', 'short',
        ])

        # Generic ungrounded phrases to avoid
        self.generic_phrases = set([
            'the image shows',
            'this is a picture of',
            'the photo depicts',
            'i can see',
        ])

    def analyze_caption_quality(
        self,
        captions: List[str],
        images: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Analyze whether generated captions reference visual content.

        Returns metrics for caption quality and visual grounding.
        """
        if not captions:
            return {'visual_grounding_score': 0.0}

        scores = []
        generic_count = 0
        visual_word_counts = []

        for caption in captions:
            caption_lower = caption.lower()

            # Count visual keywords
            visual_words = sum(1 for word in self.visual_keywords if word in caption_lower)
            visual_word_counts.append(visual_words)

            # Check for generic phrases
            is_generic = any(phrase in caption_lower for phrase in self.generic_phrases)
            if is_generic:
                generic_count += 1

            # Score based on specificity
            if visual_words > 0 and not is_generic:
                scores.append(1.0)
            elif visual_words > 0:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return {
            'visual_grounding_score': np.mean(scores) if scores else 0.0,
            'avg_visual_words': np.mean(visual_word_counts) if visual_word_counts else 0.0,
            'generic_caption_ratio': generic_count / len(captions) if captions else 0.0,
        }

    def analyze_attention_to_visual_tokens(
        self,
        attention_weights: torch.Tensor,
        num_visual_tokens: int = 8,
    ) -> Dict[str, float]:
        """
        Analyze whether the model attends to visual tokens.

        Args:
            attention_weights: [B, num_heads, seq_len, seq_len] or averaged
            num_visual_tokens: Number of visual tokens at the beginning

        Returns:
            Metrics for visual attention utilization
        """
        if attention_weights is None or attention_weights.numel() == 0:
            return {'visual_attention_score': 0.0}

        # Average over batch and heads if needed
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)  # [seq_len, seq_len]

        # Get attention TO visual tokens (columns corresponding to visual positions)
        visual_attention = attention_weights[:, :num_visual_tokens].sum(dim=1)

        # Normalize
        total_attention = attention_weights.sum(dim=1)
        visual_attention_ratio = visual_attention / (total_attention + 1e-8)

        # Statistics
        mean_visual_attention = visual_attention_ratio.mean().item()
        min_visual_attention = visual_attention_ratio.min().item()

        # Check for consistent attention (not just first few tokens)
        early_attention = visual_attention_ratio[:num_visual_tokens].mean().item()
        late_attention = visual_attention_ratio[num_visual_tokens:].mean().item()

        return {
            'visual_attention_score': mean_visual_attention,
            'min_visual_attention': min_visual_attention,
            'early_vs_late_ratio': early_attention / (late_attention + 1e-8),
            'visual_token_utilization': (visual_attention_ratio > 0.01).float().mean().item(),
        }

    def detect_mode_collapse(
        self,
        features: torch.Tensor,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Detect mode collapse in learned representations using SVD.

        Args:
            features: [B, dim] or [B, seq_len, dim]
            threshold: Threshold for effective rank ratio

        Returns:
            Metrics indicating mode collapse
        """
        if features.dim() == 3:
            # Pool over sequence
            features = features.mean(dim=1)

        # Center features
        features = features - features.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            U, S, V = torch.svd(features)
        except:
            return {'mode_collapse_detected': False, 'effective_rank_ratio': 1.0}

        # Compute effective rank
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        max_entropy = np.log(len(S))
        effective_rank_ratio = entropy / max_entropy

        # Check if top singular values dominate
        top_k_ratio = S[:10].sum() / S.sum()

        mode_collapse = effective_rank_ratio < threshold

        return {
            'mode_collapse_detected': mode_collapse.item() if isinstance(mode_collapse, torch.Tensor) else mode_collapse,
            'effective_rank_ratio': effective_rank_ratio.item(),
            'top10_singular_value_ratio': top_k_ratio.item(),
            'singular_values': S.cpu().numpy().tolist()[:20],  # First 20 for logging
        }

    def check_adapter_health(
        self,
        adapter_activations: torch.Tensor,
        exploding_threshold: float = 10.0,
        vanishing_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Check if adapter activations are in a healthy range.

        Args:
            adapter_activations: Activations from adapter module
            exploding_threshold: Max mean activation magnitude
            vanishing_threshold: Min mean activation magnitude

        Returns:
            Health metrics for adapter
        """
        mean_activation = adapter_activations.abs().mean().item()
        max_activation = adapter_activations.abs().max().item()
        std_activation = adapter_activations.std().item()

        exploding = mean_activation > exploding_threshold
        vanishing = mean_activation < vanishing_threshold
        healthy = not (exploding or vanishing)

        return {
            'adapter_healthy': healthy,
            'adapter_exploding': exploding,
            'adapter_vanishing': vanishing,
            'adapter_mean_activation': mean_activation,
            'adapter_max_activation': max_activation,
            'adapter_std_activation': std_activation,
        }

    def analyze_instruction_following(
        self,
        instructions: List[str],
        responses: List[str],
    ) -> Dict[str, float]:
        """
        Analyze whether responses actually follow the given instructions.

        Checks for:
        - Relevant keywords from instruction appearing in response
        - Appropriate response length
        - Not just repeating the instruction
        """
        if not instructions or not responses:
            return {'instruction_following_score': 0.0}

        scores = []
        for instruction, response in zip(instructions, responses):
            inst_lower = instruction.lower()
            resp_lower = response.lower()

            # Extract key content words from instruction
            inst_words = set(re.findall(r'\b\w+\b', inst_lower))
            # Remove common words
            inst_words = inst_words - {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'where', 'when'}

            # Check if response addresses key words
            resp_words = set(re.findall(r'\b\w+\b', resp_lower))
            overlap = len(inst_words & resp_words) / (len(inst_words) + 1e-8)

            # Penalize if response is just repeating instruction
            if resp_lower.startswith(inst_lower[:20]):
                overlap *= 0.3

            # Penalize very short responses
            if len(response.split()) < 5:
                overlap *= 0.5

            scores.append(min(overlap, 1.0))

        return {
            'instruction_following_score': np.mean(scores),
            'avg_response_length': np.mean([len(r.split()) for r in responses]),
        }

    def analyze_reasoning_coherence(
        self,
        reasoning_steps: List[List[str]],
        robot_selections: List[int],
    ) -> Dict[str, Any]:
        """
        Analyze coherence and quality of chain-of-thought reasoning.

        Args:
            reasoning_steps: List of reasoning chains (each is a list of steps)
            robot_selections: Final robot selections corresponding to reasoning

        Returns:
            Metrics for reasoning quality
        """
        if not reasoning_steps:
            return {'reasoning_coherence_score': 0.0}

        coherence_scores = []
        justification_scores = []

        for steps, robot_idx in zip(reasoning_steps, robot_selections):
            if not steps:
                coherence_scores.append(0.0)
                justification_scores.append(0.0)
                continue

            # Check if steps build on each other (word overlap between consecutive steps)
            step_coherence = []
            for i in range(len(steps) - 1):
                words_i = set(steps[i].lower().split())
                words_j = set(steps[i+1].lower().split())
                overlap = len(words_i & words_j) / (len(words_i) + len(words_j) + 1e-8)
                step_coherence.append(overlap)

            coherence_scores.append(np.mean(step_coherence) if step_coherence else 0.5)

            # Check if robot name appears in reasoning
            robot_name = self.robot_names[robot_idx] if robot_idx < len(self.robot_names) else ""
            mentions_robot = any(robot_name.lower() in step.lower() for step in steps)
            justification_scores.append(1.0 if mentions_robot else 0.0)

        return {
            'reasoning_coherence_score': np.mean(coherence_scores),
            'reasoning_justifies_selection': np.mean(justification_scores),
            'avg_reasoning_length': np.mean([len(steps) for steps in reasoning_steps]),
        }

    def check_counterfactual_robustness(
        self,
        original_outputs: Dict[str, torch.Tensor],
        perturbed_outputs: Dict[str, torch.Tensor],
        perturbation_type: str = "image",
    ) -> Dict[str, float]:
        """
        Check model robustness to counterfactual scenarios.

        Args:
            original_outputs: Model outputs on original input
            perturbed_outputs: Model outputs on perturbed input
            perturbation_type: Type of perturbation applied

        Returns:
            Robustness metrics
        """
        # Check if robot selection changed appropriately
        original_robot = original_outputs.get('robot_logits', torch.zeros(1, 5)).argmax(dim=-1)
        perturbed_robot = perturbed_outputs.get('robot_logits', torch.zeros(1, 5)).argmax(dim=-1)

        # For counterfactual, we EXPECT the selection to change
        # (e.g., if we change the scenario, the robot should be different)
        selection_changed = (original_robot != perturbed_robot).float().mean().item()

        # Check confidence calibration
        original_confidence = original_outputs.get('robot_confidence', torch.tensor(0.5)).mean().item()
        perturbed_confidence = perturbed_outputs.get('robot_confidence', torch.tensor(0.5)).mean().item()

        # Model should be less confident when perturbed
        confidence_appropriate = perturbed_confidence < original_confidence

        return {
            'counterfactual_selection_sensitivity': selection_changed,
            'counterfactual_confidence_calibrated': float(confidence_appropriate),
            'original_confidence': original_confidence,
            'perturbed_confidence': perturbed_confidence,
        }

    def aggregate_behavioral_checks(
        self,
        stage: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Aggregate all behavioral checks for a given stage into boolean flags.

        This is used by the StageController to make transition decisions.
        """
        if stage == 'stage1':
            return {
                'captions_reference_visual': metrics.get('visual_grounding_score', 0.0) > 0.6,
                'visual_tokens_attended': metrics.get('visual_attention_score', 0.0) > 0.15,
                'no_mode_collapse': not metrics.get('mode_collapse_detected', True),
                'adapter_healthy': metrics.get('adapter_healthy', False),
            }

        elif stage == 'stage2':
            return {
                'instruction_faithful': metrics.get('instruction_following_score', 0.0) > 0.7,
                'image_tokens_attended': metrics.get('visual_attention_score', 0.0) > 0.15,
                'semantic_failures': metrics.get('semantic_error_ratio', 0.5) > 0.5,  # Most errors are semantic, not format
                'attention_relevant': metrics.get('attention_relevance_score', 0.0) > 0.6,
                'stage1_maintained': metrics.get('visual_grounding_score', 0.0) > 0.5,
            }

        elif stage == 'stage3':
            return {
                'reasoning_justifies': metrics.get('reasoning_justifies_selection', 0.0) > 0.7,
                'reasoning_contradicts': metrics.get('reasoning_contradiction_rate', 0.0) > 0.2,  # Should be low
                'reasoning_smooth': metrics.get('reasoning_coherence_score', 0.0) > 0.6,
                'counterfactual_robust': metrics.get('counterfactual_selection_sensitivity', 0.0) > 0.5,
                'paraphrase_robust': metrics.get('paraphrase_consistency', 0.0) > 0.8,
            }

        else:
            return {}


def create_behavioral_analyzer(tokenizer: Any, robot_names: List[str] = None) -> BehavioralAnalyzer:
    """Factory function to create a BehavioralAnalyzer."""
    return BehavioralAnalyzer(tokenizer=tokenizer, robot_names=robot_names)

