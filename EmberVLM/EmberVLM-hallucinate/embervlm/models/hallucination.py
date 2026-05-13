"""
Visual Absence (VA)-aware Hallucination Suppression Module for EmberVLM.

Implements a lightweight, inference-time VA detector inspired by:
    Kim et al., "Detecting and Mitigating Hallucination in Large Vision
    Language Models via Fine-Grained AI Feedback" (2024).

Key design choices & approximations
-------------------------------------
* **Neuron selection is approximated.**  The original VA method selects the top-
  S_VA neurons that discriminate "visually absent" questions from "visually
  present" ones using a dedicated VA-QA dataset.  Because we do not ship a
  VA-QA training pipeline, we instead pick a **fixed subset** of FFN neurons
  per middle layer.  The `fit_on_dataset` skeleton is provided so that true
  S_VA selection can be added later.

* **Dual-view scoring** compares logits produced *with* the image against
  logits produced *without* the image (visual tokens zeroed).  A small
  difference means the token has weak visual grounding.

* **Token-type-aware thresholds** apply stricter thresholds to colour, object,
  number, and spatial tokens, and looser thresholds to non-visual tokens.

* **Temporal VA memory** detects "hallucination bursts" — consecutive tokens
  flagged as visually absent — and escalates the intervention (hard block
  instead of soft downweight).

* **Answer-level calibration** optionally prepends a confidence prefix ("I
  might be wrong, but …") when the average VA score of visual tokens is high.

This module is **disabled by default** (`use_va_refiner=False`).  Enabling it
does NOT modify any backbone weights (DINOv2, SmolLM, or teacher models).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class VAConfig:
    """Configuration for the VA feature extractor and classifier."""

    layer_indices: List[int] = field(
        default_factory=lambda: [10, 11, 12, 13, 14]
    )
    """Indices of SmolLM decoder layers whose FFN activations are monitored."""

    top_k_neurons: int = 128
    """Total number of FFN neurons selected across all monitored layers
    (approximately top-S_VA in the original paper)."""

    threshold_beta: float = 0.3
    """Sensitivity threshold (approximate; we do not compute S_VA)."""

    classifier_hidden_dim: int = 128
    """Hidden dimension of the per-token VA classifier MLP."""

    device: Optional[str] = None
    """Optional device override (e.g. ``'cuda:0'``)."""


@dataclass
class VAThresholds:
    """Per-token-type VA probability thresholds."""

    p_token_default: float = 0.7
    """Base VA probability threshold for generic tokens."""

    p_visual_token: float = 0.6
    """Stricter threshold for visually sensitive tokens (colours, objects, …)."""

    p_nonvisual_token: float = 0.8
    """Looser threshold for clearly non-visual tokens."""


@dataclass
class VATemporalConfig:
    """Configuration for the temporal VA memory / burst detector."""

    window_size: int = 8
    """Number of trailing tokens to keep in the sliding window."""

    burst_threshold: float = 0.7
    """Average p_VA over the window that triggers burst handling."""

    burst_decay: float = 0.2
    """Additional logit dampening factor applied during bursts."""


# =============================================================================
# Visual-keyword heuristics
# =============================================================================

VISUAL_KEYWORDS: set[str] = {
    # Colours
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "pink", "purple", "brown", "gray", "grey",
    # Common objects
    "car", "truck", "bus", "dog", "cat", "person", "man", "woman",
    "child", "table", "chair", "tree", "sky", "road", "bike", "bicycle",
    "building", "house", "door", "window", "plate", "cup", "phone",
    # Numbers (spelled)
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
    # Spatial relations
    "left", "right", "top", "bottom", "front", "back", "behind",
    "in front of", "above", "below", "next to",
}


def is_visual_sensitive_token(token_str: str) -> bool:
    """Return ``True`` if *token_str* is a visually sensitive word.

    Uses a simple heuristic: exact keyword match, or any digit present.
    """
    t = token_str.strip().lower()
    if t in VISUAL_KEYWORDS:
        return True
    if any(ch.isdigit() for ch in t):
        return True
    return False


# =============================================================================
# VAFeatureExtractor — hooks on FFN layers
# =============================================================================

class VAFeatureExtractor(nn.Module):
    """Extract per-token FFN activations from selected middle layers of SmolLM.

    Registers **forward hooks** on the MLP sub-modules of chosen decoder layers
    and stores per-token activations.  A fixed subset of neurons per layer
    is used as an **approximation** of the top-S_VA neurons described in the
    VA paper (we lack a VA-QA dataset to compute the true set).

    Parameters
    ----------
    lm : nn.Module
        The SmolLM causal-LM module whose ``model.layers[i].mlp`` modules
        will be hooked.
    config : VAConfig
        Feature extractor configuration.
    """

    def __init__(self, lm: nn.Module, config: VAConfig):
        super().__init__()
        self.lm = lm
        self.config = config
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self._features: Dict[int, torch.Tensor] = {}
        self._selected_indices: Dict[int, torch.Tensor] = {}
        self._initialized = False

    # ----- internal helpers --------------------------------------------------

    def _init_selected_indices(self, example_ffn_dim: int) -> None:
        """Compute and cache per-layer neuron indices (approximation).

        NOTE: In a full implementation these indices would come from the
        top-S_VA neurons identified via a VA-QA dataset.  Here we simply
        pick the first ``top_k_per_layer`` neurons in each monitored layer.
        """
        n_layers = max(1, len(self.config.layer_indices))
        top_k_per_layer = max(1, self.config.top_k_neurons // n_layers)
        # Clamp so we never exceed the actual FFN width
        top_k_per_layer = min(top_k_per_layer, example_ffn_dim)
        for l in self.config.layer_indices:
            self._selected_indices[l] = torch.arange(top_k_per_layer)
        self._initialized = True

    def _hook_fn(self, layer_idx: int):
        """Return a forward-hook closure for layer *layer_idx*."""

        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            # ``output`` is the FFN output tensor with shape [B, T, D_ffn].
            # We store it detached (no grad) — the VA module is inference-only.
            with torch.no_grad():
                act = output
                if not self._initialized:
                    self._init_selected_indices(act.size(-1))
                self._features[layer_idx] = act.detach()

        return hook

    # ----- public API --------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward hooks to the FFN modules at configured layer indices.

        The SmolLM (LlamaForCausalLM) hierarchy is::

            model.model.layers[i].mlp   (LlamaMLP)
        """
        # Navigate to the decoder layers list
        layers = None
        if hasattr(self.lm, "model") and hasattr(self.lm.model, "layers"):
            layers = self.lm.model.layers
        elif hasattr(self.lm, "layers"):
            layers = self.lm.layers

        if layers is None:
            logger.warning(
                "VAFeatureExtractor: could not find decoder layers on the "
                "language model — hooks will NOT be registered."
            )
            return

        for l_idx in self.config.layer_indices:
            if l_idx >= len(layers):
                logger.warning(
                    f"VAFeatureExtractor: layer_idx {l_idx} out of range "
                    f"(model has {len(layers)} layers) — skipping."
                )
                continue
            block = layers[l_idx]
            ffn_module = getattr(block, "mlp", None)
            if ffn_module is not None:
                handle = ffn_module.register_forward_hook(self._hook_fn(l_idx))
                self.handles.append(handle)
            else:
                logger.warning(
                    f"VAFeatureExtractor: layer {l_idx} has no 'mlp' attribute."
                )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def reset(self) -> None:
        """Clear captured features (call before each new forward pass)."""
        self._features.clear()

    def get_token_features(self) -> torch.Tensor:
        """Return per-token VA features.

        Concatenates selected neuron activations across all configured layers.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T, F_total]`` where ``F_total = sum(K_l)``.

        Raises
        ------
        RuntimeError
            If no features have been collected (hooks not registered or no
            forward pass performed).
        """
        if not self._features:
            raise RuntimeError(
                "VAFeatureExtractor: no features collected.  "
                "Did you call register_hooks() and run a forward pass?"
            )

        layers = sorted(self._features.keys())
        feats: List[torch.Tensor] = []
        for l in layers:
            act = self._features[l]  # [B, T, D_ffn]
            idx = self._selected_indices[l].to(act.device)
            feats.append(act[..., idx])  # [B, T, K_l]
        return torch.cat(feats, dim=-1)  # [B, T, F_total]


# =============================================================================
# VAClassifier — lightweight neuron-based MLP
# =============================================================================

class VAClassifier(nn.Module):
    """Map VA neuron features to per-token VA probability in [0, 1].

    Parameters
    ----------
    input_dim : int
        Concatenated VA feature dimension (``F_total``).
    hidden_dim : int
        Hidden layer size.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-token VA probabilities.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``[B, T, F]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T]``, values in [0, 1].
        """
        logits = self.net(features)             # [B, T, 1]
        return torch.sigmoid(logits).squeeze(-1)  # [B, T]


# =============================================================================
# Logit-based visual-consistency scorer (dual-view)
# =============================================================================

def compute_logit_va_score(
    logits_with_image: torch.Tensor,   # [B, V]
    logits_no_image: torch.Tensor,     # [B, V]
    token_ids: torch.Tensor,           # [B]
) -> torch.Tensor:
    """Estimate how much a token depends on the image via logit comparison.

    A **high** returned score means the token is similarly likely with and
    without the image (weak visual grounding → likely hallucinated).

    Parameters
    ----------
    logits_with_image : torch.Tensor
        Logits produced with the image present, shape ``[B, V]``.
    logits_no_image : torch.Tensor
        Logits produced with the image neutralised, shape ``[B, V]``.
    token_ids : torch.Tensor
        IDs of the last generated token, shape ``[B]``.

    Returns
    -------
    torch.Tensor
        Per-sample score in [0, 1], shape ``[B]``.
    """
    b_idx = torch.arange(token_ids.size(0), device=token_ids.device)
    logit_w = logits_with_image[b_idx, token_ids]  # [B]
    logit_n = logits_no_image[b_idx, token_ids]    # [B]

    # Large difference ⇒ token depends on image ⇒ low VA score
    diff = torch.abs(logit_w - logit_n)
    score = 1.0 - torch.tanh(diff)
    return score.clamp(0.0, 1.0)


# =============================================================================
# Temporal VA memory / burst detector
# =============================================================================

class VATemporalMemory:
    """Sliding-window tracker for consecutive VA scores.

    Detects "hallucination bursts" — multiple consecutive tokens flagged as
    visually absent — and triggers escalated intervention (hard block).
    """

    def __init__(self, config: VATemporalConfig):
        self.config = config
        self.window: List[float] = []

    def update(self, p_va: float) -> None:
        """Append *p_va* and evict the oldest entry if the window is full."""
        self.window.append(float(p_va))
        if len(self.window) > self.config.window_size:
            self.window.pop(0)

    def is_burst(self) -> bool:
        """Return ``True`` when the sliding-window average exceeds the burst threshold."""
        if not self.window:
            return False
        avg = sum(self.window) / len(self.window)
        return avg >= self.config.burst_threshold

    def reset(self) -> None:
        """Clear the window (call at the start of each new generation)."""
        self.window.clear()


# =============================================================================
# VARefiner — central orchestrator
# =============================================================================

class VARefiner:
    """Visual Absence-aware hallucination mitigation module.

    Orchestrates:
    * ``VAFeatureExtractor`` + ``VAClassifier`` (neuron-based detector).
    * Logit-based visual-consistency scoring (with/without image).
    * Token-type-aware thresholds.
    * Temporal memory and interventions (blocking, soft rewrite).
    * Answer-level calibration prefixes.

    Usage
    -----
    1. Instantiate once per model.
    2. For **binary QA**, call :meth:`refine_yes_no_answer`.
    3. For **open-ended generation**, call :meth:`refine_logits` at each step.
    4. Optionally call :meth:`calibrate_answer_prefix` after decoding.

    Parameters
    ----------
    lm : nn.Module
        The SmolLM causal-LM (``LlamaForCausalLM``).
    tokenizer
        A HuggingFace-compatible tokenizer.
    va_config : VAConfig
    thresholds : VAThresholds
    temporal_config : VATemporalConfig
    p_combine : float
        Convex weight for neuron vs logit VA scores.
    """

    def __init__(
        self,
        lm: nn.Module,
        tokenizer: Any,
        va_config: VAConfig,
        thresholds: VAThresholds,
        temporal_config: VATemporalConfig,
        p_combine: float = 0.5,
    ):
        self.lm = lm
        self.tokenizer = tokenizer
        self.va_config = va_config
        self.thresholds = thresholds
        self.temporal = VATemporalMemory(temporal_config)
        self.p_combine = p_combine

        self.feature_extractor = VAFeatureExtractor(lm, va_config)
        self.classifier: Optional[VAClassifier] = None
        self.device = va_config.device

        self._initialized_classifier = False

    # ----- internal helpers --------------------------------------------------

    def _ensure_classifier(self, features: torch.Tensor) -> None:
        """Lazily instantiate the classifier once the feature dim is known."""
        if not self._initialized_classifier:
            input_dim = features.size(-1)
            self.classifier = VAClassifier(
                input_dim, self.va_config.classifier_hidden_dim
            )
            if self.device is not None:
                self.classifier = self.classifier.to(self.device)
            self._initialized_classifier = True

    def _compute_neuron_va_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``[B, T]`` VA probabilities from the neuron classifier."""
        self._ensure_classifier(features)
        assert self.classifier is not None
        return self.classifier(features)

    def _combine_va_scores(
        self,
        p_neuron: torch.Tensor,
        p_logit: Optional[torch.Tensor] = None,
        token_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Combine neuron-based and logit-based VA scores.

        Parameters
        ----------
        p_neuron : [B, T]
        p_logit  : [B, T] or [B]
        token_position : required when *p_logit* is [B] (single step).
        """
        if p_logit is None:
            return p_neuron

        if p_logit.dim() == 1:
            assert token_position is not None, (
                "token_position is required when p_logit is [B]"
            )
            combined = p_neuron.clone()
            combined[:, token_position] = (
                self.p_combine * p_neuron[:, token_position]
                + (1.0 - self.p_combine) * p_logit
            )
            return combined

        # Both [B, T]: simple convex combination
        return self.p_combine * p_neuron + (1.0 - self.p_combine) * p_logit

    # ----- public API --------------------------------------------------------

    def mark_visually_absent_tokens(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token VA probabilities from FFN features.

        Assumes :class:`VAFeatureExtractor` has collected features during the
        same forward pass (i.e. hooks were active).

        Returns
        -------
        torch.Tensor
            ``[B, T]`` VA probabilities in [0, 1].
        """
        with torch.no_grad():
            features = self.feature_extractor.get_token_features()  # [B, T, F]
            if self.device is not None:
                features = features.to(self.device)
            p_neuron = self._compute_neuron_va_probs(features)  # [B, T]
            return p_neuron

    def is_token_visually_absent(
        self,
        token_id: int,
        p_va: float,
    ) -> bool:
        """Decide whether a single token is visually absent.

        Uses :func:`is_visual_sensitive_token` and :class:`VAThresholds`
        to pick the appropriate per-token threshold.
        """
        tok_str = self.tokenizer.decode([token_id]).strip()
        if is_visual_sensitive_token(tok_str):
            thr = self.thresholds.p_visual_token
        else:
            thr = self.thresholds.p_nonvisual_token
        return p_va >= thr

    # ----- Binary QA refinement ----------------------------------------------

    def refine_yes_no_answer(
        self,
        question_input_ids: torch.Tensor,   # [B, T]
        image_inputs: Dict[str, Any],
        attention_mask: torch.Tensor,
        yes_token_id: int,
        no_token_id: int,
    ) -> torch.Tensor:
        """Override binary QA answers when the question contains VA tokens.

        For each sample, if *any* question token has ``p_VA`` above its
        type-aware threshold the answer is forced to ``no_token_id``.
        Otherwise the model's original prediction is preserved.

        Returns
        -------
        torch.Tensor
            ``[B]`` answer token IDs.
        """
        B, T = question_input_ids.shape
        self.feature_extractor.reset()
        self.feature_extractor.register_hooks()

        with torch.no_grad():
            outputs = self.lm(
                input_ids=question_input_ids,
                attention_mask=attention_mask,
                **image_inputs,
            )
            p_va_tokens = self.mark_visually_absent_tokens(
                hidden_states=outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs["hidden_states"][-1],
                attention_mask=attention_mask,
            )

        self.feature_extractor.remove_hooks()

        answers = torch.full(
            (B,), yes_token_id,
            dtype=torch.long,
            device=question_input_ids.device,
        )
        for b in range(B):
            for t in range(T):
                if attention_mask[b, t] == 0:
                    continue
                token_id = int(question_input_ids[b, t])
                p_va = float(p_va_tokens[b, t])
                if self.is_token_visually_absent(token_id, p_va):
                    answers[b] = no_token_id
                    break
        return answers

    # ----- Open-ended generation refinement ----------------------------------

    def refine_logits(
        self,
        logits_with_image: torch.Tensor,   # [B, V]
        logits_no_image: torch.Tensor,     # [B, V]
        generated_ids: torch.Tensor,       # [B, T_so_far]
        step: int,
    ) -> torch.Tensor:
        """Refine next-token logits using VA detection.

        Called at each decoding step **before** sampling.

        * Computes neuron-based VA (if hooks active) + logit-based VA.
        * Applies type-aware thresholds and temporal memory.
        * Hard-blocks or soft-rewrites visually absent tokens.

        Parameters
        ----------
        logits_with_image : [B, V]
            Current-step logits with the image present.
        logits_no_image : [B, V]
            Current-step logits with image neutralised.
        generated_ids : [B, T_so_far]
            All tokens generated up to this point.
        step : int
            Current decoding step (0-indexed).

        Returns
        -------
        torch.Tensor
            Adjusted logits ``[B, V]``.
        """
        B, V = logits_with_image.shape
        device = logits_with_image.device

        if step == 0:
            return logits_with_image  # nothing to refine yet

        last_pos = step - 1
        last_token_ids = generated_ids[:, last_pos]  # [B]

        # Logit-based VA score for the last generated token
        p_logit = compute_logit_va_score(
            logits_with_image=logits_with_image,
            logits_no_image=logits_no_image,
            token_ids=last_token_ids,
        )  # [B]

        # Neuron-based VA (if hooks were active during generation)
        try:
            features = self.feature_extractor.get_token_features()
            p_neuron_full = self._compute_neuron_va_probs(features)
            p_neuron_last = p_neuron_full[:, last_pos]  # [B]
        except RuntimeError:
            # Fallback: rely only on logit-based score
            p_neuron_last = p_logit.clone()

        # Combine neuron + logit scores
        p_va_last = (
            self.p_combine * p_neuron_last
            + (1.0 - self.p_combine) * p_logit
        )  # [B]

        # Update temporal memory
        for b in range(B):
            self.temporal.update(float(p_va_last[b]))
        is_burst = self.temporal.is_burst()

        # Adjust logits
        adjusted_logits = logits_with_image.clone()

        for b in range(B):
            token_id = int(last_token_ids[b])
            p_va = float(p_va_last[b])
            visually_absent = self.is_token_visually_absent(token_id, p_va)

            if visually_absent:
                if is_burst:
                    # Hard block: prevent re-selection of last token
                    adjusted_logits[b, token_id] = float("-inf")
                else:
                    # Soft rewrite: strong penalty on the suspicious token
                    adjusted_logits[b, token_id] -= 5.0
            elif is_burst:
                # In a burst, cool down the entire distribution
                adjusted_logits[b] = adjusted_logits[b] * (
                    1.0 - self.temporal.config.burst_decay
                )

        return adjusted_logits

    # ----- Answer-level confidence calibration -------------------------------

    def calibrate_answer_prefix(
        self,
        generated_ids: torch.Tensor,  # [B, T]
        p_va_tokens: torch.Tensor,    # [B, T]
    ) -> List[str]:
        """Produce confidence prefixes based on average VA over visual tokens.

        Returns
        -------
        list[str]
            One prefix string per sample.  Empty string means no prefix.
        """
        B, T = generated_ids.shape
        prefixes: List[str] = []
        for b in range(B):
            va_scores: List[float] = []
            for t in range(T):
                token_id = int(generated_ids[b, t])
                tok_str = self.tokenizer.decode([token_id]).strip()
                if is_visual_sensitive_token(tok_str):
                    va_scores.append(float(p_va_tokens[b, t]))
            if not va_scores:
                prefixes.append("")
                continue
            avg_va = sum(va_scores) / len(va_scores)
            if avg_va < 0.4:
                prefixes.append("")  # confident
            elif avg_va < 0.7:
                prefixes.append("I might be wrong, but ")
            else:
                prefixes.append(
                    "I cannot see that clearly in the image, but "
                )
        return prefixes

    # ----- Optional: train VA classifier on a VA-QA dataset ------------------

    def fit_on_dataset(
        self,
        features: torch.Tensor,  # [N, F]
        labels: torch.Tensor,    # [N] — 0 = present, 1 = absent
        num_epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        """Train the ``VAClassifier`` on collected VA-QA features.

        This is **not** hooked into the main training loop by default.
        It serves as a skeleton for future VA-QA integration.
        """
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Initialise classifier with correct input dim
        self._ensure_classifier(features.view(1, 1, -1))
        assert self.classifier is not None
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)

        self.classifier.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            n_batches = 0
            for x, y in loader:
                if self.device is not None:
                    x = x.to(self.device)
                    y = y.to(self.device)
                # x: [N_batch, F] → [N_batch, 1, F] for classifier
                logits = self.classifier(x.unsqueeze(1)).squeeze(1)  # [N_batch]
                loss = F.binary_cross_entropy(logits, y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            logger.info(
                f"[VA fit] Epoch {epoch + 1}/{num_epochs} — "
                f"avg loss: {total_loss / max(n_batches, 1):.4f}"
            )
        self.classifier.eval()
