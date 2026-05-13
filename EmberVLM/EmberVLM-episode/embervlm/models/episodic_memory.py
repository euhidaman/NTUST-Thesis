"""
Multimodal Episodic Memory for EmberVLM

Stores fused image-text episodes and enables training-free knowledge updates.

Components:
  - EpisodicMemoryController: memory matrix M, covariance C, read/write/forget.
  - ScopeDetector: lightweight MLP that decides whether to consult memory.

All operations work on fused multimodal embeddings (after EmberVLM's Fusion Module),
*not* raw text or raw images.
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  ScopeDetector
# ---------------------------------------------------------------------------


class ScopeDetector(nn.Module):
    """
    Binary classifier over fused multimodal embeddings.

    Returns a probability in [0, 1] indicating whether a query
    is "in scope" of episodic memory (i.e. memory is likely to help).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        method: str = "internal",
    ):
        super().__init__()
        self.method = method
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C) fused multimodal embeddings.
        Returns:
            probs: (B,) probabilities in [0, 1].
        """
        # Ensure input matches weight dtype (handles bf16 from mixed-precision LM)
        weight_dtype = self.net[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        logits = self.net(x).squeeze(-1)  # (B,)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
#  EpisodicMemoryController
# ---------------------------------------------------------------------------


class EpisodicMemoryController(nn.Module):
    """
    Fixed-size episodic memory operating in fused multimodal space.

    Memory matrix  M  :  (K, C)
    Covariance-like  C :  (C, C)

    Supports Gaussian and pseudoinverse addressing, one-shot recursive
    updates, selective forgetting, novelty detection, and slot replacement.
    """

    def __init__(
        self,
        memory_slots: int = 512,
        hidden_dim: int = 576,
        addressing: str = "gaussian",
        alpha: float = 1.0,
        variance: float = 1.0,
        temperature: float = 0.1,
        novelty_threshold_novel: float = 0.7,
        novelty_threshold_similar: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.K = memory_slots
        self.C = hidden_dim
        self.addressing_mode = addressing
        self.alpha = alpha
        self.variance = variance
        self.temperature = temperature
        self.novelty_threshold_novel = novelty_threshold_novel
        self.novelty_threshold_similar = novelty_threshold_similar

        # ---------- memory state (buffers = saved with state_dict) ----------
        # M: memory matrix, small Xavier-uniform init
        M = torch.empty(memory_slots, hidden_dim)
        nn.init.xavier_uniform_(M)
        self.register_buffer("M", M)

        # Covariance-like matrix ~ scaled identity
        cov = torch.eye(hidden_dim) * (1.0 / hidden_dim)
        self.register_buffer("cov", cov)

        # Slot usage tracking
        self.register_buffer("usage_counts", torch.zeros(memory_slots))
        self.register_buffer("last_access_step", torch.zeros(memory_slots))

        # Global step counter
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # Episode metadata: indices into the original dataset so that
        # consolidation can reconstruct episodes later.
        self.register_buffer(
            "episode_meta_indices",
            torch.full((memory_slots,), -1, dtype=torch.long),
        )

        # Precomputed addressing tables (edge-optimised)
        self._M_norms: Optional[torch.Tensor] = None    # ||M_k||²  (K,)
        self._Mt: Optional[torch.Tensor] = None          # M^T       (C, K)
        self._pinv_cache: Optional[torch.Tensor] = None  # pinv(M)   (C, K)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    #  Precomputed table management
    # ------------------------------------------------------------------

    def _precompute_addressing_tables(self):
        """Recompute cached tables from current M.  Call after any write."""
        self._M_norms = (self.M * self.M).sum(dim=-1)      # (K,)
        self._Mt = self.M.t().contiguous()                   # (C, K)
        if self.addressing_mode == "pseudoinverse":
            self._pinv_cache = torch.linalg.pinv(self.M)     # (C, K)

    def _invalidate_tables(self):
        """Clear cache after memory updates."""
        self._M_norms = None
        self._Mt = None
        self._pinv_cache = None

    def prepare_for_inference(self):
        """Pre-warm addressing tables.  Call once before inference loop."""
        self._precompute_addressing_tables()
        logger.info(f"EpisodicMemoryController: addressing tables precomputed "
                    f"({self.addressing_mode}, K={self.K}, C={self.C})")

    # ------------------------------------------------------------------
    #  Addressing
    # ------------------------------------------------------------------

    def _gaussian_addressing(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian (distance-based) addressing.

        Uses expanded L2 distance to avoid materialising a (B, K, C) tensor:
            ||z - M_k||² = ||z||² - 2·z·M_k + ||M_k||²

        This replaces the naive diff-then-square with a (B,C)@(C,K) matmul,
        halving peak VRAM and improving cache locality on edge hardware.

        Args:
            z: (B, C)
        Returns:
            weights: (B, K) normalised addressing weights.
            readout: (B, C) weighted combination of memory slots.
        """
        # Lazy-init precomputed tables
        if self._M_norms is None or self._Mt is None:
            self._M_norms = (self.M * self.M).sum(dim=-1)  # (K,)
            self._Mt = self.M.t().contiguous()              # (C, K)

        # ||z||² → (B, 1)
        z_norms = (z * z).sum(dim=-1, keepdim=True)

        # z @ M^T → (B, K)
        dot = torch.matmul(z, self._Mt)

        # Expanded L2: ||z - M_k||² = ||z||² - 2·z·M_k + ||M_k||²
        d_k = z_norms - 2.0 * dot + self._M_norms.unsqueeze(0)  # (B, K)

        scores = -d_k / (2.0 * self.alpha * self.variance)  # (B, K)
        weights = F.softmax(scores / self.temperature, dim=-1)  # (B, K)

        readout = torch.matmul(weights, self.M)  # (B, C)
        return weights, readout

    def _pseudoinverse_addressing(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pseudoinverse-based addressing with cached pinv(M).

        The pinv decomposition is O(K·C²) ≈ 170M FLOPs and is cached
        after the first call.  Subsequent calls cost only O(B·C·K).

        Args:
            z: (B, C)
        Returns:
            weights: (B, K) normalised addressing weights.
            readout: (B, C) weighted combination of memory slots.
        """
        # Cache pinv(M) to avoid recomputing on every call
        if self._pinv_cache is None:
            self._pinv_cache = torch.linalg.pinv(self.M)  # (C, K)

        # raw weights: w = z @ M_pinv  →  (B, K)
        raw_w = torch.matmul(z, self._pinv_cache)

        # Clamp negatives, normalise (softmax for stability)
        weights = F.softmax(raw_w / self.temperature, dim=-1)  # (B, K)

        readout = torch.matmul(weights, self.M)  # (B, C)
        return weights, readout

    def _address(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to configured addressing method."""
        if self.addressing_mode == "pseudoinverse":
            return self._pseudoinverse_addressing(z)
        return self._gaussian_addressing(z)

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------

    def _ensure_dtype(self, z: torch.Tensor) -> torch.Tensor:
        """Cast input to match buffer dtype (handles bf16 from mixed-precision LM)."""
        buf_dtype = self.M.dtype
        return z.to(dtype=buf_dtype) if z.dtype != buf_dtype else z

    def read(self, z: torch.Tensor) -> torch.Tensor:
        """
        Memory readout for multimodal query *z*.

        Args:
            z: (B, C) fused multimodal representation.
        Returns:
            z_readout: (B, C) retrieved content.
        """
        z = self._ensure_dtype(z)
        weights, readout = self._address(z)

        # Update access stats for the most-attended slot
        with torch.no_grad():
            top_slots = weights.argmax(dim=-1)  # (B,)
            for idx in top_slots:
                self.usage_counts[idx] += 1
                self.last_access_step[idx] = self.global_step

        return readout

    # ------------------------------------------------------------------
    #  Write  (recursive one-shot update)
    # ------------------------------------------------------------------

    def write(self, z: torch.Tensor, alpha_override: Optional[float] = None) -> Dict[str, Any]:
        """
        One-shot recursive write of new multimodal episode(s).

        Uses addressing weights to compute a residual, updates M and cov.

        Args:
            z: (B, C) episode embeddings.
            alpha_override: override for update strength (negative = forget).
        Returns:
            dict with update statistics.
        """
        z = self._ensure_dtype(z)
        B = z.size(0)
        alpha_i = alpha_override if alpha_override is not None else self.alpha

        weights, _ = self._address(z)  # (B, K), _

        # Effective memory projection:  M_hat = weights @ M  → (B, C)
        M_hat = torch.matmul(weights, self.M)  # (B, C)

        # Residual
        r = z - M_hat  # (B, C)

        # Average across batch for a single update step
        r_avg = r.mean(dim=0)  # (C,)
        M_hat_avg = M_hat.mean(dim=0)  # (C,)
        w_avg = weights.mean(dim=0)  # (K,)

        # --- Covariance update ---
        # C_new = C_old + alpha * v v^T   (v = M_hat_avg)
        v = M_hat_avg  # (C,)
        self.cov = self.cov + alpha_i * torch.outer(v, v)

        # --- Memory update via linear solve ---
        # delta = C^{-1} r_avg  solved as  C @ delta = r_avg
        try:
            delta = torch.linalg.solve(self.cov, r_avg)  # (C,)
        except torch.linalg.LinAlgError:
            # Fallback to regularised solve
            reg = 1e-4 * torch.eye(self.C, device=self.cov.device, dtype=self.cov.dtype)
            delta = torch.linalg.solve(self.cov + reg, r_avg)

        # M_new = M_old + alpha * w^T ⊗ delta  (outer: slot weighting)
        # w_avg: (K,)  delta: (C,)  →  (K, C)
        update = alpha_i * torch.outer(w_avg, delta)
        self.M = self.M + update

        # Invalidate precomputed tables (M has changed)
        self._invalidate_tables()

        # --- Track slot usage ---
        with torch.no_grad():
            top_slot = w_avg.argmax().item()
            self.usage_counts[top_slot] += B
            self.last_access_step[top_slot] = self.global_step
            self.global_step += 1

        return {
            "residual_norm": r_avg.norm().item(),
            "update_norm": update.norm().item(),
            "top_slot": top_slot,
            "alpha": alpha_i,
        }

    # ------------------------------------------------------------------
    #  Forget
    # ------------------------------------------------------------------

    def forget(
        self,
        slot_index: Optional[int] = None,
        content: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Selective forgetting via negative write strength.

        Either specify *slot_index* directly or provide a *content* vector
        and the nearest slot will be targeted.
        """
        if slot_index is not None:
            target_z = self.M[slot_index].unsqueeze(0)  # (1, C)
        elif content is not None:
            if content.dim() == 1:
                content = content.unsqueeze(0)
            # Find nearest slot
            dists = ((self.M.unsqueeze(0) - content.unsqueeze(1)) ** 2).sum(-1)  # (B, K)
            slot_index = dists.argmin(dim=-1)[0].item()
            target_z = self.M[slot_index].unsqueeze(0)
        else:
            raise ValueError("Provide either slot_index or content for forgetting.")

        stats = self.write(target_z, alpha_override=-self.alpha)

        # Reset slot metadata
        with torch.no_grad():
            self.usage_counts[slot_index] = 0
            self.last_access_step[slot_index] = 0
            self.episode_meta_indices[slot_index] = -1

        stats["forgotten_slot"] = slot_index
        return stats

    # ------------------------------------------------------------------
    #  Novelty detection
    # ------------------------------------------------------------------

    def _pairwise_sq_dists(self, z: torch.Tensor) -> torch.Tensor:
        """Compute ||z_b - M_k||² using expanded L2 (no (B,K,C) alloc).

        Args:
            z: (B, C)
        Returns:
            dists: (B, K)
        """
        if self._M_norms is None or self._Mt is None:
            self._M_norms = (self.M * self.M).sum(dim=-1)
            self._Mt = self.M.t().contiguous()

        z_norms = (z * z).sum(dim=-1, keepdim=True)          # (B, 1)
        dot = torch.matmul(z, self._Mt)                       # (B, K)
        return z_norms - 2.0 * dot + self._M_norms.unsqueeze(0)  # (B, K)

    def novelty_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma(z | M) = min_k ||z - M_k||^2  for each batch element.

        Uses expanded L2 to avoid the (B, K, C) intermediate tensor.

        Args:
            z: (B, C)
        Returns:
            sigma: (B,)  minimum squared distance to any memory slot.
        """
        z = self._ensure_dtype(z)
        dists = self._pairwise_sq_dists(z)  # (B, K)
        sigma = dists.min(dim=-1).values    # (B,)
        return sigma

    def should_write(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Novelty-gated write decision.

        Returns:
            write_flags: (B,) bool   – True if episode should be written.
            sigma:       (B,) float  – novelty score.
            nearest_idx: (B,) int64  – index of nearest memory slot.
        """
        z = self._ensure_dtype(z)
        dists = self._pairwise_sq_dists(z)  # (B, K)

        sigma, nearest_idx = dists.min(dim=-1)  # both (B,)

        write_flags = sigma > self.novelty_threshold_novel

        return write_flags, sigma, nearest_idx

    # ------------------------------------------------------------------
    #  Replacement policy
    # ------------------------------------------------------------------

    def _replacement_slot(self) -> int:
        """
        Choose the best slot for replacement (LRU + LFU hybrid).

        score = 0.5 * (normalised inverse usage) + 0.5 * (normalised inverse recency)
        """
        with torch.no_grad():
            max_use = self.usage_counts.max().clamp(min=1.0)
            max_step = self.last_access_step.max().clamp(min=1.0)

            inv_use = 1.0 - self.usage_counts / max_use
            inv_rec = 1.0 - self.last_access_step / max_step

            score = 0.5 * inv_use + 0.5 * inv_rec
            return score.argmax().item()

    # ------------------------------------------------------------------
    #  Memory regularisation loss
    # ------------------------------------------------------------------

    def regularization_loss(self, lam: float = 1e-4) -> torch.Tensor:
        """
        L2 regularisation on memory slots.

        L_mem_reg = lam * sum_k ||M_k||^2
        """
        return lam * (self.M * self.M).sum()

    # ------------------------------------------------------------------
    #  Convenience: smart write with novelty gating and replacement
    # ------------------------------------------------------------------

    def smart_write(
        self,
        z: torch.Tensor,
        meta_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        High-level write that checks novelty and picks replacement slots.

        Args:
            z: (B, C) fused multimodal embeddings.
            meta_indices: optional (B,) dataset indices for consolidation.
        Returns:
            dict with per-batch stats.
        """
        write_flags, sigma, nearest_idx = self.should_write(z)

        if not write_flags.any():
            return {
                "num_written": 0,
                "sigma_mean": sigma.mean().item(),
                "sigma_max": sigma.max().item(),
            }

        # Collect episodes to write
        z_to_write = z[write_flags]

        # Write (single batched update)
        stats = self.write(z_to_write)

        # Store metadata for written episodes
        if meta_indices is not None:
            written_meta = meta_indices[write_flags]
            for mi in written_meta:
                slot = self._replacement_slot()
                self.episode_meta_indices[slot] = mi.item() if mi.dim() == 0 else mi[0].item()

        stats["num_written"] = write_flags.sum().item()
        stats["sigma_mean"] = sigma.mean().item()
        stats["sigma_max"] = sigma.max().item()
        return stats

    # ------------------------------------------------------------------
    #  Serialisation helpers
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Return all memory state tensors (M, cov, metadata)."""
        return {
            "M": self.M.clone(),
            "cov": self.cov.clone(),
            "usage_counts": self.usage_counts.clone(),
            "last_access_step": self.last_access_step.clone(),
            "global_step": self.global_step.clone(),
            "episode_meta_indices": self.episode_meta_indices.clone(),
        }

    def set_state(self, state: Dict[str, torch.Tensor], strict: bool = True):
        """Load memory state tensors."""
        for key in ["M", "cov", "usage_counts", "last_access_step",
                     "global_step", "episode_meta_indices"]:
            if key in state:
                buf = getattr(self, key)
                loaded = state[key]
                if strict and buf.shape != loaded.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: expected {buf.shape}, got {loaded.shape}"
                    )
                buf.copy_(loaded.to(buf.device))
            elif strict:
                raise KeyError(f"Missing key {key} in memory state")
        # Invalidate precomputed tables since M may have changed
        self._invalidate_tables()

    # ------------------------------------------------------------------
    #  Edge deployment
    # ------------------------------------------------------------------

    def to_edge(
        self,
        scope_detector: Optional["ScopeDetector"] = None,
        quantize: bool = False,
        cache_addressing: bool = True,
    ) -> "EdgeMemoryRuntime":  # type: ignore[name-defined]
        """Convert this controller to a streamlined edge runtime.

        Strips training-only state (cov, usage_counts, write machinery)
        and returns an ``EdgeMemoryRuntime`` optimised for low-latency
        inference on edge devices.

        Args:
            scope_detector: trained ScopeDetector (optional).
            quantize: use int8 memory matrix (saves ~4× VRAM).
            cache_addressing: precompute addressing tables.

        Returns:
            EdgeMemoryRuntime ready for deployment.
        """
        from embervlm.models.edge_memory import EdgeMemoryRuntime
        return EdgeMemoryRuntime.from_trained(
            controller=self,
            scope_detector=scope_detector,
            quantize=quantize,
            cache_addressing=cache_addressing,
        )
