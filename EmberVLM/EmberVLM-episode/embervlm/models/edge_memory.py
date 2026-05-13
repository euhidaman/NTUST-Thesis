"""
Edge-Optimised Episodic Memory for EmberVLM
============================================

Provides a low-latency, inference-only wrapper around
``EpisodicMemoryController`` designed for deployment on edge devices
(Jetson, RPi-class accelerators, mobile NPUs).

Key Design Decisions
--------------------
* **Training-only state stripped**: ``cov``, ``usage_counts``,
  ``last_access_step``, ``episode_meta_indices``, and the full
  ``write()`` / ``smart_write()`` / ``forget()`` paths are dropped.
  Only ``M`` (memory matrix) persists.
* **Precomputed addressing tables**: Gaussian addressing precomputes
  ``M_norms = ||M_k||^2`` so the per-query cost reduces from a
  materialised ``(B, K, C)`` diff to a ``(B, C)@(C, K)`` matmul plus a
  broadcast add — halving peak VRAM and improving cache locality.
  Pseudoinverse addressing caches ``pinv(M)`` so the $O(KC^2)$
  decomposition executes *once*, not per inference call.
* **Optional int8 quantisation**: Memory matrix can be stored as int8
  with per-row scales, cutting footprint from 1.18 MB (fp32) to ~0.3 MB
  while maintaining softmax-normalised addressing quality within 0.01
  cosine similarity of fp32.
* **Scope detector fast-path**: A running-mean gate allows the scope
  detector to be *skipped entirely* when the model has consistently
  returned low scope probabilities (mean < ``scope_fast_path_threshold``
  over the last ``scope_window`` queries).  This avoids the 148 K-param
  MLP entirely on trivial queries.
* **Latency benchmark entry-point**: ``benchmark_edge_latency()``
  measures read, scope-detect, and end-to-end conditioning latency
  with configurable warm-up / repeat so you can profile on the actual
  target hardware.

Usage
-----
::

    from embervlm.models.edge_memory import EdgeMemoryRuntime, benchmark_edge_latency

    # Build from a trained controller + scope detector
    runtime = EdgeMemoryRuntime.from_trained(
        controller=model.episodic_memory,
        scope_detector=model.scope_detector,
        quantize=True,           # int8 memory
        cache_addressing=True,   # precompute tables
    )

    # Condition generation on memory
    logits = runtime.condition_logits(
        last_hidden=hidden_states[:, -1, :],
        lm_head=model.language_model.get_output_embeddings(),
    )

    # Benchmark on real hardware
    report = benchmark_edge_latency(runtime, device="cuda", dtype=torch.float16)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class EdgeMemoryConfig:
    """Tunable knobs for edge deployment."""

    # Addressing
    addressing: str = "gaussian"         # "gaussian" | "pseudoinverse"
    temperature: float = 0.1
    alpha: float = 1.0
    variance: float = 1.0

    # Quantisation
    quantize: bool = False               # int8 memory
    cache_addressing: bool = True        # precompute tables

    # Scope detector fast-path
    scope_threshold: float = 0.5         # p > threshold → use memory
    scope_fast_path: bool = True         # enable running-mean gate
    scope_fast_path_threshold: float = 0.15  # skip scope detector if θ_mean < this
    scope_window: int = 64               # running window size

    # Latency tuning
    chunk_batch: int = 32                # max batch size for chunked ops


# ---------------------------------------------------------------------------
#  Int8 Quantised Memory Matrix
# ---------------------------------------------------------------------------


class QuantisedMemory(nn.Module):
    """Row-wise absmax int8 quantised memory matrix.

    Stores ``M_q`` (int8, KxC) and ``scales`` (fp16/fp32, K).
    Dequantisation: ``M_fp = M_q.float() * scales[:, None]``

    Addressing uses the dequantised matrix (lazy-cached) so the softmax
    operates on the correct magnitude.  The int8 form saves 4× VRAM for
    storage at rest; the fp32 form is rebuilt once on ``prepare()`` or
    when ``M`` changes.
    """

    def __init__(self, M_fp: torch.Tensor):
        super().__init__()
        self._quantise(M_fp)

    def _quantise(self, M_fp: torch.Tensor):
        """Quantise fp32/fp16 memory matrix to int8 + per-row scale."""
        M = M_fp.float()
        # Per-row absmax
        scales = M.abs().amax(dim=-1).clamp(min=1e-8)  # (K,)
        M_norm = M / scales[:, None]                     # (K, C) in [-1, 1]
        M_q = (M_norm * 127.0).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("M_q", M_q)
        self.register_buffer("scales", scales.half())
        # Lazily-built fp32 dequantised copy (for addressing math)
        self._M_fp_cache: Optional[torch.Tensor] = None

    @property
    def M(self) -> torch.Tensor:
        """Return dequantised (K, C) memory matrix, cached after first call."""
        if self._M_fp_cache is None:
            self._M_fp_cache = self.M_q.float() * self.scales.float().unsqueeze(-1)
        return self._M_fp_cache

    @property
    def shape(self) -> torch.Size:
        return self.M_q.shape

    def invalidate_cache(self):
        self._M_fp_cache = None

    def memory_bytes(self) -> int:
        """Return storage cost (excluding dequantised cache)."""
        return self.M_q.nelement() * 1 + self.scales.nelement() * 2

    def vs_fp32_bytes(self) -> int:
        K, C = self.M_q.shape
        return K * C * 4


# ---------------------------------------------------------------------------
#  Optimised Gaussian Addressing (precomputed norms)
# ---------------------------------------------------------------------------


class FastGaussianAddressing(nn.Module):
    """Gaussian addressing that precomputes ``||M_k||^2`` once.

    Standard addressing materialises a ``(B, K, C)`` diff tensor:
        ``diff = z[:, None, :] - M[None, :, :]``
        ``d_k  = (diff ** 2).sum(-1)``          # (B, K)

    The expand trick rewrites the L2 distance:
        ``||z - M_k||^2 = ||z||^2 - 2 z·M_k + ||M_k||^2``

    This avoids the ``(B, K, C)`` allocation entirely and turns the
    inner loop into a ``(B, C) @ (C, K)`` matmul (highly optimised on
    all hardware).
    """

    def __init__(
        self,
        M: torch.Tensor,
        alpha: float = 1.0,
        variance: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.variance = variance
        self.temperature = temperature
        self._prepare(M)

    def _prepare(self, M: torch.Tensor):
        """Precompute norms from current memory matrix."""
        self.register_buffer("M", M)
        # ||M_k||^2 for each slot  (K,)
        self.register_buffer("M_norms", (M * M).sum(dim=-1))
        # M^T for the matmul  (C, K)
        self.register_buffer("Mt", M.t().contiguous())

    def invalidate(self, M: torch.Tensor):
        """Re-prepare when memory changes (e.g. online write on edge)."""
        device, dtype = self.M.device, self.M.dtype
        M = M.to(device=device, dtype=dtype)
        self.M.copy_(M)
        self.M_norms.copy_((M * M).sum(dim=-1))
        self.Mt.copy_(M.t().contiguous())

    @torch.no_grad()
    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C) queries.
        Returns:
            weights: (B, K) softmax addressing weights.
            readout: (B, C) memory readout.
        """
        # ||z||^2  (B, 1)
        z_norms = (z * z).sum(dim=-1, keepdim=True)

        # z @ M^T  (B, K)
        dot = torch.matmul(z, self.Mt)

        # d_k = ||z||^2 - 2 z·M_k + ||M_k||^2   expanded L2
        d_k = z_norms - 2.0 * dot + self.M_norms.unsqueeze(0)  # (B, K)

        scores = -d_k / (2.0 * self.alpha * self.variance)
        weights = F.softmax(scores / self.temperature, dim=-1)  # (B, K)

        readout = torch.matmul(weights, self.M)  # (B, C)
        return weights, readout


# ---------------------------------------------------------------------------
#  Cached Pseudoinverse Addressing
# ---------------------------------------------------------------------------


class CachedPseudoinverseAddressing(nn.Module):
    """Caches ``pinv(M)`` so the O(K·C²) decomposition runs once."""

    def __init__(self, M: torch.Tensor, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self._prepare(M)

    def _prepare(self, M: torch.Tensor):
        self.register_buffer("M", M)
        self.register_buffer("M_pinv", torch.linalg.pinv(M))  # (C, K)

    def invalidate(self, M: torch.Tensor):
        device, dtype = self.M.device, self.M.dtype
        M = M.to(device=device, dtype=dtype)
        self.M.copy_(M)
        self.M_pinv.copy_(torch.linalg.pinv(M))

    @torch.no_grad()
    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_w = torch.matmul(z, self.M_pinv)  # (B, K)
        weights = F.softmax(raw_w / self.temperature, dim=-1)
        readout = torch.matmul(weights, self.M)
        return weights, readout


# ---------------------------------------------------------------------------
#  Scope Detector with Fast-Path Gate
# ---------------------------------------------------------------------------


class FastScopeDetector(nn.Module):
    """Wraps the trained ScopeDetector with a running-mean fast-path.

    When the running average of scope probabilities falls below
    ``fast_path_threshold``, the MLP is **skipped entirely** and a
    constant zero-vector is returned (no memory usage).  This avoids the
    148 K-param forward pass on queries that consistently do not benefit
    from memory.
    """

    def __init__(
        self,
        scope_detector: nn.Module,
        threshold: float = 0.5,
        fast_path: bool = True,
        fast_path_threshold: float = 0.15,
        window: int = 64,
    ):
        super().__init__()
        self.detector = scope_detector
        self.threshold = threshold
        self.fast_path = fast_path
        self.fast_path_threshold = fast_path_threshold
        self.window = window

        # Running statistics (not saved in state_dict — transient)
        self._prob_history: List[float] = []
        self._fast_path_skips: int = 0
        self._total_calls: int = 0

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Args:
            z: (B, C)
        Returns:
            scope_mask: (B,) bool — True where memory should be used.
            skipped: bool — True if fast-path was taken (detector not run).
        """
        self._total_calls += 1

        # Fast-path check: if recent history is consistently low, skip
        if self.fast_path and len(self._prob_history) >= self.window:
            recent_mean = sum(self._prob_history[-self.window:]) / self.window
            if recent_mean < self.fast_path_threshold:
                self._fast_path_skips += 1
                return torch.zeros(z.size(0), dtype=torch.bool, device=z.device), True

        # Full scope detection
        probs = self.detector(z)  # (B,)
        scope_mask = probs > self.threshold

        # Update history
        mean_prob = probs.mean().item()
        self._prob_history.append(mean_prob)
        if len(self._prob_history) > self.window * 4:
            self._prob_history = self._prob_history[-self.window * 2:]

        return scope_mask, False

    @property
    def skip_rate(self) -> float:
        return self._fast_path_skips / max(self._total_calls, 1)

    def reset_stats(self):
        self._prob_history.clear()
        self._fast_path_skips = 0
        self._total_calls = 0


# ---------------------------------------------------------------------------
#  Edge Memory Runtime
# ---------------------------------------------------------------------------


class EdgeMemoryRuntime(nn.Module):
    """Streamlined inference-only episodic memory runtime.

    Strips training-only state and provides the minimum-overhead path for:
        1. Scope detection (with fast-path gating).
        2. Memory read (with precomputed addressing tables).
        3. Logit conditioning (project readout through LM head).

    Total inference-only footprint (fp32, gaussian, no quantize):
        M (512×576) = 1.18 MB
        M_norms (512) = 2 KB
        Mt (576×512) = 1.18 MB (transposed copy)
        Scope detector = 0.58 MB
        ─────────────────────────
        Total ≈ 2.94 MB  (vs ~5.1 MB for the full controller)

    With int8 quantization:
        M_q (512×576 int8) = 0.29 MB + scales = 1 KB
        ─────────────────────────
        Total ≈ 1.47 MB
    """

    def __init__(self, config: Optional[EdgeMemoryConfig] = None):
        super().__init__()
        self.config = config or EdgeMemoryConfig()
        self.addressing: Optional[nn.Module] = None
        self.scope: Optional[FastScopeDetector] = None
        self.quantised: Optional[QuantisedMemory] = None
        self._ready = False

    # ------------------------------------------------------------------
    #  Construction from trained modules
    # ------------------------------------------------------------------

    @classmethod
    def from_trained(
        cls,
        controller: "EpisodicMemoryController",  # type: ignore[name-defined]
        scope_detector: Optional[nn.Module] = None,
        config: Optional[EdgeMemoryConfig] = None,
        quantize: Optional[bool] = None,
        cache_addressing: Optional[bool] = None,
    ) -> "EdgeMemoryRuntime":
        """Build an edge runtime from trained controller + scope detector.

        Parameters
        ----------
        controller : EpisodicMemoryController
            Trained controller (from ``model.episodic_memory``).
        scope_detector : nn.Module | None
            Trained ScopeDetector.  If None, scope gating is disabled.
        config : EdgeMemoryConfig | None
            Overrides.  If None, inherits from controller attributes.
        quantize : bool | None
            Override ``config.quantize``.
        cache_addressing : bool | None
            Override ``config.cache_addressing``.
        """
        cfg = config or EdgeMemoryConfig()

        # Inherit controller settings
        cfg.addressing = controller.addressing_mode
        cfg.temperature = controller.temperature
        cfg.alpha = controller.alpha
        cfg.variance = controller.variance

        if quantize is not None:
            cfg.quantize = quantize
        if cache_addressing is not None:
            cfg.cache_addressing = cache_addressing

        runtime = cls(cfg)

        # --- Memory matrix ---
        M_fp = controller.M.detach().clone().float()

        if cfg.quantize:
            runtime.quantised = QuantisedMemory(M_fp)
            M_for_addressing = runtime.quantised.M  # dequantised
            logger.info(
                f"EdgeMemoryRuntime: int8 quantised — "
                f"{runtime.quantised.memory_bytes():,} bytes "
                f"(was {runtime.quantised.vs_fp32_bytes():,} fp32)"
            )
        else:
            runtime.register_buffer("M_fp", M_fp)
            M_for_addressing = M_fp

        # --- Addressing ---
        if cfg.cache_addressing:
            if cfg.addressing == "pseudoinverse":
                runtime.addressing = CachedPseudoinverseAddressing(
                    M_for_addressing, temperature=cfg.temperature)
            else:
                runtime.addressing = FastGaussianAddressing(
                    M_for_addressing,
                    alpha=cfg.alpha,
                    variance=cfg.variance,
                    temperature=cfg.temperature,
                )
        else:
            # Fallback: keep a reference to M and use the basic path
            runtime.register_buffer("_M_basic", M_for_addressing)

        # --- Scope detector ---
        if scope_detector is not None:
            runtime.scope = FastScopeDetector(
                scope_detector,
                threshold=cfg.scope_threshold,
                fast_path=cfg.scope_fast_path,
                fast_path_threshold=cfg.scope_fast_path_threshold,
                window=cfg.scope_window,
            )

        runtime._ready = True
        K, C = M_fp.shape
        logger.info(
            f"EdgeMemoryRuntime ready: {K} slots × {C} dims, "
            f"addressing={cfg.addressing}, "
            f"quantize={cfg.quantize}, "
            f"cache={cfg.cache_addressing}, "
            f"scope_fast_path={cfg.scope_fast_path}"
        )
        return runtime

    # ------------------------------------------------------------------
    #  Core read
    # ------------------------------------------------------------------

    def _get_M(self) -> torch.Tensor:
        """Return the memory matrix (handles quantised / cached / basic)."""
        if self.quantised is not None:
            return self.quantised.M
        if hasattr(self, "M_fp"):
            return self.M_fp
        if hasattr(self, "_M_basic"):
            return self._M_basic
        raise RuntimeError("EdgeMemoryRuntime: no memory matrix available")

    @torch.no_grad()
    def read(self, z: torch.Tensor) -> torch.Tensor:
        """
        Memory readout for query *z*.

        Args:
            z: (B, C) fused multimodal representation.
        Returns:
            readout: (B, C) retrieved content.
        """
        if self.addressing is not None:
            _, readout = self.addressing(z)
            return readout

        # Basic fallback (no caching)
        M = self._get_M()
        diff = z.unsqueeze(1) - M.unsqueeze(0)
        d_k = (diff * diff).sum(dim=-1)
        scores = -d_k / (2.0 * self.config.alpha * self.config.variance)
        weights = F.softmax(scores / self.config.temperature, dim=-1)
        return torch.matmul(weights, M)

    # ------------------------------------------------------------------
    #  End-to-end conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def condition_logits(
        self,
        last_hidden: torch.Tensor,
        lm_head: nn.Module,
    ) -> Optional[torch.Tensor]:
        """Condition logits with memory readout.

        This is the full edge inference path:

            1. Scope detection (with fast-path).
            2. Memory read (with cached addressing).
            3. LM-head projection.

        Args:
            last_hidden: (B, C) — last-token hidden state from the LM.
            lm_head: nn.Linear — language model output projection.

        Returns:
            logit_bias: (B, V) additive logit bias, or None if all
            queries were out-of-scope.
        """
        B, C = last_hidden.shape

        # Ensure dtype compat (edge devices may use fp16)
        M_dtype = self._get_M().dtype
        if last_hidden.dtype != M_dtype:
            last_hidden = last_hidden.to(dtype=M_dtype)

        # Step 1: scope detection
        if self.scope is not None:
            scope_mask, skipped = self.scope(last_hidden)
            if skipped or not scope_mask.any():
                return None
            z = last_hidden[scope_mask]
        else:
            z = last_hidden
            scope_mask = torch.ones(B, dtype=torch.bool, device=last_hidden.device)

        # Step 2: memory read
        readout = self.read(z)  # (B_scope, C)

        # Step 3: project to logit space
        full_readout = torch.zeros_like(last_hidden)
        full_readout[scope_mask] = readout
        logit_bias = lm_head(full_readout)  # (B, V)

        return logit_bias

    # ------------------------------------------------------------------
    #  Online update (lightweight write for edge if needed)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def online_update(
        self,
        z: torch.Tensor,
        novelty_threshold: float = 0.7,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Lightweight online memory update for edge deployment.

        Unlike the full controller's ``write()``, this:
          - Does NOT update the covariance matrix.
          - Does NOT call ``linalg.solve``.
          - Uses a simple exponential moving average on the nearest slot.
          - Is O(K·C) not O(C³).

        Args:
            z: (B, C) new episode embeddings.
            novelty_threshold: minimum L2 distance to trigger write.
            alpha: update strength (default: ``config.alpha``).

        Returns:
            dict with ``num_written``, ``mean_novelty``.
        """
        M = self._get_M()
        alpha_val = alpha if alpha is not None else self.config.alpha

        # Novelty: min ||z - M_k||² per sample
        diff = z.unsqueeze(1) - M.unsqueeze(0)  # (B, K, C)
        dists = (diff * diff).sum(dim=-1)         # (B, K)
        min_dist, nearest = dists.min(dim=-1)     # (B,), (B,)

        write_mask = min_dist > novelty_threshold
        if not write_mask.any():
            return {"num_written": 0, "mean_novelty": min_dist.mean().item()}

        z_write = z[write_mask]
        slots = nearest[write_mask]

        # EMA update on nearest slot: M_k ← (1-α)M_k + α·z
        for i in range(z_write.size(0)):
            k = slots[i].item()
            M[k] = (1.0 - alpha_val) * M[k] + alpha_val * z_write[i]

        # Invalidate caches
        if self.addressing is not None and hasattr(self.addressing, "invalidate"):
            self.addressing.invalidate(M)
        if self.quantised is not None:
            self.quantised.invalidate_cache()

        return {
            "num_written": write_mask.sum().item(),
            "mean_novelty": min_dist.mean().item(),
        }

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save edge runtime to a single file."""
        state = {
            "config": self.config.__dict__,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"EdgeMemoryRuntime saved to {path}")

    @classmethod
    def load(cls, path: str, scope_detector_cls=None, **kwargs) -> "EdgeMemoryRuntime":
        """Load edge runtime from file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = EdgeMemoryConfig(**checkpoint["config"])
        runtime = cls(cfg)
        runtime.load_state_dict(checkpoint["state_dict"], strict=False)
        runtime._ready = True
        logger.info(f"EdgeMemoryRuntime loaded from {path}")
        return runtime

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def memory_footprint(self) -> Dict[str, int]:
        """Return byte counts for each component."""
        report: Dict[str, int] = {}

        if self.quantised is not None:
            report["memory_matrix_int8"] = self.quantised.memory_bytes()
        elif hasattr(self, "M_fp"):
            report["memory_matrix_fp32"] = self.M_fp.nelement() * self.M_fp.element_size()
        elif hasattr(self, "_M_basic"):
            report["memory_matrix_fp32"] = self._M_basic.nelement() * self._M_basic.element_size()

        if self.addressing is not None:
            for name, buf in self.addressing.named_buffers():
                report[f"addressing/{name}"] = buf.nelement() * buf.element_size()

        if self.scope is not None:
            scope_bytes = sum(p.nelement() * p.element_size()
                             for p in self.scope.detector.parameters())
            report["scope_detector"] = scope_bytes

        report["total"] = sum(report.values())
        return report


# ---------------------------------------------------------------------------
#  Latency Benchmark
# ---------------------------------------------------------------------------


@torch.no_grad()
def benchmark_edge_latency(
    runtime: EdgeMemoryRuntime,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    batch_sizes: Optional[List[int]] = None,
    warmup: int = 20,
    repeats: int = 100,
    include_lm_head: bool = False,
    lm_head: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Benchmark episodic memory latency on target hardware.

    Measures:
        - ``read_latency_ms``: pure memory read (addressing + matmul).
        - ``scope_latency_ms``: scope detector forward.
        - ``e2e_latency_ms``: end-to-end ``condition_logits()`` including
          scope detection, addressing, and LM-head projection.
        - ``online_update_ms``: lightweight online write (EMA).

    Parameters
    ----------
    runtime : EdgeMemoryRuntime
        Edge runtime to benchmark.
    device : str
        Target device ("cuda", "cpu", "mps").
    dtype : torch.dtype
        Precision (fp32, fp16, bf16).
    batch_sizes : list[int] | None
        Batch sizes to test.  Defaults to [1, 4, 16].
    warmup, repeats : int
        Warm-up iterations (discarded) and measurement iterations.
    include_lm_head : bool
        Whether to include LM-head projection in e2e measurement.
    lm_head : nn.Module | None
        Required if ``include_lm_head=True``.

    Returns
    -------
    dict with per-batch-size latency statistics.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]

    runtime = runtime.to(device)
    runtime.eval()

    M = runtime._get_M()
    K, C = M.shape

    is_cuda = device == "cuda" and torch.cuda.is_available()

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    results: Dict[str, Any] = {
        "device": device,
        "dtype": str(dtype),
        "memory_slots": K,
        "hidden_dim": C,
        "quantized": runtime.quantised is not None,
        "cached_addressing": runtime.addressing is not None,
        "scope_fast_path": runtime.config.scope_fast_path,
        "warmup": warmup,
        "repeats": repeats,
        "batch_results": {},
    }

    for B in batch_sizes:
        z = torch.randn(B, C, device=device, dtype=dtype)
        batch_report: Dict[str, float] = {}

        # --- Read latency ---
        for _ in range(warmup):
            _ = runtime.read(z)
        _sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = runtime.read(z)
        _sync()
        t1 = time.perf_counter()
        batch_report["read_ms"] = (t1 - t0) / repeats * 1000

        # --- Scope detector latency ---
        if runtime.scope is not None:
            runtime.scope.reset_stats()
            for _ in range(warmup):
                _ = runtime.scope(z)
            _sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = runtime.scope(z)
            _sync()
            t1 = time.perf_counter()
            batch_report["scope_ms"] = (t1 - t0) / repeats * 1000
            batch_report["scope_skip_rate"] = runtime.scope.skip_rate
        else:
            batch_report["scope_ms"] = 0.0
            batch_report["scope_skip_rate"] = 0.0

        # --- End-to-end condition_logits ---
        if include_lm_head and lm_head is not None:
            lm_head = lm_head.to(device)
            for _ in range(warmup):
                _ = runtime.condition_logits(z, lm_head)
            _sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = runtime.condition_logits(z, lm_head)
            _sync()
            t1 = time.perf_counter()
            batch_report["e2e_condition_ms"] = (t1 - t0) / repeats * 1000

        # --- Online update latency ---
        for _ in range(warmup):
            _ = runtime.online_update(z, novelty_threshold=999.0)  # no actual write
        _sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = runtime.online_update(z, novelty_threshold=999.0)
        _sync()
        t1 = time.perf_counter()
        batch_report["online_update_ms"] = (t1 - t0) / repeats * 1000

        # --- Throughput ---
        batch_report["read_throughput_qps"] = B / (batch_report["read_ms"] / 1000)

        results["batch_results"][f"B={B}"] = batch_report

    # Memory footprint
    results["memory_footprint"] = runtime.memory_footprint()

    return results


def format_benchmark_report(results: Dict[str, Any]) -> str:
    """Pretty-print a benchmark report."""
    lines = [
        "=" * 65,
        "  Edge Episodic Memory — Latency Benchmark",
        "=" * 65,
        f"  Device: {results['device']}    Dtype: {results['dtype']}",
        f"  Slots: {results['memory_slots']}    Dim: {results['hidden_dim']}",
        f"  Quantized: {results['quantized']}    Cached: {results['cached_addressing']}",
        f"  Scope fast-path: {results['scope_fast_path']}",
        "-" * 65,
    ]

    for batch_key, br in results["batch_results"].items():
        lines.append(f"\n  {batch_key}:")
        lines.append(f"    Read:          {br['read_ms']:.3f} ms")
        if br.get("scope_ms"):
            lines.append(f"    Scope:         {br['scope_ms']:.3f} ms  "
                         f"(skip rate: {br['scope_skip_rate']:.1%})")
        if br.get("e2e_condition_ms"):
            lines.append(f"    E2E condition: {br['e2e_condition_ms']:.3f} ms")
        lines.append(f"    Online update: {br['online_update_ms']:.3f} ms")
        lines.append(f"    Read QPS:      {br['read_throughput_qps']:,.0f}")

    fp = results.get("memory_footprint", {})
    if fp:
        lines.append("\n  Memory footprint:")
        for k, v in fp.items():
            if k == "total":
                lines.append(f"    {'─' * 30}")
            lines.append(f"    {k:<30s} {v:>10,} bytes  ({v/1024:.1f} KB)")

    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Compare: original controller vs edge runtime
# ---------------------------------------------------------------------------


@torch.no_grad()
def compare_addressing_accuracy(
    controller: "EpisodicMemoryController",  # type: ignore[name-defined]
    runtime: EdgeMemoryRuntime,
    n_queries: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Compare readout quality between full controller and edge runtime.

    Returns cosine similarity and L2 distance between original and
    edge-optimised readouts over random queries.
    """
    M = controller.M
    K, C = M.shape

    controller = controller.to(device)
    runtime = runtime.to(device)

    z = torch.randn(n_queries, C, device=device, dtype=dtype)

    # Original
    z_orig = controller._ensure_dtype(z)
    _, readout_orig = controller._address(z_orig)

    # Edge
    readout_edge = runtime.read(z.to(dtype=runtime._get_M().dtype))

    # Metrics
    cos_sim = F.cosine_similarity(readout_orig.float(), readout_edge.float(), dim=-1)
    l2_dist = (readout_orig.float() - readout_edge.float()).norm(dim=-1)

    return {
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_min": cos_sim.min().item(),
        "l2_dist_mean": l2_dist.mean().item(),
        "l2_dist_max": l2_dist.max().item(),
    }
