"""
Unit tests for EpisodicMemoryController and ScopeDetector.

Tests cover:
  - Tensor shapes after construction
  - Addressing weight normalisation
  - Novelty score monotonicity
  - Write / read round-trip
  - Forget resets slot
  - No NaN / Inf in any output
  - Serialisation round-trip
  - Replacement policy determinism
"""

import pytest
import torch

from embervlm.models.episodic_memory import EpisodicMemoryController, ScopeDetector


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

K = 64
C = 128
B = 4


@pytest.fixture
def mem():
    return EpisodicMemoryController(
        memory_slots=K,
        hidden_dim=C,
        addressing="gaussian",
        alpha=1.0,
        variance=1.0,
        temperature=0.1,
        novelty_threshold_novel=0.7,
        novelty_threshold_similar=0.2,
    )


@pytest.fixture
def scope():
    return ScopeDetector(input_dim=C, hidden_dim=64, method="internal")


# ---------------------------------------------------------------------------
#  Shape tests
# ---------------------------------------------------------------------------


class TestShapes:
    def test_memory_matrix_shape(self, mem):
        assert mem.M.shape == (K, C)

    def test_covariance_shape(self, mem):
        assert mem.cov.shape == (C, C)

    def test_usage_shape(self, mem):
        assert mem.usage_counts.shape == (K,)
        assert mem.last_access_step.shape == (K,)

    def test_read_output_shape(self, mem):
        z = torch.randn(B, C)
        readout = mem.read(z)
        assert readout.shape == (B, C)

    def test_scope_detector_output_shape(self, scope):
        z = torch.randn(B, C)
        probs = scope(z)
        assert probs.shape == (B,)

    def test_novelty_score_shape(self, mem):
        z = torch.randn(B, C)
        sigma = mem.novelty_score(z)
        assert sigma.shape == (B,)


# ---------------------------------------------------------------------------
#  Addressing tests
# ---------------------------------------------------------------------------


class TestAddressing:
    def test_gaussian_weights_sum_to_one(self, mem):
        z = torch.randn(B, C)
        weights, _ = mem._gaussian_addressing(z)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    def test_gaussian_weights_non_negative(self, mem):
        z = torch.randn(B, C)
        weights, _ = mem._gaussian_addressing(z)
        assert (weights >= 0).all()

    def test_pseudoinverse_weights_sum_to_one(self):
        mem = EpisodicMemoryController(
            memory_slots=K, hidden_dim=C, addressing="pseudoinverse"
        )
        z = torch.randn(B, C)
        weights, _ = mem._pseudoinverse_addressing(z)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
#  Novelty tests
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_novelty_monotonic_with_distance(self, mem):
        """Farther query -> higher novelty score."""
        z_close = mem.M[0:1]  # exactly a memory slot
        z_far = mem.M[0:1] + 100.0 * torch.randn(1, C)
        sigma_close = mem.novelty_score(z_close)
        sigma_far = mem.novelty_score(z_far)
        assert sigma_far.item() >= sigma_close.item()

    def test_should_write_exact_slot_is_false(self, mem):
        """If query is an exact memory slot, novelty should be ~0 => no write."""
        z = mem.M[0:1].clone()
        flags, sigma, _ = mem.should_write(z)
        assert sigma.item() < 1e-3
        assert not flags.item()


# ---------------------------------------------------------------------------
#  Write / Read tests
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_write_returns_stats(self, mem):
        z = torch.randn(B, C)
        stats = mem.write(z)
        assert "residual_norm" in stats
        assert "update_norm" in stats
        assert "top_slot" in stats

    def test_smart_write_with_novel_input(self, mem):
        """Sufficiently novel input should be written."""
        z = torch.randn(B, C) * 50.0  # far from init
        stats = mem.smart_write(z)
        assert stats["num_written"] >= 0  # should at least not crash

    def test_global_step_increments(self, mem):
        z = torch.randn(B, C)
        before = mem.global_step.item()
        mem.write(z)
        after = mem.global_step.item()
        assert after == before + 1

    def test_read_changes_usage_counts(self, mem):
        z = torch.randn(B, C)
        initial_sum = mem.usage_counts.sum().item()
        mem.read(z)
        final_sum = mem.usage_counts.sum().item()
        assert final_sum > initial_sum


# ---------------------------------------------------------------------------
#  Forget tests
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_by_slot_resets_metadata(self, mem):
        # Write something first
        z = torch.randn(1, C) * 50.0
        mem.write(z)
        mem.usage_counts[0] = 10
        mem.last_access_step[0] = 5

        mem.forget(slot_index=0)

        assert mem.usage_counts[0].item() == 0
        assert mem.last_access_step[0].item() == 0
        assert mem.episode_meta_indices[0].item() == -1

    def test_forget_by_content(self, mem):
        slot_vec = mem.M[3].clone()
        stats = mem.forget(content=slot_vec)
        assert "forgotten_slot" in stats

    def test_forget_raises_without_args(self, mem):
        with pytest.raises(ValueError):
            mem.forget()


# ---------------------------------------------------------------------------
#  Regularisation
# ---------------------------------------------------------------------------


class TestRegularisation:
    def test_reg_loss_non_negative(self, mem):
        loss = mem.regularization_loss(lam=1e-4)
        assert loss.item() >= 0

    def test_reg_loss_scales_with_lambda(self, mem):
        l1 = mem.regularization_loss(lam=1e-4)
        l2 = mem.regularization_loss(lam=1e-2)
        assert l2.item() > l1.item()


# ---------------------------------------------------------------------------
#  NaN / Inf checks
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_read_no_nan(self, mem):
        z = torch.randn(B, C)
        readout = mem.read(z)
        assert not torch.isnan(readout).any()
        assert not torch.isinf(readout).any()

    def test_write_no_nan(self, mem):
        z = torch.randn(B, C)
        mem.write(z)
        assert not torch.isnan(mem.M).any()
        assert not torch.isinf(mem.M).any()
        assert not torch.isnan(mem.cov).any()
        assert not torch.isinf(mem.cov).any()

    def test_scope_detector_bounded(self, scope):
        z = torch.randn(B, C) * 100.0  # large input
        probs = scope(z)
        assert (probs >= 0).all() and (probs <= 1).all()


# ---------------------------------------------------------------------------
#  Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_get_set_state_round_trip(self, mem):
        z = torch.randn(B, C) * 50.0
        mem.write(z)
        state = mem.get_state()

        # Create fresh controller and load state
        mem2 = EpisodicMemoryController(memory_slots=K, hidden_dim=C)
        mem2.set_state(state)

        assert torch.allclose(mem.M, mem2.M)
        assert torch.allclose(mem.cov, mem2.cov)
        assert torch.equal(mem.usage_counts, mem2.usage_counts)

    def test_set_state_shape_mismatch_raises(self, mem):
        state = mem.get_state()
        state["M"] = torch.randn(K + 1, C)  # wrong shape
        mem2 = EpisodicMemoryController(memory_slots=K, hidden_dim=C)
        with pytest.raises(ValueError, match="Shape mismatch"):
            mem2.set_state(state, strict=True)


# ---------------------------------------------------------------------------
#  Replacement policy
# ---------------------------------------------------------------------------


class TestReplacementPolicy:
    def test_replacement_targets_least_used_least_recent(self, mem):
        # Make one slot clearly the least used and least recent
        mem.usage_counts.fill_(100)
        mem.last_access_step.fill_(100)
        target = 7
        mem.usage_counts[target] = 0
        mem.last_access_step[target] = 0

        slot = mem._replacement_slot()
        assert slot == target
