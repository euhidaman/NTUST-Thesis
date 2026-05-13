"""
Integration tests for episodic memory inside EmberVLM.

Tests cover:
  - use_episodic_memory=False reproduces baseline (no episodic modules)
  - use_episodic_memory=True constructs without error
  - Forward pass with memory enabled runs without error
  - save / load memory state round-trip through EmberVLM public API
"""

import pytest
import tempfile
import os
import torch

from embervlm.models.embervlm import EmberVLM, EmberVLMConfig


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_config(use_memory: bool, **overrides) -> EmberVLMConfig:
    """Create a small test-friendly config."""
    cfg = EmberVLMConfig(
        vision_backbone="repvit",
        language_backbone="tinyllm",
        use_episodic_memory=use_memory,
        memory_slots=32,
        memory_addressing="gaussian",
        memory_alpha=1.0,
        memory_temperature=0.1,
        memory_variance=1.0,
        novelty_threshold_novel=0.7,
        novelty_threshold_similar=0.2,
        freeze_vision=True,
        freeze_language_base=True,
        use_pretrained_language=False,  # avoid network fetch in tests
        vision_pretrained=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_dummy_inputs(config: EmberVLMConfig, batch: int = 2):
    """Return minimal dummy tensors for a forward pass."""
    B = batch
    pixel_values = torch.randn(B, 3, config.image_size, config.image_size)
    input_ids = torch.randint(0, config.language_vocab_size, (B, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
#  Baseline flag test
# ---------------------------------------------------------------------------


class TestBaselineFlag:
    def test_no_memory_modules_when_disabled(self):
        cfg = _make_config(use_memory=False)
        model = EmberVLM(cfg)
        assert not hasattr(model, "episodic_memory") or model.episodic_memory is None
        assert not hasattr(model, "scope_detector") or model.scope_detector is None

    def test_memory_modules_created_when_enabled(self):
        cfg = _make_config(use_memory=True)
        model = EmberVLM(cfg)
        assert model.episodic_memory is not None
        assert model.scope_detector is not None
        assert model.episodic_memory.K == 32
        assert model.episodic_memory.C == cfg.language_hidden_size


# ---------------------------------------------------------------------------
#  Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Full forward needs more memory than small CPU runners allow"
    )
    def test_forward_with_memory_runs(self):
        """Forward pass completes without errors when memory is enabled."""
        cfg = _make_config(use_memory=True)
        model = EmberVLM(cfg)
        model.eval()
        inputs = _make_dummy_inputs(cfg)
        with torch.no_grad():
            _ = model(**inputs)

    def test_forward_without_memory_runs(self):
        """Forward pass completes without errors when memory is disabled."""
        cfg = _make_config(use_memory=False)
        model = EmberVLM(cfg)
        model.eval()
        inputs = _make_dummy_inputs(cfg)
        with torch.no_grad():
            _ = model(**inputs)


# ---------------------------------------------------------------------------
#  Public API: save / load memory state
# ---------------------------------------------------------------------------


class TestMemoryStateAPI:
    def test_save_load_round_trip(self):
        cfg = _make_config(use_memory=True)
        model = EmberVLM(cfg)

        # Write some data into memory
        z = torch.randn(2, cfg.language_hidden_size)
        model.update_memory(z)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mem_state.pt")
            model.save_memory_state(path)
            assert os.path.isfile(path)

            # Create a fresh model and load the state
            model2 = EmberVLM(cfg)
            model2.load_memory_state(path)

            assert torch.allclose(
                model.episodic_memory.M,
                model2.episodic_memory.M,
            )

    def test_update_memory_increments_step(self):
        cfg = _make_config(use_memory=True)
        model = EmberVLM(cfg)
        before = model.episodic_memory.global_step.item()
        z = torch.randn(2, cfg.language_hidden_size)
        model.update_memory(z)
        after = model.episodic_memory.global_step.item()
        assert after > before

    def test_forget_memory_works(self):
        cfg = _make_config(use_memory=True)
        model = EmberVLM(cfg)
        stats = model.forget_memory(slot_index=0)
        assert "forgotten_slot" in stats
        assert stats["forgotten_slot"] == 0

    def test_noop_when_memory_disabled(self):
        cfg = _make_config(use_memory=False)
        model = EmberVLM(cfg)
        # These should be no-ops (not raise)
        z = torch.randn(2, cfg.language_hidden_size)
        model.update_memory(z)
        model.forget_memory(slot_index=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mem_state.pt")
            model.save_memory_state(path)
            # File should not be created when no memory exists
            assert not os.path.isfile(path)
