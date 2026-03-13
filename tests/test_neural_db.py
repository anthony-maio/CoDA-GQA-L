"""Tests for the NeuralDatabase stateful neural database API.

Uses CoDAGQALandmarkPerf2 directly (no HF model download) to validate
state serialization, deep-copy isolation, and helper functions.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from coda_gqa_l import CoDAGQALandmarkPerf2, CoDAGQALandmarkStatePerf2
from coda_gqa_l.neural_db import (
    _deep_copy_state,
    _select_adapter_cls,
    _state_to_cpu,
    _state_to_device,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
WINDOW = 32
ME = 8
MS = 8
B = 1
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _make_model():
    model = CoDAGQALandmarkPerf2(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        window=WINDOW,
        num_landmarks_exact=ME,
        num_landmarks_summary=MS,
        collect_metrics=True,
    )
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _prefill_model(model, seq_len=64):
    """Prefill a model and return its state."""
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)
    x = torch.randn(B, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE)
    _, state = model.prefill_chunked(x, state, block_size=32)
    return state


# ---------------------------------------------------------------------------
# Tests: State helpers
# ---------------------------------------------------------------------------


class TestStateHelpers:
    def test_state_to_cpu(self):
        model = _make_model()
        state = _prefill_model(model)
        cpu_state = _state_to_cpu(state)

        assert cpu_state.k_buf.device == torch.device("cpu")
        assert cpu_state.v_buf.device == torch.device("cpu")
        assert cpu_state.pos == state.pos
        assert cpu_state.recent_filled == state.recent_filled

    def test_state_to_device_roundtrip(self):
        model = _make_model()
        state = _prefill_model(model)
        cpu_state = _state_to_cpu(state)
        _state_to_device(cpu_state, DEVICE, DTYPE)

        assert cpu_state.k_buf.device == DEVICE
        assert cpu_state.k_buf.dtype == DTYPE

    def test_deep_copy_independence(self):
        """Modifying a deep copy must not affect the original."""
        model = _make_model()
        state = _prefill_model(model)

        copy = _deep_copy_state(state)

        # Mutate the copy
        copy.k_buf.zero_()
        copy.pos = 999999

        # Original unchanged
        assert state.k_buf.abs().sum() > 0, "Original k_buf should be non-zero"
        assert state.pos != 999999

    def test_deep_copy_preserves_values(self):
        model = _make_model()
        state = _prefill_model(model)
        copy = _deep_copy_state(state)

        assert torch.equal(state.k_buf, copy.k_buf)
        assert torch.equal(state.v_buf, copy.v_buf)
        assert torch.equal(state.allowed, copy.allowed)
        assert state.pos == copy.pos
        assert state.recent_filled == copy.recent_filled
        assert state.active_exact == copy.active_exact
        assert state.active_summary == copy.active_summary


# ---------------------------------------------------------------------------
# Tests: State serialization
# ---------------------------------------------------------------------------


class TestStateSerialization:
    def test_save_load_roundtrip(self):
        """Save state to disk and load it back, verify tensors match."""
        model = _make_model()
        state = _prefill_model(model, seq_len=96)

        # Build ingested-like dict
        ingested = {
            "states": {0: state},
            "metadata": {
                "doc_length_chars": 100,
                "num_tokens": 96,
                "model_id": "test",
                "window": WINDOW,
                "num_landmarks_exact": ME,
                "num_landmarks_summary": MS,
                "timestamp": 0.0,
            },
        }

        from coda_gqa_l.neural_db import NeuralDatabase

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            file_size = NeuralDatabase.save_state(ingested, path)
            assert file_size > 0
            assert os.path.exists(path)

            loaded = NeuralDatabase.load_state(path)

            # Verify metadata preserved
            assert loaded["metadata"]["num_tokens"] == 96
            assert loaded["metadata"]["model_id"] == "test"

            # Verify state tensors match (on CPU)
            orig_cpu = _state_to_cpu(state)
            loaded_state = loaded["states"][0]

            assert torch.equal(orig_cpu.k_buf, loaded_state.k_buf)
            assert torch.equal(orig_cpu.v_buf, loaded_state.v_buf)
            assert orig_cpu.pos == loaded_state.pos
        finally:
            os.unlink(path)

    def test_save_creates_file(self):
        model = _make_model()
        state = _prefill_model(model)
        ingested = {"states": {0: state}, "metadata": {"test": True}}

        from coda_gqa_l.neural_db import NeuralDatabase

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            NeuralDatabase.save_state(ingested, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: LlamaCoDAAdapter.set_state
# ---------------------------------------------------------------------------


class TestSetState:
    def test_set_state_exists(self):
        """LlamaCoDAAdapter (and Qwen3CoDAAdapter) should have set_state."""
        from coda_gqa_l import LlamaCoDAAdapter, Qwen3CoDAAdapter

        assert hasattr(LlamaCoDAAdapter, "set_state")
        assert hasattr(Qwen3CoDAAdapter, "set_state")

    def test_set_get_roundtrip(self):
        """set_state then get_state should return the same object."""
        from coda_gqa_l import LlamaCoDAAdapter

        adapter = LlamaCoDAAdapter(
            hidden_size=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=EMBED_DIM // NUM_HEADS,
            bounded=True,
            window=WINDOW,
            num_landmarks_exact=ME,
            num_landmarks_summary=MS,
        )
        adapter.eval()

        model = adapter.coda
        state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

        adapter.set_state(state)
        retrieved = adapter.get_state()
        assert retrieved is state


# ---------------------------------------------------------------------------
# Tests: Adapter selection
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, model_type=None, architectures=None):
        self.model_type = model_type
        self.architectures = architectures or []


class _FakeModel:
    def __init__(self, model_type=None, architectures=None):
        self.config = _FakeConfig(model_type=model_type, architectures=architectures)


class TestAdapterSelection:
    def test_selects_qwen3_adapter_for_qwen_models(self):
        from coda_gqa_l import Qwen3CoDAAdapter

        model = _FakeModel(model_type="qwen3")
        assert _select_adapter_cls(model) is Qwen3CoDAAdapter

    def test_selects_llama_adapter_for_llama_family_models(self):
        from coda_gqa_l import LlamaCoDAAdapter

        for model_type in ["llama", "mistral", "smollm", "smollm2"]:
            model = _FakeModel(model_type=model_type)
            assert _select_adapter_cls(model) is LlamaCoDAAdapter

    def test_falls_back_to_architecture_name_when_model_type_missing(self):
        from coda_gqa_l import LlamaCoDAAdapter

        model = _FakeModel(architectures=["LlamaForCausalLM"])
        assert _select_adapter_cls(model) is LlamaCoDAAdapter

    def test_raises_for_unsupported_model_family(self):
        model = _FakeModel(model_type="gpt2")

        with pytest.raises(ValueError, match="Unsupported model family"):
            _select_adapter_cls(model)
