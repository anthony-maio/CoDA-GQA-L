"""Determinism tests for CoDAGQALandmarkPerf2.

Validates that outputs are reproducible across runs under identical
conditions (same seed, same input, same device).

Key invariants:
- write_cache=False is always bit-exact (no scatter ops involved).
- write_cache=True is bit-exact on CPU (scatter_reduce_ / scatter_add_
  are deterministic on CPU but not on CUDA).
- Different random seeds must produce different model weights and
  therefore different outputs.
"""

from __future__ import annotations

import copy

import pytest
import torch

from coda_gqa_l import CoDAGQALandmarkPerf2

# ---------------------------------------------------------------------------
# Shared constants (small dims for speed)
# ---------------------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
WINDOW = 64
ME = 16
MS = 16
DEVICE = torch.device("cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(seed: int = 42) -> CoDAGQALandmarkPerf2:
    """Construct a model with a fixed seed, in eval mode on CPU/float32."""
    torch.manual_seed(seed)
    model = CoDAGQALandmarkPerf2(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        window=WINDOW,
        num_landmarks_exact=ME,
        num_landmarks_summary=MS,
    )
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _make_input(seed: int, batch: int, seq_len: int) -> torch.Tensor:
    """Generate a deterministic random input tensor."""
    torch.manual_seed(seed)
    return torch.randn(batch, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE)


def _init_state(
    model: CoDAGQALandmarkPerf2, batch: int, seed: int = 99
) -> "CoDAGQALandmarkStatePerf2":
    """Init state with a fixed seed (controls random_normal mem init)."""
    torch.manual_seed(seed)
    return model.init_state(batch_size=batch, device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# 1. test_prefill_no_write_deterministic
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_no_write_deterministic():
    """prefill_chunked(write_cache=False) twice with same seed/input
    must produce bit-exact outputs. No scatter ops are involved so this
    holds on any device."""

    B, L = 2, 256
    model_seed, input_seed, state_seed = 42, 7, 99

    # --- Run A ---
    model_a = _make_model(model_seed)
    x_a = _make_input(input_seed, B, L)
    state_a = _init_state(model_a, B, state_seed)
    y_a, _ = model_a.prefill_chunked(
        x_a, state_a, block_size=128, write_cache=False, return_outputs=True
    )

    # --- Run B ---
    model_b = _make_model(model_seed)
    x_b = _make_input(input_seed, B, L)
    state_b = _init_state(model_b, B, state_seed)
    y_b, _ = model_b.prefill_chunked(
        x_b, state_b, block_size=128, write_cache=False, return_outputs=True
    )

    assert y_a is not None and y_b is not None
    assert torch.equal(y_a, y_b), (
        f"prefill(write_cache=False) outputs differ. "
        f"Max abs diff = {(y_a - y_b).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. test_prefill_write_deterministic_cpu
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_write_deterministic_cpu():
    """prefill_chunked(write_cache=True) twice on CPU with same seed/input
    must produce bit-exact outputs. CPU scatter ops are deterministic."""

    B, L = 2, 256
    model_seed, input_seed, state_seed = 42, 7, 99

    # --- Run A ---
    model_a = _make_model(model_seed)
    x_a = _make_input(input_seed, B, L)
    state_a = _init_state(model_a, B, state_seed)
    y_a, _ = model_a.prefill_chunked(
        x_a, state_a, block_size=128, write_cache=True, return_outputs=True
    )

    # --- Run B ---
    model_b = _make_model(model_seed)
    x_b = _make_input(input_seed, B, L)
    state_b = _init_state(model_b, B, state_seed)
    y_b, _ = model_b.prefill_chunked(
        x_b, state_b, block_size=128, write_cache=True, return_outputs=True
    )

    assert y_a is not None and y_b is not None
    assert torch.equal(y_a, y_b), (
        f"prefill(write_cache=True) CPU outputs differ. "
        f"Max abs diff = {(y_a - y_b).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 3. test_decode_deterministic_cpu
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_decode_deterministic_cpu():
    """50 decode steps twice on CPU with same seed/input.
    All per-step outputs must be bit-exact."""

    B = 1
    NUM_DECODE_STEPS = 50
    model_seed, prefill_seed, state_seed = 42, 7, 99

    def _run_decode():
        model = _make_model(model_seed)
        # Short prefill to populate the buffer.
        x_pf = _make_input(prefill_seed, B, 128)
        state = _init_state(model, B, state_seed)
        _, state = model.prefill_chunked(
            x_pf, state, block_size=64, write_cache=True, return_outputs=False
        )

        outputs = []
        for i in range(NUM_DECODE_STEPS):
            # Use a unique but deterministic seed per step.
            x_tok = _make_input(1000 + i, B, 1)
            y, state = model.step(x_tok, state)
            outputs.append(y.clone())
        return outputs

    outs_a = _run_decode()
    outs_b = _run_decode()

    assert len(outs_a) == len(outs_b) == NUM_DECODE_STEPS
    for i, (ya, yb) in enumerate(zip(outs_a, outs_b)):
        assert torch.equal(ya, yb), (
            f"Decode step {i} outputs differ. "
            f"Max abs diff = {(ya - yb).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. test_different_seeds_differ
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_different_seeds_differ():
    """Two runs with different model seeds should produce different outputs.
    Sanity check that random initialization actually matters."""

    B, L = 1, 128
    input_seed, state_seed = 7, 99

    # Model A (seed 42)
    model_a = _make_model(seed=42)
    x_a = _make_input(input_seed, B, L)
    state_a = _init_state(model_a, B, state_seed)
    y_a, _ = model_a.prefill_chunked(
        x_a, state_a, block_size=128, write_cache=False, return_outputs=True
    )

    # Model B (seed 123) -- different weights
    model_b = _make_model(seed=123)
    x_b = _make_input(input_seed, B, L)
    state_b = _init_state(model_b, B, state_seed)
    y_b, _ = model_b.prefill_chunked(
        x_b, state_b, block_size=128, write_cache=False, return_outputs=True
    )

    assert y_a is not None and y_b is not None
    assert not torch.equal(y_a, y_b), (
        "Outputs from models with different seeds are identical -- "
        "random initialization has no effect?"
    )


# ---------------------------------------------------------------------------
# 5. test_state_deterministic
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_state_deterministic():
    """After identical prefill runs, scalar state fields (pos, recent_filled,
    recent_idx) and buffer contents must match exactly."""

    B, L = 2, 256
    model_seed, input_seed, state_seed = 42, 7, 99

    def _run_prefill():
        model = _make_model(model_seed)
        x = _make_input(input_seed, B, L)
        state = _init_state(model, B, state_seed)
        _, state = model.prefill_chunked(
            x, state, block_size=128, write_cache=True, return_outputs=False
        )
        return state

    state_a = _run_prefill()
    state_b = _run_prefill()

    # Scalar fields
    assert state_a.pos == state_b.pos, (
        f"pos mismatch: {state_a.pos} vs {state_b.pos}"
    )
    assert state_a.recent_filled == state_b.recent_filled, (
        f"recent_filled mismatch: {state_a.recent_filled} vs {state_b.recent_filled}"
    )
    assert state_a.recent_idx == state_b.recent_idx, (
        f"recent_idx mismatch: {state_a.recent_idx} vs {state_b.recent_idx}"
    )

    # Buffer contents
    assert torch.equal(state_a.k_buf, state_b.k_buf), (
        f"k_buf differs. Max abs diff = "
        f"{(state_a.k_buf - state_b.k_buf).abs().max().item():.2e}"
    )
    assert torch.equal(state_a.v_buf, state_b.v_buf), (
        f"v_buf differs. Max abs diff = "
        f"{(state_a.v_buf - state_b.v_buf).abs().max().item():.2e}"
    )
    assert torch.equal(state_a.allowed, state_b.allowed), "allowed mask differs"
    assert torch.equal(state_a.g_recent, state_b.g_recent), (
        f"g_recent differs. Max abs diff = "
        f"{(state_a.g_recent - state_b.g_recent).abs().max().item():.2e}"
    )
    assert torch.equal(state_a.mem_last_exact, state_b.mem_last_exact), (
        "mem_last_exact differs"
    )


# ---------------------------------------------------------------------------
# 6. test_prefill_then_decode_deterministic
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_then_decode_deterministic():
    """Full pipeline: prefill L=512 then 20 decode steps, twice.
    All prefill and decode outputs must be bit-exact on CPU."""

    B = 1
    L_PREFILL = 512
    NUM_DECODE_STEPS = 20
    model_seed, input_seed, state_seed = 42, 7, 99

    def _run_full_pipeline():
        model = _make_model(model_seed)
        x_pf = _make_input(input_seed, B, L_PREFILL)
        state = _init_state(model, B, state_seed)

        y_pf, state = model.prefill_chunked(
            x_pf, state, block_size=256, write_cache=True, return_outputs=True
        )

        decode_outputs = []
        for i in range(NUM_DECODE_STEPS):
            x_tok = _make_input(2000 + i, B, 1)
            y_dec, state = model.step(x_tok, state)
            decode_outputs.append(y_dec.clone())

        return y_pf, decode_outputs, state

    y_pf_a, dec_a, state_a = _run_full_pipeline()
    y_pf_b, dec_b, state_b = _run_full_pipeline()

    # Prefill outputs
    assert y_pf_a is not None and y_pf_b is not None
    assert torch.equal(y_pf_a, y_pf_b), (
        f"Prefill outputs differ. "
        f"Max abs diff = {(y_pf_a - y_pf_b).abs().max().item():.2e}"
    )

    # Decode outputs
    assert len(dec_a) == len(dec_b) == NUM_DECODE_STEPS
    for i, (ya, yb) in enumerate(zip(dec_a, dec_b)):
        assert torch.equal(ya, yb), (
            f"Decode step {i} outputs differ. "
            f"Max abs diff = {(ya - yb).abs().max().item():.2e}"
        )

    # Final state consistency
    assert state_a.pos == state_b.pos
    assert state_a.recent_filled == state_b.recent_filled
    assert state_a.recent_idx == state_b.recent_idx
