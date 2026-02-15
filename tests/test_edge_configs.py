"""Edge-configuration tests for CoDAGQALandmarkPerf2.

Exercises unusual and boundary parameter combinations that the module
should handle without errors: zero-sized banks, tiny windows, policy
overrides, memory-init variants, and prefill-length edge cases.

All tests run on CPU with float32 and small dimensions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
import torch

from coda_gqa_l import CoDAGQALandmarkPerf2

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
DEVICE = torch.device("cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**overrides: Any) -> CoDAGQALandmarkPerf2:
    """Build a model with sensible defaults, applying *overrides* on top."""
    defaults: Dict[str, Any] = dict(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        window=64,
        num_landmarks_exact=16,
        num_landmarks_summary=16,
    )
    defaults.update(overrides)
    model = CoDAGQALandmarkPerf2(**defaults)
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _rand(batch: int, seq_len: int) -> torch.Tensor:
    return torch.randn(batch, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE)


def _prefill_then_decode(
    model: CoDAGQALandmarkPerf2,
    *,
    prefill_len: int,
    decode_steps: int = 0,
    block_size: int = 256,
    batch: int = 1,
) -> None:
    """Run prefill + decode and assert basic sanity on every output.

    Checks:
      - No exceptions raised
      - Output shape matches (B, L, embed_dim)
      - No NaN or Inf values
      - state.pos equals the expected token count
    """
    state = model.init_state(batch_size=batch, device=DEVICE, dtype=DTYPE)

    # -- Prefill --
    x = _rand(batch, prefill_len)
    y, state = model.prefill_chunked(x, state, block_size=block_size, write_cache=True)

    assert y.shape == (batch, prefill_len, EMBED_DIM), (
        f"Prefill output shape mismatch: expected {(batch, prefill_len, EMBED_DIM)}, "
        f"got {tuple(y.shape)}"
    )
    assert not torch.isnan(y).any(), "NaN detected in prefill output"
    assert not torch.isinf(y).any(), "Inf detected in prefill output"
    assert state.pos == prefill_len, (
        f"state.pos after prefill: expected {prefill_len}, got {state.pos}"
    )

    # -- Decode --
    for i in range(decode_steps):
        tok = _rand(batch, 1)
        y_dec, state = model.step(tok, state)

        assert y_dec.shape == (batch, 1, EMBED_DIM), (
            f"Decode step {i} output shape mismatch: expected {(batch, 1, EMBED_DIM)}, "
            f"got {tuple(y_dec.shape)}"
        )
        assert not torch.isnan(y_dec).any(), f"NaN detected at decode step {i}"
        assert not torch.isinf(y_dec).any(), f"Inf detected at decode step {i}"

    expected_pos = prefill_len + decode_steps
    assert state.pos == expected_pos, (
        f"state.pos after decode: expected {expected_pos}, got {state.pos}"
    )


# ---------------------------------------------------------------------------
# Tests: Zero-sized banks
# ---------------------------------------------------------------------------


class TestZeroBanks:
    """Configurations where one or both memory banks are disabled."""

    @torch.no_grad()
    def test_me_zero(self) -> None:
        """num_landmarks_exact=0: no exact bank at all."""
        model = _make_model(num_landmarks_exact=0)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_ms_zero(self) -> None:
        """num_landmarks_summary=0: no summary bank at all."""
        model = _make_model(num_landmarks_summary=0)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_both_banks_zero(self) -> None:
        """Me=0, Ms=0: only the recent window is available."""
        model = _make_model(num_landmarks_exact=0, num_landmarks_summary=0)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)


# ---------------------------------------------------------------------------
# Tests: Small / extreme windows
# ---------------------------------------------------------------------------


class TestSmallWindows:
    """Configurations with very small windows that force frequent evictions."""

    @torch.no_grad()
    def test_small_window(self) -> None:
        """window=4: frequent evictions during prefill and decode."""
        model = _make_model(window=4)
        _prefill_then_decode(model, prefill_len=64, decode_steps=20)

    @torch.no_grad()
    def test_window_one(self) -> None:
        """window=1: every new token immediately evicts the previous one."""
        model = _make_model(window=1)
        _prefill_then_decode(model, prefill_len=32, decode_steps=10)


# ---------------------------------------------------------------------------
# Tests: Write-policy and bank-behavior overrides
# ---------------------------------------------------------------------------


class TestPolicyOverrides:
    """Non-default write policies and bank behavior flags."""

    @torch.no_grad()
    def test_write_policy_none(self) -> None:
        """write_policy='none': all tokens unconditionally pass the gate."""
        model = _make_model(write_policy="none")
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_exact_refresh_on_hit(self) -> None:
        """exact_refresh_on_hit=True: hits overwrite the matched slot."""
        model = _make_model(exact_refresh_on_hit=True)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_summary_overwrite_off(self) -> None:
        """summary_overwrite_on_insert=False: new summary slots use EMA, not eta=1."""
        model = _make_model(summary_overwrite_on_insert=False)
        _prefill_then_decode(model, prefill_len=256)


# ---------------------------------------------------------------------------
# Tests: Memory initialization and masking
# ---------------------------------------------------------------------------


class TestMemoryInit:
    """Alternative memory initialization and masking configurations."""

    @torch.no_grad()
    def test_mem_init_zeros(self) -> None:
        """mem_init='zeros': bank slots start as all-zeros instead of random."""
        model = _make_model(mem_init="zeros")
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_mask_unused_memory_false(self) -> None:
        """mask_unused_memory=False: unused slots are NOT masked out of attention."""
        model = _make_model(mask_unused_memory=False)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)

    @torch.no_grad()
    def test_detach_evicted_false(self) -> None:
        """detach_evicted=False: evicted tensors keep their grad graph (training mode)."""
        model = _make_model(detach_evicted=False)
        _prefill_then_decode(model, prefill_len=256, decode_steps=10)


# ---------------------------------------------------------------------------
# Tests: Prefill-length edge cases
# ---------------------------------------------------------------------------


class TestPrefillBoundaries:
    """Prefill lengths at or near interesting eviction boundaries."""

    @torch.no_grad()
    def test_single_token_prefill(self) -> None:
        """Prefill L=1: a single token, no eviction possible."""
        model = _make_model(window=64)
        _prefill_then_decode(model, prefill_len=1)

    @torch.no_grad()
    def test_prefill_exactly_window(self) -> None:
        """Prefill L equals window size exactly (edge of eviction boundary)."""
        W = 32
        model = _make_model(window=W)
        _prefill_then_decode(model, prefill_len=W, decode_steps=5)

    @torch.no_grad()
    def test_prefill_window_plus_one(self) -> None:
        """Prefill L = window + 1: exactly one token is evicted."""
        W = 32
        model = _make_model(window=W)
        _prefill_then_decode(model, prefill_len=W + 1, decode_steps=5)

    @torch.no_grad()
    def test_large_block_size(self) -> None:
        """block_size larger than L: entire prefill processed in one chunk."""
        model = _make_model(window=64)
        _prefill_then_decode(model, prefill_len=64, block_size=1024)
