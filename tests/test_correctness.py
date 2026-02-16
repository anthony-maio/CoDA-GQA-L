"""Correctness tests for CoDAGQALandmarkPerf2.

Validates output shapes, dtypes, numerical sanity (no NaN/Inf), and
end-to-end prefill-then-decode workflows on CPU with small dimensions.
"""

from __future__ import annotations

import pytest
import torch

from coda_gqa_l import CoDAGQALandmarkPerf2

# ---------------------------------------------------------------------------
# Shared constants
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
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_model() -> CoDAGQALandmarkPerf2:
    """Return a freshly-constructed model in eval mode on CPU/float32."""
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


@pytest.fixture()
def model():
    """Per-test model instance (fresh parameters and empty caches)."""
    return _make_model()


def _rand_input(batch: int, seq_len: int) -> torch.Tensor:
    """Random input tensor on the shared device/dtype."""
    return torch.randn(batch, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# 1. test_prefill_output_shape
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_output_shape(model: CoDAGQALandmarkPerf2):
    """prefill_chunked with L=256 returns output shape (B, L, D)."""
    B, L = 1, 256
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y, state = model.prefill_chunked(x, state, block_size=256, write_cache=True, return_outputs=True)

    assert y is not None
    assert y.shape == (B, L, EMBED_DIM)


# ---------------------------------------------------------------------------
# 2. test_prefill_no_write_output_shape
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_no_write_output_shape(model: CoDAGQALandmarkPerf2):
    """prefill_chunked with write_cache=False still returns correct shape."""
    B, L = 1, 256
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y, state_out = model.prefill_chunked(
        x, state, block_size=128, write_cache=False, return_outputs=True,
    )

    assert y is not None
    assert y.shape == (B, L, EMBED_DIM)
    # State position should not advance when write_cache is False.
    assert state_out.pos == 0


# ---------------------------------------------------------------------------
# 3. test_prefill_return_none
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_return_none(model: CoDAGQALandmarkPerf2):
    """prefill_chunked with return_outputs=False returns None for output."""
    B, L = 1, 256
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y, state = model.prefill_chunked(
        x, state, block_size=128, write_cache=True, return_outputs=False,
    )

    assert y is None


# ---------------------------------------------------------------------------
# 4. test_decode_step_output_shape
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_decode_step_output_shape(model: CoDAGQALandmarkPerf2):
    """step() returns (B, 1, D) output."""
    B = 1
    # Seed the state with a short prefill so the buffer has valid tokens.
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)
    x_prefill = _rand_input(B, 32)
    _, state = model.prefill_chunked(x_prefill, state, block_size=32, write_cache=True, return_outputs=False)

    x_tok = _rand_input(B, 1)
    y, state = model.step(x_tok, state)

    assert y.shape == (B, 1, EMBED_DIM)


# ---------------------------------------------------------------------------
# 5. test_prefill_then_decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_then_decode(model: CoDAGQALandmarkPerf2):
    """Prefill L=512, then 10 decode steps -- all shapes correct."""
    B, L_prefill = 1, 512
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    x_prefill = _rand_input(B, L_prefill)
    y_pf, state = model.prefill_chunked(
        x_prefill, state, block_size=256, write_cache=True, return_outputs=True,
    )
    assert y_pf is not None
    assert y_pf.shape == (B, L_prefill, EMBED_DIM)

    for i in range(10):
        x_tok = _rand_input(B, 1)
        y_dec, state = model.step(x_tok, state)
        assert y_dec.shape == (B, 1, EMBED_DIM), f"Decode step {i} shape mismatch"

    # Position should reflect all tokens processed.
    assert state.pos == L_prefill + 10


# ---------------------------------------------------------------------------
# 6. test_no_nan_inf_prefill
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_no_nan_inf_prefill(model: CoDAGQALandmarkPerf2):
    """Prefill L=1024 -- output contains no NaN or Inf."""
    B, L = 1, 1024
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y, _ = model.prefill_chunked(x, state, block_size=256, write_cache=True, return_outputs=True)

    assert y is not None
    assert not torch.isnan(y).any(), "NaN detected in prefill output"
    assert not torch.isinf(y).any(), "Inf detected in prefill output"


# ---------------------------------------------------------------------------
# 7. test_no_nan_inf_decode_long
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_no_nan_inf_decode_long(model: CoDAGQALandmarkPerf2):
    """200 decode steps -- no NaN or Inf in any output."""
    B = 1
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    # Small prefill to warm up the buffer.
    x_pf = _rand_input(B, 64)
    _, state = model.prefill_chunked(x_pf, state, block_size=64, write_cache=True, return_outputs=False)

    for i in range(200):
        x_tok = _rand_input(B, 1)
        y, state = model.step(x_tok, state)
        assert not torch.isnan(y).any(), f"NaN at decode step {i}"
        assert not torch.isinf(y).any(), f"Inf at decode step {i}"


# ---------------------------------------------------------------------------
# 8. test_output_dtype_matches_input
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_output_dtype_matches_input(model: CoDAGQALandmarkPerf2):
    """Output dtype should match input dtype (float32)."""
    B, L = 1, 128
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y_pf, state = model.prefill_chunked(x, state, block_size=64, write_cache=True, return_outputs=True)
    assert y_pf.dtype == DTYPE, f"Prefill dtype {y_pf.dtype} != {DTYPE}"

    x_tok = _rand_input(B, 1)
    y_dec, _ = model.step(x_tok, state)
    assert y_dec.dtype == DTYPE, f"Decode dtype {y_dec.dtype} != {DTYPE}"


# ---------------------------------------------------------------------------
# 9. test_prefill_various_block_sizes
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("block_size", [64, 128, 256, 512, 1024])
def test_prefill_various_block_sizes(block_size: int):
    """Prefill L=1024 with different block_size values -- all produce same shape."""
    B, L = 1, 1024
    model = _make_model()
    x = _rand_input(B, L)
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    y, state = model.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=True)

    assert y is not None
    assert y.shape == (B, L, EMBED_DIM), f"Shape mismatch with block_size={block_size}"
    assert not torch.isnan(y).any(), f"NaN with block_size={block_size}"
    assert not torch.isinf(y).any(), f"Inf with block_size={block_size}"


# ---------------------------------------------------------------------------
# 10. test_batch_size_two
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_batch_size_two():
    """B=2, prefill + decode -- verify shapes."""
    B = 2
    L_prefill = 256
    model = _make_model()
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)

    x_pf = _rand_input(B, L_prefill)
    y_pf, state = model.prefill_chunked(
        x_pf, state, block_size=128, write_cache=True, return_outputs=True,
    )
    assert y_pf is not None
    assert y_pf.shape == (B, L_prefill, EMBED_DIM)

    for i in range(5):
        x_tok = _rand_input(B, 1)
        y_dec, state = model.step(x_tok, state)
        assert y_dec.shape == (B, 1, EMBED_DIM), f"Decode step {i} shape mismatch for B=2"

    assert state.pos == L_prefill + 5


# ---------------------------------------------------------------------------
# 11. test_head_stacked_sdpa_matches_separate (GQA alignment regression)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_head_stacked_sdpa_matches_separate():
    """Head-stacked two-stream SDPA must match two separate SDPA calls.

    Regression test for GQA head alignment bug: repeat_kv(k, 2*groups)
    breaks the GQA head mapping when num_heads > num_kv_heads.  The
    correct approach is repeat_kv(k, groups) then cat for the two streams.
    """
    import torch.nn.functional as F
    from coda_gqa_l.primitives import repeat_kv

    torch.manual_seed(42)
    B, L, Dh = 1, 32, 32
    H, Hkv = 4, 2
    groups = H // Hkv

    model = _make_model()

    # Random Q, K, V in head-space
    q_sig = torch.randn(B, H, L, Dh)
    q_noise = torch.randn(B, H, L, Dh)
    k = torch.randn(B, Hkv, L, Dh)
    v = torch.randn(B, Hkv, L, Dh)
    lam = torch.sigmoid(torch.randn(B, H, L, 1))

    # --- Reference: two separate SDPA calls (like CoDAGQA.forward) ---
    k_rep_ref = repeat_kv(k, groups)   # (B, H, L, Dh)
    v_rep_ref = repeat_kv(v, groups)
    out_sig_ref = F.scaled_dot_product_attention(q_sig, k_rep_ref, v_rep_ref, is_causal=True)
    out_noise_ref = F.scaled_dot_product_attention(q_noise, k_rep_ref, v_rep_ref, is_causal=True)
    out_ref = out_sig_ref - lam * out_noise_ref
    out_ref = model.head_norm(out_ref)

    # --- Under test: head-stacked SDPA ---
    out_stacked = model._sdpa_stacked_two_stream(
        q=q_sig, q_noise=q_noise, k=k, v=v,
        attn_mask=None, lam=lam, is_causal=True,
    )

    diff = (out_stacked - out_ref).abs().max().item()
    assert diff < 1e-4, (
        f"Head-stacked SDPA output differs from separate calls by {diff:.6f}. "
        f"GQA head alignment may be broken."
    )


# ---------------------------------------------------------------------------
# 12. test_summary_bank_hf_zero (Phase-Safe EMA invariant)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_summary_bank_hf_zero():
    """Summary bank keys maintain zero HF band after Phase-Safe EMA updates.

    The high-frequency key dimensions (0..lf_start-1) are initialized to zero
    and must remain zero after evictions trigger EMA blending in the summary
    bank.  This validates the Phase-Safe EMA invariant: only the LF band is
    blended, HF is never written.
    """
    B = 1
    model = _make_model()
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)
    lf = model.lf_start
    s = model.window + model.Me
    e = model.Lbuf

    # HF band should be zero at init
    assert state.k_buf[:, :, s:e, :lf].abs().max() == 0, "HF should be zero at init"

    # Prefill enough tokens to trigger evictions into summary bank
    x = _rand_input(B, 512)
    _, state = model.prefill_chunked(x, state, block_size=128, write_cache=True, return_outputs=False)

    hf_max = state.k_buf[:, :, s:e, :lf].abs().max().item()
    assert hf_max < 1e-6, (
        f"Summary bank HF band should remain zero after Phase-Safe EMA, "
        f"got max abs = {hf_max}"
    )
