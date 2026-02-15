"""State invariant tests for CoDAGQALandmarkPerf2.

Validates that internal state fields maintain their required invariants
after arbitrary sequences of prefill and decode operations.  These tests
catch subtle bugs in the ring buffer, memory bank update logic, LRU
timestamps, and norm caches.

All tests run on CPU with float32 and use ``@torch.no_grad()``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from coda_gqa_l import CoDAGQALandmarkPerf2
from coda_gqa_l.state import CoDAGQALandmarkStatePerf2

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
WINDOW = 32
ME = 16
MS = 16
DEVICE = torch.device("cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**overrides) -> CoDAGQALandmarkPerf2:
    """Return a freshly-constructed model in eval mode on CPU/float32."""
    kwargs = dict(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        window=WINDOW,
        num_landmarks_exact=ME,
        num_landmarks_summary=MS,
    )
    kwargs.update(overrides)
    model = CoDAGQALandmarkPerf2(**kwargs)
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _make_model_and_state(
    batch_size: int = 1, **model_overrides
) -> tuple[CoDAGQALandmarkPerf2, CoDAGQALandmarkStatePerf2]:
    """Return (model, fresh_state) pair."""
    model = _make_model(**model_overrides)
    state = model.init_state(batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    return model, state


def _rand_input(batch: int, seq_len: int) -> torch.Tensor:
    """Random non-zero input tensor on the shared device/dtype."""
    return torch.randn(batch, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE)


def _prefill(
    model: CoDAGQALandmarkPerf2,
    state: CoDAGQALandmarkStatePerf2,
    seq_len: int,
    batch: int = 1,
    block_size: int = 64,
) -> CoDAGQALandmarkStatePerf2:
    """Run a prefill of *seq_len* tokens and return the updated state."""
    x = _rand_input(batch, seq_len)
    _, state = model.prefill_chunked(
        x, state, block_size=block_size, write_cache=True, return_outputs=False,
    )
    return state


def _decode_steps(
    model: CoDAGQALandmarkPerf2,
    state: CoDAGQALandmarkStatePerf2,
    num_steps: int,
    batch: int = 1,
) -> CoDAGQALandmarkStatePerf2:
    """Run *num_steps* single-token decode steps and return the updated state."""
    for _ in range(num_steps):
        x = _rand_input(batch, 1)
        _, state = model.step(x, state)
    return state


# ---------------------------------------------------------------------------
# 1. test_pos_increments_correctly
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pos_increments_correctly():
    """After prefill L=100, pos==100.  After 50 more decode steps, pos==150."""
    model, state = _make_model_and_state()

    state = _prefill(model, state, seq_len=100)
    assert state.pos == 100, f"Expected pos=100 after prefill, got {state.pos}"

    state = _decode_steps(model, state, num_steps=50)
    assert state.pos == 150, f"Expected pos=150 after decode, got {state.pos}"


# ---------------------------------------------------------------------------
# 2. test_recent_filled_capped_at_window
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_recent_filled_capped_at_window():
    """After prefill L >> W, recent_filled should equal W (never exceeds)."""
    model, state = _make_model_and_state()
    W = model.window

    # Prefill with much more than the window size.
    state = _prefill(model, state, seq_len=W * 5)
    assert state.recent_filled == W, (
        f"recent_filled should be capped at W={W}, got {state.recent_filled}"
    )

    # Additional decode steps should not push it beyond W.
    state = _decode_steps(model, state, num_steps=W * 2)
    assert state.recent_filled == W, (
        f"recent_filled should remain W={W} after decode, got {state.recent_filled}"
    )


# ---------------------------------------------------------------------------
# 3. test_recent_idx_in_bounds
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_recent_idx_in_bounds():
    """After many operations, recent_idx is always in [0, W)."""
    model, state = _make_model_and_state()
    W = model.window

    # After prefill, _set_recent_from_sequence resets recent_idx to 0.
    state = _prefill(model, state, seq_len=W * 3)
    assert 0 <= state.recent_idx < W, (
        f"recent_idx={state.recent_idx} out of bounds [0, {W}) after prefill"
    )

    # After many decode steps (enough to wrap the ring multiple times).
    for i in range(W * 3):
        x = _rand_input(1, 1)
        _, state = model.step(x, state)
        assert 0 <= state.recent_idx < W, (
            f"recent_idx={state.recent_idx} out of bounds [0, {W}) at decode step {i}"
        )


# ---------------------------------------------------------------------------
# 4. test_allowed_recent_after_prefill
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_allowed_recent_after_prefill():
    """After prefill L >= W, all recent slots (0..W-1) should be allowed."""
    model, state = _make_model_and_state()
    W = model.window

    state = _prefill(model, state, seq_len=W * 2)

    recent_allowed = state.allowed[:, :W]
    assert recent_allowed.all(), (
        f"Not all recent slots are allowed after prefill. "
        f"Allowed count: {recent_allowed.sum().item()}/{W}"
    )


# ---------------------------------------------------------------------------
# 5. test_allowed_no_overlap
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_allowed_no_overlap():
    """The total allowed slot count never exceeds Lbuf and, when few tokens
    have been processed, never exceeds pos."""
    model, state = _make_model_and_state()
    Lbuf = model.Lbuf

    # After a short prefill (fewer tokens than Lbuf), allowed <= pos.
    short_len = Lbuf // 2
    state = _prefill(model, state, seq_len=short_len)
    total_allowed = int(state.allowed.sum().item())
    assert total_allowed <= state.pos, (
        f"allowed.sum()={total_allowed} exceeds pos={state.pos} "
        f"after short prefill of {short_len} tokens"
    )

    # After a long prefill, allowed never exceeds Lbuf.
    model2, state2 = _make_model_and_state()
    state2 = _prefill(model2, state2, seq_len=Lbuf * 5)
    total_allowed2 = int(state2.allowed.sum().item())
    assert total_allowed2 <= Lbuf, (
        f"allowed.sum()={total_allowed2} exceeds Lbuf={Lbuf}"
    )

    # After additional decode steps, still bounded.
    state2 = _decode_steps(model2, state2, num_steps=100)
    total_allowed3 = int(state2.allowed.sum().item())
    assert total_allowed3 <= Lbuf, (
        f"allowed.sum()={total_allowed3} exceeds Lbuf={Lbuf} after decode"
    )


# ---------------------------------------------------------------------------
# 6. test_exact_bank_allowed_grows
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_exact_bank_allowed_grows():
    """Exact bank starts with no allowed slots.  After prefill with evictions,
    at least some exact slots become allowed."""
    model, state = _make_model_and_state()
    W = model.window
    Me = model.Me

    # Verify initial state: no exact bank slots allowed.
    exact_allowed_init = state.allowed[:, W:W + Me]
    assert not exact_allowed_init.any(), (
        "Exact bank should start with no allowed slots"
    )

    # Prefill enough to cause evictions from the recent window into the
    # exact bank.  Use write_policy="none" to guarantee all evicted tokens
    # pass the gate check (no threshold filtering).
    model_ng, state_ng = _make_model_and_state(write_policy="none")
    state_ng = _prefill(model_ng, state_ng, seq_len=W * 4)

    exact_allowed = state_ng.allowed[:, W:W + Me]
    assert exact_allowed.any(), (
        "Expected at least some exact bank slots to become allowed after "
        f"prefill of {W * 4} tokens (window={W}, Me={Me})"
    )


# ---------------------------------------------------------------------------
# 7. test_summary_bank_allowed_grows
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_summary_bank_allowed_grows():
    """Summary bank starts with no allowed slots.  After prefill with evictions,
    at least some summary slots become allowed."""
    model, state = _make_model_and_state()
    W = model.window
    Ms = model.Ms

    # Verify initial state: no summary bank slots allowed.
    summary_allowed_init = state.allowed[:, W + ME:W + ME + Ms]
    assert not summary_allowed_init.any(), (
        "Summary bank should start with no allowed slots"
    )

    # Use write_policy="none" to guarantee evicted tokens pass the gate.
    model_ng, state_ng = _make_model_and_state(write_policy="none")
    state_ng = _prefill(model_ng, state_ng, seq_len=W * 4)

    summary_allowed = state_ng.allowed[:, W + ME:W + ME + Ms]
    assert summary_allowed.any(), (
        "Expected at least some summary bank slots to become allowed after "
        f"prefill of {W * 4} tokens (window={W}, Ms={Ms})"
    )


# ---------------------------------------------------------------------------
# 8. test_lru_timestamps_positive_when_used
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_lru_timestamps_positive_when_used():
    """For allowed exact bank slots, mem_last_exact should be >= 0.
    For disallowed slots, timestamps should be 0 (initial value)."""
    # Use write_policy="none" to ensure some exact bank activity.
    model, state = _make_model_and_state(write_policy="none")
    W = model.window
    Me = model.Me

    # Prefill enough to cause evictions and exact bank writes.
    state = _prefill(model, state, seq_len=W * 4)

    exact_allowed = state.allowed[:, W:W + Me]  # (B, Me)
    timestamps = state.mem_last_exact  # (B, Me)

    # Allowed slots should have non-negative timestamps (they were written
    # with pos_evict values, which are >= 0).
    for b in range(exact_allowed.size(0)):
        for j in range(Me):
            if exact_allowed[b, j]:
                assert timestamps[b, j] >= 0, (
                    f"Allowed exact slot [{b},{j}] has negative timestamp "
                    f"{timestamps[b, j].item()}"
                )
            else:
                assert timestamps[b, j] == 0, (
                    f"Disallowed exact slot [{b},{j}] has non-zero timestamp "
                    f"{timestamps[b, j].item()}"
                )


# ---------------------------------------------------------------------------
# 9. test_norm_cache_matches_actual
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_norm_cache_matches_actual():
    """After prefill + decode, cached V-routing norm tensors should match
    F.normalize of the corresponding v_buf slices."""
    model, state = _make_model_and_state(write_policy="none")
    W = model.window
    Me = model.Me
    Ms = model.Ms

    # Prefill + decode to exercise both code paths.
    state = _prefill(model, state, seq_len=W * 3)
    state = _decode_steps(model, state, num_steps=W)

    # Exact bank V-routing norm cache.
    if Me > 0:
        actual_exact_v = state.v_buf[:, :, W:W + Me, :]
        expected_exact_norm = F.normalize(actual_exact_v, dim=-1, eps=1e-6)
        assert state._exact_v_norm is not None, (
            "_exact_v_norm should not be None when Me > 0"
        )
        assert torch.allclose(state._exact_v_norm, expected_exact_norm, atol=1e-5), (
            "Exact bank V-routing norm cache does not match F.normalize of v_buf exact slice. "
            f"Max diff: {(state._exact_v_norm - expected_exact_norm).abs().max().item()}"
        )

    # Summary bank V-routing norm cache.
    if Ms > 0:
        actual_sum_v = state.v_buf[:, :, W + Me:W + Me + Ms, :]
        expected_sum_norm = F.normalize(actual_sum_v, dim=-1, eps=1e-6)
        assert state._sum_v_norm is not None, (
            "_sum_v_norm should not be None when Ms > 0"
        )
        assert torch.allclose(state._sum_v_norm, expected_sum_norm, atol=1e-5), (
            "Summary bank V-routing norm cache does not match F.normalize of v_buf summary slice. "
            f"Max diff: {(state._sum_v_norm - expected_sum_norm).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 10. test_norm_cache_exists
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_norm_cache_exists():
    """After init_state, both _exact_v_norm and _sum_v_norm are not None
    (when Me > 0 and Ms > 0 respectively)."""
    model, state = _make_model_and_state()

    if model.Me > 0:
        assert state._exact_v_norm is not None, (
            "_exact_v_norm should be initialized when Me > 0"
        )
        expected_shape = (1, NUM_KV_HEADS, ME, model.head_dim)
        assert state._exact_v_norm.shape == expected_shape, (
            f"_exact_v_norm shape {state._exact_v_norm.shape} != {expected_shape}"
        )

    if model.Ms > 0:
        assert state._sum_v_norm is not None, (
            "_sum_v_norm should be initialized when Ms > 0"
        )
        expected_shape = (1, NUM_KV_HEADS, MS, model.head_dim)
        assert state._sum_v_norm.shape == expected_shape, (
            f"_sum_v_norm shape {state._sum_v_norm.shape} != {expected_shape}"
        )


@torch.no_grad()
def test_norm_cache_none_when_zero_banks():
    """When Me=0 or Ms=0, the corresponding norm cache should be None."""
    # No exact bank.
    model_no_exact, state_no_exact = _make_model_and_state(num_landmarks_exact=0)
    assert state_no_exact._exact_v_norm is None, (
        "_exact_v_norm should be None when Me=0"
    )

    # No summary bank.
    model_no_sum, state_no_sum = _make_model_and_state(num_landmarks_summary=0)
    assert state_no_sum._sum_v_norm is None, (
        "_sum_v_norm should be None when Ms=0"
    )


# ---------------------------------------------------------------------------
# 11. test_kv_buf_recent_matches_allowed
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_kv_buf_recent_matches_allowed():
    """In the recent segment, allowed=True slots should contain non-zero
    keys (assuming non-zero inputs were written)."""
    model, state = _make_model_and_state()
    W = model.window

    # Prefill with a moderate number of tokens so some recent slots are filled.
    state = _prefill(model, state, seq_len=W // 2)

    for b in range(state.k_buf.size(0)):
        for i in range(W):
            if state.allowed[b, i]:
                k_slot = state.k_buf[b, :, i, :]  # (Hkv, Dh)
                # At least one KV head should have a non-zero key vector.
                assert not torch.all(k_slot == 0), (
                    f"Allowed recent slot [{b},{i}] has all-zero keys"
                )

    # Also verify after filling the full window.
    model2, state2 = _make_model_and_state()
    state2 = _prefill(model2, state2, seq_len=W * 2)

    for b in range(state2.k_buf.size(0)):
        for i in range(W):
            if state2.allowed[b, i]:
                k_slot = state2.k_buf[b, :, i, :]
                assert not torch.all(k_slot == 0), (
                    f"Allowed recent slot [{b},{i}] has all-zero keys after full fill"
                )


# ---------------------------------------------------------------------------
# 12. test_pos_matches_write_cache_false
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pos_matches_write_cache_false():
    """With write_cache=False, pos should NOT change after prefill."""
    model, state = _make_model_and_state()

    pos_before = state.pos
    x = _rand_input(1, 200)
    _, state = model.prefill_chunked(
        x, state, block_size=64, write_cache=False, return_outputs=False,
    )

    assert state.pos == pos_before, (
        f"pos changed from {pos_before} to {state.pos} with write_cache=False"
    )


# ---------------------------------------------------------------------------
# 13. test_state_after_empty_prefill
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_state_after_empty_prefill():
    """Prefill with L=0.  State should be unchanged from init."""
    model, state = _make_model_and_state()

    # Snapshot initial state values.
    pos_init = state.pos
    recent_idx_init = state.recent_idx
    recent_filled_init = state.recent_filled
    allowed_init = state.allowed.clone()
    k_buf_init = state.k_buf.clone()
    v_buf_init = state.v_buf.clone()

    # Empty prefill.
    x_empty = _rand_input(1, 0)
    _, state = model.prefill_chunked(
        x_empty, state, block_size=64, write_cache=True, return_outputs=False,
    )

    assert state.pos == pos_init, (
        f"pos changed after empty prefill: {pos_init} -> {state.pos}"
    )
    assert state.recent_idx == recent_idx_init, (
        f"recent_idx changed after empty prefill: "
        f"{recent_idx_init} -> {state.recent_idx}"
    )
    assert state.recent_filled == recent_filled_init, (
        f"recent_filled changed after empty prefill: "
        f"{recent_filled_init} -> {state.recent_filled}"
    )
    assert torch.equal(state.allowed, allowed_init), (
        "allowed mask changed after empty prefill"
    )
    assert torch.equal(state.k_buf, k_buf_init), (
        "k_buf changed after empty prefill"
    )
    assert torch.equal(state.v_buf, v_buf_init), (
        "v_buf changed after empty prefill"
    )


# ---------------------------------------------------------------------------
# 14. test_recent_ring_wraps
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_recent_ring_wraps():
    """After prefill L = 3*W, the ring should have wrapped correctly.

    The prefill path calls _set_recent_from_sequence which resets
    recent_idx to 0.  Subsequent decode steps advance recent_idx via
    _write_one:  after N decode steps (with the ring already full),
    recent_idx == N % W.
    """
    model, state = _make_model_and_state()
    W = model.window

    # After prefill, _set_recent_from_sequence sets recent_idx = 0.
    state = _prefill(model, state, seq_len=W * 3)
    assert state.recent_idx == 0, (
        f"After prefill of {W * 3} tokens, expected recent_idx=0 "
        f"(set by _set_recent_from_sequence), got {state.recent_idx}"
    )
    assert state.recent_filled == W, (
        f"After prefill of {W * 3} tokens, expected recent_filled={W}, "
        f"got {state.recent_filled}"
    )

    # Now do N decode steps.  Each step evicts at recent_idx and advances
    # it by 1 mod W.  After N steps: recent_idx == N % W.
    N = W + 7  # Enough to wrap once plus a few extra.
    for i in range(N):
        expected_idx = i % W
        assert state.recent_idx == expected_idx, (
            f"Before decode step {i}: expected recent_idx={expected_idx}, "
            f"got {state.recent_idx}"
        )
        x = _rand_input(1, 1)
        _, state = model.step(x, state)

    assert state.recent_idx == N % W, (
        f"After {N} decode steps, expected recent_idx={N % W}, "
        f"got {state.recent_idx}"
    )


# ---------------------------------------------------------------------------
# Additional compound invariant tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pos_recent_filled_consistency():
    """recent_filled == min(pos, W) should always hold, both during and
    after filling the window."""
    model, state = _make_model_and_state()
    W = model.window

    # Feed tokens one at a time to verify at every step.
    for i in range(1, W * 2 + 1):
        x = _rand_input(1, 1)
        _, state = model.step(x, state)
        expected_filled = min(i, W)
        assert state.recent_filled == expected_filled, (
            f"After {i} tokens: expected recent_filled={expected_filled}, "
            f"got {state.recent_filled}"
        )
        assert state.pos == i, (
            f"After {i} tokens: expected pos={i}, got {state.pos}"
        )


@torch.no_grad()
def test_buffer_shapes_after_init():
    """Verify that all state tensors have the expected shapes immediately
    after init_state."""
    B = 2
    model, state = _make_model_and_state(batch_size=B)
    Hkv = model.num_kv_heads
    Dh = model.head_dim
    Lbuf = model.Lbuf
    W = model.window
    Me = model.Me

    assert state.k_buf.shape == (B, Hkv, Lbuf, Dh), (
        f"k_buf shape: {state.k_buf.shape}"
    )
    assert state.v_buf.shape == (B, Hkv, Lbuf, Dh), (
        f"v_buf shape: {state.v_buf.shape}"
    )
    assert state.allowed.shape == (B, Lbuf), (
        f"allowed shape: {state.allowed.shape}"
    )
    assert state.g_recent.shape == (B, W), (
        f"g_recent shape: {state.g_recent.shape}"
    )
    assert state.mem_last_exact.shape == (B, Me), (
        f"mem_last_exact shape: {state.mem_last_exact.shape}"
    )
    assert state.pos == 0
    assert state.recent_idx == 0
    assert state.recent_filled == 0


@torch.no_grad()
def test_allowed_monotonically_nondecreasing_for_banks():
    """The number of allowed slots in exact and summary banks should
    never decrease (bank slots are never deallocated, only overwritten)."""
    model, state = _make_model_and_state(write_policy="none")
    W = model.window
    Me = model.Me
    Ms = model.Ms

    prev_exact_count = 0
    prev_sum_count = 0

    # Prefill in small chunks, checking after each.
    total_len = W * 5
    chunk = W
    for start in range(0, total_len, chunk):
        state = _prefill(model, state, seq_len=chunk)

        exact_count = int(state.allowed[:, W:W + Me].sum().item())
        sum_count = int(state.allowed[:, W + Me:W + Me + Ms].sum().item())

        assert exact_count >= prev_exact_count, (
            f"Exact bank allowed count decreased from {prev_exact_count} "
            f"to {exact_count} at pos={state.pos}"
        )
        assert sum_count >= prev_sum_count, (
            f"Summary bank allowed count decreased from {prev_sum_count} "
            f"to {sum_count} at pos={state.pos}"
        )
        prev_exact_count = exact_count
        prev_sum_count = sum_count

    # Continue with decode steps.
    for i in range(W * 2):
        x = _rand_input(1, 1)
        _, state = model.step(x, state)

        exact_count = int(state.allowed[:, W:W + Me].sum().item())
        sum_count = int(state.allowed[:, W + Me:W + Me + Ms].sum().item())

        assert exact_count >= prev_exact_count, (
            f"Exact bank allowed count decreased from {prev_exact_count} "
            f"to {exact_count} at decode step {i}"
        )
        assert sum_count >= prev_sum_count, (
            f"Summary bank allowed count decreased from {prev_sum_count} "
            f"to {sum_count} at decode step {i}"
        )
        prev_exact_count = exact_count
        prev_sum_count = sum_count
