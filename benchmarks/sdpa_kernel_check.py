#!/usr/bin/env python3
"""SDPA kernel selection probe for CoDA-GQA-L attention module.

Determines which SDPA backend (FlashAttention, Memory-Efficient, Math)
is actually selected for each call site (decode step vs chunked prefill),
and whether boolean masks force a fallback from FlashAttention.

Usage:
    python benchmarks/sdpa_kernel_check.py
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from coda_gqa_l import CoDAGQALandmarkPerf2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Model dimensions (small but representative)
EMBED_DIM = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
WINDOW = 64
NUM_LANDMARKS_EXACT = 8
NUM_LANDMARKS_SUMMARY = 8
BATCH = 1

# Prefill sequence length and block size
PREFILL_SEQ_LEN = 512
PREFILL_BLOCK_SIZE = 256

# Warmup / timing
WARMUP_ITERS = 3
TIMED_ITERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BACKEND_NAMES = {
    SDPBackend.FLASH_ATTENTION: "FlashAttention",
    SDPBackend.EFFICIENT_ATTENTION: "MemEfficient",
    SDPBackend.MATH: "Math",
    SDPBackend.CUDNN_ATTENTION: "cuDNN",
}


def _try_sdpa_with_backend(
    backend: SDPBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    enable_gqa: bool,
) -> Tuple[bool, str]:
    """Try running SDPA with exactly one backend enabled.

    Returns (success: bool, error_msg: str).
    """
    try:
        with sdpa_kernel(backend):
            kwargs = dict(
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            if enable_gqa:
                kwargs["enable_gqa"] = True
            _ = F.scaled_dot_product_attention(q, k, v, **kwargs)
            torch.cuda.synchronize()
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def _time_sdpa_with_backend(
    backend: SDPBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    enable_gqa: bool,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Time SDPA with a specific backend. Returns average ms. -1 if fails."""
    kwargs = dict(attn_mask=attn_mask, is_causal=is_causal)
    if enable_gqa:
        kwargs["enable_gqa"] = True
    try:
        with sdpa_kernel(backend):
            for _ in range(warmup):
                _ = F.scaled_dot_product_attention(q, k, v, **kwargs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = F.scaled_dot_product_attention(q, k, v, **kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        return (t1 - t0) / iters * 1000.0
    except Exception:
        return -1.0


def probe_all_backends(
    label: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    enable_gqa: bool,
) -> Dict[str, dict]:
    """Probe which backends accept the given SDPA inputs."""
    results = {}
    for backend in [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
        SDPBackend.CUDNN_ATTENTION,
    ]:
        ok, err = _try_sdpa_with_backend(
            backend, q, k, v, attn_mask, is_causal, enable_gqa
        )
        ms = -1.0
        if ok:
            ms = _time_sdpa_with_backend(
                backend, q, k, v, attn_mask, is_causal, enable_gqa
            )
        results[BACKEND_NAMES[backend]] = {"available": ok, "error": err, "ms": ms}
    return results


def print_probe_results(label: str, results: Dict[str, dict]) -> None:
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    for name, info in results.items():
        status = "OK" if info["available"] else "FAIL"
        ms_str = f"{info['ms']:.3f} ms" if info["ms"] >= 0 else "N/A"
        err_str = f"  ({info['error']})" if info["error"] else ""
        print(f"  {name:20s}  {status:4s}  {ms_str:>12s}{err_str}")


# ---------------------------------------------------------------------------
# Build model and state
# ---------------------------------------------------------------------------
def build_model_and_state():
    model = CoDAGQALandmarkPerf2(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        window=WINDOW,
        num_landmarks_exact=NUM_LANDMARKS_EXACT,
        num_landmarks_summary=NUM_LANDMARKS_SUMMARY,
        mask_unused_memory=True,
    ).to(device=DEVICE, dtype=DTYPE)
    model.eval()

    state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)
    return model, state


# ---------------------------------------------------------------------------
# TEST 1: Decode step -- empty buffer (allowed.all() == False, allowed.any() == False)
# ---------------------------------------------------------------------------
def test_decode_empty(model, state):
    """Decode into empty buffer: returns zeros, no SDPA called."""
    x = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)
    allowed = state.allowed
    print(f"\n--- Decode (empty buffer) ---")
    print(f"  allowed.any() = {bool(allowed.any())}")
    print(f"  allowed.all() = {bool(allowed.all())}")
    print(f"  -> Code returns zeros (no SDPA call)")


# ---------------------------------------------------------------------------
# TEST 2: Decode step -- partially filled buffer (boolean mask path)
# ---------------------------------------------------------------------------
def test_decode_partial(model, state):
    """Decode after partial fill: some allowed=False -> boolean mask."""
    print(f"\n--- Decode (partially filled buffer, boolean mask path) ---")

    # Fill some recent slots
    half_w = WINDOW // 2
    for i in range(half_w):
        x_tok = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)
        _, state = model.step(x_tok, state)

    allowed = state.allowed
    print(f"  allowed.any() = {bool(allowed.any())}")
    print(f"  allowed.all() = {bool(allowed.all())}")
    print(f"  -> Code builds: attn_mask = allowed[:, None, None, :]  (bool)")

    # Reconstruct what the code builds for the SDPA call
    attn_mask_bool = allowed[:, None, None, :]  # (B, 1, 1, Lbuf)

    Lbuf = model.Lbuf
    Hkv = model.num_kv_heads
    H = model.num_heads
    Dh = model.head_dim
    B = BATCH

    # Stacked query shape: (B, 2H, 1, Dh)
    q_stacked = torch.randn(B, 2 * H, 1, Dh, device=DEVICE, dtype=DTYPE)

    # GQA path (enable_gqa=True): k/v stay at Hkv heads
    k_buf = state.k_buf  # (B, Hkv, Lbuf, Dh)
    v_buf = state.v_buf  # (B, Hkv, Lbuf, Dh)

    print(f"  q shape: {q_stacked.shape}, k shape: {k_buf.shape}, v shape: {v_buf.shape}")
    print(f"  attn_mask shape: {attn_mask_bool.shape}, dtype: {attn_mask_bool.dtype}")
    print(f"  is_causal: False, enable_gqa: True")

    results = probe_all_backends(
        "Decode (partial fill, boolean mask, enable_gqa=True)",
        q_stacked, k_buf, v_buf,
        attn_mask=attn_mask_bool,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Decode (partial fill, boolean mask, enable_gqa=True)", results)

    # Also test: what if we convert bool -> float additive mask?
    attn_mask_float = torch.where(
        attn_mask_bool, torch.zeros((), device=DEVICE, dtype=DTYPE),
        torch.full((), float("-inf"), device=DEVICE, dtype=DTYPE),
    )
    results_float = probe_all_backends(
        "Decode (partial fill, FLOAT additive mask, enable_gqa=True)",
        q_stacked, k_buf, v_buf,
        attn_mask=attn_mask_float,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Decode (partial fill, FLOAT additive mask, enable_gqa=True)", results_float)

    return state


# ---------------------------------------------------------------------------
# TEST 3: Decode step -- fully filled buffer (allowed.all() fast-path)
# ---------------------------------------------------------------------------
def test_decode_full(model, state):
    """Decode after buffer is full: allowed.all() == True -> mask=None."""
    print(f"\n--- Decode (fully filled buffer, mask=None fast-path) ---")

    # Fill rest of recent window + overflow to exercise memory banks
    needed = WINDOW + 20  # overflow to populate exact/summary
    for i in range(needed):
        x_tok = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)
        _, state = model.step(x_tok, state)

    allowed = state.allowed
    n_allowed = allowed.sum().item()
    print(f"  allowed.any() = {bool(allowed.any())}")
    print(f"  allowed.all() = {bool(allowed.all())}")
    print(f"  allowed count: {n_allowed}/{model.Lbuf}")
    print(f"  -> Code sets attn_mask = None (fast-path)")

    Lbuf = model.Lbuf
    Hkv = model.num_kv_heads
    H = model.num_heads
    Dh = model.head_dim
    B = BATCH

    q_stacked = torch.randn(B, 2 * H, 1, Dh, device=DEVICE, dtype=DTYPE)
    k_buf = state.k_buf
    v_buf = state.v_buf

    print(f"  q shape: {q_stacked.shape}, k shape: {k_buf.shape}, v shape: {v_buf.shape}")
    print(f"  attn_mask: None, is_causal: False, enable_gqa: True")

    results = probe_all_backends(
        "Decode (full buffer, mask=None, enable_gqa=True)",
        q_stacked, k_buf, v_buf,
        attn_mask=None,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Decode (full buffer, mask=None, enable_gqa=True)", results)

    return state


# ---------------------------------------------------------------------------
# TEST 4: Prefill -- reproduce the exact mask construction from prefill_chunked()
# ---------------------------------------------------------------------------
def test_prefill_mask(model):
    """Prefill path: causal & allowed boolean mask."""
    print(f"\n--- Prefill (chunked, boolean mask) ---")

    state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)

    Lbuf = model.Lbuf
    blk = PREFILL_BLOCK_SIZE
    Hkv = model.num_kv_heads
    H = model.num_heads
    Dh = model.head_dim
    B = BATCH

    # Reproduce mask construction from prefill_chunked lines 663-670:
    # The first block: state is fresh, allowed_prev is all-False
    allowed_prev = state.allowed  # (B, Lbuf) all False
    block_ok = torch.ones((B, blk), device=DEVICE, dtype=torch.bool)
    allowed = torch.cat([allowed_prev, block_ok], dim=1)  # (B, Lbuf + blk)

    # Causal mask: prefix (all-True for buffer) + lower-triangular for block
    causal = model._get_causal_mask(Lprev=Lbuf, blk=blk, device=DEVICE)
    attn_mask_bool = causal.view(1, 1, blk, Lbuf + blk) & allowed[:, None, None, :]

    print(f"  Lbuf (prefix KV): {Lbuf}")
    print(f"  blk (current chunk): {blk}")
    print(f"  Total KV length: {Lbuf + blk}")
    print(f"  causal shape: {causal.shape}, dtype: {causal.dtype}")
    print(f"  allowed shape: {allowed.shape}, dtype: {allowed.dtype}")
    print(f"  attn_mask shape: {attn_mask_bool.shape}, dtype: {attn_mask_bool.dtype}")
    print(f"  attn_mask True%: {attn_mask_bool.float().mean().item()*100:.1f}%")

    # The actual shapes in SDPA:
    # q_cat: (B, 2H, blk, Dh)  -- signal + noise stacked
    # k_all: (B, Hkv, Lbuf+blk, Dh)
    # v_all: (B, Hkv, Lbuf+blk, Dh)
    q_stacked = torch.randn(B, 2 * H, blk, Dh, device=DEVICE, dtype=DTYPE)
    k_all = torch.randn(B, Hkv, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)
    v_all = torch.randn(B, Hkv, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)

    print(f"  q shape: {q_stacked.shape}, k shape: {k_all.shape}, v shape: {v_all.shape}")
    print(f"  is_causal: False, enable_gqa: True")

    # Probe with boolean mask
    results_bool = probe_all_backends(
        "Prefill (boolean mask, enable_gqa=True)",
        q_stacked, k_all, v_all,
        attn_mask=attn_mask_bool,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Prefill (boolean mask, enable_gqa=True)", results_bool)

    # Test with float additive mask
    attn_mask_float = torch.where(
        attn_mask_bool,
        torch.zeros((), device=DEVICE, dtype=DTYPE),
        torch.full((), float("-inf"), device=DEVICE, dtype=DTYPE),
    )
    results_float = probe_all_backends(
        "Prefill (FLOAT additive mask, enable_gqa=True)",
        q_stacked, k_all, v_all,
        attn_mask=attn_mask_float,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Prefill (FLOAT additive mask, enable_gqa=True)", results_float)

    # Test: what if we use is_causal=True with NO mask?
    # (This would only work if Q and K have same sequence length or specific layout)
    print(f"\n  Testing is_causal=True (no mask):")
    print(f"    Requires Q_len <= K_len, and standard causal relationship")
    q_square = torch.randn(B, 2 * H, blk, Dh, device=DEVICE, dtype=DTYPE)
    k_square = torch.randn(B, Hkv, blk, Dh, device=DEVICE, dtype=DTYPE)
    v_square = torch.randn(B, Hkv, blk, Dh, device=DEVICE, dtype=DTYPE)
    results_causal = probe_all_backends(
        "Prefill (is_causal=True, no mask, intra-block only, enable_gqa=True)",
        q_square, k_square, v_square,
        attn_mask=None,
        is_causal=True,
        enable_gqa=True,
    )
    print_probe_results("Prefill (is_causal=True, no mask, intra-block only, enable_gqa=True)", results_causal)

    # Test: mask=None, is_causal=False (cross-attention to buffer, no masking)
    q_cross = torch.randn(B, 2 * H, blk, Dh, device=DEVICE, dtype=DTYPE)
    k_cross = torch.randn(B, Hkv, Lbuf, Dh, device=DEVICE, dtype=DTYPE)
    v_cross = torch.randn(B, Hkv, Lbuf, Dh, device=DEVICE, dtype=DTYPE)
    results_cross = probe_all_backends(
        "Prefill (mask=None, cross-attn to buffer, enable_gqa=True)",
        q_cross, k_cross, v_cross,
        attn_mask=None,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Prefill (mask=None, cross-attn to buffer, enable_gqa=True)", results_cross)

    return results_bool, results_float


# ---------------------------------------------------------------------------
# TEST 5: Prefill with allowed.all() == True (hypothetical fast-path)
# ---------------------------------------------------------------------------
def test_prefill_all_allowed(model):
    """What if all buffer slots are valid (allowed.all() True) during prefill?"""
    print(f"\n--- Prefill (hypothetical: all buffer slots valid) ---")

    Lbuf = model.Lbuf
    blk = PREFILL_BLOCK_SIZE
    Hkv = model.num_kv_heads
    H = model.num_heads
    Dh = model.head_dim
    B = BATCH

    # Even if allowed is all-True, we still need the causal mask for intra-block
    allowed = torch.ones((B, Lbuf + blk), device=DEVICE, dtype=torch.bool)
    causal = model._get_causal_mask(Lprev=Lbuf, blk=blk, device=DEVICE)

    # Check: does causal & all-True = just causal?
    attn_mask_bool = causal.view(1, 1, blk, Lbuf + blk) & allowed[:, None, None, :]
    # This is still a boolean mask (just equals causal)!
    print(f"  Even with all-allowed, mask = causal (still boolean)")
    print(f"  attn_mask shape: {attn_mask_bool.shape}, dtype: {attn_mask_bool.dtype}")

    q = torch.randn(B, 2 * H, blk, Dh, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, Hkv, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, Hkv, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)

    results = probe_all_backends(
        "Prefill (causal bool mask only, all slots valid, enable_gqa=True)",
        q, k, v,
        attn_mask=attn_mask_bool,
        is_causal=False,
        enable_gqa=True,
    )
    print_probe_results("Prefill (causal bool mask only, all slots valid, enable_gqa=True)", results)


# ---------------------------------------------------------------------------
# TEST 6: Non-GQA path (repeat_kv) -- does it change kernel selection?
# ---------------------------------------------------------------------------
def test_prefill_no_gqa(model):
    """Test the repeat_kv fallback path (when enable_gqa is not used)."""
    print(f"\n--- Prefill (repeat_kv path, no enable_gqa) ---")

    Lbuf = model.Lbuf
    blk = PREFILL_BLOCK_SIZE
    H = model.num_heads
    Dh = model.head_dim
    B = BATCH

    # With repeat_kv, all head counts match: q=(B,2H,blk,Dh), k=(B,2H,Lbuf+blk,Dh)
    q = torch.randn(B, 2 * H, blk, Dh, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, 2 * H, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, 2 * H, Lbuf + blk, Dh, device=DEVICE, dtype=DTYPE)

    # Boolean mask
    state = model.init_state(batch_size=B, device=DEVICE, dtype=DTYPE)
    allowed_prev = state.allowed
    block_ok = torch.ones((B, blk), device=DEVICE, dtype=torch.bool)
    allowed = torch.cat([allowed_prev, block_ok], dim=1)
    causal = model._get_causal_mask(Lprev=Lbuf, blk=blk, device=DEVICE)
    attn_mask_bool = causal.view(1, 1, blk, Lbuf + blk) & allowed[:, None, None, :]

    results = probe_all_backends(
        "Prefill (boolean mask, repeat_kv, NO enable_gqa)",
        q, k, v,
        attn_mask=attn_mask_bool,
        is_causal=False,
        enable_gqa=False,
    )
    print_probe_results("Prefill (boolean mask, repeat_kv, NO enable_gqa)", results)


# ---------------------------------------------------------------------------
# TEST 7: End-to-end timing comparison
# ---------------------------------------------------------------------------
def test_e2e_timing(model):
    """Time full prefill_chunked with different scenarios."""
    print(f"\n{'='*72}")
    print(f"  End-to-end prefill_chunked() timing")
    print(f"{'='*72}")

    x_seq = torch.randn(BATCH, PREFILL_SEQ_LEN, EMBED_DIM, device=DEVICE, dtype=DTYPE)

    # Warm up
    for _ in range(WARMUP_ITERS):
        state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            model.prefill_chunked(x_seq, state, block_size=PREFILL_BLOCK_SIZE)
        torch.cuda.synchronize()

    # Time it
    times = []
    for _ in range(TIMED_ITERS):
        state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.prefill_chunked(x_seq, state, block_size=PREFILL_BLOCK_SIZE)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  prefill_chunked({PREFILL_SEQ_LEN} tokens, block={PREFILL_BLOCK_SIZE})")
    print(f"  avg: {avg:.2f} ms, min: {mn:.2f} ms, max: {mx:.2f} ms")

    # Now time with only MATH backend forced (to confirm it is what runs)
    for backend, name in [
        (SDPBackend.MATH, "Math"),
        (SDPBackend.EFFICIENT_ATTENTION, "MemEfficient"),
        (SDPBackend.FLASH_ATTENTION, "Flash"),
    ]:
        times_b = []
        failed = False
        for _ in range(TIMED_ITERS):
            state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                with torch.no_grad(), sdpa_kernel(backend):
                    model.prefill_chunked(x_seq, state, block_size=PREFILL_BLOCK_SIZE)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_b.append((t1 - t0) * 1000)
            except Exception as e:
                failed = True
                print(f"  {name:15s}  FAILED: {str(e)[:120]}")
                break
        if not failed and times_b:
            avg_b = sum(times_b) / len(times_b)
            mn_b = min(times_b)
            print(f"  {name:15s}  avg: {avg_b:.2f} ms, min: {mn_b:.2f} ms")


# ---------------------------------------------------------------------------
# TEST 8: Decode timing comparison
# ---------------------------------------------------------------------------
def test_decode_timing(model):
    """Time decode step with different backends."""
    print(f"\n{'='*72}")
    print(f"  Decode step() timing (buffer fully filled)")
    print(f"{'='*72}")

    # Fill the buffer fully first
    state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)
    fill_len = WINDOW + NUM_LANDMARKS_EXACT + NUM_LANDMARKS_SUMMARY + 20
    for i in range(fill_len):
        x_tok = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            _, state = model.step(x_tok, state)

    allowed = state.allowed
    print(f"  allowed.all() = {bool(allowed.all())}")
    print(f"  allowed count: {allowed.sum().item()}/{model.Lbuf}")

    x_tok = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)

    # Warmup
    for _ in range(WARMUP_ITERS):
        import copy
        st_copy = copy.deepcopy(state)
        with torch.no_grad():
            model.step(x_tok, st_copy)
        torch.cuda.synchronize()

    for backend_set, name in [
        (None, "Default"),
        (SDPBackend.FLASH_ATTENTION, "Flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "MemEfficient"),
        (SDPBackend.MATH, "Math"),
    ]:
        times_b = []
        failed = False
        for _ in range(TIMED_ITERS):
            import copy
            st_copy = copy.deepcopy(state)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                ctx = sdpa_kernel(backend_set) if backend_set is not None else contextlib.nullcontext()
                with torch.no_grad(), ctx:
                    model.step(x_tok, st_copy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_b.append((t1 - t0) * 1000)
            except Exception as e:
                failed = True
                print(f"  {name:15s}  FAILED: {str(e)[:120]}")
                break
        if not failed and times_b:
            avg_b = sum(times_b) / len(times_b)
            mn_b = min(times_b)
            print(f"  {name:15s}  avg: {avg_b:.2f} ms, min: {mn_b:.2f} ms")


# ---------------------------------------------------------------------------
# TEST 9: Allowed.all() fast-path analysis
# ---------------------------------------------------------------------------
def test_allowed_all_fastpath(model):
    """Check when allowed.all() becomes True during buffer fill."""
    print(f"\n{'='*72}")
    print(f"  allowed.all() fast-path analysis")
    print(f"{'='*72}")

    state = model.init_state(batch_size=BATCH, device=DEVICE, dtype=DTYPE)

    Lbuf = model.Lbuf
    W = model.window
    Me = model.Me
    Ms = model.Ms
    print(f"  Lbuf={Lbuf} (W={W} + Me={Me} + Ms={Ms})")
    print(f"  mask_unused_memory={model.mask_unused_memory}")

    milestones = {}
    for i in range(W + Me + Ms + 50):
        x_tok = torch.randn(BATCH, 1, EMBED_DIM, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            _, state = model.step(x_tok, state)
        n = state.allowed.sum().item()
        if n not in milestones:
            milestones[n] = i + 1
        if bool(state.allowed.all()):
            print(f"  allowed.all() became True at step {i+1}")
            print(f"  Fill milestones: {dict(sorted(milestones.items()))}")
            return state

    print(f"  WARNING: allowed.all() never became True after {W+Me+Ms+50} steps!")
    n = state.allowed.sum().item()
    print(f"  Final allowed count: {n}/{Lbuf}")
    print(f"  Fill milestones: {dict(sorted(milestones.items()))}")
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  SDPA Kernel Selection Probe for CoDA-GQA-L")
    print("=" * 72)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"  Model: embed={EMBED_DIM}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}")
    print(f"  Buffer: W={WINDOW}, Me={NUM_LANDMARKS_EXACT}, Ms={NUM_LANDMARKS_SUMMARY}")
    print(f"  Lbuf = {WINDOW + NUM_LANDMARKS_EXACT + NUM_LANDMARKS_SUMMARY}")
    print(f"  Prefill: seq_len={PREFILL_SEQ_LEN}, block_size={PREFILL_BLOCK_SIZE}")

    model, state = build_model_and_state()

    # Test 1: Empty buffer decode
    test_decode_empty(model, state)

    # Test 2: Partially filled decode (boolean mask)
    state = test_decode_partial(model, state)

    # Test 3: Fully filled decode (mask=None fast-path)
    state = test_decode_full(model, state)

    # Test 4: Prefill boolean mask probe
    test_prefill_mask(model)

    # Test 5: Prefill all-allowed hypothetical
    test_prefill_all_allowed(model)

    # Test 6: Non-GQA path
    test_prefill_no_gqa(model)

    # Test 7: Allowed.all() fast-path analysis
    test_allowed_all_fastpath(model)

    # Test 8: End-to-end prefill timing
    test_e2e_timing(model)

    # Test 9: Decode timing
    test_decode_timing(model)

    # Summary
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"""
  DECODE PATH (step / attend_step):
    - Empty buffer:     No SDPA call (returns zeros)
    - Partial buffer:   attn_mask = allowed[:, None, None, :] (bool, shape (B,1,1,Lbuf))
                        -> is_causal=False, mask is boolean
    - Full buffer:      attn_mask = None (fast-path when allowed.all())
                        -> is_causal=False, mask=None

  PREFILL PATH (prefill_chunked):
    - Always builds:    causal_mask & allowed_mask (bool, shape (1,1,blk,Lbuf+blk))
                        -> is_causal=False, mask is boolean
    - No fast-path:     Even if allowed.all(), causal mask is still needed
                        (code always does `causal & allowed`)

  KEY FINDING: In prefill_chunked(), the boolean mask prevents FlashAttention.
  The causal mask is ALWAYS applied as a boolean tensor, regardless of allowed state.
  There is no is_causal=True optimization path for the intra-block causal portion.
""")


if __name__ == "__main__":
    main()
