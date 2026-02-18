"""Triton fused differential FlashAttention kernel.

Single-pass forward kernel that:
  1. Loads K/V tiles once from HBM (GQA-aware head routing)
  2. Computes both signal and noise attention via online softmax
  3. Applies differential epilogue (subtract + optional RMSNorm) in-register
  4. Writes fused result to HBM

Algorithm:
  For each (batch, head, query_block):
    Two independent online-softmax accumulators (signal, noise) share K/V tiles.
    After the KV loop, finalize both, compute diff = sig - lam * noise,
    optionally fuse RMSNorm, then store.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _diff_flash_attn_fwd_kernel(
    # Pointers
    Q, Q_NOISE, K, V, LAM, OUT,
    RMS_W,
    # Dimensions
    B: tl.constexpr, H: tl.constexpr, H_KV: tl.constexpr,
    Q_LEN, KV_LEN,
    HEAD_DIM: tl.constexpr,
    # Strides - Q: (B, H, Lq, Dh)
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides - Q_NOISE: same layout as Q
    stride_qnb, stride_qnh, stride_qnl, stride_qnd,
    # Strides - K: (B, Hkv, Lk, Dh)
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides - V: (B, Hkv, Lk, Dh)
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Strides - LAM: (B, H, Lq)
    stride_lb, stride_lh, stride_ll,
    # Strides - OUT: (B, H, Lq, Dh)
    stride_ob, stride_oh, stride_ol, stride_od,
    # Scalars
    sm_scale,
    eps,
    # Flags
    IS_CAUSAL: tl.constexpr,
    APPLY_RMS: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ID -> (query_block, batch*head)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // H
    off_h = pid_bh % H

    # GQA: map Q head -> KV head
    groups: tl.constexpr = H // H_KV
    off_h_kv = off_h // groups

    # Query row offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_d = tl.arange(0, HEAD_DIM)                     # (HEAD_DIM,)

    # Masks for Q rows that are in-bounds
    mask_m = offs_m < Q_LEN

    # --- Load Q tile and Q_noise tile ---
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    qn_ptrs = Q_NOISE + off_b * stride_qnb + off_h * stride_qnh + offs_m[:, None] * stride_qnl + offs_d[None, :] * stride_qnd

    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)        # (BLOCK_M, HEAD_DIM)
    qn_tile = tl.load(qn_ptrs, mask=mask_m[:, None], other=0.0)      # (BLOCK_M, HEAD_DIM)

    # NOTE: sm_scale is applied after the dot product (not here) to avoid
    # promoting Q from bf16/fp16 to fp32 which would cause tl.dot dtype mismatch.

    # --- Load lambda ---
    lam_ptrs = LAM + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ll
    lam_tile = tl.load(lam_ptrs, mask=mask_m, other=0.0)  # (BLOCK_M,)

    # --- Init online softmax accumulators (fp32) ---
    # Signal stream
    m_s = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_s = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_s = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # Noise stream
    m_n = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_n = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_n = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # KV offsets
    offs_n = tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    # Causal: prefix_len = KV_LEN - Q_LEN (lower-right triangle)
    # A query at position q_idx can attend to k_idx where:
    #   q_idx + prefix_len >= k_idx  (i.e. pid_m * BLOCK_M + i + prefix_len >= j)
    if IS_CAUSAL:
        prefix_len = KV_LEN - Q_LEN
    else:
        prefix_len = 0

    # Always iterate all KV blocks.  The causal mask inside the loop
    # handles early-exit semantics; this avoids potential issues with
    # tl.cdiv on runtime kv_bound values in certain Triton versions.
    n_blocks = tl.cdiv(KV_LEN, BLOCK_N)

    # --- KV tile loop ---
    for block_n in range(0, n_blocks):
        kv_start = block_n * BLOCK_N
        offs_kv = kv_start + offs_n  # (BLOCK_N,)
        mask_n = offs_kv < KV_LEN

        # Load K tile (shared between signal and noise)
        k_ptrs = K + off_b * stride_kb + off_h_kv * stride_kh + offs_kv[:, None] * stride_kl + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # (BLOCK_N, HEAD_DIM)

        # QK^T for both streams, scaled after the dot to stay in input dtype
        # q_tile: (BLOCK_M, HEAD_DIM), k_tile: (BLOCK_N, HEAD_DIM)
        # -> qk: (BLOCK_M, BLOCK_N)
        qk_s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        qk_n = tl.dot(qn_tile, tl.trans(k_tile)) * sm_scale

        # Apply causal mask + OOB KV mask
        if IS_CAUSAL:
            # q_pos = offs_m + prefix_len, k_pos = offs_kv
            # valid where q_pos >= k_pos AND k_pos < KV_LEN
            causal_mask = (offs_m[:, None] + prefix_len) >= offs_kv[None, :]
            valid = causal_mask & mask_n[None, :]
            qk_s = tl.where(valid, qk_s, float("-inf"))
            qk_n = tl.where(valid, qk_n, float("-inf"))
        else:
            qk_s = tl.where(mask_n[None, :], qk_s, float("-inf"))
            qk_n = tl.where(mask_n[None, :], qk_n, float("-inf"))

        # Load V tile (shared)
        v_ptrs = V + off_b * stride_vb + off_h_kv * stride_vh + offs_kv[:, None] * stride_vl + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # (BLOCK_N, HEAD_DIM)

        # --- Online softmax update: signal stream ---
        m_s_new = tl.maximum(m_s, tl.max(qk_s, axis=1))
        alpha_s = tl.exp(m_s - m_s_new)
        p_s = tl.exp(qk_s - m_s_new[:, None])
        l_s = l_s * alpha_s + tl.sum(p_s, axis=1)
        acc_s = acc_s * alpha_s[:, None] + tl.dot(p_s.to(v_tile.dtype), v_tile)
        m_s = m_s_new

        # --- Online softmax update: noise stream ---
        m_n_new = tl.maximum(m_n, tl.max(qk_n, axis=1))
        alpha_n = tl.exp(m_n - m_n_new)
        p_n = tl.exp(qk_n - m_n_new[:, None])
        l_n = l_n * alpha_n + tl.sum(p_n, axis=1)
        acc_n = acc_n * alpha_n[:, None] + tl.dot(p_n.to(v_tile.dtype), v_tile)
        m_n = m_n_new

    # --- Finalize: divide by sum-of-exp ---
    acc_s = acc_s / (l_s[:, None] + 1e-10)
    acc_n = acc_n / (l_n[:, None] + 1e-10)

    # --- Differential epilogue ---
    diff = acc_s - lam_tile[:, None] * acc_n

    # --- Optional fused RMSNorm ---
    if APPLY_RMS:
        # var = mean(diff^2) over head_dim
        var = tl.sum(diff * diff, axis=1) / HEAD_DIM
        rstd = tl.math.rsqrt(var + eps)
        diff = diff * rstd[:, None]
        # Load and apply RMSNorm weight
        rms_w = tl.load(RMS_W + offs_d)  # (HEAD_DIM,)
        diff = diff * rms_w[None, :]

    # --- Store output ---
    out_ptrs = OUT + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(out_ptrs, diff.to(OUT.dtype.element_ty), mask=mask_m[:, None])


def _diff_flash_fwd(
    q: torch.Tensor,
    q_noise: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam: torch.Tensor,
    out: torch.Tensor,
    *,
    rms_weight: torch.Tensor | None,
    eps: float,
    sm_scale: float,
    is_causal: bool,
    APPLY_RMS: bool,
):
    """Launch the fused differential FlashAttention kernel."""
    B, H, Q_LEN, HEAD_DIM = q.shape
    _, H_KV, KV_LEN, _ = k.shape

    # Tile size selection: smaller tiles for decode (Lq=1), larger for prefill
    if Q_LEN <= 16:
        BLOCK_M = 16
    elif Q_LEN <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 128

    if KV_LEN <= 64:
        BLOCK_N = 32
    elif KV_LEN <= 256:
        BLOCK_N = 64
    else:
        BLOCK_N = 128

    # HEAD_DIM must fit in tile — cap for register pressure
    if HEAD_DIM > 128:
        BLOCK_M = min(BLOCK_M, 64)
        BLOCK_N = min(BLOCK_N, 64)

    # fp32 inputs use 4 bytes/element → 2x shared memory pressure.
    # Reduce tiles to stay within H100's 228KB shared memory limit.
    dtype_bytes = q.element_size()
    if dtype_bytes >= 4:
        BLOCK_M = min(BLOCK_M, 32)
        BLOCK_N = min(BLOCK_N, 64)

    # num_stages controls pipelining depth (more stages = more shared memory).
    # Use 1 stage for large tiles or fp32 to avoid OOM; 2 otherwise.
    tile_bytes = (BLOCK_M + BLOCK_N) * HEAD_DIM * dtype_bytes * 2  # Q+K rough estimate
    num_stages = 1 if tile_bytes > 64 * 1024 else 2

    # Dummy RMS weight pointer if not applying RMSNorm
    if rms_weight is None:
        rms_weight = torch.empty(0, device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)

    _diff_flash_attn_fwd_kernel[grid](
        q, q_noise, k, v, lam, out,
        rms_weight,
        B, H, H_KV,
        Q_LEN, KV_LEN,
        HEAD_DIM,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Q_NOISE strides
        q_noise.stride(0), q_noise.stride(1), q_noise.stride(2), q_noise.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # LAM strides
        lam.stride(0), lam.stride(1), lam.stride(2),
        # OUT strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        eps,
        IS_CAUSAL=is_causal,
        APPLY_RMS=APPLY_RMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps=4,
    )
