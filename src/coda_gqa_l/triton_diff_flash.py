# triton_diff_flash.py
# Fused differential FlashAttention kernel for CoDA-GQA-L.
#
# Single-pass forward kernel that loads K/V tiles once from HBM, computes both
# signal and noise attention via online softmax, applies differential epilogue
# (subtract + optional RMSNorm) in-register, then writes fused result.
#
# Derived from Makora expert-generate output (TILE_ROWS=2, num_warps=8).
# Adapted to match the public API expected by attention.py.

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _diff_flash_attn_fwd_kernel(
    Q, Q_NOISE, K, V, LAM, OUT,
    RMS_W,
    B: tl.constexpr, H: tl.constexpr, H_KV: tl.constexpr,
    Q_LEN, KV_LEN,
    HEAD_DIM: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qnb, stride_qnh, stride_qnl, stride_qnd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_lb, stride_lh, stride_ll,
    stride_ob, stride_oh, stride_ol, stride_od,
    sm_scale,
    eps,
    IS_CAUSAL: tl.constexpr,
    APPLY_RMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // H
    off_h = pid_bh % H

    groups: tl.constexpr = H // H_KV
    off_h_kv = off_h // groups

    base_m = pid_m * BLOCK_M * TILE_ROWS

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    if IS_CAUSAL:
        prefix_len = KV_LEN - Q_LEN
    else:
        prefix_len = 0

    n_blocks = tl.cdiv(KV_LEN, BLOCK_N)

    for tile_row in range(TILE_ROWS):
        offs_m = base_m + tile_row * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < Q_LEN

        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
        qn_ptrs = Q_NOISE + off_b * stride_qnb + off_h * stride_qnh + offs_m[:, None] * stride_qnl + offs_d[None, :] * stride_qnd

        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        qn_tile = tl.load(qn_ptrs, mask=mask_m[:, None], other=0.0)

        lam_ptrs = LAM + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ll
        lam_tile = tl.load(lam_ptrs, mask=mask_m, other=0.0)

        # Online softmax accumulators for signal and noise streams
        m_s = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        l_s = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_s = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m_n = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        l_n = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_n = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for block_n in range(0, n_blocks):
            kv_start = block_n * BLOCK_N
            offs_kv = kv_start + offs_n
            mask_n = offs_kv < KV_LEN

            k_ptrs = K + off_b * stride_kb + off_h_kv * stride_kh + offs_kv[:, None] * stride_kl + offs_d[None, :] * stride_kd
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            qk_s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            qk_n = tl.dot(qn_tile, tl.trans(k_tile)) * sm_scale

            if IS_CAUSAL:
                causal_mask = (offs_m[:, None] + prefix_len) >= offs_kv[None, :]
                valid = causal_mask & mask_n[None, :]
                qk_s = tl.where(valid, qk_s, float("-inf"))
                qk_n = tl.where(valid, qk_n, float("-inf"))
            else:
                qk_s = tl.where(mask_n[None, :], qk_s, float("-inf"))
                qk_n = tl.where(mask_n[None, :], qk_n, float("-inf"))

            v_ptrs = V + off_b * stride_vb + off_h_kv * stride_vh + offs_kv[:, None] * stride_vl + offs_d[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # Signal stream online softmax
            m_s_new = tl.maximum(m_s, tl.max(qk_s, axis=1))
            alpha_s = tl.exp(m_s - m_s_new)
            p_s = tl.exp(qk_s - m_s_new[:, None])
            l_s = l_s * alpha_s + tl.sum(p_s, axis=1)
            acc_s = acc_s * alpha_s[:, None] + tl.dot(p_s.to(v_tile.dtype), v_tile)
            m_s = m_s_new

            # Noise stream online softmax
            m_n_new = tl.maximum(m_n, tl.max(qk_n, axis=1))
            alpha_n = tl.exp(m_n - m_n_new)
            p_n = tl.exp(qk_n - m_n_new[:, None])
            l_n = l_n * alpha_n + tl.sum(p_n, axis=1)
            acc_n = acc_n * alpha_n[:, None] + tl.dot(p_n.to(v_tile.dtype), v_tile)
            m_n = m_n_new

        acc_s = acc_s / (l_s[:, None] + 1e-10)
        acc_n = acc_n / (l_n[:, None] + 1e-10)

        # Differential epilogue: signal - lambda * noise
        diff = acc_s - lam_tile[:, None] * acc_n

        # Optional fused RMSNorm
        if APPLY_RMS:
            var = tl.sum(diff * diff, axis=1) / HEAD_DIM
            rstd = tl.math.rsqrt(var + eps)
            diff = diff * rstd[:, None]
            rms_w = tl.load(RMS_W + offs_d)
            diff = diff * rms_w[None, :]

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
    rms_weight: Optional[torch.Tensor],
    eps: float,
    sm_scale: float,
    is_causal: bool,
    apply_rms: bool,
) -> None:
    B, H, Q_LEN, HEAD_DIM = q.shape
    _, H_KV, KV_LEN, _ = k.shape

    # Auto-tune block sizes based on problem dimensions and dtype
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

    if HEAD_DIM > 128:
        BLOCK_M = min(BLOCK_M, 64)
        BLOCK_N = min(BLOCK_N, 64)

    dtype_bytes = q.element_size()
    if dtype_bytes >= 4:
        BLOCK_M = min(BLOCK_M, 32)
        BLOCK_N = min(BLOCK_N, 64)

    TILE_ROWS = 2
    effective_block_m = BLOCK_M * TILE_ROWS

    tile_bytes = (effective_block_m + BLOCK_N) * HEAD_DIM * dtype_bytes * 2
    num_stages = 1 if tile_bytes > 64 * 1024 else 2

    if rms_weight is None:
        rms_weight = torch.empty(0, device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(Q_LEN, effective_block_m), B * H)

    _diff_flash_attn_fwd_kernel[grid](
        q, q_noise, k, v, lam, out,
        rms_weight,
        B, H, H_KV,
        Q_LEN, KV_LEN,
        HEAD_DIM,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_noise.stride(0), q_noise.stride(1), q_noise.stride(2), q_noise.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        lam.stride(0), lam.stride(1), lam.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        eps,
        IS_CAUSAL=is_causal,
        APPLY_RMS=apply_rms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        TILE_ROWS=TILE_ROWS,
        num_stages=num_stages,
        num_warps=8,
    )


def diff_flash_attn(
    q: torch.Tensor,
    q_noise: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam: torch.Tensor,
    *,
    rms_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused differential FlashAttention: single-pass dual-stream attention.

    Loads K/V tiles once from HBM, computes both signal and noise attention
    via online softmax, applies differential epilogue (subtract + optional
    RMSNorm) in-register, then writes fused result.

    Args:
        q:         (B, H, Lq, Dh) signal queries
        q_noise:   (B, H, Lq, Dh) noise queries (orthogonally rotated)
        k:         (B, Hkv, Lk, Dh) keys
        v:         (B, Hkv, Lk, Dh) values
        lam:       (B, H, Lq, 1) per-query lambda gate
        rms_weight: (Dh,) HeadwiseRMSNorm weight, or None to skip norm
        eps:       RMSNorm epsilon
        is_causal: apply causal masking (lower-right aligned)

    Returns:
        out: (B, H, Lq, Dh) differential attention output
    """
    B, H, Lq, Dh = q.shape
    sm_scale = 1.0 / math.sqrt(Dh)
    out = torch.empty_like(q)

    # lam is (B, H, Lq, 1) from attention.py; kernel expects (B, H, Lq)
    lam_squeezed = lam.squeeze(-1).contiguous()

    apply_rms = rms_weight is not None
    _diff_flash_fwd(
        q, q_noise, k, v, lam_squeezed, out,
        rms_weight=rms_weight,
        eps=eps,
        sm_scale=sm_scale,
        is_causal=is_causal,
        apply_rms=apply_rms,
    )
    return out


def is_available() -> bool:
    """Check if the Triton fused differential FlashAttention kernel is usable."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F811
        return True
    except ImportError:
        return False
