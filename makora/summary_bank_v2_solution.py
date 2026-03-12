"""Makora solution file: Triton fused summary bank update kernel (v2).

Two-phase design fixing the O(H_KV^2) routing redundancy in v1:
  Phase 1: Route all T candidates to slots ONCE (LF-K cosine similarity
           averaged across H_KV heads). Store routing as per-slot weight
           accumulator + per-candidate best_slot decisions.
  Phase 2: For each head, iterate candidates, use Phase 1 routing decisions
           to scatter-add K/V into per-slot accumulators. Apply Phase-Safe
           EMA (LF-only for keys, full for values). Write back.

Complexity: O(H_KV * T * MS * DH_LF) for routing (Phase 1)
          + O(H_KV * T * DH) for accumulation (Phase 2)
vs v1:    O(H_KV^2 * T * MS * DH_LF) due to re-routing per head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _summary_bank_update_v2_kernel(
    # Candidate data
    K_SEL,          # (B, Hkv, T, Dh)
    V_SEL,          # (B, Hkv, T, Dh)
    W_TOK,          # (B, T) float32
    # Bank data (read)
    MEM_K,          # (B, Hkv, Ms, Dh)
    MEM_V,          # (B, Hkv, Ms, Dh)
    USED,           # (B, Ms) int8
    MEM_LF_K_NORM,  # (B, Hkv, Ms, Dh_lf)
    # Routing decisions (intermediate, written by Phase 1, read by Phase 2)
    ROUTE_IDX,      # (B, T) int32 - which slot each candidate routes to
    # Outputs
    MEM_K_OUT,      # (B, Hkv, Ms, Dh)
    MEM_V_OUT,      # (B, Hkv, Ms, Dh)
    USED_OUT,       # (B, Ms) int8
    # Strides - K_SEL
    stride_kb, stride_kh, stride_kt, stride_kd,
    # Strides - V_SEL
    stride_vb, stride_vh, stride_vt, stride_vd,
    # Strides - W_TOK
    stride_wb, stride_wt,
    # Strides - MEM_K
    stride_mkb, stride_mkh, stride_mkm, stride_mkd,
    # Strides - MEM_V
    stride_mvb, stride_mvh, stride_mvm, stride_mvd,
    # Strides - USED
    stride_ub, stride_um,
    # Strides - MEM_LF_K_NORM
    stride_nb, stride_nh, stride_nm, stride_nd,
    # Strides - ROUTE_IDX
    stride_rb, stride_rt,
    # Strides - MEM_K_OUT
    stride_mob, stride_moh, stride_mom, stride_mod,
    # Strides - MEM_V_OUT
    stride_vob, stride_voh, stride_vom, stride_vod,
    # Strides - USED_OUT
    stride_uob, stride_uom,
    # Scalars
    ETA_BASE,
    ENERGY_SCALE,
    # Dimensions
    H_KV: tl.constexpr,
    T: tl.constexpr,
    MS: tl.constexpr,
    DH: tl.constexpr,
    LF_START: tl.constexpr,
    DH_LF: tl.constexpr,
    OVERWRITE_ON_INSERT: tl.constexpr,
    BLOCK_DH: tl.constexpr,
):
    off_b = tl.program_id(0)

    offs_ms = tl.arange(0, MS)
    offs_lf = tl.arange(0, DH_LF)

    # Load used flags
    used_ptrs = USED + off_b * stride_ub + offs_ms * stride_um
    sram_used = tl.load(used_ptrs).to(tl.int1)

    # =============================================================
    # Phase 1: Route all candidates ONCE
    # =============================================================
    # For each candidate t, compute LF-K cosine similarity averaged
    # across all H_KV heads, then argmax to find best slot.
    # Store routing decision to ROUTE_IDX and accumulate slot weights.

    slot_count = tl.zeros([MS], dtype=tl.float32)

    for t in range(T):
        w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)

        best_slot = tl.zeros([], dtype=tl.int32)

        if w > 0:
            avg_scores = tl.zeros([MS], dtype=tl.float32)

            for h in range(H_KV):
                # Load candidate LF keys and normalize
                k_lf_ptrs = (K_SEL + off_b * stride_kb + h * stride_kh
                             + t * stride_kt + (LF_START + offs_lf) * stride_kd)
                k_lf = tl.load(k_lf_ptrs).to(tl.float32)
                k_lf_norm_val = tl.sqrt(tl.sum(k_lf * k_lf) + 1e-12)
                k_lf = k_lf / k_lf_norm_val

                # Load bank LF key norms
                n_ptrs = (MEM_LF_K_NORM + off_b * stride_nb + h * stride_nh
                          + offs_ms[:, None] * stride_nm + offs_lf[None, :] * stride_nd)
                mem_lf = tl.load(n_ptrs).to(tl.float32)

                scores_h = tl.sum(mem_lf * k_lf[None, :], axis=1)
                avg_scores += scores_h

            avg_scores = avg_scores / H_KV
            best_slot = tl.argmax(avg_scores, axis=0).to(tl.int32)

            # Accumulate weight to target slot
            slot_count += tl.where(offs_ms == best_slot, w, tl.zeros([MS], dtype=tl.float32))

        # Store routing decision
        tl.store(ROUTE_IDX + off_b * stride_rb + t * stride_rt, best_slot)

    # =============================================================
    # Phase 2: Accumulate K/V per head using Phase 1 routing
    # =============================================================
    # No re-routing! Just load the pre-computed slot index per candidate.

    for h in range(H_KV):
        # Process DH in blocks to manage register pressure
        num_dh_blocks = tl.cdiv(DH, BLOCK_DH)

        for dh_block in range(num_dh_blocks):
            d_start = dh_block * BLOCK_DH
            offs_dh = d_start + tl.arange(0, BLOCK_DH)
            mask_dh = offs_dh < DH

            # Per-slot accumulators for this DH block
            acc_k = tl.zeros([MS, BLOCK_DH], dtype=tl.float32)
            acc_v = tl.zeros([MS, BLOCK_DH], dtype=tl.float32)

            for t in range(T):
                w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)

                if w > 0:
                    # Load pre-computed routing decision (NO re-routing)
                    best_slot = tl.load(ROUTE_IDX + off_b * stride_rb + t * stride_rt)

                    # Load candidate K, V for this head and DH block
                    k_ptrs = (K_SEL + off_b * stride_kb + h * stride_kh
                              + t * stride_kt + offs_dh * stride_kd)
                    k_tok = tl.load(k_ptrs, mask=mask_dh, other=0.0).to(tl.float32) * w

                    v_ptrs = (V_SEL + off_b * stride_vb + h * stride_vh
                              + t * stride_vt + offs_dh * stride_vd)
                    v_tok = tl.load(v_ptrs, mask=mask_dh, other=0.0).to(tl.float32) * w

                    # Scatter-add: one-hot slot mask
                    slot_mask = (offs_ms == best_slot)  # (MS,)
                    acc_k += slot_mask[:, None] * k_tok[None, :]
                    acc_v += slot_mask[:, None] * v_tok[None, :]

            # Apply EMA for this head and DH block
            count4 = slot_count[:, None]  # (MS, 1)
            active = count4 > 0

            avg_k = acc_k / tl.maximum(count4, 1e-6)
            avg_v = acc_v / tl.maximum(count4, 1e-6)

            # Load current memory
            mk_ptrs = (MEM_K + off_b * stride_mkb + h * stride_mkh
                        + offs_ms[:, None] * stride_mkm + offs_dh[None, :] * stride_mkd)
            cur_mk = tl.load(mk_ptrs, mask=mask_dh[None, :], other=0.0).to(tl.float32)

            mv_ptrs = (MEM_V + off_b * stride_mvb + h * stride_mvh
                        + offs_ms[:, None] * stride_mvm + offs_dh[None, :] * stride_mvd)
            cur_mv = tl.load(mv_ptrs, mask=mask_dh[None, :], other=0.0).to(tl.float32)

            # ETA computation
            eta = tl.full([MS, BLOCK_DH], value=ETA_BASE, dtype=tl.float32)
            if OVERWRITE_ON_INSERT:
                new_slot = (~sram_used)[:, None] & active
                eta = tl.where(new_slot, tl.full([MS, BLOCK_DH], value=1.0, dtype=tl.float32), eta)

            # Phase-Safe EMA: only LF key band (dims >= LF_START)
            delta_k = tl.zeros([MS, BLOCK_DH], dtype=tl.float32)
            lf_mask = offs_dh[None, :] >= LF_START
            incoming_lf = avg_k * ENERGY_SCALE
            delta_k = tl.where(lf_mask & mask_dh[None, :], incoming_lf - cur_mk, delta_k)

            new_mk = tl.where(active, cur_mk + eta * delta_k, cur_mk)
            new_mv = tl.where(active, cur_mv + eta * (avg_v - cur_mv), cur_mv)

            # Store updated memory
            out_mk_ptrs = (MEM_K_OUT + off_b * stride_mob + h * stride_moh
                           + offs_ms[:, None] * stride_mom + offs_dh[None, :] * stride_mod)
            tl.store(out_mk_ptrs, new_mk.to(MEM_K_OUT.dtype.element_ty), mask=mask_dh[None, :])

            out_mv_ptrs = (MEM_V_OUT + off_b * stride_vob + h * stride_voh
                           + offs_ms[:, None] * stride_vom + offs_dh[None, :] * stride_vod)
            tl.store(out_mv_ptrs, new_mv.to(MEM_V_OUT.dtype.element_ty), mask=mask_dh[None, :])

    # Update used flags
    new_used = sram_used | (slot_count > 0)
    used_out_ptrs = USED_OUT + off_b * stride_uob + offs_ms * stride_uom
    tl.store(used_out_ptrs, new_used.to(tl.int8))


def _launch_summary_bank_update(
    k_sel, v_sel, w_tok, mem_k, mem_v, used, mem_lf_k_norm,
    eta_base, energy_scale, lf_start, overwrite_on_insert,
):
    B, H_KV, T, DH = k_sel.shape
    MS = mem_k.shape[2]
    DH_LF = DH - lf_start
    device = k_sel.device

    mem_k_out = torch.empty_like(mem_k)
    mem_v_out = torch.empty_like(mem_v)
    used_i8 = used.to(torch.int8).contiguous()
    used_out = torch.empty_like(used_i8)

    # Routing decisions buffer (written by Phase 1, read by Phase 2)
    route_idx = torch.empty(B, T, device=device, dtype=torch.int32)

    BLOCK_DH = 64  # Process DH in blocks of 64 to manage register pressure

    grid = (B,)
    _summary_bank_update_v2_kernel[grid](
        k_sel, v_sel, w_tok,
        mem_k, mem_v, used_i8, mem_lf_k_norm,
        route_idx,
        mem_k_out, mem_v_out, used_out,
        # K_SEL strides
        k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
        # V_SEL strides
        v_sel.stride(0), v_sel.stride(1), v_sel.stride(2), v_sel.stride(3),
        # W_TOK strides
        w_tok.stride(0), w_tok.stride(1),
        # MEM_K strides
        mem_k.stride(0), mem_k.stride(1), mem_k.stride(2), mem_k.stride(3),
        # MEM_V strides
        mem_v.stride(0), mem_v.stride(1), mem_v.stride(2), mem_v.stride(3),
        # USED strides
        used_i8.stride(0), used_i8.stride(1),
        # MEM_LF_K_NORM strides
        mem_lf_k_norm.stride(0), mem_lf_k_norm.stride(1), mem_lf_k_norm.stride(2), mem_lf_k_norm.stride(3),
        # ROUTE_IDX strides
        route_idx.stride(0), route_idx.stride(1),
        # MEM_K_OUT strides
        mem_k_out.stride(0), mem_k_out.stride(1), mem_k_out.stride(2), mem_k_out.stride(3),
        # MEM_V_OUT strides
        mem_v_out.stride(0), mem_v_out.stride(1), mem_v_out.stride(2), mem_v_out.stride(3),
        # USED_OUT strides
        used_out.stride(0), used_out.stride(1),
        # Scalars
        eta_base, energy_scale,
        # Dimensions
        H_KV=H_KV, T=T, MS=MS, DH=DH,
        LF_START=lf_start, DH_LF=DH_LF,
        OVERWRITE_ON_INSERT=overwrite_on_insert,
        BLOCK_DH=BLOCK_DH,
    )
    return mem_k_out, mem_v_out, used_out.to(torch.bool)


class ModelNew(nn.Module):
    """Triton fused summary bank update (v2: two-phase, no re-routing)."""

    def __init__(
        self,
        H_KV: int,
        T: int,
        MS: int,
        DH: int,
        lf_start: int = 64,
        eta_logit: float = -2.0,
        overwrite_on_insert: bool = True,
    ):
        super().__init__()
        self.H_KV = H_KV
        self.T = T
        self.MS = MS
        self.DH = DH
        self.lf_start = lf_start
        self.eta_base = float(torch.sigmoid(torch.tensor(eta_logit)).item())
        self.energy_scale = math.sqrt(DH / (DH - lf_start))
        self.overwrite_on_insert = overwrite_on_insert

    def forward(
        self,
        k_sel: torch.Tensor,
        v_sel: torch.Tensor,
        w_tok: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        used: torch.Tensor,
        mem_lf_k_norm: torch.Tensor,
    ) -> tuple:
        return _launch_summary_bank_update(
            k_sel, v_sel, w_tok, mem_k, mem_v, used, mem_lf_k_norm,
            self.eta_base, self.energy_scale, self.lf_start,
            self.overwrite_on_insert,
        )


# Problem dimensions (must match problem.py)
B = 2
H_KV = 8
T = 8
MS = 64
DH = 128
LF_START = 64


def get_inputs():
    dtype = torch.float32
    k_sel = torch.randn(B, H_KV, T, DH, dtype=dtype)
    v_sel = torch.randn(B, H_KV, T, DH, dtype=dtype)
    w_tok = torch.rand(B, T, dtype=dtype) * 0.5 + 0.1
    mem_k = torch.randn(B, H_KV, MS, DH, dtype=dtype)
    mem_v = torch.randn(B, H_KV, MS, DH, dtype=dtype)
    used = torch.ones(B, MS, dtype=torch.bool)
    used[:, -16:] = False
    mem_lf_k_norm = F.normalize(mem_k[..., LF_START:], dim=-1, eps=1e-6)
    return [k_sel, v_sel, w_tok, mem_k, mem_v, used, mem_lf_k_norm]


def get_init_inputs():
    return [H_KV, T, MS, DH, LF_START, -2.0, True]
