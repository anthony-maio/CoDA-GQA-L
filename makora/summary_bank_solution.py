"""Makora solution file: Triton fused summary bank update kernel.

Fuses LF-K routing (cosine similarity on low-frequency key band),
scatter-add weighted aggregation, and Phase-Safe EMA blending into
a single kernel pass per batch element.

Architecture:
  Grid: (B,) — one program per batch element.
  Each program:
    1. Routes candidates to summary slots via LF-K cosine similarity
    2. Accumulates weighted K/V contributions per slot (scatter-add in SRAM)
    3. Applies Phase-Safe EMA (LF-only for keys, full for values)
    4. Writes updated bank back to HBM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _summary_bank_update_kernel(
    # Candidate data
    K_SEL,          # (B, Hkv, T, Dh)
    V_SEL,          # (B, Hkv, T, Dh)
    W_TOK,          # (B, T) float32
    # Bank data (read + write)
    MEM_K,          # (B, Hkv, Ms, Dh)
    MEM_V,          # (B, Hkv, Ms, Dh)
    USED,           # (B, Ms) bool as int8
    MEM_LF_K_NORM,  # (B, Hkv, Ms, Dh_lf) cached normalized LF keys
    # Outputs
    MEM_K_OUT,      # (B, Hkv, Ms, Dh)
    MEM_V_OUT,      # (B, Hkv, Ms, Dh)
    USED_OUT,       # (B, Ms) int8
    # Strides - K_SEL: (B, Hkv, T, Dh)
    stride_kb, stride_kh, stride_kt, stride_kd,
    # Strides - V_SEL: (B, Hkv, T, Dh)
    stride_vb, stride_vh, stride_vt, stride_vd,
    # Strides - W_TOK: (B, T)
    stride_wb, stride_wt,
    # Strides - MEM_K: (B, Hkv, Ms, Dh)
    stride_mkb, stride_mkh, stride_mkm, stride_mkd,
    # Strides - MEM_V: (B, Hkv, Ms, Dh)
    stride_mvb, stride_mvh, stride_mvm, stride_mvd,
    # Strides - USED: (B, Ms)
    stride_ub, stride_um,
    # Strides - MEM_LF_K_NORM: (B, Hkv, Ms, Dh_lf)
    stride_nb, stride_nh, stride_nm, stride_nd,
    # Strides - MEM_K_OUT: (B, Hkv, Ms, Dh)
    stride_mob, stride_moh, stride_mom, stride_mod,
    # Strides - MEM_V_OUT: (B, Hkv, Ms, Dh)
    stride_vob, stride_voh, stride_vom, stride_vod,
    # Strides - USED_OUT: (B, Ms)
    stride_uob, stride_uom,
    # Scalars
    ETA_BASE,       # sigmoid(eta_logit) float32
    ENERGY_SCALE,   # sqrt(Dh / Dh_lf) float32
    # Dimensions
    H_KV: tl.constexpr,
    T: tl.constexpr,
    MS: tl.constexpr,
    DH: tl.constexpr,
    LF_START: tl.constexpr,
    DH_LF: tl.constexpr,
    OVERWRITE_ON_INSERT: tl.constexpr,
):
    off_b = tl.program_id(0)

    offs_ms = tl.arange(0, MS)
    offs_dh = tl.arange(0, DH)
    offs_lf = tl.arange(0, DH_LF)

    # Load used flags
    used_ptrs = USED + off_b * stride_ub + offs_ms * stride_um
    sram_used = tl.load(used_ptrs).to(tl.int1)

    # -------------------------------------------------------
    # Phase 1: Route each candidate to best matching slot
    # -------------------------------------------------------
    # Store per-candidate routing decisions
    # We'll accumulate weighted contributions in per-slot accumulators

    # Per-slot accumulators for weighted K and V (in fp32)
    # We process candidates sequentially and accumulate into slot buffers
    slot_count = tl.zeros([MS], dtype=tl.float32)

    # We need per-slot, per-head accumulators but that's too much SRAM
    # for (MS * Hkv * Dh). Instead, process per-head sequentially.

    # First pass: route all candidates and accumulate counts
    route_idx = tl.zeros([T], dtype=tl.int32)  # which slot each candidate maps to

    for t in range(T):
        w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)

        if w > 0:
            # LF-K cosine similarity across heads
            avg_scores = tl.zeros([MS], dtype=tl.float32)

            for h in range(H_KV):
                # Load candidate LF keys and normalize
                k_lf_ptrs = (K_SEL + off_b * stride_kb + h * stride_kh
                             + t * stride_kt + (LF_START + offs_lf) * stride_kd)
                k_lf = tl.load(k_lf_ptrs).to(tl.float32)
                # Normalize
                k_lf_norm_val = tl.sqrt(tl.sum(k_lf * k_lf) + 1e-12)
                k_lf = k_lf / k_lf_norm_val

                # Load bank LF key norms: (MS, DH_LF)
                n_ptrs = (MEM_LF_K_NORM + off_b * stride_nb + h * stride_nh
                          + offs_ms[:, None] * stride_nm + offs_lf[None, :] * stride_nd)
                mem_lf = tl.load(n_ptrs).to(tl.float32)

                # Dot product: (MS, DH_LF) @ (DH_LF,) -> (MS,)
                scores_h = tl.sum(mem_lf * k_lf[None, :], axis=1)
                avg_scores += scores_h

            avg_scores = avg_scores / H_KV
            best_slot = tl.argmax(avg_scores, axis=0)
            # Accumulate weight to this slot
            slot_count += tl.where(offs_ms == best_slot, w, tl.zeros([MS], dtype=tl.float32))
        else:
            best_slot = 0

        # Store routing decision (can't index into route_idx dynamically in Triton,
        # so we store to a temporary output or use a different strategy)
        # Instead, we'll do a second pass per head

    # -------------------------------------------------------
    # Phase 2: For each head, re-route and accumulate K/V
    # -------------------------------------------------------
    # This re-computes routing (cheap) to avoid storing per-candidate decisions

    for h in range(H_KV):
        # Per-slot K accumulator: (MS, DH)
        acc_k = tl.zeros([MS, DH], dtype=tl.float32)
        acc_v = tl.zeros([MS, DH], dtype=tl.float32)

        for t in range(T):
            w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)

            if w > 0:
                # Re-route: LF-K cosine similarity across ALL heads
                avg_scores = tl.zeros([MS], dtype=tl.float32)
                for hh in range(H_KV):
                    k_lf_ptrs = (K_SEL + off_b * stride_kb + hh * stride_kh
                                 + t * stride_kt + (LF_START + offs_lf) * stride_kd)
                    k_lf = tl.load(k_lf_ptrs).to(tl.float32)
                    k_lf_norm_val = tl.sqrt(tl.sum(k_lf * k_lf) + 1e-12)
                    k_lf = k_lf / k_lf_norm_val

                    n_ptrs = (MEM_LF_K_NORM + off_b * stride_nb + hh * stride_nh
                              + offs_ms[:, None] * stride_nm + offs_lf[None, :] * stride_nd)
                    mem_lf = tl.load(n_ptrs).to(tl.float32)
                    scores_h = tl.sum(mem_lf * k_lf[None, :], axis=1)
                    avg_scores += scores_h
                avg_scores = avg_scores / H_KV
                best_slot = tl.argmax(avg_scores, axis=0)

                # Load candidate K, V for this head
                k_ptrs = K_SEL + off_b * stride_kb + h * stride_kh + t * stride_kt + offs_dh * stride_kd
                k_tok = tl.load(k_ptrs).to(tl.float32) * w
                v_ptrs = V_SEL + off_b * stride_vb + h * stride_vh + t * stride_vt + offs_dh * stride_vd
                v_tok = tl.load(v_ptrs).to(tl.float32) * w

                # Scatter-add into slot
                slot_mask = (offs_ms == best_slot)  # (MS,)
                acc_k += slot_mask[:, None] * k_tok[None, :]
                acc_v += slot_mask[:, None] * v_tok[None, :]

        # Apply EMA to this head's memory
        count4 = slot_count[:, None]  # (MS, 1)
        active = count4 > 0

        avg_k = acc_k / tl.maximum(count4, 1e-6)
        avg_v = acc_v / tl.maximum(count4, 1e-6)

        # Load current memory for this head
        mk_ptrs = (MEM_K + off_b * stride_mkb + h * stride_mkh
                    + offs_ms[:, None] * stride_mkm + offs_dh[None, :] * stride_mkd)
        cur_mk = tl.load(mk_ptrs).to(tl.float32)
        mv_ptrs = (MEM_V + off_b * stride_mvb + h * stride_mvh
                    + offs_ms[:, None] * stride_mvm + offs_dh[None, :] * stride_mvd)
        cur_mv = tl.load(mv_ptrs).to(tl.float32)

        # ETA: base or 1.0 for new slots
        eta = tl.full([MS, DH], value=ETA_BASE, dtype=tl.float32)
        if OVERWRITE_ON_INSERT:
            new_slot = (~sram_used)[:, None] & active
            eta = tl.where(new_slot, tl.full([MS, DH], value=1.0, dtype=tl.float32), eta)

        # Phase-Safe EMA: only LF key band
        # Build delta_k: zeros for HF, (avg_k_lf * energy_scale - mem_k_lf) for LF
        delta_k = tl.zeros([MS, DH], dtype=tl.float32)
        # Create LF mask: dims >= LF_START
        lf_mask = offs_dh[None, :] >= LF_START  # (1, DH) broadcast to (MS, DH)
        incoming_lf = avg_k * ENERGY_SCALE
        delta_k = tl.where(lf_mask, incoming_lf - cur_mk, delta_k)

        new_mk = tl.where(active, cur_mk + eta * delta_k, cur_mk)
        new_mv = tl.where(active, cur_mv + eta * (avg_v - cur_mv), cur_mv)

        # Store updated memory
        out_mk_ptrs = (MEM_K_OUT + off_b * stride_mob + h * stride_moh
                       + offs_ms[:, None] * stride_mom + offs_dh[None, :] * stride_mod)
        tl.store(out_mk_ptrs, new_mk.to(MEM_K_OUT.dtype.element_ty))
        out_mv_ptrs = (MEM_V_OUT + off_b * stride_vob + h * stride_voh
                       + offs_ms[:, None] * stride_vom + offs_dh[None, :] * stride_vod)
        tl.store(out_mv_ptrs, new_mv.to(MEM_V_OUT.dtype.element_ty))

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

    grid = (B,)
    _summary_bank_update_kernel[grid](
        k_sel, v_sel, w_tok,
        mem_k, mem_v, used_i8, mem_lf_k_norm,
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
        # MEM_K_OUT strides
        mem_k_out.stride(0), mem_k_out.stride(1), mem_k_out.stride(2), mem_k_out.stride(3),
        # MEM_V_OUT strides
        mem_v_out.stride(0), mem_v_out.stride(1), mem_v_out.stride(2), mem_v_out.stride(3),
        # USED_OUT strides
        used_out.stride(0), used_out.stride(1),
        # Scalars
        eta_base,
        energy_scale,
        # Dimensions
        H_KV=H_KV, T=T, MS=MS, DH=DH,
        LF_START=lf_start, DH_LF=DH_LF,
        OVERWRITE_ON_INSERT=overwrite_on_insert,
    )
    return mem_k_out, mem_v_out, used_out.to(torch.bool)


class ModelNew(nn.Module):
    """Triton fused summary bank update."""

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
