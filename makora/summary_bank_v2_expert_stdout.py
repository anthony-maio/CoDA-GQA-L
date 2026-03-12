"""Makora solution file: Triton fused summary bank update kernel (v3).

Optimized two-phase design:
  Phase 1: Route all T candidates to slots ONCE (LF-K cosine similarity
           averaged across H_KV heads). Store routing as per-slot weight
           accumulator + per-candidate best_slot decisions.
  Phase 2: For each head, iterate candidates, use Phase 1 routing decisions
           to scatter-add K/V into per-slot accumulators. Apply Phase-Safe
           EMA (LF-only for keys, full for values). Write back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _summary_bank_update_v3_kernel(
    K_SEL, V_SEL, W_TOK,
    MEM_K, MEM_V, USED, MEM_LF_K_NORM,
    ROUTE_IDX,
    MEM_K_OUT, MEM_V_OUT, USED_OUT,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_wb, stride_wt,
    stride_mkb, stride_mkh, stride_mkm, stride_mkd,
    stride_mvb, stride_mvh, stride_mvm, stride_mvd,
    stride_ub, stride_um,
    stride_nb, stride_nh, stride_nm, stride_nd,
    stride_rb, stride_rt,
    stride_mob, stride_moh, stride_mom, stride_mod,
    stride_vob, stride_voh, stride_vom, stride_vod,
    stride_uob, stride_uom,
    ETA_BASE, ENERGY_SCALE,
    H_KV: tl.constexpr, T: tl.constexpr, MS: tl.constexpr, DH: tl.constexpr,
    LF_START: tl.constexpr, DH_LF: tl.constexpr,
    OVERWRITE_ON_INSERT: tl.constexpr,
    BLOCK_MS: tl.constexpr, BLOCK_DH: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_ms_block = tl.program_id(2)
    
    offs_ms = off_ms_block * BLOCK_MS + tl.arange(0, BLOCK_MS)
    mask_ms = offs_ms < MS
    
    used_ptrs = USED + off_b * stride_ub + offs_ms * stride_um
    sram_used = tl.load(used_ptrs, mask=mask_ms, other=1).to(tl.int1)
    
    if off_h == 0:
        slot_count = tl.zeros([BLOCK_MS], dtype=tl.float32)
        
        for t in range(T):
            w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)
            best_slot = tl.zeros([], dtype=tl.int32)
            
            if w > 0:
                avg_scores = tl.zeros([BLOCK_MS], dtype=tl.float32)
                
                for h in range(H_KV):
                    k_lf_sum = tl.zeros([DH_LF], dtype=tl.float32)
                    for d_start in range(0, DH_LF, BLOCK_DH):
                        offs_lf = d_start + tl.arange(0, BLOCK_DH)
                        mask_lf = offs_lf < DH_LF
                        k_lf_ptrs = (K_SEL + off_b * stride_kb + h * stride_kh
                                     + t * stride_kt + (LF_START + offs_lf) * stride_kd)
                        k_lf_chunk = tl.load(k_lf_ptrs, mask=mask_lf, other=0.0).to(tl.float32)
                        k_lf_sum += tl.where(mask_lf, k_lf_chunk, 0.0)
                    
                    k_lf_norm_val = tl.sqrt(tl.sum(k_lf_sum * k_lf_sum) + 1e-12)
                    k_lf_normed = k_lf_sum / k_lf_norm_val
                    
                    scores_h = tl.zeros([BLOCK_MS], dtype=tl.float32)
                    for d_start in range(0, DH_LF, BLOCK_DH):
                        offs_lf = d_start + tl.arange(0, BLOCK_DH)
                        mask_lf = offs_lf < DH_LF
                        n_ptrs = (MEM_LF_K_NORM + off_b * stride_nb + h * stride_nh
                                  + offs_ms[:, None] * stride_nm + offs_lf[None, :] * stride_nd)
                        mem_lf = tl.load(n_ptrs, mask=mask_ms[:, None] & mask_lf[None, :], other=0.0).to(tl.float32)
                        scores_h += tl.sum(mem_lf * k_lf_normed[None, :], axis=1)
                    
                    avg_scores += scores_h
                
                avg_scores = avg_scores / H_KV
                local_best = tl.argmax(avg_scores, axis=0).to(tl.int32)
                global_best = off_ms_block * BLOCK_MS + local_best
                best_slot = global_best
                
                is_best_in_block = (offs_ms == local_best)
                slot_count += tl.where(is_best_in_block & mask_ms, w, 0.0)
            
            tl.store(ROUTE_IDX + off_b * stride_rb + t * stride_rt, best_slot)
    
    tl.debug_barrier()
    
    num_dh_blocks = tl.cdiv(DH, BLOCK_DH)
    
    for dh_block in range(num_dh_blocks):
        d_start = dh_block * BLOCK_DH
        offs_dh = d_start + tl.arange(0, BLOCK_DH)
        mask_dh = offs_dh < DH
        
        acc_k = tl.zeros([BLOCK_MS, BLOCK_DH], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_MS, BLOCK_DH], dtype=tl.float32)
        slot_count = tl.zeros([BLOCK_MS], dtype=tl.float32)
        
        for t in range(T):
            w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)
            
            if w > 0:
                best_slot = tl.load(ROUTE_IDX + off_b * stride_rb + t * stride_rt)
                local_slot = best_slot - off_ms_block * BLOCK_MS
                
                if local_slot >= 0 and local_slot < BLOCK_MS:
                    k_ptrs = (K_SEL + off_b * stride_kb + off_h * stride_kh
                              + t * stride_kt + offs_dh * stride_kd)
                    k_tok = tl.load(k_ptrs, mask=mask_dh, other=0.0).to(tl.float32) * w
                    
                    v_ptrs = (V_SEL + off_b * stride_vb + off_h * stride_vh
                              + t * stride_vt + offs_dh * stride_vd)
                    v_tok = tl.load(v_ptrs, mask=mask_dh, other=0.0).to(tl.float32) * w
                    
                    slot_mask = (offs_ms == local_slot)
                    acc_k += slot_mask[:, None] * k_tok[None, :]
                    acc_v += slot_mask[:, None] * v_tok[None, :]
                    slot_count += tl.where(slot_mask, w, 0.0)
        
        count4 = slot_count[:, None]
        active = count4 > 0
        
        avg_k = acc_k / tl.maximum(count4, 1e-6)
        avg_v = acc_v / tl.maximum(count4, 1e-6)
        
        mk_ptrs = (MEM_K + off_b * stride_mkb + off_h * stride_mkh
                    + offs_ms[:, None] * stride_mkm + offs_dh[None, :] * stride_mkd)
        cur_mk = tl.load(mk_ptrs, mask=mask_ms[:, None] & mask_dh[None, :], other=0.0).to(tl.float32)
        
        mv_ptrs = (MEM_V + off_b * stride_mvb + off_h * stride_mvh
                    + offs_ms[:, None] * stride_mvm + offs_dh[None, :] * stride_mvd)
        cur_mv = tl.load(mv_ptrs, mask=mask_ms[:, None] & mask_dh[None, :], other=0.0).to(tl.float32)
        
        eta = tl.full([BLOCK_MS, BLOCK_DH], value=ETA_BASE, dtype=tl.float32)
        if OVERWRITE_ON_INSERT:
            new_slot = (~sram_used)[:, None] & active
            eta = tl.where(new_slot, 1.0, eta)
        
        delta_k = tl.zeros([BLOCK_MS, BLOCK_DH], dtype=tl.float32)
        lf_mask = offs_dh[None, :] >= LF_START
        incoming_lf = avg_k * ENERGY_SCALE
        delta_k = tl.where(lf_mask & mask_dh[None, :], incoming_lf - cur_mk, delta_k)
        
        new_mk = tl.where(active, cur_mk + eta * delta_k, cur_mk)
        new_mv = tl.where(active, cur_mv + eta * (avg_v - cur_mv), cur_mv)
        
        out_mk_ptrs = (MEM_K_OUT + off_b * stride_mob + off_h * stride_moh
                       + offs_ms[:, None] * stride_mom + offs_dh[None, :] * stride_mod)
        tl.store(out_mk_ptrs, new_mk.to(MEM_K_OUT.dtype.element_ty), mask=mask_ms[:, None] & mask_dh[None, :])
        
        out_mv_ptrs = (MEM_V_OUT + off_b * stride_vob + off_h * stride_voh
                       + offs_ms[:, None] * stride_vom + offs_dh[None, :] * stride_vod)
        tl.store(out_mv_ptrs, new_mv.to(MEM_V_OUT.dtype.element_ty), mask=mask_ms[:, None] & mask_dh[None, :])
    
    if off_h == 0:
        slot_count_final = tl.zeros([BLOCK_MS], dtype=tl.float32)
        for t in range(T):
            w = tl.load(W_TOK + off_b * stride_wb + t * stride_wt)
            if w > 0:
                best_slot = tl.load(ROUTE_IDX + off_b * stride_rb + t * stride_rt)
                local_slot = best_slot - off_ms_block * BLOCK_MS
                if local_slot >= 0 and local_slot < BLOCK_MS:
                    slot_mask = (offs_ms == local_slot)
                    slot_count_final += tl.where(slot_mask, w, 0.0)
        
        new_used = sram_used | (slot_count_final > 0)
        used_out_ptrs = USED_OUT + off_b * stride_uob + offs_ms * stride_uom
        tl.store(used_out_ptrs, new_used.to(tl.int8), mask=mask_ms)


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

    route_idx = torch.empty(B, T, device=device, dtype=torch.int32)

    BLOCK_MS = 32
    BLOCK_DH = 64
    
    num_ms_blocks = triton.cdiv(MS, BLOCK_MS)

    grid = (B, H_KV, num_ms_blocks)
    _summary_bank_update_v3_kernel[grid](
        k_sel, v_sel, w_tok,
        mem_k, mem_v, used_i8, mem_lf_k_norm,
        route_idx,
        mem_k_out, mem_v_out, used_out,
        k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
        v_sel.stride(0), v_sel.stride(1), v_sel.stride(2), v_sel.stride(3),
        w_tok.stride(0), w_tok.stride(1),
        mem_k.stride(0), mem_k.stride(1), mem_k.stride(2), mem_k.stride(3),
        mem_v.stride(0), mem_v.stride(1), mem_v.stride(2), mem_v.stride(3),
        used_i8.stride(0), used_i8.stride(1),
        mem_lf_k_norm.stride(0), mem_lf_k_norm.stride(1), mem_lf_k_norm.stride(2), mem_lf_k_norm.stride(3),
        route_idx.stride(0), route_idx.stride(1),
        mem_k_out.stride(0), mem_k_out.stride(1), mem_k_out.stride(2), mem_k_out.stride(3),
        mem_v_out.stride(0), mem_v_out.stride(1), mem_v_out.stride(2), mem_v_out.stride(3),
        used_out.stride(0), used_out.stride(1),
        eta_base, energy_scale,
        H_KV=H_KV, T=T, MS=MS, DH=DH,
        LF_START=lf_start, DH_LF=DH_LF,
        OVERWRITE_ON_INSERT=overwrite_on_insert,
        BLOCK_MS=BLOCK_MS, BLOCK_DH=BLOCK_DH,
    )
    return mem_k_out, mem_v_out, used_out.to(torch.bool)


class ModelNew(nn.Module):
    """Triton fused summary bank update (v3: optimized multi-head parallelism)."""

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
