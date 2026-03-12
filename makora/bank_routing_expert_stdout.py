"""Makora solution file: Triton fused exact-bank routing kernel.

Single kernel replacing ~15 PyTorch launches: cosine similarity scoring,
candidate classification (novel/hit/skip), and LRU victim assignment with
sequential state updates for deterministic slot allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _exact_bank_route_kernel(
    V_SEL_NORM,
    MEM_V_NORM,
    USED,
    LAST_IN,
    GATE_SEL,
    POS_SEL,
    IDX_TOK,
    OVERWRITE_TOK,
    TOUCH_TOK,
    LAST_OUT,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_mb, stride_mh, stride_mm, stride_md,
    stride_ub, stride_um,
    stride_lb, stride_lm,
    stride_gb, stride_gt,
    stride_pb, stride_pt,
    stride_ib, stride_it,
    stride_ob, stride_ot,
    stride_tb, stride_tt,
    stride_lob, stride_lom,
    TAU_GATE,
    TAU_MATCH,
    TAU_NOVEL,
    H_KV: tl.constexpr,
    T: tl.constexpr,
    ME: tl.constexpr,
    DH: tl.constexpr,
    REFRESH_ON_HIT: tl.constexpr,
    AE: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_ME: tl.constexpr,
):
    off_b = tl.program_id(0)

    offs_me = tl.arange(0, BLOCK_ME)
    active_mask = offs_me < AE

    used_ptrs = USED + off_b * stride_ub + offs_me * stride_um
    sram_used = tl.load(used_ptrs, mask=active_mask, other=0).to(tl.int1)

    last_ptrs = LAST_IN + off_b * stride_lb + offs_me * stride_lm
    sram_last = tl.load(last_ptrs, mask=active_mask, other=0)

    any_used = tl.max(sram_used.to(tl.int32), axis=0)

    for t in range(T):
        gate = tl.load(GATE_SEL + off_b * stride_gb + t * stride_gt)

        idx_t: tl.int64 = tl.zeros([], dtype=tl.int64)
        overwrite_t: tl.int1 = tl.zeros([], dtype=tl.int1)
        touch_t: tl.int1 = tl.zeros([], dtype=tl.int1)

        keep = gate >= TAU_GATE

        if keep:
            avg_scores = tl.zeros([BLOCK_ME], dtype=tl.float32)

            for h in range(H_KV):
                for d_chunk in range(0, DH, BLOCK_DH):
                    offs_dh = d_chunk + tl.arange(0, BLOCK_DH)
                    dh_mask = offs_dh < DH

                    v_ptrs = (V_SEL_NORM + off_b * stride_vb
                              + h * stride_vh + t * stride_vt
                              + offs_dh * stride_vd)
                    v_tok = tl.load(v_ptrs, mask=dh_mask, other=0.0).to(tl.float32)

                    m_ptrs = (MEM_V_NORM + off_b * stride_mb
                              + h * stride_mh
                              + offs_me[:, None] * stride_mm
                              + offs_dh[None, :] * stride_md)
                    mem_mask = active_mask[:, None] & dh_mask[None, :]
                    mem_tile = tl.load(m_ptrs, mask=mem_mask, other=0.0).to(tl.float32)

                    scores_h = tl.sum(mem_tile * v_tok[None, :], axis=1)
                    avg_scores += scores_h

            avg_scores = avg_scores / H_KV
            avg_scores = tl.where(sram_used & active_mask, avg_scores, -1e9)

            best_score = tl.max(avg_scores, axis=0)
            best_idx = tl.argmax(avg_scores, axis=0).to(tl.int64)

            is_novel = (any_used == 0) | (best_score < TAU_NOVEL)
            is_hit = (any_used != 0) & (best_score >= TAU_MATCH)

            if is_novel:
                lru_key = tl.where(
                    active_mask,
                    tl.where(sram_used, sram_last, tl.full([BLOCK_ME], value=-2**62, dtype=tl.int64)),
                    tl.full([BLOCK_ME], value=2**62, dtype=tl.int64),
                )
                victim = tl.argmin(lru_key, axis=0).to(tl.int64)

                idx_t = victim
                overwrite_t = tl.full([], value=1, dtype=tl.int1)
                touch_t = tl.full([], value=1, dtype=tl.int1)

                pos_t = tl.load(POS_SEL + off_b * stride_pb + t * stride_pt)
                sram_used = sram_used | (offs_me == victim)
                sram_last = tl.where(offs_me == victim, pos_t, sram_last)
                any_used = 1

            elif is_hit:
                idx_t = best_idx
                overwrite_t = tl.full([], value=1, dtype=tl.int1) if REFRESH_ON_HIT else tl.zeros([], dtype=tl.int1)
                touch_t = tl.full([], value=1, dtype=tl.int1)

                pos_t = tl.load(POS_SEL + off_b * stride_pb + t * stride_pt)
                sram_last = tl.where(offs_me == best_idx, pos_t, sram_last)

            else:
                idx_t = best_idx
                overwrite_t = tl.zeros([], dtype=tl.int1)
                touch_t = tl.zeros([], dtype=tl.int1)

        idx_ptr = IDX_TOK + off_b * stride_ib + t * stride_it
        overwrite_ptr = OVERWRITE_TOK + off_b * stride_ob + t * stride_ot
        touch_ptr = TOUCH_TOK + off_b * stride_tb + t * stride_tt

        tl.inline_asm_elementwise(
            "st.global.cs.b64 [$0], $1;", "l,l", [idx_ptr, idx_t], dtype=tl.int64, is_pure=False, pack=1
        )
        tl.inline_asm_elementwise(
            "st.global.cs.b8 [$0], $1;", "l,r", [overwrite_ptr, overwrite_t], dtype=tl.int1, is_pure=False, pack=1
        )
        tl.inline_asm_elementwise(
            "st.global.cs.b8 [$0], $1;", "l,r", [touch_ptr, touch_t], dtype=tl.int1, is_pure=False, pack=1
        )

    last_out_ptrs = LAST_OUT + off_b * stride_lob + offs_me * stride_lom
    tl.store(last_out_ptrs, sram_last, mask=active_mask)


def _launch_exact_bank_route(
    v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel,
    tau_gate, tau_match, tau_novel, refresh_on_hit, active_exact,
):
    B, H_KV, T, DH = v_sel_norm.shape
    ME = mem_v_norm.shape[2]
    device = v_sel_norm.device

    idx_tok = torch.zeros(B, T, device=device, dtype=torch.int64)
    overwrite_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
    touch_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
    last_out = last.clone()

    used_i8 = used.to(torch.int8).contiguous()

    BLOCK_DH = min(128, triton.next_power_of_2(DH))
    BLOCK_ME = triton.next_power_of_2(ME)

    grid = (B,)
    _exact_bank_route_kernel[grid](
        v_sel_norm, mem_v_norm, used_i8, last, gate_sel, pos_sel,
        idx_tok, overwrite_tok, touch_tok, last_out,
        v_sel_norm.stride(0), v_sel_norm.stride(1), v_sel_norm.stride(2), v_sel_norm.stride(3),
        mem_v_norm.stride(0), mem_v_norm.stride(1), mem_v_norm.stride(2), mem_v_norm.stride(3),
        used_i8.stride(0), used_i8.stride(1),
        last.stride(0), last.stride(1),
        gate_sel.stride(0), gate_sel.stride(1),
        pos_sel.stride(0), pos_sel.stride(1),
        idx_tok.stride(0), idx_tok.stride(1),
        overwrite_tok.stride(0), overwrite_tok.stride(1),
        touch_tok.stride(0), touch_tok.stride(1),
        last_out.stride(0), last_out.stride(1),
        tau_gate, tau_match, tau_novel,
        H_KV=H_KV, T=T, ME=ME, DH=DH,
        REFRESH_ON_HIT=refresh_on_hit,
        AE=active_exact,
        BLOCK_DH=BLOCK_DH,
        BLOCK_ME=BLOCK_ME,
    )
    return idx_tok, overwrite_tok, touch_tok, last_out


class ModelNew(nn.Module):
    """Triton fused exact-bank routing."""

    def __init__(
        self,
        H_KV: int,
        T: int,
        ME: int,
        DH: int,
        tau_gate: float = 0.1,
        tau_match: float = 0.8,
        tau_novel: float = 0.3,
        refresh_on_hit: bool = True,
    ):
        super().__init__()
        self.H_KV = H_KV
        self.T = T
        self.ME = ME
        self.DH = DH
        self.tau_gate = tau_gate
        self.tau_match = tau_match
        self.tau_novel = tau_novel
        self.refresh_on_hit = refresh_on_hit

    def forward(
        self,
        v_sel_norm: torch.Tensor,
        mem_v_norm: torch.Tensor,
        used: torch.Tensor,
        last: torch.Tensor,
        gate_sel: torch.Tensor,
        pos_sel: torch.Tensor,
    ) -> tuple:
        return _launch_exact_bank_route(
            v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel,
            self.tau_gate, self.tau_match, self.tau_novel,
            self.refresh_on_hit, self.ME,
        )
