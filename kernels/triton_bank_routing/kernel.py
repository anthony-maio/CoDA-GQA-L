"""Triton fused exact-bank routing kernel.

Replaces the scoring + classification + LRU assignment phases of
_exact_update_block_vectorized with a single kernel launch.

Architecture:
  Grid: (B,) — one thread block per batch element.
  Each block iterates over KV heads to compute per-head cosine similarity,
  then averages across heads (matching the PyTorch cross-head routing
  semantics).  Candidates are processed sequentially to produce deterministic
  LRU victim assignments: each novel candidate claims an LRU slot that is
  immediately visible to subsequent candidates.

Replaces ~15 PyTorch kernel launches (matmul, mean, max, masked_fill,
topk, cumsum, clamp, gather, 6x comparison/where) with 1 Triton kernel.

Inputs:
  v_sel_norm: (B, Hkv, T, Dh)  pre-normalized candidate values
  mem_v_norm: (B, Hkv, Me, Dh) cached normalized bank values
  used:       (B, Me)           which bank slots are valid
  last:       (B, Me)           LRU timestamps (int64)
  gate_sel:   (B, T)            write gate values
  pos_sel:    (B, T)            position timestamps

Outputs (written in-place):
  idx_tok:       (B, T)   target slot index per candidate
  overwrite_tok: (B, T)   True if candidate should overwrite its target slot
  touch_tok:     (B, T)   True if candidate should update LRU timestamp
  last_out:      (B, Me)  updated LRU timestamps
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _exact_bank_route_kernel(
    # Candidate data
    V_SEL_NORM,   # (B, Hkv, T, Dh) fp16/bf16/fp32
    # Bank data
    MEM_V_NORM,   # (B, Hkv, Me, Dh) fp16/bf16/fp32
    USED,         # (B, Me) bool (stored as int8)
    LAST_IN,      # (B, Me) int64
    GATE_SEL,     # (B, T) fp32
    POS_SEL,      # (B, T) int64
    # Outputs
    IDX_TOK,      # (B, T) int64
    OVERWRITE_TOK,  # (B, T) bool
    TOUCH_TOK,    # (B, T) bool
    LAST_OUT,     # (B, Me) int64
    # Strides — V_SEL_NORM: (B, Hkv, T, Dh)
    stride_vb, stride_vh, stride_vt, stride_vd,
    # Strides — MEM_V_NORM: (B, Hkv, Me, Dh)
    stride_mb, stride_mh, stride_mm, stride_md,
    # Strides — USED: (B, Me)
    stride_ub, stride_um,
    # Strides — LAST: (B, Me)
    stride_lb, stride_lm,
    # Strides — GATE_SEL: (B, T)
    stride_gb, stride_gt,
    # Strides — POS_SEL: (B, T)
    stride_pb, stride_pt,
    # Strides — IDX_TOK: (B, T)
    stride_ib, stride_it,
    # Strides — OVERWRITE_TOK: (B, T)
    stride_ob, stride_ot,
    # Strides — TOUCH_TOK: (B, T)
    stride_tb, stride_tt,
    # Strides — LAST_OUT: (B, Me)
    stride_lob, stride_lom,
    # Scalars
    TAU_GATE,         # exact gate threshold (fp32)
    TAU_MATCH,        # match threshold (fp32)
    TAU_NOVEL,        # novelty threshold (fp32)
    # Dimensions
    H_KV: tl.constexpr,
    T: tl.constexpr,       # num candidates (typ 8-16)
    ME: tl.constexpr,      # exact bank size (typ 64)
    DH: tl.constexpr,      # head dim (typ 128)
    REFRESH_ON_HIT: tl.constexpr,  # bool
):
    off_b = tl.program_id(0)

    offs_me = tl.arange(0, ME)
    offs_dh = tl.arange(0, DH)

    # ------------------------------------------------------------------
    # 1. Load shared routing state into SRAM (registers)
    # ------------------------------------------------------------------
    # used: (ME,) bool — loaded as int8
    used_ptrs = USED + off_b * stride_ub + offs_me * stride_um
    sram_used = tl.load(used_ptrs).to(tl.int1)

    # last: (ME,) int64 — LRU timestamps
    last_ptrs = LAST_IN + off_b * stride_lb + offs_me * stride_lm
    sram_last = tl.load(last_ptrs)

    any_used = tl.max(sram_used.to(tl.int32), axis=0) > 0

    # ------------------------------------------------------------------
    # 2. Process each candidate sequentially
    # ------------------------------------------------------------------
    # Sequential processing gives deterministic LRU assignment: each
    # novel candidate claims a victim slot that is immediately visible
    # to the next candidate.

    for t in range(T):
        # Load gate for this candidate
        gate = tl.load(GATE_SEL + off_b * stride_gb + t * stride_gt)

        # Default outputs: not overwrite, not touch, idx = 0
        idx_t: tl.int64 = tl.zeros([], dtype=tl.int64)
        overwrite_t: tl.int1 = tl.zeros([], dtype=tl.int1)
        touch_t: tl.int1 = tl.zeros([], dtype=tl.int1)

        keep = gate >= TAU_GATE

        if keep:
            # -------------------------------------------------------
            # Cosine similarity: average across KV heads
            # -------------------------------------------------------
            # Accumulate scores in fp32: (ME,)
            avg_scores = tl.zeros([ME], dtype=tl.float32)

            for h in range(H_KV):
                # Load candidate v_norm for this head: (DH,)
                v_ptrs = (V_SEL_NORM + off_b * stride_vb
                          + h * stride_vh + t * stride_vt
                          + offs_dh * stride_vd)
                v_tok = tl.load(v_ptrs).to(tl.float32)

                # Load bank v_norms for this head: (ME, DH)
                m_ptrs = (MEM_V_NORM + off_b * stride_mb
                          + h * stride_mh
                          + offs_me[:, None] * stride_mm
                          + offs_dh[None, :] * stride_md)
                mem_tile = tl.load(m_ptrs).to(tl.float32)

                # Dot product: (ME, DH) · (DH,) → (ME,)
                scores_h = tl.sum(mem_tile * v_tok[None, :], axis=1)
                avg_scores += scores_h

            avg_scores = avg_scores / H_KV

            # Mask unused slots
            avg_scores = tl.where(sram_used, avg_scores, -1e9)

            best_score = tl.max(avg_scores, axis=0)
            best_idx = tl.argmax(avg_scores, axis=0).to(tl.int64)

            # -------------------------------------------------------
            # Classify: novel, hit, or skip
            # -------------------------------------------------------
            is_novel = (any_used == 0) | (best_score < TAU_NOVEL)
            is_hit = (any_used != 0) & (best_score >= TAU_MATCH)

            if is_novel:
                # Find LRU victim: slot with minimum timestamp.
                # Unused slots get -inf timestamp so they're picked first.
                lru_key = tl.where(sram_used, sram_last, tl.full([ME], value=-2**62, dtype=tl.int64))
                victim = tl.argmin(lru_key, axis=0).to(tl.int64)

                idx_t = victim
                overwrite_t = tl.full([], value=1, dtype=tl.int1)
                touch_t = tl.full([], value=1, dtype=tl.int1)

                # Update SRAM: mark victim as used, update LRU timestamp.
                # This is immediately visible to the next candidate.
                pos_t = tl.load(POS_SEL + off_b * stride_pb + t * stride_pt)
                sram_used = sram_used | (offs_me == victim)
                sram_last = tl.where(offs_me == victim, pos_t, sram_last)
                any_used = 1

                # Note: we do NOT update sram mem_v_norm here because the
                # actual K/V write happens in PyTorch after this kernel.
                # The norm cache is updated selectively in Python.
                # For back-to-back novel tokens targeting different victims,
                # the routing is still correct because the LRU state (sram_last,
                # sram_used) IS updated, preventing double-eviction.

            elif is_hit:
                idx_t = best_idx
                overwrite_t = tl.full([], value=1, dtype=tl.int1) if REFRESH_ON_HIT else tl.zeros([], dtype=tl.int1)
                touch_t = tl.full([], value=1, dtype=tl.int1)

                # Update LRU timestamp for hit
                pos_t = tl.load(POS_SEL + off_b * stride_pb + t * stride_pt)
                sram_last = tl.where(offs_me == best_idx, pos_t, sram_last)

            else:
                # Neither novel nor hit: skip (between thresholds)
                idx_t = best_idx
                overwrite_t = tl.zeros([], dtype=tl.int1)
                touch_t = tl.zeros([], dtype=tl.int1)

        # -------------------------------------------------------
        # Store per-candidate outputs
        # -------------------------------------------------------
        tl.store(IDX_TOK + off_b * stride_ib + t * stride_it, idx_t)
        tl.store(OVERWRITE_TOK + off_b * stride_ob + t * stride_ot, overwrite_t)
        tl.store(TOUCH_TOK + off_b * stride_tb + t * stride_tt, touch_t)

    # ------------------------------------------------------------------
    # 3. Store updated LRU timestamps back to HBM
    # ------------------------------------------------------------------
    last_out_ptrs = LAST_OUT + off_b * stride_lob + offs_me * stride_lom
    tl.store(last_out_ptrs, sram_last)
