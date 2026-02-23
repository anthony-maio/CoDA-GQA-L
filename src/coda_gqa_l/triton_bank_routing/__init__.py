"""Fused exact-bank routing kernel for CoDA-GQA inference.

Replaces ~15 PyTorch kernel launches (matmul, mean, max, topk, cumsum,
masked_fill, 6x comparison/where) with a single Triton kernel that
computes V-routing cosine similarity, classifies candidates as
novel/hit/skip, and assigns LRU victim slots — all with deterministic
sequential state updates.

Forward-pass only (inference).  Falls back to PyTorch when Triton is
unavailable.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    from .kernel import _exact_bank_route_kernel
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    pass


def is_available() -> bool:
    """True when the Triton kernel is importable and can be launched."""
    return _HAS_TRITON


def diagnose() -> str:
    """Human-readable status string for debugging kernel availability."""
    if _HAS_TRITON:
        import triton
        return f"triton_bank_routing: available (triton {triton.__version__})"
    try:
        import triton  # noqa: F811
        return f"triton imported ({triton.__version__}) but kernel failed to load"
    except ImportError:
        return "triton_bank_routing: triton not installed"


def exact_bank_route(
    v_sel_norm: Tensor,   # (B, Hkv, T, Dh)
    mem_v_norm: Tensor,   # (B, Hkv, Me, Dh)
    used: Tensor,         # (B, Me)  bool
    last: Tensor,         # (B, Me)  int64
    gate_sel: Tensor,     # (B, T)   fp32
    pos_sel: Tensor,      # (B, T)   int64
    *,
    tau_gate: float,
    tau_match: float,
    tau_novel: float,
    refresh_on_hit: bool = False,
    active_exact: int = -1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused exact-bank routing: score + classify + LRU assign in one kernel.

    Args:
        v_sel_norm:  Pre-normalized candidate values (B, Hkv, T, Dh).
        mem_v_norm:  Cached normalized bank values (B, Hkv, Me, Dh).
        used:        Valid-slot mask (B, Me).
        last:        LRU timestamps (B, Me), int64.
        gate_sel:    Write gate values (B, T), fp32.
        pos_sel:     Position timestamps (B, T), int64.
        tau_gate:    Minimum gate threshold for exact bank consideration.
        tau_match:   Cosine similarity threshold for a "hit".
        tau_novel:   Maximum cosine similarity for "novel" classification.
        refresh_on_hit: Whether hits overwrite the slot (refresh K/V).
        active_exact: Active bank capacity (slots [0, active_exact) are usable).
                      -1 or >= Me means use all slots (no expansion restriction).

    Returns:
        idx_tok:       (B, T) int64  — target slot index per candidate
        overwrite_tok: (B, T) bool   — True if candidate should overwrite
        touch_tok:     (B, T) bool   — True if LRU timestamp should update
        last_out:      (B, Me) int64 — updated LRU timestamps
    """
    if not _HAS_TRITON or not v_sel_norm.is_cuda:
        return _fallback(
            v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel,
            tau_gate=tau_gate, tau_match=tau_match, tau_novel=tau_novel,
            refresh_on_hit=refresh_on_hit, active_exact=active_exact,
        )

    B, Hkv, T, Dh = v_sel_norm.shape
    _, _, Me, _ = mem_v_norm.shape
    Ae = active_exact if 0 < active_exact < Me else Me

    # Ensure contiguous
    v_sel_norm = v_sel_norm.contiguous()
    mem_v_norm = mem_v_norm.contiguous()
    used = used.contiguous()
    last = last.contiguous()
    gate_sel = gate_sel.float().contiguous()
    pos_sel = pos_sel.to(torch.int64).contiguous()

    # Allocate outputs
    idx_tok = torch.zeros((B, T), device=v_sel_norm.device, dtype=torch.int64)
    overwrite_tok = torch.zeros((B, T), device=v_sel_norm.device, dtype=torch.bool)
    touch_tok = torch.zeros((B, T), device=v_sel_norm.device, dtype=torch.bool)
    last_out = last.clone()

    grid = (B,)

    _exact_bank_route_kernel[grid](
        v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel,
        idx_tok, overwrite_tok, touch_tok, last_out,
        # V_SEL_NORM strides
        v_sel_norm.stride(0), v_sel_norm.stride(1),
        v_sel_norm.stride(2), v_sel_norm.stride(3),
        # MEM_V_NORM strides
        mem_v_norm.stride(0), mem_v_norm.stride(1),
        mem_v_norm.stride(2), mem_v_norm.stride(3),
        # USED strides
        used.stride(0), used.stride(1),
        # LAST_IN strides
        last.stride(0), last.stride(1),
        # GATE_SEL strides
        gate_sel.stride(0), gate_sel.stride(1),
        # POS_SEL strides
        pos_sel.stride(0), pos_sel.stride(1),
        # IDX_TOK strides
        idx_tok.stride(0), idx_tok.stride(1),
        # OVERWRITE_TOK strides
        overwrite_tok.stride(0), overwrite_tok.stride(1),
        # TOUCH_TOK strides
        touch_tok.stride(0), touch_tok.stride(1),
        # LAST_OUT strides
        last_out.stride(0), last_out.stride(1),
        # Scalars
        tau_gate, tau_match, tau_novel,
        # Dimensions
        H_KV=Hkv, T=T, ME=Me, DH=Dh,
        REFRESH_ON_HIT=refresh_on_hit,
        AE=Ae,
    )

    return idx_tok, overwrite_tok, touch_tok, last_out


def _fallback(
    v_sel_norm: Tensor,
    mem_v_norm: Tensor,
    used: Tensor,
    last: Tensor,
    gate_sel: Tensor,
    pos_sel: Tensor,
    *,
    tau_gate: float,
    tau_match: float,
    tau_novel: float,
    refresh_on_hit: bool,
    active_exact: int = -1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch reference fallback (same semantics as the Triton kernel)."""
    import torch.nn.functional as F

    B, Hkv, T, Dh = v_sel_norm.shape
    Me = mem_v_norm.shape[2]
    Ae = active_exact if 0 < active_exact < Me else Me
    device = v_sel_norm.device

    keep_tok = gate_sel >= tau_gate

    # V-routing cosine similarity averaged across heads
    scores_h = torch.matmul(
        v_sel_norm.reshape(B * Hkv, T, Dh),
        mem_v_norm.reshape(B * Hkv, Me, Dh).transpose(1, 2),
    ).view(B, Hkv, T, Me)
    scores = scores_h.mean(dim=1)
    # Mask inactive slots (beyond active capacity)
    if Ae < Me:
        inactive = torch.arange(Me, device=device).unsqueeze(0) >= Ae
        scores.masked_fill_(inactive.unsqueeze(1), -1e9)
    scores.masked_fill_(~used[:, None, :], -1e9)

    best_score, best_idx = scores.max(dim=-1)
    any_used = used.any(dim=-1)

    novel = (~any_used[:, None]) | (best_score < tau_novel)
    hit = any_used[:, None] & (best_score >= tau_match)
    novel_keep = novel & keep_tok

    # LRU victim assignment (sequential for determinism)
    last_out = last.clone()
    idx_tok = best_idx.clone()
    overwrite_tok = torch.zeros((B, T), device=device, dtype=torch.bool)
    touch_tok = torch.zeros((B, T), device=device, dtype=torch.bool)

    sram_used = used.clone()
    sram_last = last_out

    for t in range(T):
        for b in range(B):
            if not keep_tok[b, t]:
                continue

            if novel_keep[b, t]:
                # Find LRU victim (only among active slots)
                lru_key = torch.where(
                    sram_used[b], sram_last[b],
                    torch.full((Me,), -(2**62), device=device, dtype=torch.int64),
                )
                if Ae < Me:
                    lru_key[Ae:] = 2**62  # inactive slots never picked
                victim = lru_key.argmin().item()
                idx_tok[b, t] = victim
                overwrite_tok[b, t] = True
                touch_tok[b, t] = True
                sram_used[b, victim] = True
                sram_last[b, victim] = pos_sel[b, t]
            elif hit[b, t]:
                idx_tok[b, t] = best_idx[b, t]
                overwrite_tok[b, t] = refresh_on_hit
                touch_tok[b, t] = True
                sram_last[b, best_idx[b, t]] = pos_sel[b, t]

    last_out = sram_last
    return idx_tok, overwrite_tok, touch_tok, last_out
