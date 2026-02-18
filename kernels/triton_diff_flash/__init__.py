"""Fused differential FlashAttention for CoDA-GQA inference.

Loads K/V tiles once from HBM, computes both signal and noise attention
streams, and applies the differential epilogue (subtract + optional
RMSNorm) in-register before writing back.  Eliminates the K/V bandwidth
doubling and intermediate tensor allocation of two separate SDPA calls.

Forward-pass only (inference).  Falls back to PyTorch SDPA when Triton
is unavailable.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    from .kernel import _diff_flash_fwd
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
        return f"triton_diff_flash: available (triton {triton.__version__})"
    try:
        import triton  # noqa: F811
        return f"triton imported ({triton.__version__}) but kernel failed to load"
    except ImportError:
        return "triton_diff_flash: triton not installed"


def _fallback(
    q: Tensor,
    q_noise: Tensor,
    k: Tensor,
    v: Tensor,
    lam: Tensor,
    rms_weight: Optional[Tensor],
    eps: float,
    is_causal: bool,
    sm_scale: Optional[float],
) -> Tensor:
    """Two-call PyTorch SDPA + epilogue (reference fallback)."""
    import torch.nn.functional as F

    B, H, Lq, Dh = q.shape
    _, Hkv, Lk, _ = k.shape
    groups = H // Hkv

    # GQA expansion
    if groups > 1:
        k_exp = k[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
        v_exp = v[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
    else:
        k_exp, v_exp = k, v

    out_sig = F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=None, is_causal=is_causal,
    )
    out_noise = F.scaled_dot_product_attention(
        q_noise, k_exp, v_exp, attn_mask=None, is_causal=is_causal,
    )

    out = out_sig - lam * out_noise

    if rms_weight is not None:
        var = out.pow(2).mean(dim=-1, keepdim=True)
        out = out * torch.rsqrt(var + eps) * rms_weight

    return out


def diff_flash_attn(
    q: Tensor,
    q_noise: Tensor,
    k: Tensor,
    v: Tensor,
    lam: Tensor,
    rms_weight: Optional[Tensor] = None,
    eps: float = 1e-6,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Fused differential attention: out = RMSNorm(Attn(q,k,v) - lam * Attn(q_noise,k,v)).

    Args:
        q:          (B, H, Lq, Dh)  signal queries
        q_noise:    (B, H, Lq, Dh)  noise queries
        k:          (B, Hkv, Lk, Dh) keys (GQA: Hkv <= H)
        v:          (B, Hkv, Lk, Dh) values
        lam:        (B, H, Lq, 1)   per-head per-token lambda gate
        rms_weight: (Dh,) or None.  When provided, fuses HeadwiseRMSNorm.
        eps:        RMSNorm epsilon
        is_causal:  lower-right causal mask (prefix_len = Lk - Lq)
        sm_scale:   softmax scale, defaults to 1/sqrt(Dh)

    Returns:
        (B, H, Lq, Dh) differential attention output
    """
    if not _HAS_TRITON or not q.is_cuda:
        return _fallback(q, q_noise, k, v, lam, rms_weight, eps, is_causal, sm_scale)

    B, H, Lq, Dh = q.shape
    _, Hkv, Lk, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / (Dh ** 0.5)

    # Squeeze lambda from (B,H,Lq,1) -> (B,H,Lq) for kernel
    lam_sq = lam.squeeze(-1).contiguous()

    # Ensure contiguous layout
    q = q.contiguous()
    q_noise = q_noise.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    apply_rms = rms_weight is not None
    if rms_weight is not None:
        rms_weight = rms_weight.contiguous()

    _diff_flash_fwd(
        q, q_noise, k, v, lam_sq, out,
        rms_weight=rms_weight,
        eps=eps,
        sm_scale=sm_scale,
        is_causal=is_causal,
        APPLY_RMS=apply_rms,
    )

    return out
