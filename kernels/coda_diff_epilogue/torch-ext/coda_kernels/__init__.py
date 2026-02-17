"""Fused differential attention epilogue kernel for CoDA-GQA-L.

Fuses: (out_sig - lam * out_noise) -> RMSNorm -> weight scaling
into a single CUDA kernel pass.

Usage:
    from coda_kernels import diff_epilogue

    output = diff_epilogue(out_sig, out_noise, lam, weight, eps=1e-6)
"""

from typing import Optional

import torch

# Load the compiled ops
try:
    from torch.ops import coda_kernels as ops
    _HAS_KERNEL = True
except (ImportError, RuntimeError):
    _HAS_KERNEL = False


def diff_epilogue(
    out_sig: torch.Tensor,
    out_noise: torch.Tensor,
    lam: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused differential attention epilogue.

    Computes: RMSNorm(out_sig - lam * out_noise) * weight
    in a single CUDA kernel pass.

    Args:
        out_sig:   (B, H, Lq, Dh) signal stream attention output
        out_noise: (B, H, Lq, Dh) noise stream attention output
        lam:       (B, H, Lq, 1)  per-token cancellation gate
        weight:    (Dh,)           RMSNorm weight
        eps:       RMSNorm epsilon (default: 1e-6)
        out:       Optional pre-allocated output tensor

    Returns:
        (B, H, Lq, Dh) fused output
    """
    if not _HAS_KERNEL:
        # Pure PyTorch fallback (unfused)
        diff = out_sig - lam * out_noise
        rms = diff.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        return diff * rms * weight

    if out is None:
        out = torch.empty_like(out_sig)

    # Flatten lam from (B, H, Lq, 1) to (B*H*Lq,)
    lam_flat = lam.reshape(-1).contiguous()

    ops.diff_epilogue_forward(out, out_sig, out_noise, lam_flat, weight, eps)
    return out


def is_available() -> bool:
    """Check if the compiled CUDA kernel is available."""
    return _HAS_KERNEL
