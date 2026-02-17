"""Fused differential attention epilogue kernel for CoDA-GQA-L.

Fuses: (out_sig - lam * out_noise) -> RMSNorm -> weight scaling
into a single CUDA kernel pass.

Usage:
    from coda_kernels import diff_epilogue

    output = diff_epilogue(out_sig, out_noise, lam, weight, eps=1e-6)
"""

from typing import Optional

import torch

# Load the compiled extension first (triggers TORCH_LIBRARY registration),
# then access the registered ops via torch.ops.
_KERNEL_ERROR = None
try:
    import coda_kernels._ops  # noqa: F401 -- side-effect: registers ops
    ops = torch.ops.coda_kernels
    _HAS_KERNEL = True
except (ImportError, RuntimeError, AttributeError) as e:
    _HAS_KERNEL = False
    _KERNEL_ERROR = e


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


def diagnose() -> str:
    """Return diagnostic info about kernel loading."""
    import sys
    lines = [
        f"_HAS_KERNEL: {_HAS_KERNEL}",
        f"_KERNEL_ERROR: {_KERNEL_ERROR!r}",
        f"torch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        cc = torch.cuda.get_device_capability(0)
        lines.append(f"Compute capability: {cc[0]}.{cc[1]}")
    # Check if _ops module exists in installed package
    try:
        import importlib.util
        spec = importlib.util.find_spec("coda_kernels._ops")
        lines.append(f"_ops spec: {spec}")
        if spec and spec.origin:
            lines.append(f"_ops path: {spec.origin}")
    except Exception as e:
        lines.append(f"_ops lookup error: {e}")
    return "\n".join(lines)
