"""Fused differential attention epilogue kernels for CoDA-GQA-L.

Two kernels:
  1. diff_epilogue:  (out_sig - lam * out_noise) -> RMSNorm -> weight scaling
     Used when head_norm_mode="full" (Eve-2, standalone CoDA)

  2. diff_sub_only:  out_sig - lam * out_noise
     Used when head_norm_mode="identity" (Llama/Mistral cold-swap)

Both fall back to unfused PyTorch when the compiled extension is missing.

Usage:
    from coda_kernels import diff_epilogue, diff_sub_only, is_available

    # Full (with RMSNorm):
    output = diff_epilogue(out_sig, out_noise, lam, weight, eps=1e-6)

    # Subtract-only (identity norm):
    output = diff_sub_only(out_sig, out_noise, lam)
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
    """Fused differential attention epilogue (subtract + RMSNorm + weight).

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
        diff = out_sig - lam * out_noise
        rms = diff.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        return diff * rms * weight

    if out is None:
        out = torch.empty_like(out_sig)

    lam_flat = lam.reshape(-1).contiguous()
    ops.diff_epilogue_forward(out, out_sig, out_noise, lam_flat, weight, eps)
    return out


def diff_sub_only(
    out_sig: torch.Tensor,
    out_noise: torch.Tensor,
    lam: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused subtract-only epilogue (identity head norm, no RMSNorm).

    Computes: out_sig - lam * out_noise
    in a single CUDA kernel pass, avoiding intermediate tensor allocation.

    Args:
        out_sig:   (B, H, Lq, Dh) signal stream attention output
        out_noise: (B, H, Lq, Dh) noise stream attention output
        lam:       (B, H, Lq, 1)  per-token cancellation gate
        out:       Optional pre-allocated output tensor

    Returns:
        (B, H, Lq, Dh) fused output
    """
    if not _HAS_KERNEL:
        return out_sig - lam * out_noise

    if out is None:
        out = torch.empty_like(out_sig)

    lam_flat = lam.reshape(-1).contiguous()
    ops.diff_sub_only_forward(out, out_sig, out_noise, lam_flat)
    return out


def is_available() -> bool:
    """Check if the compiled CUDA kernel is available."""
    return _HAS_KERNEL


def diagnose() -> str:
    """Return diagnostic info about kernel loading with actionable fixes."""
    lines = [
        f"_HAS_KERNEL: {_HAS_KERNEL}",
        f"_KERNEL_ERROR: {_KERNEL_ERROR!r}",
        f"torch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        lines.append(f"GPU: {gpu_name}")
        lines.append(f"Compute capability: sm_{cc[0]}{cc[1]}")
    try:
        import importlib.util
        spec = importlib.util.find_spec("coda_kernels._ops")
        lines.append(f"_ops spec: {spec}")
        if spec and spec.origin:
            lines.append(f"_ops path: {spec.origin}")
    except Exception as e:
        lines.append(f"_ops lookup error: {e}")

    if not _HAS_KERNEL:
        lines.append("")
        lines.append("--- Fix ---")
        if _KERNEL_ERROR and "No module named" in str(_KERNEL_ERROR):
            lines.append("Kernel not compiled. Build with:")
            lines.append("  pip install -e kernels/coda_diff_epilogue/")
        elif _KERNEL_ERROR and "undefined symbol" in str(_KERNEL_ERROR):
            lines.append("Binary incompatible (likely CUDA/PyTorch version mismatch).")
            lines.append("Rebuild: pip install -e kernels/coda_diff_epilogue/ --no-build-isolation --force-reinstall")
        elif _KERNEL_ERROR and "CUDA" in str(_KERNEL_ERROR):
            lines.append("CUDA error during load. Check nvidia-smi and CUDA toolkit version.")
            lines.append("Rebuild: pip install -e kernels/coda_diff_epilogue/ --force-reinstall")
        else:
            lines.append("Unknown error. Try rebuilding:")
            lines.append("  pip install -e kernels/coda_diff_epilogue/ --force-reinstall")
    return "\n".join(lines)
