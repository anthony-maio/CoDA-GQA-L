# primitives.py
# Shared building blocks for CoDA-GQA attention mechanisms.
#
# Requires PyTorch >= 2.0 (for torch.nn.functional.scaled_dot_product_attention)

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm across the head_dim dimension, applied independently per (batch, head, token)."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, Dh)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """Standard RoPE (rotary positional embeddings) for even head_dim.

    Returns cos/sin tables for a given (seq_len, offset).
    """

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        *,
        seq_len: int,
        offset: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim/2)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def _apply_pairwise_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies a 2D rotation independently to each (even, odd) feature pair.

    x:   (B, H, L, Dh)
    cos: broadcastable to (1, H, 1, Dh/2)
    sin: broadcastable to (1, H, 1, Dh/2)
    """
    B, H, L, Dh = x.shape
    half = Dh // 2

    cos = cos.view(1, H, 1, half)
    sin = sin.view(1, H, 1, half)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE (position-dependent pairwise rotation) to x."""
    B, H, L, Dh = x.shape
    half = Dh // 2
    cos = cos.view(1, 1, L, half)
    sin = sin.view(1, 1, L, half)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def repeat_kv(x: torch.Tensor, num_repeat: int) -> torch.Tensor:
    """Repeat KV heads to match Q heads (GQA -> SDPA shape compatibility).

    x: (B, H_kv, L, Dh) -> (B, H_kv * num_repeat, L, Dh)

    Uses expand+reshape to avoid materializing the repeated tensor when possible.
    """
    if num_repeat == 1:
        return x
    B, H_kv, L, Dh = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, num_repeat, L, Dh)
    return x.reshape(B, H_kv * num_repeat, L, Dh)


def _sdpa_supports_enable_gqa() -> bool:
    """Conservative feature probe: enable_gqa introduced in newer PyTorch versions."""
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 1, 1, 4)
    v = torch.randn(1, 1, 1, 4)
    try:
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=True)
        return True
    except TypeError:
        return False


SDPA_ENABLE_GQA: bool = _sdpa_supports_enable_gqa()
