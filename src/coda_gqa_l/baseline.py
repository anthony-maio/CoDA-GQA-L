# baseline.py
# Unbounded CoDA-GQA and standard GQA attention for comparison/testing.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    HeadwiseRMSNorm,
    RotaryEmbedding,
    _apply_pairwise_rotation,
    apply_rope,
    repeat_kv,
)


class CoDAGQA(nn.Module):
    """CoDA-GQA: Constrained Orthogonal Differential Attention + GQA.

    Signal attention:
        out_sig = Attn(q, k, v)

    Noise attention:
        q_noise = Rotate(q; theta)   # orthogonal per-head pairwise rotation
        out_noise = Attn(q_noise, k, v)

    Differential output:
        out = RMSNorm(out_sig - lambda * out_noise)

    Key property:
      - Only one KV cache is stored (k, v), identical to standard GQA.
      - Parameter overhead is small: theta (per-head pairwise angles) + lambda_proj.
      - FlashAttention compatibility comes from using torch SDPA.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float = 10_000.0,
        lambda_init_bias: float = -6.0,
        theta_init: float = math.pi / 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads (GQA constraint).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (required by RoPE + pairwise rotations).")

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)

        self.theta = nn.Parameter(torch.full((num_heads, self.head_dim // 2), theta_init))

        self.lambda_proj = nn.Linear(embed_dim, num_heads, bias=True)
        nn.init.constant_(self.lambda_proj.bias, lambda_init_bias)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

        self.head_norm = HeadwiseRMSNorm(self.head_dim, eps=eps)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (B, L, D)
        past_key_value: (k_cache, v_cache) each (B, H_kv, L_past, Dh)
        returns: (y, (k_total, v_total))
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].size(2)

        cos, sin = self.rope(seq_len=L, offset=offset, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        current_kv = (k, v)

        num_repeat = self.num_heads // self.num_kv_heads
        k_rep = repeat_kv(k, num_repeat)
        v_rep = repeat_kv(v, num_repeat)

        lambda_gate = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)

        cos_t = torch.cos(self.theta).to(dtype=q.dtype, device=q.device)
        sin_t = torch.sin(self.theta).to(dtype=q.dtype, device=q.device)
        q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

        causal_flag = is_causal if attention_mask is None else False
        out_sig = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attention_mask, is_causal=causal_flag)
        out_noise = F.scaled_dot_product_attention(q_noise, k_rep, v_rep, attn_mask=attention_mask, is_causal=causal_flag)

        out = out_sig - lambda_gate * out_noise
        out = self.head_norm(out)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        return out, current_kv


class BaselineGQA(nn.Module):
    """Baseline GQA attention (single SDPA call), for comparison/testing."""

    def __init__(self, *, embed_dim: int, num_heads: int, num_kv_heads: int, rope_base: float = 10_000.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads (GQA constraint).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (required by RoPE).")

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].size(2)

        cos, sin = self.rope(seq_len=L, offset=offset, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        current_kv = (k, v)

        num_repeat = self.num_heads // self.num_kv_heads
        k_rep = repeat_kv(k, num_repeat)
        v_rep = repeat_kv(v, num_repeat)

        causal_flag = is_causal if attention_mask is None else False
        out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attention_mask, is_causal=causal_flag)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        return out, current_kv
