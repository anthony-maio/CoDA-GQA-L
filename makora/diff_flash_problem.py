"""Makora problem file: Differential FlashAttention (dual-stream SDPA + diff epilogue).

Reference PyTorch implementation computing:
  out_sig = softmax(Q @ K^T * scale) @ V
  out_noise = softmax(Q_noise @ K^T * scale) @ V
  diff = out_sig - lambda * out_noise
  diff = RMSNorm(diff)  [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """Differential attention: signal - lambda * noise, with optional fused RMSNorm."""

    def __init__(self, H: int, H_KV: int, HEAD_DIM: int, apply_rms: bool = True):
        super().__init__()
        self.H = H
        self.H_KV = H_KV
        self.HEAD_DIM = HEAD_DIM
        self.apply_rms = apply_rms
        self.groups = H // H_KV
        if apply_rms:
            self.rms_weight = nn.Parameter(torch.ones(HEAD_DIM))

    def forward(
        self,
        q: torch.Tensor,
        q_noise: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q:       (B, H, Lq, Dh)   signal queries
            q_noise: (B, H, Lq, Dh)   noise queries
            k:       (B, Hkv, Lk, Dh) keys
            v:       (B, Hkv, Lk, Dh) values
            lam:     (B, H, Lq)        per-query lambda gate

        Returns:
            out: (B, H, Lq, Dh) differential attention output
        """
        B, H, Lq, Dh = q.shape
        _, H_KV, Lk, _ = k.shape
        sm_scale = 1.0 / math.sqrt(Dh)

        # Expand KV heads for GQA
        k_exp = k.repeat_interleave(self.groups, dim=1)  # (B, H, Lk, Dh)
        v_exp = v.repeat_interleave(self.groups, dim=1)  # (B, H, Lk, Dh)

        # Signal stream
        attn_s = torch.matmul(q, k_exp.transpose(-2, -1)) * sm_scale
        attn_s = F.softmax(attn_s, dim=-1)
        out_s = torch.matmul(attn_s, v_exp)

        # Noise stream
        attn_n = torch.matmul(q_noise, k_exp.transpose(-2, -1)) * sm_scale
        attn_n = F.softmax(attn_n, dim=-1)
        out_n = torch.matmul(attn_n, v_exp)

        # Differential epilogue
        diff = out_s - lam.unsqueeze(-1) * out_n

        # Optional RMSNorm
        if self.apply_rms:
            var = (diff * diff).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + 1e-6)
            diff = diff * rstd * self.rms_weight

        return diff


# Problem dimensions
B = 2
H = 32
H_KV = 8
Lq = 512
Lk = 512
HEAD_DIM = 128


def get_inputs():
    dtype = torch.float32
    q = torch.randn(B, H, Lq, HEAD_DIM, dtype=dtype)
    q_noise = torch.randn(B, H, Lq, HEAD_DIM, dtype=dtype)
    k = torch.randn(B, H_KV, Lk, HEAD_DIM, dtype=dtype)
    v = torch.randn(B, H_KV, Lk, HEAD_DIM, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, H, Lq, dtype=dtype))
    return [q, q_noise, k, v, lam]


def get_init_inputs():
    return [H, H_KV, HEAD_DIM, True]
