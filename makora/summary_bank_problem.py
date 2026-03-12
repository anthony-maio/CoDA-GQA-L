"""Makora problem file: Summary bank update (LF-K routing + Phase-Safe EMA).

Reference PyTorch implementation of summary bank update:
  1. LF-K routing: cosine similarity on low-frequency key band
  2. Scatter-add weighted aggregation per target slot
  3. Phase-Safe EMA: blend only LF key band, full EMA on values
  4. Mark newly-used slots

This is the _summary_update_block_hard operation from CoDA-GQA-L.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """Summary bank update: LF-K routing + Phase-Safe EMA blending."""

    def __init__(
        self,
        H_KV: int,
        T: int,
        MS: int,
        DH: int,
        lf_start: int = 64,
        eta_logit: float = -2.0,
        overwrite_on_insert: bool = True,
    ):
        super().__init__()
        self.H_KV = H_KV
        self.T = T
        self.MS = MS
        self.DH = DH
        self.lf_start = lf_start
        self.eta_logit = eta_logit
        self.overwrite_on_insert = overwrite_on_insert
        self.energy_scale = math.sqrt(DH / (DH - lf_start))

    def forward(
        self,
        k_sel: torch.Tensor,
        v_sel: torch.Tensor,
        w_tok: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        used: torch.Tensor,
        mem_lf_k_norm: torch.Tensor,
    ) -> tuple:
        """
        Args:
            k_sel:         (B, Hkv, T, Dh) candidate keys
            v_sel:         (B, Hkv, T, Dh) candidate values
            w_tok:         (B, T)           per-candidate weight (gate * keep)
            mem_k:         (B, Hkv, Ms, Dh) current summary bank keys
            mem_v:         (B, Hkv, Ms, Dh) current summary bank values
            used:          (B, Ms)          which slots are occupied
            mem_lf_k_norm: (B, Hkv, Ms, Dh_lf) cached normalized LF keys

        Returns:
            mem_k_out:  (B, Hkv, Ms, Dh) updated keys
            mem_v_out:  (B, Hkv, Ms, Dh) updated values
            used_out:   (B, Ms) updated occupancy
        """
        B, Hkv, T, Dh = k_sel.shape
        Ms = mem_k.shape[2]
        device = k_sel.device
        lf = self.lf_start
        Dh_lf = Dh - lf

        # LF-K routing: cosine sim on low-frequency key band
        lf_k_sel_norm = F.normalize(k_sel[..., lf:], dim=-1, eps=1e-6)
        scores_h = torch.matmul(
            lf_k_sel_norm.reshape(B * Hkv, T, Dh_lf),
            mem_lf_k_norm.reshape(B * Hkv, Ms, Dh_lf).transpose(1, 2),
        ).view(B, Hkv, T, Ms)
        scores = scores_h.mean(dim=1)  # (B, T, Ms)
        idx_tok = scores.argmax(dim=-1)  # (B, T)

        w = w_tok.to(device=device, dtype=mem_k.dtype)

        # Scatter-add weighted aggregation
        count = torch.zeros(B, Ms, device=device, dtype=mem_k.dtype)
        count.scatter_add_(1, idx_tok, w)

        idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
        w4 = w.view(B, 1, T, 1)
        src_k = k_sel * w4
        src_v = v_sel * w4
        sum_k = torch.zeros_like(mem_k)
        sum_v = torch.zeros_like(mem_v)
        sum_k.scatter_add_(2, idx_exp, src_k)
        sum_v.scatter_add_(2, idx_exp, src_v)

        # EMA
        eta_base = torch.sigmoid(torch.tensor(self.eta_logit, device=device, dtype=mem_k.dtype))
        eta = eta_base.view(1, 1, 1, 1).expand(B, Hkv, Ms, Dh)

        count4 = count.view(B, 1, Ms, 1)
        active = count4 > 0
        avg_k = sum_k / count4.clamp_min(1e-6)
        avg_v = sum_v / count4.clamp_min(1e-6)

        new_slots = (~used).view(B, 1, Ms, 1) & active
        if self.overwrite_on_insert:
            eta_eff = torch.where(new_slots, torch.ones_like(eta), eta)
        else:
            eta_eff = eta

        # Phase-Safe EMA: only blend LF key band
        delta_k = torch.zeros_like(avg_k)
        delta_k[..., lf:] = avg_k[..., lf:] * self.energy_scale - mem_k[..., lf:]
        mem_k_out = torch.where(active, mem_k + eta_eff * delta_k, mem_k)
        mem_v_out = torch.where(active, mem_v + eta_eff * (avg_v - mem_v), mem_v)
        used_out = used | (count > 0)

        return mem_k_out, mem_v_out, used_out


# Problem dimensions
B = 2
H_KV = 8
T = 8
MS = 64
DH = 128
LF_START = 64


def get_inputs():
    dtype = torch.float32
    k_sel = torch.randn(B, H_KV, T, DH, dtype=dtype)
    v_sel = torch.randn(B, H_KV, T, DH, dtype=dtype)
    w_tok = torch.rand(B, T, dtype=dtype) * 0.5 + 0.1
    mem_k = torch.randn(B, H_KV, MS, DH, dtype=dtype)
    mem_v = torch.randn(B, H_KV, MS, DH, dtype=dtype)
    used = torch.ones(B, MS, dtype=torch.bool)
    used[:, -16:] = False
    mem_lf_k_norm = F.normalize(mem_k[..., LF_START:], dim=-1, eps=1e-6)
    return [k_sel, v_sel, w_tok, mem_k, mem_v, used, mem_lf_k_norm]


def get_init_inputs():
    return [H_KV, T, MS, DH, LF_START, -2.0, True]
