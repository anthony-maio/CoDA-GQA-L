"""Makora problem file: Exact bank routing (cosine similarity + classification + LRU).

Reference PyTorch implementation of exact-bank routing logic:
  1. Compute cross-head average cosine similarity between candidates and bank
  2. Classify each candidate as novel/hit/skip
  3. Assign LRU victim slots for novel candidates (sequential)
  4. Update LRU timestamps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Exact bank routing: score, classify, assign LRU slots."""

    def __init__(
        self,
        H_KV: int,
        T: int,
        ME: int,
        DH: int,
        tau_gate: float = 0.1,
        tau_match: float = 0.8,
        tau_novel: float = 0.3,
        refresh_on_hit: bool = True,
    ):
        super().__init__()
        self.H_KV = H_KV
        self.T = T
        self.ME = ME
        self.DH = DH
        self.tau_gate = tau_gate
        self.tau_match = tau_match
        self.tau_novel = tau_novel
        self.refresh_on_hit = refresh_on_hit

    def forward(
        self,
        v_sel_norm: torch.Tensor,
        mem_v_norm: torch.Tensor,
        used: torch.Tensor,
        last: torch.Tensor,
        gate_sel: torch.Tensor,
        pos_sel: torch.Tensor,
    ) -> tuple:
        """
        Args:
            v_sel_norm: (B, Hkv, T, Dh) pre-normalized candidate values
            mem_v_norm: (B, Hkv, Me, Dh) cached normalized bank values
            used:       (B, Me) bool - which bank slots are valid
            last:       (B, Me) int64 - LRU timestamps
            gate_sel:   (B, T)  float32 - write gate values
            pos_sel:    (B, T)  int64 - position timestamps

        Returns:
            idx_tok:       (B, T) int64 - target slot index per candidate
            overwrite_tok: (B, T) bool  - should overwrite
            touch_tok:     (B, T) bool  - should update LRU
            last_out:      (B, Me) int64 - updated LRU timestamps
        """
        B, Hkv, T, Dh = v_sel_norm.shape
        Me = mem_v_norm.shape[2]
        device = v_sel_norm.device

        idx_tok = torch.zeros(B, T, device=device, dtype=torch.int64)
        overwrite_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
        touch_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
        last_out = last.clone()

        for b in range(B):
            sram_used = used[b].clone()
            sram_last = last_out[b].clone()
            any_used = sram_used.any().item()

            for t in range(T):
                gate = gate_sel[b, t].item()
                if gate < self.tau_gate:
                    continue

                # Cross-head average cosine similarity
                scores = torch.zeros(Me, device=device, dtype=torch.float32)
                for h in range(Hkv):
                    scores += torch.mv(mem_v_norm[b, h], v_sel_norm[b, h, t])
                scores /= Hkv

                scores[~sram_used] = -1e9

                best_score = scores.max().item()
                best_idx = scores.argmax().item()

                is_novel = (not any_used) or (best_score < self.tau_novel)
                is_hit = any_used and (best_score >= self.tau_match)

                if is_novel:
                    # LRU victim
                    lru_key = torch.where(
                        sram_used,
                        sram_last,
                        torch.full_like(sram_last, -(2**62)),
                    )
                    victim = lru_key.argmin().item()
                    idx_tok[b, t] = victim
                    overwrite_tok[b, t] = True
                    touch_tok[b, t] = True
                    sram_used[victim] = True
                    sram_last[victim] = pos_sel[b, t]
                    any_used = True
                elif is_hit:
                    idx_tok[b, t] = best_idx
                    overwrite_tok[b, t] = self.refresh_on_hit
                    touch_tok[b, t] = True
                    sram_last[best_idx] = pos_sel[b, t]
                else:
                    idx_tok[b, t] = best_idx

            last_out[b] = sram_last

        return idx_tok, overwrite_tok, touch_tok, last_out


# Problem dimensions
B = 2
H_KV = 8
T = 8
ME = 64
DH = 128


def get_inputs():
    v_sel_norm = F.normalize(torch.randn(B, H_KV, T, DH, dtype=torch.float32), dim=-1)
    mem_v_norm = F.normalize(torch.randn(B, H_KV, ME, DH, dtype=torch.float32), dim=-1)
    used = torch.ones(B, ME, dtype=torch.bool)
    used[:, -16:] = False  # some empty slots
    last = torch.randint(0, 10000, (B, ME), dtype=torch.int64)
    gate_sel = torch.rand(B, T, dtype=torch.float32) * 0.5 + 0.5
    pos_sel = torch.arange(T, dtype=torch.int64).unsqueeze(0).expand(B, T) + 10000
    return [v_sel_norm, mem_v_norm, used, last, gate_sel, pos_sel]


def get_init_inputs():
    return [H_KV, T, ME, DH, 0.1, 0.8, 0.3, True]
