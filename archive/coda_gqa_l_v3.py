
# coda_gqa_l_v3.py
# CoDA-GQA-L v3: CoDA-GQA + bounded KV cache with:
#   (A) importance-gated writes (write policy),
#   (B) two-bank memory: exact (snapshot) + summary (EMA),
#   (C) chunked prefill (blockwise causal processing).
#
# Design goals:
#   - Memory per layer is O(W + Me + Ms), independent of prompt length L.
#   - Uses PyTorch SDPA (FlashAttention path when available) twice (signal + inhibitory).
#   - Keeps KV cache in KV-head space (GQA), using enable_gqa when supported.
#
# This is a research-grade reference implementation that is runnable and auditable.
# It is not "production-optimized" (mask construction + Python loops for memory updates remain).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coda_gqa import (
    HeadwiseRMSNorm,
    RotaryEmbedding,
    apply_rope,
    repeat_kv,
    _apply_pairwise_rotation,
)


# ----------------------------
# SDPA feature detection
# ----------------------------

def _sdpa_supports_enable_gqa() -> bool:
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 1, 1, 4)
    v = torch.randn(1, 1, 1, 4)
    try:
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_ENABLE_GQA: bool = _sdpa_supports_enable_gqa()


# ----------------------------
# RoPE helper for arbitrary positions (for memory slots)
# ----------------------------

def _rotate_with_positions(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE-like pairwise rotation for arbitrary (possibly non-integer) positions.

    x:         (B, H, L, Dh)
    positions: (B, H, L) float or int
    inv_freq:  (Dh/2,)
    """
    B, H, L, Dh = x.shape
    half = Dh // 2

    freqs = positions.to(dtype=inv_freq.dtype).unsqueeze(-1) * inv_freq.view(1, 1, 1, half)
    cos = freqs.cos().to(dtype=x.dtype)
    sin = freqs.sin().to(dtype=x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ----------------------------
# State
# ----------------------------

@dataclass
class CoDAGQALandmarkStateV3:
    # Recent ring-buffer (RAW K, V) + write gates (scalar per token)
    k_recent: torch.Tensor      # (B, Hkv, W, Dh)
    v_recent: torch.Tensor      # (B, Hkv, W, Dh)
    g_recent: torch.Tensor      # (B, W) float32 in [0,1]
    recent_idx: int
    recent_filled: int

    # Exact memory bank (snapshot)
    mem_k_exact: torch.Tensor   # (B, Hkv, Me, Dh)
    mem_v_exact: torch.Tensor   # (B, Hkv, Me, Dh)
    mem_pos_exact: torch.Tensor # (B, Hkv, Me) int64 (absolute position)
    mem_used_exact: torch.Tensor# (B, Hkv, Me) bool
    mem_last_exact: torch.Tensor# (B, Hkv, Me) int64 (LRU timestamp)

    # Summary memory bank (EMA prototypes)
    mem_k_sum: torch.Tensor     # (B, Hkv, Ms, Dh)
    mem_v_sum: torch.Tensor     # (B, Hkv, Ms, Dh)
    mem_pos_sum: torch.Tensor   # (B, Hkv, Ms) float32 (EMA position)
    mem_used_sum: torch.Tensor  # (B, Hkv, Ms) bool
    mem_last_sum: torch.Tensor  # (B, Hkv, Ms) int64 (LRU timestamp)

    # Absolute position of next token to be written
    pos: int


# ----------------------------
# Module
# ----------------------------

class CoDAGQALandmarkV3(nn.Module):
    """
    CoDA-GQA-L v3.

    Attention core:
      out = RMSNorm( Attn(q, K, V) - lambda * Attn(R_theta(q), K, V) )
    where:
      - q uses one projection; noise query is an orthogonal per-head rotation.
      - K/V are in KV-head space (GQA); SDPA enable_gqa used when available.

    Bounded memory per layer:
      - recent window W (exact, raw tokens)
      - exact memory Me (snapshots, novelty-filtered + LRU)
      - summary memory Ms (EMA prototypes, novelty-filtered + LRU)

    Write policy:
      - write gate g ∈ (0,1) predicted per token from x.
      - evicted tokens only update memory if g >= threshold.
      - summary EMA rate is scaled by g (eta_eff = eta * g).

    Prefill:
      - prefill_chunked ingests sequences in blocks. Each block attends to:
          [prev_recent (<=W) + exact_mem + summary_mem] + [current_block causal prefix]
        This is not identical to per-token streaming semantics; it is a deliberate "chunked prefill"
        policy that keeps prefill throughput reasonable while retaining bounded memory.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        window: int,
        num_landmarks_exact: int,
        num_landmarks_summary: int,
        rope_base: float = 10_000.0,
        lambda_init_bias: float = -6.0,
        theta_init: float = math.pi / 2,
        # Write policy
        write_policy: Literal["none", "gated"] = "gated",
        write_gate_init_bias: float = -2.0,        # sigmoid(-2)~0.119
        write_gate_threshold_exact: float = 0.10,  # skip exact update if g < thr
        write_gate_threshold_summary: float = 0.05,# skip summary update if g < thr
        # Exact bank
        exact_match_threshold: float = 0.90,        # cosine sim -> "already represented"
        exact_novelty_threshold: float = 0.70,      # cosine sim < thr -> novel insert/replace
        exact_refresh_on_hit: bool = False,         # if True, overwrite slot on "hit"
        # Summary bank
        summary_novelty_threshold: float = 0.50,
        summary_overwrite_on_insert: bool = True,
        summary_eta_init_logit: float = -3.0,       # sigmoid(-3)~0.047
        # Memory init
        mem_init: Literal["zeros", "random_normal"] = "random_normal",
        # RoPE handling for memory banks
        mem_rope_exact: Literal["none", "pos"] = "pos",
        mem_rope_summary: Literal["none", "mean_pos"] = "mean_pos",
        # Masking
        mask_unused_memory: bool = True,
        # Detach evicted tokens in memory update (inference-friendly)
        detach_evicted: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if window <= 0:
            raise ValueError("window must be positive.")
        if num_landmarks_exact < 0 or num_landmarks_summary < 0:
            raise ValueError("num_landmarks_* must be non-negative.")
        if num_landmarks_exact == 0 and num_landmarks_summary == 0:
            # still valid: bounded cache degenerates to pure windowed attention
            pass

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (required by RoPE + pairwise rotations).")

        self.window = int(window)
        self.Me = int(num_landmarks_exact)
        self.Ms = int(num_landmarks_summary)

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        # Orthogonal inhibitory rotation params (per Q-head, per pair)
        self.theta = nn.Parameter(torch.full((self.num_heads, self.head_dim // 2), float(theta_init)))

        # λ gate: (B, L, D) -> (B, L, Hq)
        self.lambda_proj = nn.Linear(self.embed_dim, self.num_heads, bias=True)
        nn.init.constant_(self.lambda_proj.bias, float(lambda_init_bias))

        # Write gate: (B, L, D) -> (B, L, 1)
        self.write_policy = write_policy
        self.write_proj = nn.Linear(self.embed_dim, 1, bias=True)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, float(write_gate_init_bias))
        self.write_gate_threshold_exact = float(write_gate_threshold_exact)
        self.write_gate_threshold_summary = float(write_gate_threshold_summary)

        self.rope = RotaryEmbedding(self.head_dim, base=float(rope_base))

        # Memory hyperparams
        self.exact_match_threshold = float(exact_match_threshold)
        self.exact_novelty_threshold = float(exact_novelty_threshold)
        self.exact_refresh_on_hit = bool(exact_refresh_on_hit)

        self.summary_novelty_threshold = float(summary_novelty_threshold)
        self.summary_overwrite_on_insert = bool(summary_overwrite_on_insert)
        self.summary_eta_logit = nn.Parameter(torch.full((self.num_kv_heads,), float(summary_eta_init_logit)))

        self.mem_init = mem_init
        self.mem_rope_exact = mem_rope_exact
        self.mem_rope_summary = mem_rope_summary

        self.mask_unused_memory = bool(mask_unused_memory)
        self.detach_evicted = bool(detach_evicted)

        self.head_norm = HeadwiseRMSNorm(self.head_dim, eps=eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    # ----------------------------
    # Init state
    # ----------------------------

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> CoDAGQALandmarkStateV3:
        B = int(batch_size)
        Hkv = self.num_kv_heads
        W = self.window
        Dh = self.head_dim
        Me = self.Me
        Ms = self.Ms

        k_recent = torch.zeros((B, Hkv, W, Dh), device=device, dtype=dtype)
        v_recent = torch.zeros((B, Hkv, W, Dh), device=device, dtype=dtype)
        g_recent = torch.zeros((B, W), device=device, dtype=torch.float32)

        def init_mem_k(shape):
            if self.mem_init == "zeros":
                return torch.zeros(shape, device=device, dtype=dtype)
            if self.mem_init == "random_normal":
                return torch.randn(shape, device=device, dtype=dtype) * 0.02
            raise ValueError(f"Unknown mem_init: {self.mem_init}")

        mem_k_exact = init_mem_k((B, Hkv, Me, Dh)) if Me > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_v_exact = torch.zeros((B, Hkv, Me, Dh), device=device, dtype=dtype) if Me > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_pos_exact = torch.zeros((B, Hkv, Me), device=device, dtype=torch.int64) if Me > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.int64)
        mem_used_exact = torch.zeros((B, Hkv, Me), device=device, dtype=torch.bool) if Me > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.bool)
        mem_last_exact = torch.zeros((B, Hkv, Me), device=device, dtype=torch.int64) if Me > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.int64)

        mem_k_sum = init_mem_k((B, Hkv, Ms, Dh)) if Ms > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_v_sum = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=dtype) if Ms > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_pos_sum = torch.zeros((B, Hkv, Ms), device=device, dtype=torch.float32) if Ms > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.float32)
        mem_used_sum = torch.zeros((B, Hkv, Ms), device=device, dtype=torch.bool) if Ms > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.bool)
        mem_last_sum = torch.zeros((B, Hkv, Ms), device=device, dtype=torch.int64) if Ms > 0 else torch.zeros((B, Hkv, 0), device=device, dtype=torch.int64)

        return CoDAGQALandmarkStateV3(
            k_recent=k_recent,
            v_recent=v_recent,
            g_recent=g_recent,
            recent_idx=0,
            recent_filled=0,
            mem_k_exact=mem_k_exact,
            mem_v_exact=mem_v_exact,
            mem_pos_exact=mem_pos_exact,
            mem_used_exact=mem_used_exact,
            mem_last_exact=mem_last_exact,
            mem_k_sum=mem_k_sum,
            mem_v_sum=mem_v_sum,
            mem_pos_sum=mem_pos_sum,
            mem_used_sum=mem_used_sum,
            mem_last_sum=mem_last_sum,
            pos=0,
        )

    # ----------------------------
    # Projections
    # ----------------------------

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,Hq,L,Dh)

    def _project_kv_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hkv,L,Dh)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def _write_gate(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, L) float32
        if self.write_policy == "none":
            return torch.ones((x.size(0), x.size(1)), device=x.device, dtype=torch.float32)
        g = torch.sigmoid(self.write_proj(x)).squeeze(-1)  # (B,L)
        return g.to(dtype=torch.float32)

    # ----------------------------
    # Recent ring helpers
    # ----------------------------

    def _get_recent_in_order(self, state: CoDAGQALandmarkStateV3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns (k_recent_raw, v_recent, g_recent, first_pos)
          k_recent_raw: (B,Hkv,Lr,Dh) in time order
          v_recent:     (B,Hkv,Lr,Dh)
          g_recent:     (B,Lr) float32
          first_pos:    absolute position of first element
        """
        B, Hkv, W, Dh = state.k_recent.shape
        n = int(state.recent_filled)
        if n == 0:
            empty_k = state.k_recent[:, :, :0, :]
            empty_v = state.v_recent[:, :, :0, :]
            empty_g = state.g_recent[:, :0]
            return empty_k, empty_v, empty_g, state.pos

        if n < W:
            k = state.k_recent[:, :, :n, :]
            v = state.v_recent[:, :, :n, :]
            g = state.g_recent[:, :n]
            first_pos = state.pos - n
            return k, v, g, first_pos

        i = int(state.recent_idx)
        k = torch.cat([state.k_recent[:, :, i:, :], state.k_recent[:, :, :i, :]], dim=2)
        v = torch.cat([state.v_recent[:, :, i:, :], state.v_recent[:, :, :i, :]], dim=2)
        g = torch.cat([state.g_recent[:, i:], state.g_recent[:, :i]], dim=1)
        first_pos = state.pos - W
        return k, v, g, first_pos

    def _set_recent_from_sequence(
        self,
        state: CoDAGQALandmarkStateV3,
        *,
        k_seq: torch.Tensor,  # (B,Hkv,Lr,Dh) time order, Lr<=W
        v_seq: torch.Tensor,  # (B,Hkv,Lr,Dh)
        g_seq: torch.Tensor,  # (B,Lr)
    ) -> None:
        B, Hkv, Lr, Dh = k_seq.shape
        W = self.window
        if Lr > W:
            raise ValueError("k_seq longer than window.")
        # zero out then fill
        state.k_recent.zero_()
        state.v_recent.zero_()
        state.g_recent.zero_()

        if Lr > 0:
            state.k_recent[:, :, :Lr, :] = k_seq
            state.v_recent[:, :, :Lr, :] = v_seq
            state.g_recent[:, :Lr] = g_seq.to(dtype=torch.float32)

        state.recent_filled = int(Lr)
        state.recent_idx = 0  # oldest at slot 0 when full

    # ----------------------------
    # Memory update: exact bank
    # ----------------------------

    def _exact_update_one(
        self,
        state: CoDAGQALandmarkStateV3,
        *,
        k_evict: torch.Tensor,  # (B,Hkv,Dh)
        v_evict: torch.Tensor,  # (B,Hkv,Dh)
        pos_evict: int,
        gate: torch.Tensor,     # (B,) float32
    ) -> None:
        """Update exact snapshot bank with one evicted token (vectorized over B,Hkv)."""
        if self.Me == 0:
            return

        B, Hkv, Dh = k_evict.shape
        Me = self.Me
        device = k_evict.device

        # gate threshold
        if self.write_policy != "none":
            keep = gate >= self.write_gate_threshold_exact  # (B,)
            if not bool(keep.any()):
                return
        else:
            keep = torch.ones((B,), device=device, dtype=torch.bool)

        used = state.mem_used_exact  # (B,Hkv,Me)

        # cosine similarity to used slots
        k_n = F.normalize(k_evict, dim=-1, eps=1e-6)
        mem_n = F.normalize(state.mem_k_exact, dim=-1, eps=1e-6)
        scores = torch.einsum("bhd,bhmd->bhm", k_n, mem_n)  # (B,Hkv,Me)
        scores = scores.masked_fill(~used, -1e9)

        best_score, best_idx = scores.max(dim=-1)  # (B,Hkv)

        # Determine "hit" vs "novel"
        hit = best_score >= self.exact_match_threshold  # (B,Hkv)
        novel = best_score < self.exact_novelty_threshold  # (B,Hkv)
        # Between thresholds: treat as hit-ish (touch), but do not overwrite (unless refresh_on_hit)

        # Free slot selection (first free)
        free = ~used
        free_exists = free.any(dim=-1)  # (B,Hkv)
        free_idx = free.to(dtype=torch.int64).argmax(dim=-1)  # (B,Hkv) (0 if none)

        # LRU replacement: argmin of last_used among USED slots (smaller = older)
        # For unused slots, set last_used to +inf so they are never selected here.
        last = state.mem_last_exact
        inf = torch.full_like(last, 2**62)
        last_masked = torch.where(used, last, inf)
        lru_idx = last_masked.argmin(dim=-1)  # (B,Hkv)

        insert_idx = torch.where(free_exists, free_idx, lru_idx)  # (B,Hkv)

        # Decide final index:
        #   - if hit and not refresh_on_hit: just touch best_idx
        #   - else if novel: insert/replace into insert_idx
        #   - else: touch best_idx (and optionally refresh)
        if self.exact_refresh_on_hit:
            # overwrite on hit too
            idx = torch.where(novel, insert_idx, best_idx)
            overwrite = torch.ones_like(idx, dtype=torch.bool)
        else:
            idx = torch.where(novel, insert_idx, best_idx)
            overwrite = novel  # only overwrite on novel insert/replace

        # Apply per-batch gate mask: if keep[b]=False, skip updates for all heads for that batch
        # We do this by setting overwrite=False and touch=False for those batches.
        keep_b = keep.view(B, 1)  # (B,1)
        overwrite = overwrite & keep_b  # broadcast to (B,Hkv)

        # Touch always for kept batches (even if not overwriting)
        touch = keep_b.expand(B, Hkv)

        # Update content for overwrite entries
        if bool(overwrite.any()):
            # Build one-hot mask for overwrite slots
            w = torch.zeros((B, Hkv, Me), device=device, dtype=state.mem_k_exact.dtype)
            w.scatter_(-1, idx.unsqueeze(-1), overwrite.to(dtype=state.mem_k_exact.dtype).unsqueeze(-1))
            w4 = w.unsqueeze(-1)  # (B,Hkv,Me,1)

            state.mem_k_exact = state.mem_k_exact + w4 * (k_evict.unsqueeze(2) - state.mem_k_exact)
            state.mem_v_exact = state.mem_v_exact + w4 * (v_evict.unsqueeze(2) - state.mem_v_exact)

            # positions (overwrite only)
            pos_t = torch.full((B, Hkv, Me), int(pos_evict), device=device, dtype=state.mem_pos_exact.dtype)
            state.mem_pos_exact = torch.where(w.to(torch.bool), pos_t, state.mem_pos_exact)

            # mark used
            state.mem_used_exact = state.mem_used_exact | w.to(torch.bool)

        # Touch: update last_used timestamp for idx for kept batches
        if bool(touch.any()):
            t = torch.full((B, Hkv), int(pos_evict), device=device, dtype=state.mem_last_exact.dtype)
            # scatter per idx, but only where touch=True
            # Create a copy for gather/scatter
            last = state.mem_last_exact
            # Scatter requires value tensor matching idx shape
            update_vals = torch.where(touch, t, last.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
            last.scatter_(-1, idx.unsqueeze(-1), update_vals.unsqueeze(-1))
            state.mem_last_exact = last

            # If slot was free and we touched it (e.g. not novel but best_idx might be -inf?), handle:
            # In practice, if no used slots, best_idx will be 0 but scores are -inf; novel=True will trigger insert anyway.

    # ----------------------------
    # Memory update: summary bank
    # ----------------------------

    def _summary_update_one(
        self,
        state: CoDAGQALandmarkStateV3,
        *,
        k_evict: torch.Tensor,  # (B,Hkv,Dh)
        v_evict: torch.Tensor,  # (B,Hkv,Dh)
        pos_evict: int,
        gate: torch.Tensor,     # (B,) float32
    ) -> None:
        if self.Ms == 0:
            return

        B, Hkv, Dh = k_evict.shape
        Ms = self.Ms
        device = k_evict.device

        # gate threshold
        if self.write_policy != "none":
            keep = gate >= self.write_gate_threshold_summary  # (B,)
            if not bool(keep.any()):
                return
        else:
            keep = torch.ones((B,), device=device, dtype=torch.bool)

        used = state.mem_used_sum  # (B,Hkv,Ms)

        # If no used slots at all for some (B,Hkv), we will insert
        any_used = used.any(dim=-1)  # (B,Hkv)

        # cosine similarity against used slots (mask unused)
        k_n = F.normalize(k_evict, dim=-1, eps=1e-6)
        mem_n = F.normalize(state.mem_k_sum, dim=-1, eps=1e-6)
        scores = torch.einsum("bhd,bhmd->bhm", k_n, mem_n)  # (B,Hkv,Ms)
        scores = scores.masked_fill(~used, -1e9)
        best_score, best_idx = scores.max(dim=-1)  # (B,Hkv)

        # free slot selection
        free = ~used
        free_exists = free.any(dim=-1)
        free_idx = free.to(dtype=torch.int64).argmax(dim=-1)

        # LRU replacement among used
        last = state.mem_last_sum
        inf = torch.full_like(last, 2**62)
        last_masked = torch.where(used, last, inf)
        lru_idx = last_masked.argmin(dim=-1)

        insert_idx = torch.where(free_exists, free_idx, lru_idx)

        # decide insert/replace vs update
        novel = best_score < self.summary_novelty_threshold
        need_insert = novel | (~any_used)  # if no used, must insert

        idx = torch.where(need_insert, insert_idx, best_idx)  # (B,Hkv)

        # One-hot mask for selected slot
        w = torch.zeros((B, Hkv, Ms), device=device, dtype=state.mem_k_sum.dtype)
        w.scatter_(-1, idx.unsqueeze(-1), 1.0)
        w4 = w.unsqueeze(-1)

        # eta_eff (scaled by gate)
        eta_base = torch.sigmoid(self.summary_eta_logit).to(dtype=state.mem_k_sum.dtype, device=device)  # (Hkv,)
        eta = eta_base.view(1, Hkv, 1, 1).expand(B, Hkv, 1, 1)

        if self.write_policy != "none":
            g = gate.to(device=device, dtype=state.mem_k_sum.dtype).view(B, 1, 1, 1)
            eta = eta * g

        if self.summary_overwrite_on_insert:
            overwrite = need_insert.unsqueeze(-1).unsqueeze(-1)  # (B,Hkv,1,1)
            eta = torch.where(overwrite, torch.ones_like(eta), eta)

        # Apply keep mask per batch
        keep_b = keep.to(device=device).view(B, 1, 1, 1)
        eta = eta * keep_b.to(dtype=eta.dtype)

        # EMA update for selected slots
        state.mem_k_sum = state.mem_k_sum + eta * w4 * (k_evict.unsqueeze(2) - state.mem_k_sum)
        state.mem_v_sum = state.mem_v_sum + eta * w4 * (v_evict.unsqueeze(2) - state.mem_v_sum)

        # position EMA (float)
        pos_t = torch.full((B, Hkv, 1), float(pos_evict), device=device, dtype=state.mem_pos_sum.dtype)
        eta_pos = eta.squeeze(-1).squeeze(-1).to(dtype=state.mem_pos_sum.dtype)  # (B,Hkv)
        state.mem_pos_sum = state.mem_pos_sum + (eta_pos.unsqueeze(-1) * w.to(dtype=state.mem_pos_sum.dtype)) * (pos_t - state.mem_pos_sum)

        # mark used for inserts; in practice we can just OR with w>0 where keep
        if self.write_policy != "none":
            # only mark used where keep_b True
            used_update = (w > 0) & keep.to(device=device).view(B, 1, 1)
        else:
            used_update = (w > 0)
        state.mem_used_sum = state.mem_used_sum | used_update

        # touch last_used timestamps where keep
        t = torch.full((B, Hkv), int(pos_evict), device=device, dtype=state.mem_last_sum.dtype)
        touch = keep.to(device=device).view(B, 1).expand(B, Hkv)
        last = state.mem_last_sum
        update_vals = torch.where(touch, t, last.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
        last.scatter_(-1, idx.unsqueeze(-1), update_vals.unsqueeze(-1))
        state.mem_last_sum = last

    # ----------------------------
    # Single-token cache write
    # ----------------------------

    def _write_one(self, state: CoDAGQALandmarkStateV3, *, k_new: torch.Tensor, v_new: torch.Tensor, g_new: torch.Tensor) -> None:
        """
        k_new, v_new: (B,Hkv,1,Dh) raw
        g_new:        (B,1) float32
        """
        B, Hkv, L, Dh = k_new.shape
        assert L == 1
        W = self.window

        if state.recent_filled < W:
            idx = state.recent_filled
            state.k_recent[:, :, idx:idx+1, :] = k_new
            state.v_recent[:, :, idx:idx+1, :] = v_new
            state.g_recent[:, idx:idx+1] = g_new
            state.recent_filled += 1
        else:
            ev_idx = state.recent_idx
            k_evict = state.k_recent[:, :, ev_idx, :]
            v_evict = state.v_recent[:, :, ev_idx, :]
            g_evict = state.g_recent[:, ev_idx]  # (B,)
            if self.detach_evicted:
                k_evict = k_evict.detach()
                v_evict = v_evict.detach()
                g_evict = g_evict.detach()

            pos_evict = int(state.pos - W)

            # Update both banks
            self._exact_update_one(state, k_evict=k_evict, v_evict=v_evict, pos_evict=pos_evict, gate=g_evict)
            self._summary_update_one(state, k_evict=k_evict, v_evict=v_evict, pos_evict=pos_evict, gate=g_evict)

            # Overwrite evicted slot with new
            state.k_recent[:, :, ev_idx:ev_idx+1, :] = k_new
            state.v_recent[:, :, ev_idx:ev_idx+1, :] = v_new
            state.g_recent[:, ev_idx:ev_idx+1] = g_new
            state.recent_idx = (ev_idx + 1) % W

        state.pos += 1

    # ----------------------------
    # Build KV for attention (prev context only)
    # ----------------------------

    def _build_prev_kv_for_attention(
        self,
        state: CoDAGQALandmarkStateV3,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds K/V in KV-head space for the CURRENT STATE (recent + exact + summary),
        excluding any "current token" (step() handles that by writing before attend if desired).

        Returns:
          k_prev: (B,Hkv,Lprev,Dh) with RoPE applied appropriately
          v_prev: (B,Hkv,Lprev,Dh)
          allowed_kv: (B,Hkv,Lprev) bool (True=keep)
        """
        k_recent_raw, v_recent, _g_recent, first_pos = self._get_recent_in_order(state)
        B, Hkv, Lr, Dh = k_recent_raw.shape

        # recent keys: consecutive positions -> use RotaryEmbedding tables
        if Lr > 0:
            cos, sin = self.rope(seq_len=Lr, offset=first_pos, device=device, dtype=dtype)
            k_recent = apply_rope(k_recent_raw.to(device=device, dtype=dtype), cos, sin)
        else:
            k_recent = k_recent_raw.to(device=device, dtype=dtype)

        v_recent = v_recent.to(device=device, dtype=dtype)

        # exact memory
        if self.Me > 0:
            k_exact_raw = state.mem_k_exact.to(device=device, dtype=dtype)
            v_exact = state.mem_v_exact.to(device=device, dtype=dtype)
            used_exact = state.mem_used_exact.to(device=device)
            if self.mem_rope_exact == "none":
                k_exact = k_exact_raw
            elif self.mem_rope_exact == "pos":
                pos = state.mem_pos_exact.to(device=device).to(dtype=torch.float32)
                k_exact = _rotate_with_positions(k_exact_raw, pos, self.rope.inv_freq.to(device=device))
            else:
                raise ValueError(f"Unknown mem_rope_exact: {self.mem_rope_exact}")
        else:
            k_exact = k_recent[:, :, :0, :]
            v_exact = v_recent[:, :, :0, :]
            used_exact = torch.zeros((B, Hkv, 0), device=device, dtype=torch.bool)

        # summary memory
        if self.Ms > 0:
            k_sum_raw = state.mem_k_sum.to(device=device, dtype=dtype)
            v_sum = state.mem_v_sum.to(device=device, dtype=dtype)
            used_sum = state.mem_used_sum.to(device=device)
            if self.mem_rope_summary == "none":
                k_sum = k_sum_raw
            elif self.mem_rope_summary == "mean_pos":
                pos = state.mem_pos_sum.to(device=device)  # float32
                k_sum = _rotate_with_positions(k_sum_raw, pos, self.rope.inv_freq.to(device=device))
            else:
                raise ValueError(f"Unknown mem_rope_summary: {self.mem_rope_summary}")
        else:
            k_sum = k_recent[:, :, :0, :]
            v_sum = v_recent[:, :, :0, :]
            used_sum = torch.zeros((B, Hkv, 0), device=device, dtype=torch.bool)

        # concatenate
        k_prev = torch.cat([k_recent, k_exact, k_sum], dim=2)
        v_prev = torch.cat([v_recent, v_exact, v_sum], dim=2)

        if self.mask_unused_memory:
            recent_ok = torch.ones((B, Hkv, Lr), device=device, dtype=torch.bool)
            allowed = torch.cat([recent_ok, used_exact, used_sum], dim=2)
        else:
            allowed = torch.ones((B, Hkv, k_prev.size(2)), device=device, dtype=torch.bool)

        return k_prev, v_prev, allowed

    # ----------------------------
    # Attention primitives (single token and block)
    # ----------------------------

    def _sdpa_two_stream(
        self,
        *,
        q: torch.Tensor,          # (B,Hq,Lq,Dh) RoPE already applied
        q_noise: torch.Tensor,    # (B,Hq,Lq,Dh)
        k_all: torch.Tensor,      # (B,Hkv,Lk,Dh) RoPE already applied
        v_all: torch.Tensor,      # (B,Hkv,Lk,Dh)
        attn_mask_h: Optional[torch.Tensor],  # (B,Hq,Lq,Lk) bool True=keep
        lambda_gate: torch.Tensor,# (B,Hq,Lq,1)
    ) -> torch.Tensor:
        """Returns (B,Hq,Lq,Dh)."""
        if _SDPA_ENABLE_GQA:
            out_sig = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask_h, is_causal=False, enable_gqa=True)
            out_noise = F.scaled_dot_product_attention(q_noise, k_all, v_all, attn_mask=attn_mask_h, is_causal=False, enable_gqa=True)
        else:
            num_repeat = self.num_heads // self.num_kv_heads
            k_rep = repeat_kv(k_all, num_repeat)
            v_rep = repeat_kv(v_all, num_repeat)
            out_sig = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask_h, is_causal=False)
            out_noise = F.scaled_dot_product_attention(q_noise, k_rep, v_rep, attn_mask=attn_mask_h, is_causal=False)

        out = out_sig - lambda_gate * out_noise
        out = self.head_norm(out)
        return out

    def _to_output(self, out_heads: torch.Tensor) -> torch.Tensor:
        """(B,H,L,Dh) -> (B,L,D)"""
        B, H, L, Dh = out_heads.shape
        x = out_heads.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return self.o_proj(x)

    def attend_step(self, x: torch.Tensor, state: CoDAGQALandmarkStateV3, *, query_pos: Optional[int] = None) -> torch.Tensor:
        """
        Attention only (no cache write), x: (B,1,D) -> y: (B,1,D)
        """
        B, L, D = x.shape
        assert L == 1
        device, dtype = x.device, x.dtype

        k_prev, v_prev, allowed_kv = self._build_prev_kv_for_attention(state, device=device, dtype=dtype)
        Lk = k_prev.size(2)
        if Lk == 0:
            return torch.zeros_like(x)

        q = self._project_q(x)  # (B,H,1,Dh)
        q_pos = state.pos if query_pos is None else int(query_pos)
        cos, sin = self.rope(seq_len=1, offset=q_pos, device=device, dtype=dtype)
        q = apply_rope(q, cos, sin)

        # noise query
        cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
        sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
        q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

        # λ gate
        lam = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)  # (B,H,1,1)

        # build mask in Q-head space
        num_repeat = self.num_heads // self.num_kv_heads
        allowed_h = allowed_kv.repeat_interleave(num_repeat, dim=1)  # (B,H,Lk)
        attn_mask = allowed_h.unsqueeze(-2)  # (B,H,1,Lk)

        out_heads = self._sdpa_two_stream(
            q=q, q_noise=q_noise, k_all=k_prev, v_all=v_prev, attn_mask_h=attn_mask, lambda_gate=lam
        )
        y = self._to_output(out_heads)
        return y

    # ----------------------------
    # Step API (decode)
    # ----------------------------

    def step(
        self,
        x: torch.Tensor,
        state: CoDAGQALandmarkStateV3,
        *,
        attend: bool = True,
        write_cache: bool = True,
        include_current_in_attention: bool = True,
    ) -> Tuple[torch.Tensor, CoDAGQALandmarkStateV3]:
        """
        Autoregressive step: x is (B,1,D).
        If include_current_in_attention and write_cache, the token is written first (bounded) then attended,
        with query_pos fixed to the pre-write position.
        """
        B, L, D = x.shape
        assert L == 1

        q_pos = state.pos

        if attend and include_current_in_attention and write_cache:
            k_new, v_new = self._project_kv_raw(x)
            g_new = self._write_gate(x)  # (B,1)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)
            y = self.attend_step(x, state, query_pos=q_pos)
            return y, state

        if attend:
            y = self.attend_step(x, state, query_pos=q_pos)
        else:
            y = torch.zeros_like(x)

        if write_cache:
            k_new, v_new = self._project_kv_raw(x)
            g_new = self._write_gate(x)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)

        return y, state

    # ----------------------------
    # Chunked prefill
    # ----------------------------

    def prefill_chunked(
        self,
        x_seq: torch.Tensor,             # (B,L,D)
        state: CoDAGQALandmarkStateV3,
        *,
        block_size: int = 256,
        write_cache: bool = True,
        return_outputs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], CoDAGQALandmarkStateV3]:
        """
        Chunked prefill policy:

        For each block of tokens:
          - Build prev context from (recent<=W + exact + summary) from the *start of the block*.
          - Attend over [prev context] + [current block causal prefix] using an explicit boolean mask.
          - After the block, write the block into the bounded cache, evicting older tokens and updating memory.

        This yields bounded KV memory while avoiding a per-token Python loop for attention.
        Memory updates still loop over evicted tokens (but per block).
        """
        B, L, D = x_seq.shape
        if L == 0:
            return (x_seq[:, :0, :].clone() if return_outputs else None), state
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        device, dtype = x_seq.device, x_seq.dtype
        outs = [] if return_outputs else None

        t = 0
        while t < L:
            blk = min(int(block_size), L - t)
            x_blk = x_seq[:, t:t+blk, :]  # (B,blk,D)
            pos0 = int(state.pos)

            # Build prev KV (RoPE applied)
            k_prev, v_prev, allowed_prev = self._build_prev_kv_for_attention(state, device=device, dtype=dtype)
            Lprev = int(k_prev.size(2))

            # Project Q, K, V for block (raw)
            q = self._project_q(x_blk)             # (B,Hq,blk,Dh) raw
            k_blk_raw, v_blk = self._project_kv_raw(x_blk)  # (B,Hkv,blk,Dh) raw
            g_blk = self._write_gate(x_blk)        # (B,blk)

            # RoPE on Q and K(block) using consecutive positions
            cos_q, sin_q = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            q = apply_rope(q, cos_q, sin_q)

            cos_k, sin_k = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            k_blk = apply_rope(k_blk_raw.to(device=device, dtype=dtype), cos_k, sin_k)

            # Build combined KV for attention: prev + block
            k_all = torch.cat([k_prev, k_blk], dim=2)  # (B,Hkv,Lprev+blk,Dh)
            v_all = torch.cat([v_prev, v_blk.to(device=device, dtype=dtype)], dim=2)

            # Allowed mask for keys: prev mask + all-true for block keys
            if self.mask_unused_memory:
                block_ok = torch.ones((B, self.num_kv_heads, blk), device=device, dtype=torch.bool)
                allowed_kv = torch.cat([allowed_prev, block_ok], dim=2)  # (B,Hkv,Lk)
            else:
                allowed_kv = torch.ones((B, self.num_kv_heads, Lprev + blk), device=device, dtype=torch.bool)

            # Build causal mask for block queries: (blk, Lprev+blk)
            # allow all prev keys, and causal within block.
            if Lprev > 0:
                prev_part = torch.ones((blk, Lprev), device=device, dtype=torch.bool)
            else:
                prev_part = torch.ones((blk, 0), device=device, dtype=torch.bool)
            block_part = torch.tril(torch.ones((blk, blk), device=device, dtype=torch.bool), diagonal=0)
            causal = torch.cat([prev_part, block_part], dim=1)  # (blk, Lk)

            # Expand masks to Q-head space and combine
            num_repeat = self.num_heads // self.num_kv_heads
            allowed_h = allowed_kv.repeat_interleave(num_repeat, dim=1)  # (B,Hq,Lk)
            # combined: (B,Hq,blk,Lk)
            attn_mask = causal.view(1, 1, blk, Lprev + blk) & allowed_h.unsqueeze(2)

            # λ gate per token/head
            lam = torch.sigmoid(self.lambda_proj(x_blk)).transpose(1, 2).unsqueeze(-1)  # (B,Hq,blk,1)

            # Noise query
            cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
            sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
            q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

            out_heads = self._sdpa_two_stream(
                q=q, q_noise=q_noise, k_all=k_all, v_all=v_all, attn_mask_h=attn_mask, lambda_gate=lam
            )
            y_blk = self._to_output(out_heads)  # (B,blk,D)

            if return_outputs:
                outs.append(y_blk)

            # Cache write for block
            if write_cache:
                self._write_block(state, k_blk_raw=k_blk_raw, v_blk=v_blk, g_blk=g_blk, pos0=pos0)

            t += blk

        if return_outputs:
            return torch.cat(outs, dim=1), state
        return None, state

    # ----------------------------
    # Block cache write (bounded) + memory update
    # ----------------------------

    def _write_block(
        self,
        state: CoDAGQALandmarkStateV3,
        *,
        k_blk_raw: torch.Tensor,   # (B,Hkv,blk,Dh)
        v_blk: torch.Tensor,       # (B,Hkv,blk,Dh)
        g_blk: torch.Tensor,       # (B,blk) float32
        pos0: int,
    ) -> None:
        """
        Update recent + memory after ingesting a block, in time order.

        Policy:
          concat = [old_recent] + [block]
          keep last W as new recent
          evict older -> update exact+summary in time order
        """
        device = k_blk_raw.device
        dtype = k_blk_raw.dtype

        k_old, v_old, g_old, first_pos = self._get_recent_in_order(state)
        B, Hkv, Lr, Dh = k_old.shape
        blk = int(k_blk_raw.size(2))
        W = self.window

        # Build concatenated sequences in time order
        k_cat = torch.cat([k_old, k_blk_raw], dim=2)  # (B,Hkv,Lr+blk,Dh)
        v_cat = torch.cat([v_old, v_blk], dim=2)
        g_cat = torch.cat([g_old, g_blk.to(device=device, dtype=torch.float32)], dim=1)  # (B,Lr+blk)

        total = Lr + blk
        keep_len = min(W, total)
        evict_len = total - keep_len

        # positions for concatenated sequence (shared across B,Hkv)
        # old recent positions: [first_pos .. first_pos+Lr-1]
        # block positions:      [pos0 .. pos0+blk-1] == [state.pos ..]
        pos_old = torch.arange(first_pos, first_pos + Lr, device=device, dtype=torch.int64)
        pos_blk = torch.arange(pos0, pos0 + blk, device=device, dtype=torch.int64)
        pos_cat = torch.cat([pos_old, pos_blk], dim=0)  # (total,)

        # Evict oldest tokens and update memory in time order
        if evict_len > 0:
            for i in range(evict_len):
                k_e = k_cat[:, :, i, :]  # (B,Hkv,Dh)
                v_e = v_cat[:, :, i, :]
                g_e = g_cat[:, i]        # (B,)
                if self.detach_evicted:
                    k_e = k_e.detach()
                    v_e = v_e.detach()
                    g_e = g_e.detach()
                pos_e = int(pos_cat[i].item())
                self._exact_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)
                self._summary_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)

        # New recent = last keep_len tokens
        if keep_len > 0:
            k_new = k_cat[:, :, total - keep_len : total, :]
            v_new = v_cat[:, :, total - keep_len : total, :]
            g_new = g_cat[:, total - keep_len : total]
        else:
            k_new = k_cat[:, :, :0, :]
            v_new = v_cat[:, :, :0, :]
            g_new = g_cat[:, :0]

        self._set_recent_from_sequence(state, k_seq=k_new, v_seq=v_new, g_seq=g_new)

        # Advance position
        state.pos = int(pos0 + blk)

    # ----------------------------
    # Utility: estimate cache bytes
    # ----------------------------

    def cache_bytes(self, *, batch_size: int, dtype: torch.dtype) -> int:
        """
        Rough bytes for K/V storage + gates + positions for one layer.
        This is a lower bound; Python object overhead not included.
        """
        B = int(batch_size)
        Hkv = self.num_kv_heads
        Dh = self.head_dim
        W = self.window
        Me = self.Me
        Ms = self.Ms

        bytes_per = torch.tensor([], dtype=dtype).element_size()
        # K + V
        kv = 2 * B * Hkv * (W + Me + Ms) * Dh * bytes_per
        # gates (float32) and positions (int64 + float32)
        gates = B * W * 4
        pos_exact = B * Hkv * Me * 8
        pos_sum = B * Hkv * Ms * 4
        return int(kv + gates + pos_exact + pos_sum)
