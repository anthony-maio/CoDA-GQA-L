# coda_gqa_l_perf1.py
# Performance Iteration 1 for CoDA-GQA-L:
#   - Store RoPE-applied keys in caches (recent + memory) to avoid per-step re-rotation.
#   - Use per-batch (not per-kv-head) slot occupancy masks so attention masks broadcast across heads.
#   - Reduce prefill chunk mask size to (B,1,Lq,Lk) and avoid repeat_interleave across heads.
#   - Replace O(evict_len) Python loops in block cache writes with:
#       * exact bank: top-K gated snapshot updates (small loop K)
#       * summary bank: vectorized hard-assignment + scatter_add update
#
# This keeps the CoDA core (2x SDPA calls, subtract + HeadwiseRMSNorm) and a bounded cache:
#   recent window W + exact bank Me + summary bank Ms
# Memory per layer is O(W + Me + Ms) independent of prompt length.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coda_gqa import HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv, _apply_pairwise_rotation


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


@dataclass
class CoDAGQALandmarkStatePerf1:
    # Recent ring-buffer (RoPE-applied keys)
    k_recent: torch.Tensor     # (B, Hkv, W, Dh)
    v_recent: torch.Tensor     # (B, Hkv, W, Dh)
    g_recent: torch.Tensor     # (B, W) float32
    recent_idx: int
    recent_filled: int

    # Exact snapshot memory (RoPE-applied keys)
    mem_k_exact: torch.Tensor  # (B, Hkv, Me, Dh)
    mem_v_exact: torch.Tensor  # (B, Hkv, Me, Dh)
    mem_used_exact: torch.Tensor  # (B, Me) bool
    mem_last_exact: torch.Tensor  # (B, Me) int64

    # Summary prototype memory (RoPE-applied keys)
    mem_k_sum: torch.Tensor    # (B, Hkv, Ms, Dh)
    mem_v_sum: torch.Tensor    # (B, Hkv, Ms, Dh)
    mem_used_sum: torch.Tensor # (B, Ms) bool

    # Absolute position of next token to be written
    pos: int


class CoDAGQALandmarkPerf1(nn.Module):
    """
    CoDA-GQA-L (Performance Iteration 1).

    Core attention:
      out = RMSNorm( Attn(q, K, V) - λ * Attn(Rθ(q), K, V) )

    Bounded KV (per layer):
      K/V = [recent window W] + [exact snapshots Me] + [summary prototypes Ms]

    Key performance change vs v3:
      - store RoPE-applied keys in caches so decode does NOT re-rotate K every step.

    Notes / assumptions:
      - Memory slot occupancy is shared across KV heads (per batch) to allow a compact
        attention mask (B,1,Lq,Lk) that broadcasts across heads.
      - Novelty/hit decisions use cosine similarity over RoPE-applied keys.
        This makes slot selection position-sensitive (tradeoff for performance).
      - Chunked prefill uses a "block policy": attention uses prev-memory from block start;
        cache+memory updates happen after the block.
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
        write_gate_init_bias: float = -2.0,
        write_gate_threshold_exact: float = 0.10,
        write_gate_threshold_summary: float = 0.05,
        # Exact bank behavior
        exact_match_threshold: float = 0.90,
        exact_novelty_threshold: float = 0.70,
        exact_refresh_on_hit: bool = False,
        exact_candidates_per_block: int = 8,
        # Summary bank behavior
        summary_eta_init_logit: float = -3.0,
        summary_candidates_per_block: int = 64,
        summary_overwrite_on_insert: bool = True,
        # Memory init
        mem_init: Literal["zeros", "random_normal"] = "random_normal",
        mask_unused_memory: bool = True,
        detach_evicted: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if window <= 0:
            raise ValueError("window must be positive")
        if num_landmarks_exact < 0 or num_landmarks_summary < 0:
            raise ValueError("num_landmarks_* must be non-negative")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (RoPE requirement)")

        self.window = int(window)
        self.Me = int(num_landmarks_exact)
        self.Ms = int(num_landmarks_summary)

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.theta = nn.Parameter(torch.full((self.num_heads, self.head_dim // 2), float(theta_init)))

        self.lambda_proj = nn.Linear(self.embed_dim, self.num_heads, bias=True)
        nn.init.constant_(self.lambda_proj.bias, float(lambda_init_bias))

        self.write_policy = write_policy
        self.write_proj = nn.Linear(self.embed_dim, 1, bias=True)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, float(write_gate_init_bias))
        self.write_gate_threshold_exact = float(write_gate_threshold_exact)
        self.write_gate_threshold_summary = float(write_gate_threshold_summary)

        self.exact_match_threshold = float(exact_match_threshold)
        self.exact_novelty_threshold = float(exact_novelty_threshold)
        self.exact_refresh_on_hit = bool(exact_refresh_on_hit)
        self.exact_candidates_per_block = int(max(0, exact_candidates_per_block))

        self.summary_eta_logit = nn.Parameter(torch.full((self.num_kv_heads,), float(summary_eta_init_logit)))
        self.summary_candidates_per_block = int(max(0, summary_candidates_per_block))
        self.summary_overwrite_on_insert = bool(summary_overwrite_on_insert)

        self.mem_init = mem_init
        self.mask_unused_memory = bool(mask_unused_memory)
        self.detach_evicted = bool(detach_evicted)

        self.rope = RotaryEmbedding(self.head_dim, base=float(rope_base))

        self.head_norm = HeadwiseRMSNorm(self.head_dim, eps=eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        # small causal mask cache: keyed by (Lprev, blk, device)
        self._causal_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    # ----------------------------
    # State init
    # ----------------------------

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> CoDAGQALandmarkStatePerf1:
        B = int(batch_size)
        Hkv = self.num_kv_heads
        W = self.window
        Dh = self.head_dim
        Me = self.Me
        Ms = self.Ms

        k_recent = torch.zeros((B, Hkv, W, Dh), device=device, dtype=dtype)
        v_recent = torch.zeros((B, Hkv, W, Dh), device=device, dtype=dtype)
        g_recent = torch.zeros((B, W), device=device, dtype=torch.float32)

        def init_mem(shape):
            if self.mem_init == "zeros":
                return torch.zeros(shape, device=device, dtype=dtype)
            if self.mem_init == "random_normal":
                return torch.randn(shape, device=device, dtype=dtype) * 0.02
            raise ValueError(f"Unknown mem_init: {self.mem_init}")

        mem_k_exact = init_mem((B, Hkv, Me, Dh)) if Me > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_v_exact = torch.zeros((B, Hkv, Me, Dh), device=device, dtype=dtype) if Me > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_used_exact = torch.zeros((B, Me), device=device, dtype=torch.bool) if Me > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool)
        mem_last_exact = torch.zeros((B, Me), device=device, dtype=torch.int64) if Me > 0 else torch.zeros((B, 0), device=device, dtype=torch.int64)

        mem_k_sum = init_mem((B, Hkv, Ms, Dh)) if Ms > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_v_sum = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=dtype) if Ms > 0 else torch.zeros((B, Hkv, 0, Dh), device=device, dtype=dtype)
        mem_used_sum = torch.zeros((B, Ms), device=device, dtype=torch.bool) if Ms > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool)

        return CoDAGQALandmarkStatePerf1(
            k_recent=k_recent,
            v_recent=v_recent,
            g_recent=g_recent,
            recent_idx=0,
            recent_filled=0,
            mem_k_exact=mem_k_exact,
            mem_v_exact=mem_v_exact,
            mem_used_exact=mem_used_exact,
            mem_last_exact=mem_last_exact,
            mem_k_sum=mem_k_sum,
            mem_v_sum=mem_v_sum,
            mem_used_sum=mem_used_sum,
            pos=0,
        )

    # ----------------------------
    # Projections
    # ----------------------------

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,Hq,L,Dh)

    def _project_kv_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hkv,L,Dh)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def _write_gate(self, x: torch.Tensor) -> torch.Tensor:
        # (B,L) float32
        if self.write_policy == "none":
            return torch.ones((x.size(0), x.size(1)), device=x.device, dtype=torch.float32)
        g = torch.sigmoid(self.write_proj(x)).squeeze(-1)
        return g.to(dtype=torch.float32)

    # ----------------------------
    # KV assembly
    # ----------------------------

    def _build_kv(self, state: CoDAGQALandmarkStatePerf1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (k_all, v_all, allowed) where allowed is (B, Lk) bool True=keep."""
        B, Hkv, W, Dh = state.k_recent.shape

        # recent mask
        if state.recent_filled >= W:
            recent_ok = torch.ones((B, W), device=state.k_recent.device, dtype=torch.bool)
        else:
            n = int(state.recent_filled)
            # build once per call; W is small.
            recent_ok = torch.zeros((B, W), device=state.k_recent.device, dtype=torch.bool)
            if n > 0:
                recent_ok[:, :n] = True

        k_all = [state.k_recent]
        v_all = [state.v_recent]
        allowed = [recent_ok]

        if self.Me > 0:
            k_all.append(state.mem_k_exact)
            v_all.append(state.mem_v_exact)
            allowed.append(state.mem_used_exact)

        if self.Ms > 0:
            k_all.append(state.mem_k_sum)
            v_all.append(state.mem_v_sum)
            allowed.append(state.mem_used_sum)

        k_cat = torch.cat(k_all, dim=2)
        v_cat = torch.cat(v_all, dim=2)
        allowed_cat = torch.cat(allowed, dim=1)
        return k_cat, v_cat, allowed_cat

    # ----------------------------
    # SDPA two-stream
    # ----------------------------

    def _sdpa_two_stream(
        self,
        *,
        q: torch.Tensor,       # (B,Hq,Lq,Dh) RoPE applied
        q_noise: torch.Tensor, # (B,Hq,Lq,Dh)
        k: torch.Tensor,       # (B,Hkv,Lk,Dh) RoPE applied
        v: torch.Tensor,       # (B,Hkv,Lk,Dh)
        attn_mask: Optional[torch.Tensor],  # broadcastable to (B,Hq,Lq,Lk), bool True=keep
        lam: torch.Tensor,     # (B,Hq,Lq,1)
    ) -> torch.Tensor:
        if _SDPA_ENABLE_GQA:
            out_sig = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
            out_noise = F.scaled_dot_product_attention(q_noise, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
        else:
            num_repeat = self.num_heads // self.num_kv_heads
            k_rep = repeat_kv(k, num_repeat)
            v_rep = repeat_kv(v, num_repeat)
            out_sig = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, is_causal=False)
            out_noise = F.scaled_dot_product_attention(q_noise, k_rep, v_rep, attn_mask=attn_mask, is_causal=False)

        out = out_sig - lam * out_noise
        out = self.head_norm(out)
        return out

    def _to_output(self, out_heads: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = out_heads.shape
        x = out_heads.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return self.o_proj(x)

    # ----------------------------
    # Memory updates (decode: one token)
    # ----------------------------

    def _cos_scores_mean_heads(self, k_vec: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """k_vec: (B,Hkv,Dh), mem: (B,Hkv,M,Dh) -> scores (B,M) mean over Hkv."""
        k_n = F.normalize(k_vec, dim=-1, eps=1e-6)
        mem_n = F.normalize(mem, dim=-1, eps=1e-6)
        scores_h = torch.einsum("bhd,bhmd->bhm", k_n, mem_n)  # (B,Hkv,M)
        return scores_h.mean(dim=1)  # (B,M)

    def _exact_update_one(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_evict: torch.Tensor,  # (B,Hkv,Dh)
        v_evict: torch.Tensor,  # (B,Hkv,Dh)
        pos_evict: torch.Tensor,  # (B,) int64
        gate: torch.Tensor,     # (B,) float32
    ) -> None:
        if self.Me == 0:
            return
        B = k_evict.size(0)
        device = k_evict.device

        if self.write_policy != "none":
            keep_b = gate >= self.write_gate_threshold_exact
            if not bool(keep_b.any()):
                return
        else:
            keep_b = torch.ones((B,), device=device, dtype=torch.bool)

        used = state.mem_used_exact  # (B,Me)
        Me = self.Me

        # similarity to USED slots (mean over kv heads)
        scores = self._cos_scores_mean_heads(k_evict, state.mem_k_exact)  # (B,Me)
        scores = scores.masked_fill(~used, -1e9)
        best_score, best_idx = scores.max(dim=-1)  # (B,)

        any_used = used.any(dim=-1)  # (B,)
        # novel if (no used) OR best below novelty threshold
        novel = (~any_used) | (best_score < self.exact_novelty_threshold)
        hit = any_used & (best_score >= self.exact_match_threshold)

        free = ~used
        free_exists = free.any(dim=-1)
        free_idx = free.to(torch.int64).argmax(dim=-1)

        # LRU among used, else +inf
        last = state.mem_last_exact
        inf = torch.full_like(last, 2**62)
        last_masked = torch.where(used, last, inf)
        lru_idx = last_masked.argmin(dim=-1)
        insert_idx = torch.where(free_exists, free_idx, lru_idx)

        idx = torch.where(novel, insert_idx, best_idx)  # (B,)

        # overwrite decision
        if self.exact_refresh_on_hit:
            overwrite = novel | hit
        else:
            overwrite = novel

        overwrite = overwrite & keep_b
        touch = keep_b

        # one-hot mask (B,Me)
        w = torch.zeros((B, Me), device=device, dtype=state.mem_k_exact.dtype)
        w.scatter_(1, idx.view(B, 1), overwrite.to(dtype=state.mem_k_exact.dtype).view(B, 1))
        w4 = w[:, None, :, None]  # (B,1,Me,1), broadcast over Hkv

        if bool(overwrite.any()):
            state.mem_k_exact = state.mem_k_exact + w4 * (k_evict.unsqueeze(2) - state.mem_k_exact)
            state.mem_v_exact = state.mem_v_exact + w4 * (v_evict.unsqueeze(2) - state.mem_v_exact)
            state.mem_used_exact = state.mem_used_exact | (w.to(torch.bool))

        # touch last_used
        if bool(touch.any()):
            last = state.mem_last_exact
            # scatter positions at idx for touched batches
            # Create update values: pos_evict where touch else existing
            pos_vals = pos_evict.to(device=device, dtype=last.dtype)
            update_mask = touch
            # gather current
            cur = last.gather(1, idx.view(B, 1)).squeeze(1)
            new = torch.where(update_mask, pos_vals, cur)
            last.scatter_(1, idx.view(B, 1), new.view(B, 1))
            state.mem_last_exact = last

    def _summary_update_one(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_evict: torch.Tensor,  # (B,Hkv,Dh)
        v_evict: torch.Tensor,  # (B,Hkv,Dh)
        pos_evict: torch.Tensor,  # (B,) int64
        gate: torch.Tensor,     # (B,) float32
    ) -> None:
        if self.Ms == 0:
            return
        B = k_evict.size(0)
        device = k_evict.device
        Ms = self.Ms

        if self.write_policy != "none":
            keep_b = gate >= self.write_gate_threshold_summary
            if not bool(keep_b.any()):
                return
        else:
            keep_b = torch.ones((B,), device=device, dtype=torch.bool)

        used = state.mem_used_sum  # (B,Ms)

        scores = self._cos_scores_mean_heads(k_evict, state.mem_k_sum)  # (B,Ms)
        idx = scores.argmax(dim=-1)  # (B,)

        # eta base per kv head
        eta_base = torch.sigmoid(self.summary_eta_logit).to(device=device, dtype=state.mem_k_sum.dtype)  # (Hkv,)
        eta = eta_base.view(1, self.num_kv_heads, 1, 1).expand(B, self.num_kv_heads, 1, 1)

        if self.write_policy != "none":
            g = gate.to(device=device, dtype=state.mem_k_sum.dtype).view(B, 1, 1, 1)
            eta = eta * g

        # overwrite if inserting into unused slot
        if self.summary_overwrite_on_insert:
            is_new = (~used.gather(1, idx.view(B, 1)).squeeze(1))
            eta = torch.where(is_new.view(B, 1, 1, 1), torch.ones_like(eta), eta)

        # apply keep mask
        eta = eta * keep_b.to(dtype=eta.dtype, device=device).view(B, 1, 1, 1)

        # one-hot (B,Ms)
        w = torch.zeros((B, Ms), device=device, dtype=state.mem_k_sum.dtype)
        w.scatter_(1, idx.view(B, 1), 1.0)
        w4 = w[:, None, :, None]  # (B,1,Ms,1)

        state.mem_k_sum = state.mem_k_sum + eta * w4 * (k_evict.unsqueeze(2) - state.mem_k_sum)
        state.mem_v_sum = state.mem_v_sum + eta * w4 * (v_evict.unsqueeze(2) - state.mem_v_sum)

        # mark used for touched batches
        used_update = keep_b.view(B, 1) & (w.to(torch.bool))
        state.mem_used_sum = state.mem_used_sum | used_update

    # ----------------------------
    # Recent helpers
    # ----------------------------

    def _get_recent_time_order(self, state: CoDAGQALandmarkStatePerf1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (k_seq, v_seq, g_seq) in chronological order, length Lr<=W."""
        B, Hkv, W, Dh = state.k_recent.shape
        n = int(state.recent_filled)
        if n == 0:
            return state.k_recent[:, :, :0, :], state.v_recent[:, :, :0, :], state.g_recent[:, :0]
        if n < W:
            return state.k_recent[:, :, :n, :], state.v_recent[:, :, :n, :], state.g_recent[:, :n]
        i = int(state.recent_idx)
        k = torch.cat([state.k_recent[:, :, i:, :], state.k_recent[:, :, :i, :]], dim=2)
        v = torch.cat([state.v_recent[:, :, i:, :], state.v_recent[:, :, :i, :]], dim=2)
        g = torch.cat([state.g_recent[:, i:], state.g_recent[:, :i]], dim=1)
        return k, v, g

    def _set_recent_from_sequence(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_seq: torch.Tensor,  # (B,Hkv,Lr,Dh)
        v_seq: torch.Tensor,  # (B,Hkv,Lr,Dh)
        g_seq: torch.Tensor,  # (B,Lr)
    ) -> None:
        B, Hkv, Lr, Dh = k_seq.shape
        W = self.window
        if Lr > W:
            raise ValueError("k_seq longer than window")
        state.k_recent.zero_()
        state.v_recent.zero_()
        state.g_recent.zero_()
        if Lr > 0:
            state.k_recent[:, :, :Lr, :] = k_seq
            state.v_recent[:, :, :Lr, :] = v_seq
            state.g_recent[:, :Lr] = g_seq.to(dtype=torch.float32)
        state.recent_filled = int(Lr)
        state.recent_idx = 0

    # ----------------------------
    # Write one token (decode)
    # ----------------------------

    def _write_one(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_new: torch.Tensor,  # (B,Hkv,1,Dh) RoPE-applied
        v_new: torch.Tensor,  # (B,Hkv,1,Dh)
        g_new: torch.Tensor,  # (B,1) float32
    ) -> None:
        B, Hkv, L, Dh = k_new.shape
        assert L == 1
        W = self.window

        if state.recent_filled < W:
            idx = int(state.recent_filled)
            state.k_recent[:, :, idx:idx+1, :] = k_new
            state.v_recent[:, :, idx:idx+1, :] = v_new
            state.g_recent[:, idx:idx+1] = g_new
            state.recent_filled += 1
        else:
            ev_idx = int(state.recent_idx)
            k_e = state.k_recent[:, :, ev_idx, :]
            v_e = state.v_recent[:, :, ev_idx, :]
            g_e = state.g_recent[:, ev_idx]  # (B,)
            if self.detach_evicted:
                k_e = k_e.detach()
                v_e = v_e.detach()
                g_e = g_e.detach()

            pos_e = torch.full((B,), int(state.pos - W), device=k_new.device, dtype=torch.int64)

            self._exact_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)
            self._summary_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)

            # overwrite evicted slot
            state.k_recent[:, :, ev_idx:ev_idx+1, :] = k_new
            state.v_recent[:, :, ev_idx:ev_idx+1, :] = v_new
            state.g_recent[:, ev_idx:ev_idx+1] = g_new
            state.recent_idx = (ev_idx + 1) % W

        state.pos += 1

    # ----------------------------
    # Attention (step)
    # ----------------------------

    def attend_step(self, x: torch.Tensor, state: CoDAGQALandmarkStatePerf1, *, query_pos: Optional[int] = None) -> torch.Tensor:
        B, L, _ = x.shape
        assert L == 1
        device, dtype = x.device, x.dtype

        k_all, v_all, allowed = self._build_kv(state)
        if not bool(allowed.any()):
            return torch.zeros_like(x)

        # Project + RoPE query
        q = self._project_q(x)  # (B,Hq,1,Dh)
        q_pos = state.pos if query_pos is None else int(query_pos)
        cos, sin = self.rope(seq_len=1, offset=q_pos, device=device, dtype=dtype)
        q = apply_rope(q, cos, sin)

        # Noise query
        cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
        sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
        q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

        # λ gate
        lam = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)  # (B,Hq,1,1)

        # attn mask: (B,1,1,Lk)
        attn_mask = allowed[:, None, None, :]
        out_h = self._sdpa_two_stream(q=q, q_noise=q_noise, k=k_all, v=v_all, attn_mask=attn_mask, lam=lam)
        return self._to_output(out_h)

    def step(
        self,
        x: torch.Tensor,
        state: CoDAGQALandmarkStatePerf1,
        *,
        attend: bool = True,
        write_cache: bool = True,
        include_current_in_attention: bool = True,
    ) -> Tuple[torch.Tensor, CoDAGQALandmarkStatePerf1]:
        B, L, _ = x.shape
        assert L == 1
        q_pos = int(state.pos)

        if attend and include_current_in_attention and write_cache:
            # compute KV, apply RoPE to K at q_pos, write, then attend with query_pos=q_pos
            k_raw, v_new = self._project_kv_raw(x)  # (B,Hkv,1,Dh)
            cosk, sink = self.rope(seq_len=1, offset=q_pos, device=x.device, dtype=x.dtype)
            k_new = apply_rope(k_raw, cosk, sink)
            g_new = self._write_gate(x)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)
            y = self.attend_step(x, state, query_pos=q_pos)
            return y, state

        y = self.attend_step(x, state, query_pos=q_pos) if attend else torch.zeros_like(x)

        if write_cache:
            k_raw, v_new = self._project_kv_raw(x)
            cosk, sink = self.rope(seq_len=1, offset=q_pos, device=x.device, dtype=x.dtype)
            k_new = apply_rope(k_raw, cosk, sink)
            g_new = self._write_gate(x)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)

        return y, state

    # ----------------------------
    # Chunked prefill
    # ----------------------------

    def _get_causal_mask(self, *, Lprev: int, blk: int, device: torch.device) -> torch.Tensor:
        key = (int(Lprev), int(blk), device)
        m = self._causal_cache.get(key)
        if m is not None:
            return m
        # allow all prefix keys, causal within block
        if Lprev > 0:
            prefix = torch.ones((blk, Lprev), device=device, dtype=torch.bool)
        else:
            prefix = torch.ones((blk, 0), device=device, dtype=torch.bool)
        tril = torch.tril(torch.ones((blk, blk), device=device, dtype=torch.bool), diagonal=0)
        m = torch.cat([prefix, tril], dim=1)  # (blk, Lprev+blk)
        self._causal_cache[key] = m
        return m

    def prefill_chunked(
        self,
        x_seq: torch.Tensor,  # (B,L,D)
        state: CoDAGQALandmarkStatePerf1,
        *,
        block_size: int = 256,
        write_cache: bool = True,
        return_outputs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], CoDAGQALandmarkStatePerf1]:
        B, L, _ = x_seq.shape
        if L == 0:
            return (x_seq[:, :0, :].clone() if return_outputs else None), state
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        device, dtype = x_seq.device, x_seq.dtype
        outs = [] if return_outputs else None

        t = 0
        while t < L:
            blk = min(int(block_size), L - t)
            x_blk = x_seq[:, t:t+blk, :]
            pos0 = int(state.pos)

            # prev memory KV
            k_prev, v_prev, allowed_prev = self._build_kv(state)  # fixed length Lprev
            Lprev = int(k_prev.size(2))

            # project Q and block KV
            q = self._project_q(x_blk)  # (B,Hq,blk,Dh)
            k_raw, v_blk = self._project_kv_raw(x_blk)  # (B,Hkv,blk,Dh)
            g_blk = self._write_gate(x_blk)  # (B,blk)

            # RoPE apply to Q and K(block)
            cos_q, sin_q = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            q = apply_rope(q, cos_q, sin_q)

            cos_k, sin_k = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            k_blk = apply_rope(k_raw, cos_k, sin_k)

            # combine KV
            k_all = torch.cat([k_prev, k_blk], dim=2)  # (B,Hkv,Lprev+blk,Dh)
            v_all = torch.cat([v_prev, v_blk], dim=2)

            # allowed keys: prev allowed + block all-true
            if self.mask_unused_memory:
                block_ok = torch.ones((B, blk), device=device, dtype=torch.bool)
                allowed = torch.cat([allowed_prev, block_ok], dim=1)  # (B,Lk)
            else:
                allowed = torch.ones((B, Lprev + blk), device=device, dtype=torch.bool)

            # causal mask (blk, Lk)
            causal = self._get_causal_mask(Lprev=Lprev, blk=blk, device=device)
            # broadcast combine -> (B,1,blk,Lk)
            attn_mask = causal.view(1, 1, blk, Lprev + blk) & allowed[:, None, None, :]

            # λ gate
            lam = torch.sigmoid(self.lambda_proj(x_blk)).transpose(1, 2).unsqueeze(-1)  # (B,Hq,blk,1)

            # noise query
            cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
            sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
            q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

            out_h = self._sdpa_two_stream(q=q, q_noise=q_noise, k=k_all, v=v_all, attn_mask=attn_mask, lam=lam)
            y_blk = self._to_output(out_h)

            if return_outputs:
                outs.append(y_blk)

            if write_cache:
                self._write_block_fast(state, k_blk=k_blk, v_blk=v_blk, g_blk=g_blk, pos0=pos0)

            t += blk

        if return_outputs:
            return torch.cat(outs, dim=1), state
        return None, state

    # ----------------------------
    # Block cache write + memory update
    # ----------------------------

    def _write_block_fast(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_blk: torch.Tensor,   # (B,Hkv,blk,Dh) RoPE-applied
        v_blk: torch.Tensor,   # (B,Hkv,blk,Dh)
        g_blk: torch.Tensor,   # (B,blk) float32
        pos0: int,
    ) -> None:
        """Bounded write for a whole block. Uses reduced-loop exact updates and vectorized summary updates."""
        device = k_blk.device
        dtype = k_blk.dtype
        B, Hkv, blk, Dh = k_blk.shape
        W = self.window

        # Old recent in time order
        k_old, v_old, g_old = self._get_recent_time_order(state)
        Lr = int(k_old.size(2))

        # concat in time order
        k_cat = torch.cat([k_old, k_blk], dim=2)  # (B,Hkv,Lr+blk,Dh)
        v_cat = torch.cat([v_old, v_blk], dim=2)
        g_cat = torch.cat([g_old, g_blk.to(device=device, dtype=torch.float32)], dim=1)  # (B,total)
        total = Lr + blk
        keep_len = min(W, total)
        evict_len = total - keep_len

        # base position for concatenated stream
        # old recent spans [pos0-Lr, ..., pos0-1], block spans [pos0, ..., pos0+blk-1]
        pos_base = int(pos0 - Lr)

        if evict_len > 0:
            # evicted slices
            k_e = k_cat[:, :, :evict_len, :]
            v_e = v_cat[:, :, :evict_len, :]
            g_e = g_cat[:, :evict_len]
            if self.detach_evicted:
                k_e = k_e.detach()
                v_e = v_e.detach()
                g_e = g_e.detach()

            # -----------------
            # Exact bank: top-K candidates by gate
            # -----------------
            if self.Me > 0 and self.exact_candidates_per_block > 0:
                K = min(self.exact_candidates_per_block, evict_len)
                # topk per batch
                vals, idxs = torch.topk(g_e, k=K, dim=1, largest=True, sorted=True)  # (B,K)

                for r in range(K):
                    gate_r = vals[:, r]  # (B,)
                    idx_r = idxs[:, r]   # (B,)
                    # gather k/v at idx_r
                    idx_exp = idx_r.view(B, 1, 1, 1).expand(B, Hkv, 1, Dh)
                    k_r = k_e.gather(2, idx_exp).squeeze(2)
                    v_r = v_e.gather(2, idx_exp).squeeze(2)
                    pos_r = (torch.full((B,), pos_base, device=device, dtype=torch.int64) + idx_r.to(torch.int64))
                    self._exact_update_one(state, k_evict=k_r, v_evict=v_r, pos_evict=pos_r, gate=gate_r)

            # -----------------
            # Summary bank: vectorized hard assignment + scatter_add
            # -----------------
            if self.Ms > 0 and self.summary_candidates_per_block > 0:
                T = min(self.summary_candidates_per_block, evict_len)
                vals_s, idxs_s = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)  # (B,T)

                # optional: if gated, skip if all below threshold
                if self.write_policy != "none":
                    keep_tok = vals_s >= self.write_gate_threshold_summary  # (B,T)
                    if bool(keep_tok.any()):
                        # gather selected tokens
                        idx_exp = idxs_s.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                        k_sel = k_e.gather(2, idx_exp)
                        v_sel = v_e.gather(2, idx_exp)

                        w_tok = vals_s.to(device=device, dtype=k_sel.dtype)  # (B,T)
                        w_tok = torch.where(keep_tok, w_tok, torch.zeros_like(w_tok))

                        self._summary_update_block_hard(
                            state,
                            k_sel=k_sel,
                            v_sel=v_sel,
                            pos_sel=(torch.full((B, T), pos_base, device=device, dtype=torch.int64) + idxs_s.to(torch.int64)),
                            w_tok=w_tok,
                        )
                else:
                    # no gating => all tokens have weights vals_s
                    idx_exp = idxs_s.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                    k_sel = k_e.gather(2, idx_exp)
                    v_sel = v_e.gather(2, idx_exp)
                    w_tok = vals_s.to(device=device, dtype=k_sel.dtype)
                    self._summary_update_block_hard(
                        state,
                        k_sel=k_sel,
                        v_sel=v_sel,
                        pos_sel=(torch.full((B, T), pos_base, device=device, dtype=torch.int64) + idxs_s.to(torch.int64)),
                        w_tok=w_tok,
                    )

        # keep last W as new recent
        if keep_len > 0:
            k_new = k_cat[:, :, total - keep_len : total, :]
            v_new = v_cat[:, :, total - keep_len : total, :]
            g_new = g_cat[:, total - keep_len : total]
        else:
            k_new = k_cat[:, :, :0, :]
            v_new = v_cat[:, :, :0, :]
            g_new = g_cat[:, :0]

        self._set_recent_from_sequence(state, k_seq=k_new, v_seq=v_new, g_seq=g_new)
        state.pos = int(pos0 + blk)

    def _summary_update_block_hard(
        self,
        state: CoDAGQALandmarkStatePerf1,
        *,
        k_sel: torch.Tensor,     # (B,Hkv,T,Dh)
        v_sel: torch.Tensor,     # (B,Hkv,T,Dh)
        pos_sel: torch.Tensor,   # (B,T) int64 (currently unused)
        w_tok: torch.Tensor,     # (B,T) float weights (0 for skipped tokens)
    ) -> None:
        """Vectorized hard-assignment update of summary bank using scatter_add."""
        if self.Ms == 0:
            return
        B, Hkv, T, Dh = k_sel.shape
        Ms = self.Ms
        device = k_sel.device

        # If all weights are zero, nothing to do.
        if not bool((w_tok > 0).any()):
            return

        used = state.mem_used_sum  # (B,Ms)

        # scores: (B,T,Ms) mean over kv heads
        k_n = F.normalize(k_sel, dim=-1, eps=1e-6)
        mem_n = F.normalize(state.mem_k_sum, dim=-1, eps=1e-6)
        scores_h = torch.einsum("bhtd,bhmd->bhtm", k_n, mem_n)  # (B,Hkv,T,Ms)
        scores = scores_h.mean(dim=1)  # (B,T,Ms)

        idx_tok = scores.argmax(dim=-1)  # (B,T)

        # weights
        w = w_tok.to(device=device, dtype=k_sel.dtype)  # (B,T)

        # counts per slot: (B,Ms)
        count = torch.zeros((B, Ms), device=device, dtype=k_sel.dtype)
        count.scatter_add_(1, idx_tok, w)

        # weighted sums (B,Hkv,Ms,Dh)
        sum_k = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=k_sel.dtype)
        sum_v = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=v_sel.dtype)

        idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
        src_k = k_sel * w.view(B, 1, T, 1)
        src_v = v_sel * w.view(B, 1, T, 1)
        sum_k.scatter_add_(2, idx_exp, src_k)
        sum_v.scatter_add_(2, idx_exp, src_v)

        # eta per kv head
        eta_base = torch.sigmoid(self.summary_eta_logit).to(device=device, dtype=k_sel.dtype)  # (Hkv,)
        eta = eta_base.view(1, Hkv, 1, 1)

        # update slots with count>0
        count4 = count.view(B, 1, Ms, 1)
        active = count4 > 0
        avg_k = sum_k / (count4.clamp_min(1e-6))
        avg_v = sum_v / (count4.clamp_min(1e-6))

        # overwrite new slots where previously unused and now got mass
        new_slots = (~used).view(B, 1, Ms, 1) & active
        eta_eff = torch.where(new_slots & self.summary_overwrite_on_insert, torch.ones_like(eta), eta)

        state.mem_k_sum = torch.where(active, state.mem_k_sum + eta_eff * (avg_k - state.mem_k_sum), state.mem_k_sum)
        state.mem_v_sum = torch.where(active, state.mem_v_sum + eta_eff * (avg_v - state.mem_v_sum), state.mem_v_sum)

        # mark used
        state.mem_used_sum = state.mem_used_sum | (count > 0)

    # ----------------------------
    # Utility: estimate cache bytes
    # ----------------------------

    def cache_bytes(self, *, batch_size: int, dtype: torch.dtype) -> int:
        B = int(batch_size)
        Hkv = self.num_kv_heads
        Dh = self.head_dim
        W = self.window
        Me = self.Me
        Ms = self.Ms

        bytes_per = torch.tensor([], dtype=dtype).element_size()
        kv = 2 * B * Hkv * (W + Me + Ms) * Dh * bytes_per
        gates = B * W * 4
        used_masks = B * (Me + Ms)  # bool ~1 byte (approx)
        lru = B * Me * 8
        return int(kv + gates + used_masks + lru)
