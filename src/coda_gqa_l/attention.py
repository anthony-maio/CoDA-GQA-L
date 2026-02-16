# attention.py
# CoDA-GQA-L: Bounded-memory differential attention with GQA.
#
# Core attention:
#   out = RMSNorm( Attn(q, K, V) - lambda * Attn(R_theta(q), K, V) )
#
# Bounded KV (per layer):
#   K/V = [recent window W | exact landmarks Me | summary landmarks Ms]
#
# Key optimizations (vs earlier iterations):
#   - Single paged KV buffer (no torch.cat during decode)
#   - Single SDPA call via head-stacking (signal + noise queries)
#   - Vectorized block memory updates (scatter_reduce/scatter_add)
#   - RoPE applied at write time (no per-step re-rotation)
#   - Frequency-aware routing: exact bank uses V-routing, summary bank uses LF-K routing
#   - Phase-Safe EMA: summary bank blends only LF key band (HF zeroed)

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_banks import MemoryBankMixin
from .primitives import (
    HeadwiseRMSNorm,
    RotaryEmbedding,
    _apply_pairwise_rotation,
    apply_rope,
    repeat_kv,
)
from .state import CoDAGQALandmarkStatePerf2

# Re-export so `from .attention import CoDAGQALandmarkStatePerf2` still works.
__all__ = ["CoDAGQALandmarkPerf2", "CoDAGQALandmarkStatePerf2"]


class CoDAGQALandmarkPerf2(MemoryBankMixin, nn.Module):
    """CoDA-GQA-L: Bounded-memory differential attention with GQA.

    Attention produces two streams from a single query projection:
      - Signal: standard SDPA(q, K, V)
      - Inhibitory: SDPA(R_theta(q), K, V) where R_theta is a learned orthogonal rotation

    Output: RMSNorm(signal - lambda * inhibitory), where lambda is a learned per-token gate.

    Memory is bounded to O(W + Me + Ms) per layer via:
      - Recent window (W): exact ring buffer of recent tokens
      - Exact landmark bank (Me): novelty-filtered LRU cache of important evicted tokens
      - Summary landmark bank (Ms): EMA prototypes compressing older context

    Memory bank update logic lives in MemoryBankMixin (memory_banks.py).
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
        write_policy: str = "gated",
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
        mem_init: str = "random_normal",
        mask_unused_memory: bool = True,
        detach_evicted: bool = True,
        eps: float = 1e-6,
        # Metrics
        collect_metrics: bool = False,
        # Head norm mode: "full" = HeadwiseRMSNorm (default), "identity" = bypass
        # WARNING: These defaults (rope_interleaved=True, head_norm_mode="full")
        # differ from LlamaCoDAAdapter (rope_interleaved=False, head_norm_mode=
        # "identity"). When transferring weights between standalone usage and
        # LlamaCoDAAdapter, ensure settings match to avoid silent numerical
        # divergence. Llama-family models use non-interleaved (contiguous-half)
        # RoPE and identity head norm for cold-swap evaluation.
        head_norm_mode: str = "full",
        # RoPE dimension pairing: True = interleaved (0,1),(2,3),...
        # False = contiguous halves (0,D/2),(1,D/2+1),... (Llama convention)
        rope_interleaved: bool = True,
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

        # Frequency-aware split: LF band (position-invariant) starts at head_dim // 2.
        self.lf_start = self.head_dim // 2
        self.lf_dim = self.head_dim - self.lf_start
        self.energy_scale = math.sqrt(self.head_dim / self.lf_dim)

        self.window = int(window)
        self.Me = int(num_landmarks_exact)
        self.Ms = int(num_landmarks_summary)
        self.Lbuf = self.window + self.Me + self.Ms

        # Projections
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        # Orthogonal rotation angles (noise query)
        self.theta = nn.Parameter(torch.full((self.num_heads, self.head_dim // 2), float(theta_init)))

        # Cancellation strength lambda
        self.lambda_proj = nn.Linear(self.embed_dim, self.num_heads, bias=True)
        nn.init.constant_(self.lambda_proj.bias, float(lambda_init_bias))

        # Write policy gate
        self.write_policy = write_policy
        self.write_proj = nn.Linear(self.embed_dim, 1, bias=True)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, float(write_gate_init_bias))
        self.write_gate_threshold_exact = float(write_gate_threshold_exact)
        self.write_gate_threshold_summary = float(write_gate_threshold_summary)

        # Exact bank
        self.exact_match_threshold = float(exact_match_threshold)
        self.exact_novelty_threshold = float(exact_novelty_threshold)
        self.exact_refresh_on_hit = bool(exact_refresh_on_hit)
        self.exact_candidates_per_block = int(max(0, exact_candidates_per_block))

        # Summary bank
        self.summary_eta_logit = nn.Parameter(torch.full((self.num_kv_heads,), float(summary_eta_init_logit)))
        self.summary_candidates_per_block = int(max(0, summary_candidates_per_block))
        self.summary_overwrite_on_insert = bool(summary_overwrite_on_insert)

        self.mem_init = mem_init
        self.mask_unused_memory = bool(mask_unused_memory)
        self.detach_evicted = bool(detach_evicted)
        self.collect_metrics = bool(collect_metrics)
        self.rope_interleaved = bool(rope_interleaved)

        self.rope = RotaryEmbedding(self.head_dim, base=float(rope_base))

        self.head_norm_mode = head_norm_mode
        if head_norm_mode == "full":
            self.head_norm = HeadwiseRMSNorm(self.head_dim, eps=eps)
        elif head_norm_mode == "identity":
            self.head_norm = nn.Identity()
        else:
            raise ValueError(f"head_norm_mode must be 'full' or 'identity', got {head_norm_mode!r}")
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        # Causal mask cache for chunked prefill: key = (Lprev_fixed, blk, device)
        self._causal_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

        # Scratch buffer cache: eliminates per-block allocations in bank updates.
        # Keyed by (B, device, dtype) so different batch sizes / devices don't collide.
        self._scratch: Dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # State init
    # ------------------------------------------------------------------

    def init_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mem_init: Optional[str] = None,
    ) -> CoDAGQALandmarkStatePerf2:
        """Create a fresh KV cache state.

        Args:
            batch_size: Batch size.
            device: Target device.
            dtype: Data type for KV buffers.
            mem_init: Override for memory bank initialization strategy.
                ``None`` (default) uses ``self.mem_init``.  Pass ``"zeros"``
                during training for deterministic initialization that is safe
                with gradient checkpointing (avoids RNG-dependent shapes on
                recomputation).
        """
        B = int(batch_size)
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        k_buf = torch.zeros((B, Hkv, self.Lbuf, Dh), device=device, dtype=dtype)
        v_buf = torch.zeros((B, Hkv, self.Lbuf, Dh), device=device, dtype=dtype)
        allowed = torch.zeros((B, self.Lbuf), device=device, dtype=torch.bool)

        g_recent = torch.zeros((B, self.window), device=device, dtype=torch.float32)

        mem_last_exact = torch.zeros((B, self.Me), device=device, dtype=torch.int64) if self.Me > 0 else torch.zeros((B, 0), device=device, dtype=torch.int64)

        _mem_init = mem_init if mem_init is not None else self.mem_init
        if _mem_init == "random_normal":
            if self.Me > 0:
                k_buf[:, :, self.window:self.window + self.Me, :].normal_(mean=0.0, std=0.02)
                v_buf[:, :, self.window:self.window + self.Me, :].normal_(mean=0.0, std=0.02)
            if self.Ms > 0:
                # Phase-Safe: HF key band stays zero, only LF band is initialized.
                k_buf[:, :, self.window + self.Me:, self.lf_start:].normal_(mean=0.0, std=0.02)
                v_buf[:, :, self.window + self.Me:, :].normal_(mean=0.0, std=0.02)
        elif _mem_init == "zeros":
            pass
        else:
            raise ValueError("mem_init must be 'zeros' or 'random_normal'")

        state = CoDAGQALandmarkStatePerf2(
            k_buf=k_buf,
            v_buf=v_buf,
            allowed=allowed,
            g_recent=g_recent,
            recent_idx=0,
            recent_filled=0,
            mem_last_exact=mem_last_exact,
            pos=0,
        )

        # Initialize cached normalized routing targets.
        # Exact bank: V-routing (RoPE-free values preserve semantic matching).
        # Summary bank: LF-K routing (low-frequency key band is position-invariant).
        if self.Me > 0:
            state._exact_v_norm = F.normalize(
                v_buf[:, :, self.window:self.window + self.Me, :], dim=-1, eps=1e-6
            ).clone()
        if self.Ms > 0:
            state._sum_lf_k_norm = F.normalize(
                k_buf[:, :, self.window + self.Me:, self.lf_start:], dim=-1, eps=1e-6
            ).clone()

        # Initialize lightweight metrics counters when collection is enabled.
        if self.collect_metrics:
            state.metrics = {
                "exact_hits": 0,
                "exact_inserts": 0,
                "exact_overwrites": 0,
                "exact_fill_ratio": 0.0,
                "summary_updates": 0,
                "summary_inserts": 0,
                "summary_fill_ratio": 0.0,
                "tokens_gated_out": 0,
                "total_evictions": 0,
            }

        return state

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_kv_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def _write_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.write_policy == "none":
            return torch.ones((x.size(0), x.size(1)), device=x.device, dtype=torch.float32)
        g = torch.sigmoid(self.write_proj(x)).squeeze(-1)
        return g.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # SDPA (stacked heads)
    # ------------------------------------------------------------------

    def _sdpa_stacked_two_stream(
        self,
        *,
        q: torch.Tensor,        # (B,H,Lq,Dh)
        q_noise: torch.Tensor,  # (B,H,Lq,Dh)
        k: torch.Tensor,        # (B,Hkv,Lk,Dh)
        v: torch.Tensor,        # (B,Hkv,Lk,Dh)
        attn_mask: Optional[torch.Tensor],  # broadcastable to (B,2H,Lq,Lk)
        lam: torch.Tensor,      # (B,H,Lq,1)
        is_causal: bool = False,
    ) -> torch.Tensor:
        q_cat = torch.cat([q, q_noise], dim=1)  # (B,2H,Lq,Dh)

        # GQA head alignment for stacked two-stream attention:
        # q_cat = [sig_h0..sig_h{H-1}, noise_h0..noise_h{H-1}]
        # We need each signal head and its noise counterpart to attend
        # to the SAME KV head.  First expand KV for normal GQA (Hkv→H),
        # then duplicate for the noise stream (H→2H).
        # This gives [kv0xG, kv1xG, ..., kv0xG, kv1xG, ...] which
        # correctly aligns with [signal heads | noise heads].
        groups = self.num_heads // self.num_kv_heads
        k_gqa = repeat_kv(k, groups)   # (B, H, Lk, Dh)
        v_gqa = repeat_kv(v, groups)   # (B, H, Lk, Dh)
        k_rep = torch.cat([k_gqa, k_gqa], dim=1)  # (B, 2H, Lk, Dh)
        v_rep = torch.cat([v_gqa, v_gqa], dim=1)  # (B, 2H, Lk, Dh)
        out_cat = F.scaled_dot_product_attention(
            q_cat, k_rep, v_rep, attn_mask=attn_mask, is_causal=is_causal,
        )

        H = self.num_heads
        out_sig = out_cat[:, :H, :, :]
        out_noise = out_cat[:, H:, :, :]
        out = out_sig - lam * out_noise
        out = self.head_norm(out)
        return out

    def _to_output(self, out_heads: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = out_heads.shape
        x = out_heads.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return self.o_proj(x)

    # ------------------------------------------------------------------
    # Attention (decode step)
    # ------------------------------------------------------------------

    def attend_step(self, x: torch.Tensor, state: CoDAGQALandmarkStatePerf2, *, query_pos: Optional[int] = None) -> torch.Tensor:
        B, L, _ = x.shape
        assert L == 1
        device, dtype = x.device, x.dtype

        k_attend = state.k_buf
        v_attend = state.v_buf
        attn_mask = None

        if self.mask_unused_memory:
            allowed = state.allowed
            if not bool(allowed.any()):
                return torch.zeros_like(x)
            if not bool(allowed.all()):
                if B == 1:
                    # Dense packing: select only valid slots → no mask needed
                    # → unlocks FlashAttention / MemEfficient SDPA backends.
                    valid_idx = allowed[0].nonzero(as_tuple=True)[0]
                    k_attend = state.k_buf[:, :, valid_idx, :]
                    v_attend = state.v_buf[:, :, valid_idx, :]
                else:
                    attn_mask = allowed[:, None, None, :]

        q = self._project_q(x)
        q_pos = state.pos if query_pos is None else int(query_pos)
        cos, sin = self.rope(seq_len=1, offset=q_pos, device=device, dtype=dtype)
        q = apply_rope(q, cos, sin, interleaved=self.rope_interleaved)

        cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
        sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
        q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

        lam = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)

        out_h = self._sdpa_stacked_two_stream(
            q=q, q_noise=q_noise, k=k_attend, v=v_attend,
            attn_mask=attn_mask, lam=lam,
        )
        return self._to_output(out_h)

    def step(
        self,
        x: torch.Tensor,
        state: CoDAGQALandmarkStatePerf2,
        *,
        attend: bool = True,
        write_cache: bool = True,
        include_current_in_attention: bool = True,
    ) -> Tuple[torch.Tensor, CoDAGQALandmarkStatePerf2]:
        B, L, _ = x.shape
        assert L == 1
        q_pos = int(state.pos)

        if attend and include_current_in_attention and write_cache:
            k_raw, v_new = self._project_kv_raw(x)
            cosk, sink = self.rope(seq_len=1, offset=q_pos, device=x.device, dtype=x.dtype)
            k_new = apply_rope(k_raw, cosk, sink, interleaved=self.rope_interleaved)
            g_new = self._write_gate(x)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)
            y = self.attend_step(x, state, query_pos=q_pos)
            return y, state

        y = self.attend_step(x, state, query_pos=q_pos) if attend else torch.zeros_like(x)

        if write_cache:
            k_raw, v_new = self._project_kv_raw(x)
            cosk, sink = self.rope(seq_len=1, offset=q_pos, device=x.device, dtype=x.dtype)
            k_new = apply_rope(k_raw, cosk, sink, interleaved=self.rope_interleaved)
            g_new = self._write_gate(x)
            self._write_one(state, k_new=k_new, v_new=v_new, g_new=g_new)

        return y, state

    # ------------------------------------------------------------------
    # Chunked prefill
    # ------------------------------------------------------------------

    def _get_causal_mask(self, *, Lprev: int, blk: int, device: torch.device) -> torch.Tensor:
        key = (int(Lprev), int(blk), device)
        m = self._causal_cache.get(key)
        if m is not None:
            return m
        if Lprev > 0:
            prefix = torch.ones((blk, Lprev), device=device, dtype=torch.bool)
        else:
            prefix = torch.ones((blk, 0), device=device, dtype=torch.bool)
        tril = torch.tril(torch.ones((blk, blk), device=device, dtype=torch.bool), diagonal=0)
        m = torch.cat([prefix, tril], dim=1)
        self._causal_cache[key] = m
        return m

    def prefill_chunked(
        self,
        x_seq: torch.Tensor,
        state: CoDAGQALandmarkStatePerf2,
        *,
        block_size: int = 256,
        write_cache: bool = True,
        return_outputs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], CoDAGQALandmarkStatePerf2]:
        B, L, _ = x_seq.shape
        if L == 0:
            return (x_seq[:, :0, :].clone() if return_outputs else None), state
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        device, dtype = x_seq.device, x_seq.dtype
        outs = [] if return_outputs else None

        Lprev_fixed = self.Lbuf

        t = 0
        while t < L:
            blk = min(int(block_size), L - t)
            x_blk = x_seq[:, t:t+blk, :]
            pos0 = int(state.pos)

            k_prev = state.k_buf
            v_prev = state.v_buf
            allowed_prev = state.allowed

            q = self._project_q(x_blk)
            k_raw, v_blk = self._project_kv_raw(x_blk)
            g_blk = self._write_gate(x_blk)

            cos_q, sin_q = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            q = apply_rope(q, cos_q, sin_q, interleaved=self.rope_interleaved)
            cos_k, sin_k = self.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
            k_blk = apply_rope(k_raw, cos_k, sin_k, interleaved=self.rope_interleaved)

            # Dense packing: when B==1 and there are invalid prefix slots,
            # pack only valid prefix slots to eliminate unnecessary keys.
            use_packing = (
                B == 1
                and self.mask_unused_memory
                and not bool(allowed_prev[0].all())
            )

            if use_packing:
                valid_idx = allowed_prev[0].nonzero(as_tuple=True)[0]
                k_prefix = k_prev[:, :, valid_idx, :]
                v_prefix = v_prev[:, :, valid_idx, :]
                k_all = torch.cat([k_prefix, k_blk], dim=2)
                v_all = torch.cat([v_prefix, v_blk], dim=2)
                Lpacked = int(valid_idx.numel())
            else:
                k_all = torch.cat([k_prev, k_blk], dim=2)
                v_all = torch.cat([v_prev, v_blk], dim=2)
                Lpacked = Lprev_fixed

            # Build attention mask.
            # NOTE: is_causal=True uses upper-left alignment in PyTorch (>=2.5),
            # which is WRONG when Lq < Lk (prefix-LM pattern). We need all
            # queries to see the full prefix plus causal within the block,
            # so we always construct an explicit mask when there is a prefix.
            #
            # SDPA bool mask: True = keep/participate, False = mask out (-inf).
            Lk_total = k_all.size(2)
            if Lk_total == blk:
                # No prefix → square attention → is_causal=True is correct.
                attn_mask = None
                is_causal = True
            else:
                # Prefix + block → explicit prefix-causal mask.
                causal = self._get_causal_mask(Lprev=Lk_total - blk, blk=blk, device=device)
                if use_packing:
                    # Packed: all remaining slots are valid, mask is just the causal pattern.
                    attn_mask = causal.view(1, 1, blk, Lk_total)
                    is_causal = False
                else:
                    # Non-packed: also AND with allowed mask to hide invalid slots.
                    if self.mask_unused_memory:
                        block_ok = torch.ones((B, blk), device=device, dtype=torch.bool)
                        allowed = torch.cat([allowed_prev, block_ok], dim=1)
                    else:
                        allowed = torch.ones((B, Lk_total), device=device, dtype=torch.bool)
                    attn_mask = causal.view(1, 1, blk, Lk_total) & allowed[:, None, None, :]
                    is_causal = False

            cos_t = torch.cos(self.theta).to(device=device, dtype=dtype)
            sin_t = torch.sin(self.theta).to(device=device, dtype=dtype)
            q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)

            lam = torch.sigmoid(self.lambda_proj(x_blk)).transpose(1, 2).unsqueeze(-1)

            out_h = self._sdpa_stacked_two_stream(
                q=q, q_noise=q_noise, k=k_all, v=v_all,
                attn_mask=attn_mask, lam=lam, is_causal=is_causal,
            )
            y_blk = self._to_output(out_h)

            if return_outputs:
                outs.append(y_blk)

            if write_cache:
                self._write_block_fast(state, k_blk=k_blk, v_blk=v_blk, g_blk=g_blk, pos0=pos0)

            t += blk

        if return_outputs:
            return torch.cat(outs, dim=1), state
        return None, state

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def cache_bytes(self, *, batch_size: int, dtype: torch.dtype) -> int:
        """Return total inference state size in bytes.

        Includes KV buffers, routing norm caches, gate history, allowed
        mask, and LRU metadata.
        """
        B = int(batch_size)
        Hkv = self.num_kv_heads
        Dh = self.head_dim
        bytes_per = torch.tensor([], dtype=dtype).element_size()
        kv = 2 * B * Hkv * self.Lbuf * Dh * bytes_per
        # Routing norm caches: exact bank V-routing + summary bank LF-K routing.
        exact_norm = B * Hkv * self.Me * Dh * bytes_per if self.Me > 0 else 0
        sum_norm = B * Hkv * self.Ms * self.lf_dim * bytes_per if self.Ms > 0 else 0
        gates = B * self.window * 4  # float32 write gates
        used_masks = B * self.Lbuf   # bool allowed mask
        lru = B * self.Me * 8        # int64 LRU timestamps
        return int(kv + exact_norm + sum_norm + gates + used_masks + lru)

    def reset_metrics(self, state: CoDAGQALandmarkStatePerf2) -> None:
        """Reset all metrics counters to zero.

        No-op if metrics collection is not enabled on this state.
        """
        if state.metrics is not None:
            for k in state.metrics:
                state.metrics[k] = 0 if isinstance(state.metrics[k], int) else 0.0
