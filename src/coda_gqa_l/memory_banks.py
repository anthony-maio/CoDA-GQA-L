# memory_banks.py
# Memory bank update logic for CoDA-GQA-L (exact + summary banks).
#
# This module defines MemoryBankMixin, which is mixed into
# CoDAGQALandmarkPerf2 to keep bank logic separate from attention/SDPA.
# All methods access `self.*` attributes defined on the main module class.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .state import CoDAGQALandmarkStatePerf2


class MemoryBankMixin:
    """Memory bank update methods, mixed into CoDAGQALandmarkPerf2.

    Frequency-aware routing and EMA:
      - Exact bank: V-routing (cosine similarity on Values, RoPE-free) for
        semantic deduplication.  Full key overwrites, no EMA, no phase issue.
      - Summary bank: LF-K routing (cosine similarity on low-frequency key
        band).  Phase-Safe EMA blends only the LF band of keys; HF band stays
        zero.  Values use full EMA (no RoPE, no phase issue).  Energy
        calibration scales the LF band by sqrt(Dh / Dh_lf) to compensate for
        reduced dimensionality.

    RoPE splits key dimensions into high-frequency (position-sensitive, HF)
    and low-frequency (position-invariant, LF) bands.  EMA-blending HF keys
    from different positions causes destructive interference (Prism, 2026).
    Phase-Safe EMA avoids this by zeroing the HF band and only blending LF.

    Provides:
      - Scratch buffer management (_get_scratch)
      - KV buffer views (_view_recent, _view_exact, _view_sum)
      - Recent ring helpers (_get_recent_time_order, _set_recent_from_sequence)
      - Single-token writes (_write_one, _exact_update_one, _summary_update_one)
      - Block writes (_write_block_fast, _exact_update_block_vectorized,
        _summary_update_block_hard)
    """

    # ------------------------------------------------------------------
    # Scratch buffers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_scratch(B: int, Hkv: int, Me: int, Ms: int, Dh: int, device: torch.device, dtype: torch.dtype) -> dict:
        """Allocate a fresh set of scratch buffers."""
        return {
            'exact_sum_k': torch.zeros(B, Hkv, Me, Dh, device=device, dtype=dtype),
            'exact_sum_v': torch.zeros(B, Hkv, Me, Dh, device=device, dtype=dtype),
            'exact_upd': torch.zeros(B, Me, device=device, dtype=dtype),
            'exact_max_score': torch.empty(B, Me, device=device, dtype=torch.float32),
            'exact_max_pos': torch.empty(B, Me, device=device, dtype=torch.int64),
            'exact_pos_slot': torch.zeros(B, Me, device=device, dtype=torch.int64),
            'sum_count': torch.zeros(B, Ms, device=device, dtype=dtype),
            'sum_sum_k': torch.zeros(B, Hkv, Ms, Dh, device=device, dtype=dtype),
            'sum_sum_v': torch.zeros(B, Hkv, Ms, Dh, device=device, dtype=dtype),
        }

    def _get_scratch(self, B: int, Hkv: int, Dh: int, device: torch.device, dtype: torch.dtype) -> dict:
        """Lazily allocate and return reusable scratch tensors for bank updates.

        When ``detach_evicted=False``, returns **fresh** buffers every call.
        Reusing scratch across chunks causes autograd version conflicts:
        ``zero_()`` + ``scatter_add_()`` on graph-connected tensors increment
        the version counter, and the next chunk's ``zero_()`` invalidates the
        prior chunk's saved state for backward.
        """
        if not self.detach_evicted:
            return self._new_scratch(B, Hkv, self.Me, self.Ms, Dh, device, dtype)
        key = (B, Hkv, Dh, device, dtype)
        if key not in self._scratch:
            self._scratch[key] = self._new_scratch(B, Hkv, self.Me, self.Ms, Dh, device, dtype)
        return self._scratch[key]

    # ------------------------------------------------------------------
    # KV buffer views
    # ------------------------------------------------------------------

    def _view_recent(self, state: CoDAGQALandmarkStatePerf2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W = self.window
        return (
            state.k_buf[:, :, :W, :],
            state.v_buf[:, :, :W, :],
            state.allowed[:, :W],
        )

    def _view_exact(self, state: CoDAGQALandmarkStatePerf2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Me == 0:
            empty_k = state.k_buf[:, :, :0, :]
            empty_v = state.v_buf[:, :, :0, :]
            empty_m = state.allowed[:, :0]
            return empty_k, empty_v, empty_m
        s = self.window
        e = self.window + self.Me
        return (
            state.k_buf[:, :, s:e, :],
            state.v_buf[:, :, s:e, :],
            state.allowed[:, s:e],
        )

    def _view_sum(self, state: CoDAGQALandmarkStatePerf2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Ms == 0:
            empty_k = state.k_buf[:, :, :0, :]
            empty_v = state.v_buf[:, :, :0, :]
            empty_m = state.allowed[:, :0]
            return empty_k, empty_v, empty_m
        s = self.window + self.Me
        e = self.Lbuf
        return (
            state.k_buf[:, :, s:e, :],
            state.v_buf[:, :, s:e, :],
            state.allowed[:, s:e],
        )

    # ------------------------------------------------------------------
    # Recent ring helpers
    # ------------------------------------------------------------------

    def _get_recent_time_order(self, state: CoDAGQALandmarkStatePerf2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return recent (k_seq, v_seq, g_seq) in chronological order length Lr<=W."""
        W = self.window
        n = int(state.recent_filled)
        k_recent = state.k_buf[:, :, :W, :]
        v_recent = state.v_buf[:, :, :W, :]
        g_recent = state.g_recent
        if n == 0:
            return k_recent[:, :, :0, :], v_recent[:, :, :0, :], g_recent[:, :0]
        if n < W:
            return k_recent[:, :, :n, :], v_recent[:, :, :n, :], g_recent[:, :n]
        i = int(state.recent_idx)
        k = torch.cat([k_recent[:, :, i:, :], k_recent[:, :, :i, :]], dim=2)
        v = torch.cat([v_recent[:, :, i:, :], v_recent[:, :, :i, :]], dim=2)
        g = torch.cat([g_recent[:, i:], g_recent[:, :i]], dim=1)
        return k, v, g

    def _set_recent_from_sequence(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        g_seq: torch.Tensor,
    ) -> None:
        B, Hkv, Lr, Dh = k_seq.shape
        W = self.window
        if Lr > W:
            raise ValueError("k_seq longer than window")

        state.k_buf[:, :, :W, :].zero_()
        state.v_buf[:, :, :W, :].zero_()
        state.g_recent.zero_()
        state.allowed[:, :W].fill_(False)

        if Lr > 0:
            state.k_buf[:, :, :Lr, :] = k_seq
            state.v_buf[:, :, :Lr, :] = v_seq
            state.g_recent[:, :Lr] = g_seq.to(dtype=torch.float32)
            state.allowed[:, :Lr] = True

        state.recent_filled = int(Lr)
        state.recent_idx = 0

    # ------------------------------------------------------------------
    # Write one token (decode path)
    # ------------------------------------------------------------------

    def _write_one(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        g_new: torch.Tensor,
    ) -> None:
        B, Hkv, L, Dh = k_new.shape
        assert L == 1
        W = self.window

        if state.recent_filled < W:
            idx = int(state.recent_filled)
            state.k_buf[:, :, idx:idx+1, :] = k_new
            state.v_buf[:, :, idx:idx+1, :] = v_new
            state.g_recent[:, idx:idx+1] = g_new
            state.allowed[:, idx:idx+1] = True
            state.recent_filled += 1
        else:
            ev_idx = int(state.recent_idx)
            k_e = state.k_buf[:, :, ev_idx, :]
            v_e = state.v_buf[:, :, ev_idx, :]
            g_e = state.g_recent[:, ev_idx]
            if self.detach_evicted:
                k_e = k_e.detach()
                v_e = v_e.detach()
                g_e = g_e.detach()

            if state.metrics is not None:
                state.metrics["total_evictions"] += 1

            pos_e = torch.full((B,), int(state.pos - W), device=k_new.device, dtype=torch.int64)
            self._exact_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)
            self._summary_update_one(state, k_evict=k_e, v_evict=v_e, pos_evict=pos_e, gate=g_e)

            state.k_buf[:, :, ev_idx:ev_idx+1, :] = k_new
            state.v_buf[:, :, ev_idx:ev_idx+1, :] = v_new
            state.g_recent[:, ev_idx:ev_idx+1] = g_new
            state.recent_idx = (ev_idx + 1) % W

        state.pos += 1

    # ------------------------------------------------------------------
    # Exact bank update (single token, decode path)
    # ------------------------------------------------------------------

    def _exact_update_one(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_evict: torch.Tensor,
        v_evict: torch.Tensor,
        pos_evict: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        """Update exact bank with a single evicted token (decode path).

        Runs unconditionally without CPU-GPU sync points.
        """
        if self.Me == 0:
            return
        B = k_evict.size(0)
        device = k_evict.device

        if self.write_policy != "none":
            keep_b = gate >= self.write_gate_threshold_exact
        else:
            keep_b = torch.ones((B,), device=device, dtype=torch.bool)

        mem_k, mem_v, used = self._view_exact(state)
        Me = self.Me
        Ae = state.active_exact  # active capacity (may be < Me)

        # V-routing: cosine similarity on Values (RoPE-free) for semantic matching.
        v_n = F.normalize(v_evict, dim=-1, eps=1e-6)
        mem_n = state._exact_v_norm
        scores_h = torch.einsum("bhd,bhmd->bhm", v_n, mem_n)
        scores = scores_h.mean(dim=1)
        # Mask inactive slots beyond active capacity.
        if Ae < Me:
            scores[:, Ae:] = -1e9
        scores = scores.masked_fill(~used, -1e9)
        best_score, best_idx = scores.max(dim=-1)

        any_used = used[:, :Ae].any(dim=-1)
        novel = (~any_used) | (best_score < self.exact_novelty_threshold)
        hit = any_used & (best_score >= self.exact_match_threshold)

        # Only search for free/LRU slots within active capacity.
        free = ~used.clone()
        if Ae < Me:
            free[:, Ae:] = False
        free_exists = free.any(dim=-1)
        free_idx = free.to(torch.int64).argmax(dim=-1)

        last = state.mem_last_exact
        inf = torch.full_like(last, 2**62)
        last_masked = torch.where(used, last, inf)
        if Ae < Me:
            last_masked[:, Ae:] = 2**62  # inactive slots won't be LRU candidates
        lru_idx = last_masked.argmin(dim=-1)

        insert_idx = torch.where(free_exists, free_idx, lru_idx)
        idx = torch.where(novel, insert_idx, best_idx)

        if self.exact_refresh_on_hit:
            overwrite = novel | hit
        else:
            overwrite = novel
        overwrite = overwrite & keep_b
        touch = keep_b

        w = torch.zeros((B, Me), device=device, dtype=mem_k.dtype)
        w.scatter_(1, idx.view(B, 1), overwrite.to(dtype=mem_k.dtype).view(B, 1))
        w4 = w[:, None, :, None]

        # Run unconditionally (w4 zeros out non-overwrite cases)
        mem_k = mem_k + w4 * (k_evict.unsqueeze(2) - mem_k)
        mem_v = mem_v + w4 * (v_evict.unsqueeze(2) - mem_v)
        used = used | w.to(torch.bool)

        s = self.window
        e = self.window + self.Me
        state.k_buf[:, :, s:e, :] = mem_k
        state.v_buf[:, :, s:e, :] = mem_v
        state.allowed[:, s:e] = used

        # Update V-routing norm cache
        state._exact_v_norm = F.normalize(mem_v, dim=-1, eps=1e-6)

        # Metrics (opt-in; guarded so zero overhead when disabled)
        if state.metrics is not None:
            m = state.metrics
            if self.write_policy != "none":
                m["tokens_gated_out"] += int((~keep_b).sum().item())
            m["exact_hits"] += int(hit.sum().item())
            m["exact_inserts"] += int((novel & keep_b).sum().item())
            m["exact_overwrites"] += int(overwrite.sum().item())
            if self.Me > 0:
                m["exact_fill_ratio"] = float(used.any(dim=0).float().mean().item())

        # LRU timestamp (unconditional)
        last = state.mem_last_exact
        pos_vals = pos_evict.to(device=device, dtype=last.dtype)
        cur = last.gather(1, idx.view(B, 1)).squeeze(1)
        new = torch.where(touch, pos_vals, cur)
        last.scatter_(1, idx.view(B, 1), new.view(B, 1))
        state.mem_last_exact = last

        # Pressure-gated expansion: frustrated = novel token forced to LRU-evict
        # when all active slots are occupied.
        frustrated = int((novel & keep_b & ~free_exists).any().item())
        if frustrated > 0 and not self.training:
            self._try_expand(state, frustrated_exact=frustrated, frustrated_summary=0)

    # ------------------------------------------------------------------
    # Summary bank update (single token, decode path)
    # ------------------------------------------------------------------

    def _summary_update_one(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_evict: torch.Tensor,
        v_evict: torch.Tensor,
        pos_evict: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        """Update summary bank with a single evicted token (decode path).

        Runs unconditionally without CPU-GPU sync points.
        """
        if self.Ms == 0:
            return
        B = k_evict.size(0)
        device = k_evict.device
        Ms = self.Ms
        As = state.active_summary  # active capacity (may be < Ms)

        if self.write_policy != "none":
            keep_b = gate >= self.write_gate_threshold_summary
        else:
            keep_b = torch.ones((B,), device=device, dtype=torch.bool)

        mem_k, mem_v, used = self._view_sum(state)

        # LF-K routing: cosine similarity on low-frequency key band.
        lf = self.lf_start
        k_lf_n = F.normalize(k_evict[..., lf:], dim=-1, eps=1e-6)
        mem_n = state._sum_lf_k_norm
        scores_h = torch.einsum("bhd,bhmd->bhm", k_lf_n, mem_n)
        scores = scores_h.mean(dim=1)
        # Mask inactive slots beyond active capacity.
        if As < Ms:
            scores[:, As:] = -1e9
        idx = scores.argmax(dim=-1)

        eta_base = torch.sigmoid(self.summary_eta_logit).to(device=device, dtype=mem_k.dtype)
        eta = eta_base.view(1, self.num_kv_heads, 1, 1).expand(B, self.num_kv_heads, 1, 1)

        if self.write_policy != "none":
            g = gate.to(device=device, dtype=mem_k.dtype).view(B, 1, 1, 1)
            eta = eta * g

        if self.summary_overwrite_on_insert:
            is_new = (~used.gather(1, idx.view(B, 1)).squeeze(1))
            eta = torch.where(is_new.view(B, 1, 1, 1), torch.ones_like(eta), eta)

        eta = eta * keep_b.to(dtype=eta.dtype, device=device).view(B, 1, 1, 1)

        w = torch.zeros((B, Ms), device=device, dtype=mem_k.dtype)
        w.scatter_(1, idx.view(B, 1), 1.0)
        w4 = w[:, None, :, None]

        # Phase-Safe EMA: only blend LF key band; HF stays zero.
        delta_k = torch.zeros_like(mem_k)
        incoming_lf = k_evict[..., lf:].unsqueeze(2) * self.energy_scale
        delta_k[..., lf:] = incoming_lf - mem_k[..., lf:]
        mem_k = mem_k + eta * w4 * delta_k
        # Values: full EMA (RoPE-free, no phase issue)
        mem_v = mem_v + eta * w4 * (v_evict.unsqueeze(2) - mem_v)
        used = used | (keep_b.view(B, 1) & w.to(torch.bool))

        s = self.window + self.Me
        e = self.Lbuf
        state.k_buf[:, :, s:e, :] = mem_k
        state.v_buf[:, :, s:e, :] = mem_v
        state.allowed[:, s:e] = used

        # Update LF-K routing norm cache
        state._sum_lf_k_norm = F.normalize(mem_k[..., lf:], dim=-1, eps=1e-6)

        # Pressure-gated expansion: all active slots used and incoming data.
        all_active_used = used[:, :As].all()
        frustrated_s = int((all_active_used & keep_b.any()).item())
        if frustrated_s > 0 and not self.training:
            self._try_expand(state, frustrated_exact=0, frustrated_summary=frustrated_s)

        # Metrics (opt-in; guarded so zero overhead when disabled)
        if state.metrics is not None:
            m = state.metrics
            m["summary_updates"] += int(keep_b.sum().item())
            m["summary_frustrated"] += frustrated_s
            # Count inserts into previously-empty slots
            if self.summary_overwrite_on_insert:
                # is_new was computed above as (~used_before_write for the target slot)
                m["summary_inserts"] += int((is_new & keep_b).sum().item())
            if self.Ms > 0:
                m["summary_fill_ratio"] = float(used.any(dim=0).float().mean().item())

    # ------------------------------------------------------------------
    # Block write (cache + memory, prefill path)
    # ------------------------------------------------------------------

    def _write_block_fast(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_blk: torch.Tensor,
        v_blk: torch.Tensor,
        g_blk: torch.Tensor,
        pos0: int,
    ) -> None:
        B, Hkv, blk, Dh = k_blk.shape
        W = self.window

        k_old, v_old, g_old = self._get_recent_time_order(state)
        Lr = int(k_old.size(2))

        # When gradients flow through evicted tokens (detach_evicted=False),
        # clone state buffers so subsequent in-place writes (bank updates,
        # ring buffer zero/fill) don't invalidate tensors saved by autograd
        # during the SDPA computation above.  k_old/v_old/g_old already
        # reference the originals; writes will go to the clones only.
        if not self.detach_evicted:
            state.k_buf = state.k_buf.clone()
            state.v_buf = state.v_buf.clone()
            state.g_recent = state.g_recent.clone()

        k_cat = torch.cat([k_old, k_blk], dim=2)
        v_cat = torch.cat([v_old, v_blk], dim=2)
        g_cat = torch.cat([g_old, g_blk.to(device=k_blk.device, dtype=torch.float32)], dim=1)
        total = Lr + blk
        keep_len = min(W, total)
        evict_len = total - keep_len

        pos_base = int(pos0 - Lr)

        if state.metrics is not None and evict_len > 0:
            state.metrics["total_evictions"] += evict_len

        if evict_len > 0:
            k_e = k_cat[:, :, :evict_len, :]
            v_e = v_cat[:, :, :evict_len, :]
            g_e = g_cat[:, :evict_len]
            if self.detach_evicted:
                k_e = k_e.detach()
                v_e = v_e.detach()
                g_e = g_e.detach()

            pos_e = (torch.arange(evict_len, device=k_blk.device, dtype=torch.int64).view(1, evict_len) + pos_base)
            pos_e = pos_e.expand(B, evict_len)

            need_exact = self.Me > 0 and self.exact_candidates_per_block > 0
            need_summary = self.Ms > 0 and self.summary_candidates_per_block > 0

            if need_exact and need_summary:
                # Single topk for the larger candidate set, slice for the smaller
                Te = min(self.exact_candidates_per_block, evict_len)
                Ts = min(self.summary_candidates_per_block, evict_len)
                T_max = max(Te, Ts)
                vals_all, idxs_all = torch.topk(g_e, k=T_max, dim=1, largest=True, sorted=False)
                idx_exp_all = idxs_all.view(B, 1, T_max, 1).expand(B, Hkv, T_max, Dh)
                k_sel_all = k_e.gather(2, idx_exp_all)
                v_sel_all = v_e.gather(2, idx_exp_all)
                pos_sel_all = pos_e.gather(1, idxs_all)

                # V-routing norms for exact bank (full values, RoPE-free)
                v_sel_norm_all = F.normalize(v_sel_all, dim=-1, eps=1e-6)
                # LF-K routing norms for summary bank (LF key band only)
                lf_k_sel_norm_all = F.normalize(k_sel_all[..., self.lf_start:], dim=-1, eps=1e-6)

                # Exact bank uses first Te candidates
                self._exact_update_block_vectorized(
                    state,
                    k_sel=k_sel_all[:, :, :Te, :],
                    v_sel=v_sel_all[:, :, :Te, :],
                    pos_sel=pos_sel_all[:, :Te],
                    gate_sel=vals_all[:, :Te],
                    v_sel_norm=v_sel_norm_all[:, :, :Te, :],
                )

                # Summary bank uses first Ts candidates
                k_sel_s = k_sel_all[:, :, :Ts, :]
                v_sel_s = v_sel_all[:, :, :Ts, :]
                pos_sel_s = pos_sel_all[:, :Ts]
                vals_s = vals_all[:, :Ts]

                if self.write_policy != "none":
                    keep_tok = vals_s >= self.write_gate_threshold_summary
                    w_tok = vals_s.to(device=k_blk.device, dtype=k_sel_s.dtype)
                    w_tok = torch.where(keep_tok, w_tok, torch.zeros_like(w_tok))
                else:
                    w_tok = vals_s.to(device=k_blk.device, dtype=k_sel_s.dtype)

                self._summary_update_block_hard(
                    state, k_sel=k_sel_s, v_sel=v_sel_s, pos_sel=pos_sel_s, w_tok=w_tok,
                    lf_k_sel_norm=lf_k_sel_norm_all[:, :, :Ts, :],
                )

            elif need_exact:
                T = min(self.exact_candidates_per_block, evict_len)
                vals, idxs = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs)
                self._exact_update_block_vectorized(state, k_sel=k_sel, v_sel=v_sel, pos_sel=pos_sel, gate_sel=vals)

            elif need_summary:
                T = min(self.summary_candidates_per_block, evict_len)
                vals_s, idxs_s = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs_s.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs_s)
                if self.write_policy != "none":
                    keep_tok = vals_s >= self.write_gate_threshold_summary
                    w_tok = vals_s.to(device=k_blk.device, dtype=k_sel.dtype)
                    w_tok = torch.where(keep_tok, w_tok, torch.zeros_like(w_tok))
                else:
                    w_tok = vals_s.to(device=k_blk.device, dtype=k_sel.dtype)
                self._summary_update_block_hard(state, k_sel=k_sel, v_sel=v_sel, pos_sel=pos_sel, w_tok=w_tok)

        if keep_len > 0:
            k_new = k_cat[:, :, total - keep_len: total, :]
            v_new = v_cat[:, :, total - keep_len: total, :]
            g_new = g_cat[:, total - keep_len: total]
        else:
            k_new = k_cat[:, :, :0, :]
            v_new = v_cat[:, :, :0, :]
            g_new = g_cat[:, :0]

        self._set_recent_from_sequence(state, k_seq=k_new, v_seq=v_new, g_seq=g_new)
        state.pos = int(pos0 + blk)

    # ------------------------------------------------------------------
    # Exact bank update (block, vectorized, prefill path)
    # ------------------------------------------------------------------

    def _exact_update_block_vectorized(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_sel: torch.Tensor,
        v_sel: torch.Tensor,
        pos_sel: torch.Tensor,
        gate_sel: torch.Tensor,
        v_sel_norm: Optional[torch.Tensor] = None,
    ) -> None:
        """Update exact landmark bank with evicted token candidates.

        V-routing: cosine similarity uses Values (RoPE-free) for semantic
        matching.  Avoids CPU-GPU sync points on the hot path.  Uses scratch
        buffers and cached normalized memory values.
        """
        if self.Me == 0:
            return
        B, Hkv, T, Dh = k_sel.shape
        Me = self.Me
        Ae = state.active_exact  # active capacity (may be < Me)
        device = k_sel.device
        sc = self._get_scratch(B, Hkv, Dh, device, k_sel.dtype)

        mem_k, mem_v, used = self._view_exact(state)
        # Clone views when gradients flow through bank updates: the
        # write-back (state.k_buf[:, :, s:e, :] = ...) increments the base
        # tensor's version, invalidating autograd-saved references to these
        # views.  Cloning makes them independent tensors.
        if not self.detach_evicted:
            mem_k = mem_k.clone()
            mem_v = mem_v.clone()
        last = state.mem_last_exact

        if self.write_policy != "none":
            keep_tok = gate_sel >= self.write_gate_threshold_exact
        else:
            keep_tok = torch.ones((B, T), device=device, dtype=torch.bool)

        # V-routing: cosine similarity on Values (RoPE-free) for semantic matching.
        if v_sel_norm is None:
            v_sel_norm = F.normalize(v_sel, dim=-1, eps=1e-6)
        mem_n = state._exact_v_norm  # cached; updated on writes
        scores_h = torch.matmul(
            v_sel_norm.reshape(B * Hkv, T, Dh),
            mem_n.reshape(B * Hkv, Me, Dh).transpose(1, 2),
        ).view(B, Hkv, T, Me)
        scores = scores_h.mean(dim=1)

        # Mask inactive slots (beyond active capacity) before used-mask.
        if Ae < Me:
            inactive = torch.arange(Me, device=device).unsqueeze(0) >= Ae
            scores.masked_fill_(inactive.unsqueeze(1), -1e9)
        scores.masked_fill_(~used[:, None, :], -1e9)
        best_score, best_idx = scores.max(dim=-1)
        any_used = used.any(dim=-1)

        novel = (~any_used[:, None]) | (best_score < self.exact_novelty_threshold)
        hit = any_used[:, None] & (best_score >= self.exact_match_threshold)

        novel_keep = novel & keep_tok

        rank_key = last.clone()
        rank_key[~used] = -(2**62)
        if Ae < Me:
            rank_key[:, Ae:] = -(2**62)  # inactive slots are not LRU candidates
        lru_k = min(T, Ae)
        _, lru_slots = torch.topk(rank_key, k=lru_k, dim=1, largest=False, sorted=True)

        novel_rank = novel_keep.to(torch.int64).cumsum(dim=1) - 1
        novel_cap = novel_keep & (novel_rank < Ae)

        rank_clamped = novel_rank.clamp(min=0, max=max(lru_k - 1, 0))
        slot_assigned = lru_slots.gather(1, rank_clamped)

        idx_tok = torch.where(novel_cap, slot_assigned, best_idx)

        touch_tok = keep_tok & (~novel | novel_cap)

        if self.exact_refresh_on_hit:
            overwrite_tok = keep_tok & (novel_cap | hit)
        else:
            overwrite_tok = novel_cap

        # LRU timestamp update (scratch buffer: max_pos)
        pos_touch = torch.where(touch_tok, pos_sel, torch.full_like(pos_sel, -1))
        max_pos = sc['exact_max_pos']
        max_pos.fill_(-1)
        max_pos.scatter_reduce_(1, idx_tok, pos_touch, reduce="amax", include_self=True)
        last = torch.where(max_pos >= 0, max_pos, last)
        state.mem_last_exact = last

        # Winner-take-all per slot (scratch buffer: max_score)
        tie = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0) * 1e-6
        score_tok = gate_sel.float() + tie
        neginf = torch.tensor(-float("inf"), device=device, dtype=torch.float32)
        score_tok = torch.where(overwrite_tok, score_tok, neginf)

        max_score = sc['exact_max_score']
        max_score.fill_(-float("inf"))
        max_score.scatter_reduce_(1, idx_tok, score_tok, reduce="amax", include_self=True)
        winner = overwrite_tok & (score_tok == max_score.gather(1, idx_tok))

        # KV scatter (scratch buffers: sum_k, sum_v, upd, pos_slot)
        idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
        w4 = winner.to(dtype=mem_k.dtype, device=device).view(B, 1, T, 1)
        weighted_k = k_sel * w4
        weighted_v = v_sel * w4

        sum_k = sc['exact_sum_k']
        sum_v = sc['exact_sum_v']
        sum_k.zero_()
        sum_v.zero_()
        sum_k.scatter_add_(2, idx_exp, weighted_k)
        sum_v.scatter_add_(2, idx_exp, weighted_v)

        upd = sc['exact_upd']
        upd.zero_()
        upd.scatter_add_(1, idx_tok, winner.to(dtype=mem_k.dtype, device=device))
        slot_updated = upd > 0

        pos_vals = pos_sel * winner.to(dtype=torch.int64)
        pos_slot = sc['exact_pos_slot']
        pos_slot.zero_()
        pos_slot.scatter_add_(1, idx_tok, pos_vals)

        slot4 = slot_updated.view(B, 1, Me, 1)
        mem_k = torch.where(slot4, sum_k, mem_k)
        mem_v = torch.where(slot4, sum_v, mem_v)
        used = used | slot_updated
        last = state.mem_last_exact
        last = torch.where(slot_updated, pos_slot, last)
        state.mem_last_exact = last

        s = self.window
        e = self.window + Me
        state.k_buf[:, :, s:e, :] = mem_k
        state.v_buf[:, :, s:e, :] = mem_v
        state.allowed[:, s:e] = used

        # Update V-routing norm cache for overwritten slots
        state._exact_v_norm = F.normalize(mem_v, dim=-1, eps=1e-6)

        # Pressure-gated expansion: count frustrated tokens (novel but capped).
        frustrated = int((novel_keep & ~novel_cap).sum().item())
        if frustrated > 0 and not self.training:
            self._try_expand(state, frustrated_exact=frustrated, frustrated_summary=0)

        # Metrics (opt-in; guarded so zero overhead when disabled)
        if state.metrics is not None:
            m = state.metrics
            m["exact_inserts"] += int(novel_cap.sum().item())
            m["exact_hits"] += int((hit & keep_tok).sum().item())
            m["exact_overwrites"] += int(overwrite_tok.sum().item())
            m["exact_frustrated"] += frustrated
            if self.write_policy != "none":
                m["tokens_gated_out"] += int((~keep_tok).sum().item())
            if self.Me > 0:
                m["exact_fill_ratio"] = float(used.any(dim=0).float().mean().item())

    # ------------------------------------------------------------------
    # Summary bank update (block, prefill path)
    # ------------------------------------------------------------------

    def _summary_update_block_hard(
        self,
        state: CoDAGQALandmarkStatePerf2,
        *,
        k_sel: torch.Tensor,
        v_sel: torch.Tensor,
        pos_sel: torch.Tensor,
        w_tok: torch.Tensor,
        lf_k_sel_norm: Optional[torch.Tensor] = None,
    ) -> None:
        """Update summary landmark bank with evicted token candidates.

        LF-K routing: cosine similarity on low-frequency key band for semantic
        matching.  Phase-Safe EMA blends only LF keys; HF stays zero.
        Uses scratch buffers and cached normalized memory values.
        Zero-weight tokens produce zero scatter contributions.
        """
        if self.Ms == 0:
            return
        B, Hkv, T, Dh = k_sel.shape
        Ms = self.Ms
        As = state.active_summary  # active capacity (may be < Ms)
        device = k_sel.device
        sc = self._get_scratch(B, Hkv, Dh, device, k_sel.dtype)

        mem_k, mem_v, used = self._view_sum(state)
        if not self.detach_evicted:
            mem_k = mem_k.clone()
            mem_v = mem_v.clone()

        # LF-K routing: cosine similarity on low-frequency key band.
        lf = self.lf_start
        Dh_lf = Dh - lf
        if lf_k_sel_norm is None:
            lf_k_sel_norm = F.normalize(k_sel[..., lf:], dim=-1, eps=1e-6)
        mem_n = state._sum_lf_k_norm  # cached; updated on writes
        scores_h = torch.matmul(
            lf_k_sel_norm.reshape(B * Hkv, T, Dh_lf),
            mem_n.reshape(B * Hkv, Ms, Dh_lf).transpose(1, 2),
        ).view(B, Hkv, T, Ms)
        scores = scores_h.mean(dim=1)
        # Mask inactive slots beyond active capacity.
        if As < Ms:
            inactive_s = torch.arange(Ms, device=device).unsqueeze(0) >= As
            scores.masked_fill_(inactive_s.unsqueeze(1), -1e9)
        idx_tok = scores.argmax(dim=-1)

        w = w_tok.to(device=device, dtype=mem_k.dtype)

        # Scratch buffers instead of torch.zeros
        count = sc['sum_count']
        count.zero_()
        count.scatter_add_(1, idx_tok, w)

        idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
        w4 = w.view(B, 1, T, 1)
        src_k = k_sel * w4
        src_v = v_sel * w4
        sum_k = sc['sum_sum_k']
        sum_v = sc['sum_sum_v']
        sum_k.zero_()
        sum_v.zero_()
        sum_k.scatter_add_(2, idx_exp, src_k)
        sum_v.scatter_add_(2, idx_exp, src_v)

        eta_base = torch.sigmoid(self.summary_eta_logit).to(device=device, dtype=mem_k.dtype)
        eta = eta_base.view(1, Hkv, 1, 1)

        count4 = count.view(B, 1, Ms, 1)
        active = count4 > 0
        avg_k = sum_k / count4.clamp_min(1e-6)
        avg_v = sum_v / count4.clamp_min(1e-6)

        new_slots = (~used).view(B, 1, Ms, 1) & active
        if self.summary_overwrite_on_insert:
            eta_eff = torch.where(new_slots, torch.ones_like(eta), eta)
        else:
            eta_eff = eta

        # Phase-Safe EMA: only blend LF key band; HF stays zero.
        delta_k = torch.zeros_like(avg_k)
        delta_k[..., lf:] = avg_k[..., lf:] * self.energy_scale - mem_k[..., lf:]
        mem_k = torch.where(active, mem_k + eta_eff * delta_k, mem_k)
        # Values: full EMA (RoPE-free, no phase issue)
        mem_v = torch.where(active, mem_v + eta_eff * (avg_v - mem_v), mem_v)
        used = used | (count > 0)

        s = self.window + self.Me
        e = self.Lbuf
        state.k_buf[:, :, s:e, :] = mem_k
        state.v_buf[:, :, s:e, :] = mem_v
        state.allowed[:, s:e] = used

        # Update LF-K routing norm cache for EMA-blended slots
        state._sum_lf_k_norm = F.normalize(mem_k[..., lf:], dim=-1, eps=1e-6)

        # Pressure-gated expansion: frustrated when all active slots are used
        # and new tokens are arriving (being merged rather than creating new prototypes).
        all_active_used = used[:, :As].all()
        has_incoming = (count[:, :As] > 0).any()
        frustrated_s = int(all_active_used.item() and has_incoming.item())
        if frustrated_s > 0 and not self.training:
            self._try_expand(state, frustrated_exact=0, frustrated_summary=frustrated_s)

        # Metrics (opt-in; guarded so zero overhead when disabled)
        if state.metrics is not None:
            m = state.metrics
            m["summary_updates"] += int((count > 0).sum().item())
            m["summary_frustrated"] += frustrated_s
            if self.summary_overwrite_on_insert:
                m["summary_inserts"] += int(new_slots.sum().item())
            if self.Ms > 0:
                m["summary_fill_ratio"] = float(used.any(dim=0).float().mean().item())
