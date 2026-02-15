#!/usr/bin/env python3
"""Sub-phase profiler for memory bank update internals.

Breaks down the two dominant hotspots (exact bank: ~40%, summary bank: ~21%)
into their constituent operations to identify exactly which ops to optimize.

Usage:
    python benchmarks/profile_memory_banks.py
    DIM=1024 HEADS=32 KV_HEADS=4 python benchmarks/profile_memory_banks.py
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from coda_gqa_l import CoDAGQALandmarkPerf2


class CudaTimer:
    """Accumulates CUDA-event timings per named region."""

    def __init__(self):
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.setdefault(name, [])
        self.events[name].append((ev, None))

    def stop(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events[name][-1] = (self.events[name][-1][0], ev)

    def report(self) -> dict[str, float]:
        torch.cuda.synchronize()
        results = {}
        for name, pairs in self.events.items():
            total = sum(s.elapsed_time(e) for s, e in pairs if e is not None)
            results[name] = total
        return results


def profiled_exact_update(
    model: CoDAGQALandmarkPerf2,
    state,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    pos_sel: torch.Tensor,
    gate_sel: torch.Tensor,
    timer: CudaTimer,
) -> None:
    """Reimplements _exact_update_block_vectorized with sub-phase timing."""
    if model.Me == 0:
        return
    B, Hkv, T, Dh = k_sel.shape
    Me = model.Me
    device = k_sel.device

    # --- E1: View + gate threshold ---
    timer.start("E1_view_gate")
    mem_k, mem_v, used = model._view_exact(state)
    last = state.mem_last_exact
    if model.write_policy != "none":
        keep_tok = gate_sel >= model.write_gate_threshold_exact
    else:
        keep_tok = torch.ones((B, T), device=device, dtype=torch.bool)
    if not bool(keep_tok.any()):
        timer.stop("E1_view_gate")
        return
    timer.stop("E1_view_gate")

    # --- E2: Normalize + cosine similarity ---
    timer.start("E2_cosine_sim")
    k_n = F.normalize(k_sel, dim=-1, eps=1e-6)
    mem_n = F.normalize(mem_k, dim=-1, eps=1e-6)
    scores_h = torch.einsum("bhtd,bhmd->bhtm", k_n, mem_n)
    scores = scores_h.mean(dim=1)
    timer.stop("E2_cosine_sim")

    # --- E3: Score masking + novelty/hit classification ---
    timer.start("E3_novelty_class")
    scores = scores.masked_fill(~used[:, None, :], -1e9)
    best_score, best_idx = scores.max(dim=-1)
    any_used = used.any(dim=-1)
    novel = (~any_used[:, None]) | (best_score < model.exact_novelty_threshold)
    hit = any_used[:, None] & (best_score >= model.exact_match_threshold)
    novel_keep = novel & keep_tok
    timer.stop("E3_novelty_class")

    # --- E4: LRU slot assignment ---
    timer.start("E4_lru_assign")
    very_small = torch.full_like(last, -2**62)
    rank_key = torch.where(used, last, very_small)
    slot_order = torch.argsort(rank_key, dim=1, descending=False)
    novel_rank = novel_keep.to(torch.int64).cumsum(dim=1) - 1
    novel_cap = novel_keep & (novel_rank < Me)
    rank_clamped = novel_rank.clamp(min=0, max=max(Me - 1, 0))
    slot_assigned = slot_order.gather(1, rank_clamped)
    idx_tok = torch.where(novel_cap, slot_assigned, best_idx)
    touch_tok = keep_tok & (~novel | novel_cap)
    if model.exact_refresh_on_hit:
        overwrite_tok = keep_tok & (novel_cap | hit)
    else:
        overwrite_tok = novel_cap
    timer.stop("E4_lru_assign")

    # --- E5: Touch scatter (LRU timestamp) ---
    timer.start("E5_touch_scatter")
    if bool(touch_tok.any()):
        pos_touch = torch.where(touch_tok, pos_sel, torch.full_like(pos_sel, -1))
        max_pos = torch.full((B, Me), -1, device=device, dtype=torch.int64)
        max_pos.scatter_reduce_(1, idx_tok, pos_touch, reduce="amax", include_self=True)
        last = torch.where(max_pos >= 0, max_pos, last)
        state.mem_last_exact = last
    timer.stop("E5_touch_scatter")

    if not bool(overwrite_tok.any()):
        return

    # --- E6: Winner-take-all scatter ---
    timer.start("E6_winner_scatter")
    tie = torch.arange(T, device=device, dtype=torch.float32).view(1, T) * 1e-6
    score_tok = gate_sel.to(device=device, dtype=torch.float32) + tie
    score_tok = torch.where(overwrite_tok, score_tok, torch.full_like(score_tok, -float("inf")))
    max_score = torch.full((B, Me), -float("inf"), device=device, dtype=torch.float32)
    max_score.scatter_reduce_(1, idx_tok, score_tok, reduce="amax", include_self=True)
    winner = overwrite_tok & (score_tok == max_score.gather(1, idx_tok))
    if not bool(winner.any()):
        timer.stop("E6_winner_scatter")
        return
    timer.stop("E6_winner_scatter")

    # --- E7: KV scatter_add + writeback ---
    timer.start("E7_kv_writeback")
    sum_k = torch.zeros((B, Hkv, Me, Dh), device=device, dtype=mem_k.dtype)
    sum_v = torch.zeros((B, Hkv, Me, Dh), device=device, dtype=mem_v.dtype)
    idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
    w4 = winner.to(dtype=mem_k.dtype, device=device).view(B, 1, T, 1)
    sum_k.scatter_add_(2, idx_exp, k_sel * w4)
    sum_v.scatter_add_(2, idx_exp, v_sel * w4)
    upd = torch.zeros((B, Me), device=device, dtype=mem_k.dtype)
    upd.scatter_add_(1, idx_tok, winner.to(dtype=mem_k.dtype, device=device))
    slot_updated = upd > 0
    pos_vals = pos_sel * winner.to(dtype=torch.int64)
    pos_slot = torch.zeros((B, Me), device=device, dtype=torch.int64)
    pos_slot.scatter_add_(1, idx_tok, pos_vals)
    slot4 = slot_updated.view(B, 1, Me, 1)
    mem_k = torch.where(slot4, sum_k, mem_k)
    mem_v = torch.where(slot4, sum_v, mem_v)
    used = used | slot_updated
    last = state.mem_last_exact
    last = torch.where(slot_updated, pos_slot, last)
    state.mem_last_exact = last
    s = model.window
    e = model.window + Me
    state.k_buf[:, :, s:e, :] = mem_k
    state.v_buf[:, :, s:e, :] = mem_v
    state.allowed[:, s:e] = used
    timer.stop("E7_kv_writeback")


def profiled_summary_update(
    model: CoDAGQALandmarkPerf2,
    state,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    pos_sel: torch.Tensor,
    w_tok: torch.Tensor,
    timer: CudaTimer,
) -> None:
    """Reimplements _summary_update_block_hard with sub-phase timing."""
    if model.Ms == 0:
        return
    B, Hkv, T, Dh = k_sel.shape
    Ms = model.Ms
    device = k_sel.device

    if not bool((w_tok > 0).any()):
        return

    # --- S1: View + normalize + cosine sim ---
    timer.start("S1_cosine_sim")
    mem_k, mem_v, used = model._view_sum(state)
    k_n = F.normalize(k_sel, dim=-1, eps=1e-6)
    mem_n = F.normalize(mem_k, dim=-1, eps=1e-6)
    scores_h = torch.einsum("bhtd,bhmd->bhtm", k_n, mem_n)
    scores = scores_h.mean(dim=1)
    idx_tok = scores.argmax(dim=-1)
    timer.stop("S1_cosine_sim")

    # --- S2: Weighted scatter_add ---
    timer.start("S2_scatter_add")
    w = w_tok.to(device=device, dtype=mem_k.dtype)
    count = torch.zeros((B, Ms), device=device, dtype=mem_k.dtype)
    count.scatter_add_(1, idx_tok, w)
    sum_k = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=mem_k.dtype)
    sum_v = torch.zeros((B, Hkv, Ms, Dh), device=device, dtype=mem_v.dtype)
    idx_exp = idx_tok.unsqueeze(1).unsqueeze(-1).expand(B, Hkv, T, Dh)
    src_k = k_sel * w.view(B, 1, T, 1)
    src_v = v_sel * w.view(B, 1, T, 1)
    sum_k.scatter_add_(2, idx_exp, src_k)
    sum_v.scatter_add_(2, idx_exp, src_v)
    timer.stop("S2_scatter_add")

    # --- S3: EMA blend + writeback ---
    timer.start("S3_ema_writeback")
    eta_base = torch.sigmoid(model.summary_eta_logit).to(device=device, dtype=mem_k.dtype)
    eta = eta_base.view(1, Hkv, 1, 1)
    count4 = count.view(B, 1, Ms, 1)
    active = count4 > 0
    avg_k = sum_k / count4.clamp_min(1e-6)
    avg_v = sum_v / count4.clamp_min(1e-6)
    new_slots = (~used).view(B, 1, Ms, 1) & active
    if model.summary_overwrite_on_insert:
        eta_eff = torch.where(new_slots, torch.ones_like(eta), eta)
    else:
        eta_eff = eta
    mem_k = torch.where(active, mem_k + eta_eff * (avg_k - mem_k), mem_k)
    mem_v = torch.where(active, mem_v + eta_eff * (avg_v - mem_v), mem_v)
    used = used | (count > 0)
    s = model.window + model.Me
    e = model.Lbuf
    state.k_buf[:, :, s:e, :] = mem_k
    state.v_buf[:, :, s:e, :] = mem_v
    state.allowed[:, s:e] = used
    timer.stop("S3_ema_writeback")


@torch.no_grad()
def profiled_prefill_with_bank_detail(model, x_seq, state, block_size, timer):
    """Runs prefill with sub-phase timing on memory bank ops."""
    from coda_gqa_l.primitives import _apply_pairwise_rotation, apply_rope

    B, L, _ = x_seq.shape
    device, dtype = x_seq.device, x_seq.dtype
    Lprev_fixed = model.Lbuf
    W = model.window
    Hkv = model.num_kv_heads
    Dh = model.head_dim

    t = 0
    while t < L:
        blk = min(block_size, L - t)
        x_blk = x_seq[:, t:t+blk, :]
        pos0 = int(state.pos)

        k_prev = state.k_buf
        v_prev = state.v_buf
        allowed_prev = state.allowed

        # Projections
        q = model._project_q(x_blk)
        k_raw, v_blk = model._project_kv_raw(x_blk)
        g_blk = model._write_gate(x_blk)

        # RoPE
        cos_q, sin_q = model.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
        q = apply_rope(q, cos_q, sin_q)
        cos_k, sin_k = model.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
        k_blk = apply_rope(k_raw, cos_k, sin_k)

        # Concat + mask
        k_all = torch.cat([k_prev, k_blk], dim=2)
        v_all = torch.cat([v_prev, v_blk], dim=2)
        if model.mask_unused_memory:
            block_ok = torch.ones((B, blk), device=device, dtype=torch.bool)
            allowed = torch.cat([allowed_prev, block_ok], dim=1)
        else:
            allowed = torch.ones((B, Lprev_fixed + blk), device=device, dtype=torch.bool)
        causal = model._get_causal_mask(Lprev=Lprev_fixed, blk=blk, device=device)
        attn_mask = causal.view(1, 1, blk, Lprev_fixed + blk) & allowed[:, None, None, :]
        cos_t = torch.cos(model.theta).to(device=device, dtype=dtype)
        sin_t = torch.sin(model.theta).to(device=device, dtype=dtype)
        q_noise = _apply_pairwise_rotation(q, cos_t, sin_t)
        lam = torch.sigmoid(model.lambda_proj(x_blk)).transpose(1, 2).unsqueeze(-1)

        # SDPA
        out_h = model._sdpa_stacked_two_stream(
            q=q, q_noise=q_noise, k=k_all, v=v_all, attn_mask=attn_mask, lam=lam
        )
        y_blk = model._to_output(out_h)

        # Memory updates
        k_old, v_old, g_old = model._get_recent_time_order(state)
        Lr = int(k_old.size(2))
        k_cat = torch.cat([k_old, k_blk], dim=2)
        v_cat = torch.cat([v_old, v_blk], dim=2)
        g_cat = torch.cat([g_old, g_blk.to(device=device, dtype=torch.float32)], dim=1)
        total = Lr + blk
        keep_len = min(W, total)
        evict_len = total - keep_len

        if evict_len > 0:
            k_e = k_cat[:, :, :evict_len, :]
            v_e = v_cat[:, :, :evict_len, :]
            g_e = g_cat[:, :evict_len]
            if model.detach_evicted:
                k_e = k_e.detach()
                v_e = v_e.detach()
                g_e = g_e.detach()
            pos_e = (torch.arange(evict_len, device=device, dtype=torch.int64).view(1, evict_len)
                     + int(pos0 - Lr)).expand(B, evict_len)

            # Exact bank (sub-phased)
            if model.Me > 0 and model.exact_candidates_per_block > 0:
                T = min(model.exact_candidates_per_block, evict_len)
                vals, idxs = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs)
                profiled_exact_update(model, state, k_sel, v_sel, pos_sel, vals, timer)

            # Summary bank (sub-phased)
            if model.Ms > 0 and model.summary_candidates_per_block > 0:
                T = min(model.summary_candidates_per_block, evict_len)
                vals_s, idxs_s = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs_s.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs_s)
                if model.write_policy != "none":
                    keep_tok = vals_s >= model.write_gate_threshold_summary
                    w_tok_val = vals_s.to(device=device, dtype=k_sel.dtype)
                    w_tok_val = torch.where(keep_tok, w_tok_val, torch.zeros_like(w_tok_val))
                else:
                    w_tok_val = vals_s.to(device=device, dtype=k_sel.dtype)
                profiled_summary_update(model, state, k_sel, v_sel, pos_sel, w_tok_val, timer)

        # Ring reorder
        if keep_len > 0:
            k_new = k_cat[:, :, total - keep_len: total, :]
            v_new = v_cat[:, :, total - keep_len: total, :]
            g_new = g_cat[:, total - keep_len: total]
        else:
            k_new = k_cat[:, :, :0, :]
            v_new = v_cat[:, :, :0, :]
            g_new = g_cat[:, :0]
        model._set_recent_from_sequence(state, k_seq=k_new, v_seq=v_new, g_seq=g_new)
        state.pos = int(pos0 + blk)

        t += blk

    return state


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    B = int(os.environ.get("BATCH", "1"))
    D = int(os.environ.get("DIM", "512"))
    H = int(os.environ.get("HEADS", "8"))
    Hkv = int(os.environ.get("KV_HEADS", "2"))
    W = int(os.environ.get("W", "128"))
    Me = int(os.environ.get("ME", "32"))
    Ms = int(os.environ.get("MS", "32"))
    block_size = int(os.environ.get("BLOCK", "256"))
    L = int(os.environ.get("SEQ_LEN", "4096"))
    n_runs = int(os.environ.get("RUNS", "5"))

    model = CoDAGQALandmarkPerf2(
        embed_dim=D, num_heads=H, num_kv_heads=Hkv,
        window=W, num_landmarks_exact=Me, num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(B, L, D, device=device, dtype=dtype)

    print(f"device={device} dtype={dtype}")
    print(f"B={B} D={D} H={H} Hkv={Hkv} W={W} Me={Me} Ms={Ms}")
    print(f"L={L} block_size={block_size} n_runs={n_runs}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print()

    # Warmup
    for _ in range(2):
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        timer = CudaTimer()
        profiled_prefill_with_bank_detail(model, x, state, block_size, timer)
    torch.cuda.synchronize()

    # Timed runs
    all_results: list[dict[str, float]] = []
    for _ in range(n_runs):
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        timer = CudaTimer()
        profiled_prefill_with_bank_detail(model, x, state, block_size, timer)
        all_results.append(timer.report())

    # Aggregate
    keys = list(all_results[0].keys())
    keys.sort()

    print(f"{'Sub-Phase':<25s}  {'Avg ms':>10s}  {'Min ms':>10s}  {'% of total':>10s}")
    print("-" * 60)

    avg_totals = {}
    grand_total = 0.0
    for k in keys:
        vals = [r[k] for r in all_results]
        avg = sum(vals) / len(vals)
        avg_totals[k] = avg
        grand_total += avg

    # Group by prefix
    groups = {}
    for k in keys:
        prefix = k.split("_")[0]
        groups.setdefault(prefix, []).append(k)

    for prefix in sorted(groups.keys()):
        group_total = sum(avg_totals[k] for k in groups[prefix])
        for k in groups[prefix]:
            avg = avg_totals[k]
            vals = [r[k] for r in all_results]
            mn = min(vals)
            pct = avg / grand_total * 100 if grand_total > 0 else 0
            print(f"  {k:<23s}  {avg:>10.3f}  {mn:>10.3f}  {pct:>9.1f}%")
        pct_group = group_total / grand_total * 100 if grand_total > 0 else 0
        print(f"  {'':23s}  {'':>10s}  {'':>10s}  [{pct_group:.1f}%]")
        print()

    print("-" * 60)
    print(f"{'TOTAL':<25s}  {grand_total:>10.3f}")


if __name__ == "__main__":
    main()
