#!/usr/bin/env python3
"""Fine-grained CUDA-event profiler for prefill_chunked internals.

Instruments one prefill pass to break down wall time into:
  1. Projections (q/k/v proj + write gate)
  2. RoPE application
  3. KV concat + mask construction
  4. SDPA (two-stream stacked)
  5. Memory: exact bank update
  6. Memory: summary bank update
  7. Memory: ring reorder (recent buffer update)
  8. Output projection

Usage:
    python benchmarks/profile_prefill.py
    DIM=1024 HEADS=32 KV_HEADS=4 python benchmarks/profile_prefill.py
"""

from __future__ import annotations

import os

import torch

from coda_gqa_l import CoDAGQALandmarkPerf2
from coda_gqa_l.primitives import _apply_pairwise_rotation, apply_rope


class CudaTimer:
    """Accumulates CUDA-event timings per named region."""

    def __init__(self):
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self, name: str) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.setdefault(name, [])
        self.events[name].append((ev, None))
        return ev

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


@torch.no_grad()
def profiled_prefill(model: CoDAGQALandmarkPerf2, x_seq: torch.Tensor,
                     state, block_size: int, timer: CudaTimer):
    """Reimplements prefill_chunked with CUDA-event instrumentation."""
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

        # --- 1. Projections ---
        timer.start("1_projections")
        q = model._project_q(x_blk)
        k_raw, v_blk = model._project_kv_raw(x_blk)
        g_blk = model._write_gate(x_blk)
        timer.stop("1_projections")

        # --- 2. RoPE ---
        timer.start("2_rope")
        cos_q, sin_q = model.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
        q = apply_rope(q, cos_q, sin_q)
        cos_k, sin_k = model.rope(seq_len=blk, offset=pos0, device=device, dtype=dtype)
        k_blk = apply_rope(k_raw, cos_k, sin_k)
        timer.stop("2_rope")

        # --- 3. KV concat + mask ---
        timer.start("3_concat_mask")
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
        timer.stop("3_concat_mask")

        # --- 4. SDPA ---
        timer.start("4_sdpa")
        out_h = model._sdpa_stacked_two_stream(
            q=q, q_noise=q_noise, k=k_all, v=v_all, attn_mask=attn_mask, lam=lam
        )
        y_blk = model._to_output(out_h)
        timer.stop("4_sdpa")

        # --- 5-7. Memory updates (only when write_cache=True and eviction happens) ---
        timer.start("5_mem_ring_setup")
        k_old, v_old, g_old = model._get_recent_time_order(state)
        Lr = int(k_old.size(2))

        k_cat = torch.cat([k_old, k_blk], dim=2)
        v_cat = torch.cat([v_old, v_blk], dim=2)
        g_cat = torch.cat([g_old, g_blk.to(device=device, dtype=torch.float32)], dim=1)
        total = Lr + blk
        keep_len = min(W, total)
        evict_len = total - keep_len
        timer.stop("5_mem_ring_setup")

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

            # --- 6. Exact bank ---
            timer.start("6_exact_bank")
            if model.Me > 0 and model.exact_candidates_per_block > 0:
                T = min(model.exact_candidates_per_block, evict_len)
                vals, idxs = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs)
                model._exact_update_block_vectorized(state, k_sel=k_sel, v_sel=v_sel, pos_sel=pos_sel, gate_sel=vals)
            timer.stop("6_exact_bank")

            # --- 7. Summary bank ---
            timer.start("7_summary_bank")
            if model.Ms > 0 and model.summary_candidates_per_block > 0:
                T = min(model.summary_candidates_per_block, evict_len)
                vals_s, idxs_s = torch.topk(g_e, k=T, dim=1, largest=True, sorted=False)
                idx_exp = idxs_s.view(B, 1, T, 1).expand(B, Hkv, T, Dh)
                k_sel = k_e.gather(2, idx_exp)
                v_sel = v_e.gather(2, idx_exp)
                pos_sel = pos_e.gather(1, idxs_s)

                if model.write_policy != "none":
                    keep_tok = vals_s >= model.write_gate_threshold_summary
                    w_tok = vals_s.to(device=device, dtype=k_sel.dtype)
                    w_tok = torch.where(keep_tok, w_tok, torch.zeros_like(w_tok))
                else:
                    w_tok = vals_s.to(device=device, dtype=k_sel.dtype)

                model._summary_update_block_hard(state, k_sel=k_sel, v_sel=v_sel, pos_sel=pos_sel, w_tok=w_tok)
            timer.stop("7_summary_bank")

        # --- 8. Ring reorder ---
        timer.start("8_ring_reorder")
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
        timer.stop("8_ring_reorder")

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
        profiled_prefill(model, x, state, block_size, timer)
    torch.cuda.synchronize()

    # Timed runs
    all_results: list[dict[str, float]] = []
    for _ in range(n_runs):
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        timer = CudaTimer()
        profiled_prefill(model, x, state, block_size, timer)
        all_results.append(timer.report())

    # Aggregate
    keys = list(all_results[0].keys())
    keys.sort()

    print(f"{'Phase':<25s}  {'Avg ms':>10s}  {'Min ms':>10s}  {'% of total':>10s}")
    print("-" * 60)

    avg_totals = {}
    grand_total = 0.0
    for k in keys:
        vals = [r[k] for r in all_results]
        avg = sum(vals) / len(vals)
        mn = min(vals)
        avg_totals[k] = avg
        grand_total += avg

    for k in keys:
        avg = avg_totals[k]
        vals = [r[k] for r in all_results]
        mn = min(vals)
        pct = avg / grand_total * 100 if grand_total > 0 else 0
        print(f"{k:<25s}  {avg:>10.3f}  {mn:>10.3f}  {pct:>9.1f}%")

    print("-" * 60)
    print(f"{'TOTAL':<25s}  {grand_total:>10.3f}")


if __name__ == "__main__":
    main()
