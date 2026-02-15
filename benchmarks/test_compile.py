#!/usr/bin/env python3
"""Test torch.compile on memory bank methods and measure speedup.

Compiles _write_block_fast (which calls exact + summary updates)
and measures wall time improvement.

Usage:
    python benchmarks/test_compile.py
"""

from __future__ import annotations

import os
import time

import torch

from coda_gqa_l import CoDAGQALandmarkPerf2


def sync():
    torch.cuda.synchronize()


def bench_prefill(model, x, device, dtype, B, block_size, iters=5, warmup=2):
    """Returns average seconds per prefill call."""
    for _ in range(warmup):
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        model.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=False)
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        model.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=False)
    sync()
    return (time.perf_counter() - t0) / iters


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = int(os.environ.get("BATCH", "1"))
    D = int(os.environ.get("DIM", "512"))
    H = int(os.environ.get("HEADS", "8"))
    Hkv = int(os.environ.get("KV_HEADS", "2"))
    W = int(os.environ.get("W", "128"))
    Me = int(os.environ.get("ME", "32"))
    Ms = int(os.environ.get("MS", "32"))
    block_size = int(os.environ.get("BLOCK", "256"))
    L = int(os.environ.get("SEQ_LEN", "4096"))

    print(f"device={device} dtype={dtype}")
    print(f"B={B} D={D} H={H} Hkv={Hkv} W={W} Me={Me} Ms={Ms}")
    print(f"L={L} block_size={block_size}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print()

    # --- Baseline (no compile) ---
    model = CoDAGQALandmarkPerf2(
        embed_dim=D, num_heads=H, num_kv_heads=Hkv,
        window=W, num_landmarks_exact=Me, num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(B, L, D, device=device, dtype=dtype)

    dt_baseline = bench_prefill(model, x, device, dtype, B, block_size, iters=5, warmup=3)
    print(f"Baseline (eager):    {dt_baseline*1000:.2f} ms")

    # --- Compiled: _write_block_fast ---
    model2 = CoDAGQALandmarkPerf2(
        embed_dim=D, num_heads=H, num_kv_heads=Hkv,
        window=W, num_landmarks_exact=Me, num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    # Compile the memory update method
    original_write = model2._write_block_fast
    model2._write_block_fast = torch.compile(original_write, mode="reduce-overhead")

    print("Compiling _write_block_fast (first run will be slow)...")
    dt_compiled = bench_prefill(model2, x, device, dtype, B, block_size, iters=5, warmup=3)
    print(f"Compiled (write):    {dt_compiled*1000:.2f} ms")
    speedup = dt_baseline / dt_compiled if dt_compiled > 0 else float("inf")
    print(f"Speedup:             {speedup:.2f}x")

    print()

    # --- Compiled: full prefill_chunked ---
    model3 = CoDAGQALandmarkPerf2(
        embed_dim=D, num_heads=H, num_kv_heads=Hkv,
        window=W, num_landmarks_exact=Me, num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    model3.load_state_dict(model.state_dict())
    model3.eval()

    # Try compiling the full prefill_chunked
    try:
        original_prefill = model3.prefill_chunked
        model3.prefill_chunked = torch.compile(original_prefill, mode="reduce-overhead")

        print("Compiling full prefill_chunked...")
        dt_full = bench_prefill(model3, x, device, dtype, B, block_size, iters=5, warmup=3)
        print(f"Compiled (full):     {dt_full*1000:.2f} ms")
        speedup2 = dt_baseline / dt_full if dt_full > 0 else float("inf")
        print(f"Speedup:             {speedup2:.2f}x")
    except Exception as e:
        print(f"Full compile failed: {e}")


if __name__ == "__main__":
    main()
