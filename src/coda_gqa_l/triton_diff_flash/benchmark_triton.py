#!/usr/bin/env python3
"""Benchmark fused differential FlashAttention Triton kernel.

Compares:
  1. Reference (two PyTorch SDPA calls + epilogue)
  2. Triton fused kernel (single pass)

Reports correctness, throughput (median ms), and bandwidth utilization.

Usage:
    python kernels/triton_diff_flash/benchmark_triton.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from coda_gqa_l.triton_diff_flash import diff_flash_attn, is_available, diagnose


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def reference_diff_attn(q, q_noise, k, v, lam, rms_weight=None, eps=1e-6,
                        is_causal=False):
    B, H, Lq, Dh = q.shape
    _, Hkv, Lk, _ = k.shape
    groups = H // Hkv
    if groups > 1:
        k_exp = k[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
        v_exp = v[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
    else:
        k_exp, v_exp = k, v

    out_sig = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
    out_noise = F.scaled_dot_product_attention(q_noise, k_exp, v_exp, is_causal=is_causal)
    out = out_sig - lam * out_noise
    if rms_weight is not None:
        var = out.pow(2).mean(dim=-1, keepdim=True)
        out = out * torch.rsqrt(var + eps) * rms_weight
    return out


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "decode_7B":           dict(B=1, H=32, Hkv=32, Lq=1,   Lk=384, Dh=128, desc="7B decode"),
    "decode_70B":          dict(B=1, H=64, Hkv=8,  Lq=1,   Lk=384, Dh=128, desc="70B decode (GQA 8:1)"),
    "decode_eve":          dict(B=1, H=8,  Hkv=8,  Lq=1,   Lk=384, Dh=64,  desc="Eve-2 decode"),
    "prefill_7B":          dict(B=1, H=32, Hkv=32, Lq=256, Lk=640, Dh=128, desc="7B prefill blk=256"),
    "prefill_small_block": dict(B=1, H=32, Hkv=32, Lq=32,  Lk=416, Dh=128, desc="7B prefill blk=32"),
    "batched_decode":      dict(B=4, H=32, Hkv=32, Lq=1,   Lk=384, Dh=128, desc="7B batched B=4"),
}


def make_inputs(cfg, device, dtype):
    B, H, Hkv = cfg["B"], cfg["H"], cfg["Hkv"]
    Lq, Lk, Dh = cfg["Lq"], cfg["Lk"], cfg["Dh"]
    q = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    q_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, Lk, Dh, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, Lk, Dh, device=device, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
    rms_w = torch.ones(Dh, device=device, dtype=dtype)
    return q, q_noise, k, v, lam, rms_w


def bytes_moved(cfg, dtype_bytes=2):
    """Estimate bytes read from HBM (lower bound)."""
    B, H, Hkv = cfg["B"], cfg["H"], cfg["Hkv"]
    Lq, Lk, Dh = cfg["Lq"], cfg["Lk"], cfg["Dh"]
    # Reads: Q + Q_noise + K + V + lam + rms_weight + output write
    q_bytes = 2 * B * H * Lq * Dh * dtype_bytes  # Q and Q_noise
    kv_bytes = B * Hkv * Lk * Dh * dtype_bytes * 2  # K + V (read once)
    lam_bytes = B * H * Lq * dtype_bytes
    rms_bytes = Dh * dtype_bytes
    out_bytes = B * H * Lq * Dh * dtype_bytes
    return q_bytes + kv_bytes + lam_bytes + rms_bytes + out_bytes


def benchmark_fn(fn, warmup=50, trials=200):
    """Returns median, p10, p90 in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    return times[n // 2], times[n // 10], times[9 * n // 10]


def main():
    device = "cuda"
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Kernel: {diagnose()}")
    print()

    # --- Correctness ---
    print("=" * 80)
    print("CORRECTNESS (bf16)")
    print("=" * 80)
    for name, cfg in CONFIGS.items():
        q, q_noise, k, v, lam, rms_w = make_inputs(cfg, device, dtype)
        with torch.no_grad():
            out_ref = reference_diff_attn(q, q_noise, k, v, lam, rms_weight=rms_w)
            out_fused = diff_flash_attn(q, q_noise, k, v, lam, rms_weight=rms_w)
        d = (out_ref - out_fused).abs()
        mx, mn = d.max().item(), d.mean().item()
        ok = "PASS" if mx < 6e-2 else "FAIL"
        print(f"  {name:25s} max={mx:.2e} mean={mn:.2e} [{ok}]")

    print()

    # --- Throughput ---
    print("=" * 80)
    print("THROUGHPUT (median ms, lower is better)")
    print("=" * 80)
    hdr = f"  {'Config':25s} {'Reference':>10s} {'Triton':>10s} {'Speedup':>8s} {'BW(GB/s)':>10s}"
    print(hdr)
    print("  " + "-" * 70)

    for name, cfg in CONFIGS.items():
        q, q_noise, k, v, lam, rms_w = make_inputs(cfg, device, dtype)

        def ref_fn():
            reference_diff_attn(q, q_noise, k, v, lam, rms_weight=rms_w)

        def fused_fn():
            diff_flash_attn(q, q_noise, k, v, lam, rms_weight=rms_w)

        with torch.no_grad():
            med_ref, _, _ = benchmark_fn(ref_fn)
            med_fused, p10, p90 = benchmark_fn(fused_fn)

        speedup = med_ref / med_fused if med_fused > 0 else float("inf")
        bw = bytes_moved(cfg) / (med_fused * 1e-3) / 1e9 if med_fused > 0 else 0

        desc = cfg.get("desc", "")
        print(f"  {name:25s} {med_ref:8.3f}ms {med_fused:8.3f}ms {speedup:7.2f}x {bw:8.1f}  ({desc})")

    print()

    # --- Peak Memory ---
    print("=" * 80)
    print("PEAK MEMORY (decode_7B)")
    print("=" * 80)
    cfg = CONFIGS["decode_7B"]
    q, q_noise, k, v, lam, rms_w = make_inputs(cfg, device, dtype)

    for label, fn in [
        ("Reference", lambda: reference_diff_attn(q, q_noise, k, v, lam, rms_weight=rms_w)),
        ("Triton",    lambda: diff_flash_attn(q, q_noise, k, v, lam, rms_weight=rms_w)),
    ]:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  {label:20s}: {peak:.1f} MB")


if __name__ == "__main__":
    main()
