#!/usr/bin/env python3
"""Benchmark: fused diff epilogue kernel vs PyTorch baseline.

Run:
    cd kernels/coda_diff_epilogue
    pip install -e .
    python benchmark_diff_epilogue.py

Or without compilation (uses PyTorch fallback for correctness check):
    python benchmark_diff_epilogue.py --no-kernel
"""

import argparse
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference (unfused PyTorch)
# ---------------------------------------------------------------------------

def diff_epilogue_pytorch(out_sig, out_noise, lam, weight, eps=1e-6):
    """Three separate ops: subtract, RMSNorm, scale."""
    diff = out_sig - lam * out_noise
    rms = diff.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return diff * rms * weight


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(fn, *args, warmup=50, trials=200, sync=True):
    """Returns median time in ms."""
    for _ in range(warmup):
        fn(*args)
    if sync:
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-kernel", action="store_true",
                        help="Skip fused kernel, only run correctness check with fallback")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Try to import fused kernel
    has_kernel = False
    if not args.no_kernel:
        try:
            from coda_kernels import diff_epilogue, diff_sub_only, is_available, diagnose
            has_kernel = is_available()
            print(f"Fused kernel: {'LOADED' if has_kernel else 'FALLBACK (not compiled)'}")
            if not has_kernel:
                print(f"\n--- Kernel diagnostics ---")
                print(diagnose())
                print(f"--- end diagnostics ---\n")
        except ImportError:
            print("Fused kernel: NOT INSTALLED (run pip install -e .)")
    else:
        print("Fused kernel: SKIPPED (--no-kernel)")

    if not has_kernel:
        from coda_kernels import diff_epilogue, diff_sub_only  # fallback mode

    # ---------------------------------------------------------------------------
    # Configs matching real CoDA-GQA-L usage
    # ---------------------------------------------------------------------------

    configs = [
        # (name, B, H, Lq, Dh, dtype)
        ("decode_7B",   1, 32,   1, 128, torch.bfloat16),
        ("decode_70B",  1, 64,   1, 128, torch.bfloat16),
        ("decode_eve",  1,  8,   1,  64, torch.bfloat16),
        ("prefill_7B",  1, 32, 256, 128, torch.bfloat16),
        ("prefill_70B", 1, 64, 256, 128, torch.bfloat16),
        ("batched_7B",  4, 32,   1, 128, torch.bfloat16),
        ("fp16_7B",     1, 32,   1, 128, torch.float16),
        ("fp32_7B",     1, 32,   1, 128, torch.float32),
    ]

    # ---------------------------------------------------------------------------
    # Correctness: full epilogue (subtract + RMSNorm + weight)
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRECTNESS: diff_epilogue (subtract + RMSNorm + weight)")
    print("=" * 70)

    for name, B, H, Lq, Dh, dtype in configs:
        torch.manual_seed(42)
        out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
        weight    = torch.ones(Dh, device=device, dtype=dtype)

        ref = diff_epilogue_pytorch(out_sig, out_noise, lam, weight)
        fused = diff_epilogue(out_sig, out_noise, lam, weight)

        max_diff = (ref - fused).abs().max().item()
        mean_diff = (ref - fused).abs().mean().item()

        # bf16 tolerance is higher because kernel accumulates in fp32 while
        # PyTorch reference uses bf16 intermediates (kernel is more precise)
        tol = 5e-2 if dtype == torch.bfloat16 else (1e-2 if dtype == torch.float16 else 1e-5)
        status = "PASS" if max_diff < tol else "FAIL"
        print(f"  {name:20s}  max={max_diff:.2e}  mean={mean_diff:.2e}  [{status}]")

    # ---------------------------------------------------------------------------
    # Correctness: subtract-only (identity head norm)
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRECTNESS: diff_sub_only (identity mode)")
    print("=" * 70)

    def diff_sub_pytorch(out_sig, out_noise, lam):
        return out_sig - lam * out_noise

    for name, B, H, Lq, Dh, dtype in configs:
        torch.manual_seed(42)
        out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))

        ref = diff_sub_pytorch(out_sig, out_noise, lam)
        fused = diff_sub_only(out_sig, out_noise, lam)

        max_diff = (ref - fused).abs().max().item()
        mean_diff = (ref - fused).abs().mean().item()

        tol = 5e-2 if dtype == torch.bfloat16 else (1e-2 if dtype == torch.float16 else 1e-5)
        status = "PASS" if max_diff < tol else "FAIL"
        print(f"  {name:20s}  max={max_diff:.2e}  mean={mean_diff:.2e}  [{status}]")

    # ---------------------------------------------------------------------------
    # Throughput: full epilogue
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("THROUGHPUT: diff_epilogue (median ms)")
    print("=" * 70)
    print(f"  {'Config':20s} {'PyTorch':>10s} {'Fused':>10s} {'Speedup':>8s}  {'Rows':>8s}")
    print("  " + "-" * 62)

    for name, B, H, Lq, Dh, dtype in configs:
        torch.manual_seed(42)
        out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
        weight    = torch.ones(Dh, device=device, dtype=dtype)

        rows = B * H * Lq

        t_ref = benchmark(diff_epilogue_pytorch, out_sig, out_noise, lam, weight)
        t_fused = benchmark(diff_epilogue, out_sig, out_noise, lam, weight)

        speedup = t_ref / t_fused if t_fused > 0 else float('inf')
        print(f"  {name:20s} {t_ref:8.4f}ms {t_fused:8.4f}ms {speedup:7.2f}x  {rows:>8d}")

    # ---------------------------------------------------------------------------
    # Throughput: subtract-only (identity mode, used by Mistral/Llama)
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("THROUGHPUT: diff_sub_only (median ms)")
    print("=" * 70)
    print(f"  {'Config':20s} {'PyTorch':>10s} {'Fused':>10s} {'Speedup':>8s}  {'Rows':>8s}")
    print("  " + "-" * 62)

    for name, B, H, Lq, Dh, dtype in configs:
        torch.manual_seed(42)
        out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
        lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))

        rows = B * H * Lq

        t_ref = benchmark(diff_sub_pytorch, out_sig, out_noise, lam)
        t_fused = benchmark(diff_sub_only, out_sig, out_noise, lam)

        speedup = t_ref / t_fused if t_fused > 0 else float('inf')
        print(f"  {name:20s} {t_ref:8.4f}ms {t_fused:8.4f}ms {speedup:7.2f}x  {rows:>8d}")

    # ---------------------------------------------------------------------------
    # Memory
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PEAK MEMORY (decode_7B)")
    print("=" * 70)

    B, H, Lq, Dh, dtype = 1, 32, 1, 128, torch.bfloat16
    out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
    weight    = torch.ones(Dh, device=device, dtype=dtype)

    # PyTorch baseline
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        _ = diff_epilogue_pytorch(out_sig, out_noise, lam, weight)
    torch.cuda.synchronize()
    peak_ref = torch.cuda.max_memory_allocated() / 1024

    # Fused
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        _ = diff_epilogue(out_sig, out_noise, lam, weight)
    torch.cuda.synchronize()
    peak_fused = torch.cuda.max_memory_allocated() / 1024

    print(f"  PyTorch unfused: {peak_ref:.1f} KB")
    print(f"  Fused kernel:    {peak_fused:.1f} KB")
    if peak_ref > 0:
        print(f"  Savings:         {(1 - peak_fused/peak_ref)*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Bandwidth analysis (prefill, where it matters most)
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BANDWIDTH ANALYSIS (prefill_7B: B=1, H=32, Lq=256, Dh=128)")
    print("=" * 70)

    B, H, Lq, Dh, dtype = 1, 32, 256, 128, torch.bfloat16
    bytes_per = 2  # bf16

    # Reads: out_sig + out_noise + lam + weight = 2*B*H*Lq*Dh + B*H*Lq + Dh
    # Writes: output = B*H*Lq*Dh
    # Total = 3*B*H*Lq*Dh + B*H*Lq + Dh (all in bytes_per)
    rows = B * H * Lq
    bytes_rw = (3 * rows * Dh + rows + Dh) * bytes_per
    print(f"  Data moved (optimal): {bytes_rw / 1024:.1f} KB")

    out_sig   = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    out_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    lam       = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
    weight    = torch.ones(Dh, device=device, dtype=dtype)

    if has_kernel:
        t_ms = benchmark(diff_epilogue, out_sig, out_noise, lam, weight, trials=500)
        bandwidth_gbps = (bytes_rw / 1e9) / (t_ms / 1e3)
        # Peak memory bandwidths: H100=3350, A100=2000, RTX 4090=1008, RTX 4080=716
        print(f"  Kernel time:    {t_ms:.4f} ms")
        print(f"  Bandwidth:      {bandwidth_gbps:.1f} GB/s")
        gpu_name = torch.cuda.get_device_name(0)
        if "4080" in gpu_name:
            print(f"  RTX 4080 efficiency: {bandwidth_gbps/716*100:.1f}%")
        elif "4090" in gpu_name:
            print(f"  RTX 4090 efficiency: {bandwidth_gbps/1008*100:.1f}%")
        print(f"  H100 efficiency: {bandwidth_gbps/3350*100:.1f}%")
        print(f"  A100 efficiency: {bandwidth_gbps/2000*100:.1f}%")
    else:
        print("  (Bandwidth analysis requires compiled kernel)")


if __name__ == "__main__":
    main()
