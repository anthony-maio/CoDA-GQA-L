#!/usr/bin/env python3
"""Test harness for fused differential attention kernels.

Usage:
  1. Paste the generated kernel class into fused_kernel.py (same directory)
  2. Run: python benchmarks/test_fused_kernel.py

Or import and call test_kernel() directly with your kernel module.
"""

import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementation (known-correct, what we currently do)
# ---------------------------------------------------------------------------

class ReferenceModel(nn.Module):
    """Reference dual-stream differential attention -- two SDPA calls."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, q_sig, q_noise, k, v, lam):
        out_sig = F.scaled_dot_product_attention(q_sig, k, v)
        out_noise = F.scaled_dot_product_attention(q_noise, k, v)
        out = out_sig - lam * out_noise
        rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        out = out * rms * self.weight
        return out


class HeadStackedModel(nn.Module):
    """Current CoDA-GQA-L approach: head-stacking trick, single SDPA call."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, q_sig, q_noise, k, v, lam):
        q_cat = torch.cat([q_sig, q_noise], dim=1)  # (B, 2H, Lq, Dh)
        k_rep = torch.cat([k, k], dim=1)
        v_rep = torch.cat([v, v], dim=1)
        out_cat = F.scaled_dot_product_attention(q_cat, k_rep, v_rep)
        H = self.num_heads
        out_sig = out_cat[:, :H]
        out_noise = out_cat[:, H:]
        out = out_sig - lam * out_noise
        rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        out = out * rms * self.weight
        return out


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "decode_7B": dict(B=1, H=32, Lq=1, Lk=384, Dh=128, desc="7B decode (bounded cache)"),
    "decode_70B": dict(B=1, H=64, Lq=1, Lk=384, Dh=128, desc="70B decode (bounded cache)"),
    "decode_eve": dict(B=1, H=8, Lq=1, Lk=384, Dh=64, desc="Eve-2 decode"),
    "prefill_7B": dict(B=1, H=32, Lq=256, Lk=640, Dh=128, desc="7B prefill block=256"),
    "prefill_small_block": dict(B=1, H=32, Lq=32, Lk=416, Dh=128, desc="7B prefill block=32"),
    "batched_decode": dict(B=4, H=32, Lq=1, Lk=384, Dh=128, desc="7B batched decode B=4"),
}


def make_inputs(cfg, device="cuda", dtype=torch.bfloat16):
    B, H, Lq, Lk, Dh = cfg["B"], cfg["H"], cfg["Lq"], cfg["Lk"], cfg["Dh"]
    q_sig = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    q_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    k = torch.randn(B, H, Lk, Dh, device=device, dtype=dtype)
    v = torch.randn(B, H, Lk, Dh, device=device, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
    return q_sig, q_noise, k, v, lam


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness(fused_model, ref_model, cfg, device="cuda", dtype=torch.bfloat16):
    """Compare fused kernel output against reference. Returns max abs diff."""
    q_sig, q_noise, k, v, lam = make_inputs(cfg, device, dtype)

    with torch.no_grad():
        out_ref = ref_model(q_sig, q_noise, k, v, lam)
        out_fused = fused_model(q_sig, q_noise, k, v, lam)

    diff = (out_ref - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    return max_diff, mean_diff


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(model, cfg, device="cuda", dtype=torch.bfloat16,
              warmup=50, trials=200):
    """Benchmark model throughput. Returns median time in ms."""
    inputs = make_inputs(cfg, device, dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
    torch.cuda.synchronize()

    # Timed trials
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[9 * len(times) // 10]
    return median, p10, p90


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_all(fused_model_class=None, fused_init_args=None):
    device = "cuda"
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA required for kernel testing")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # If no fused model provided, try to import from fused_kernel.py
    if fused_model_class is None:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from fused_kernel import Model as FusedModel
            from fused_kernel import get_init_inputs
            fused_model_class = FusedModel
            fused_init_args = get_init_inputs()
            print(f"Loaded fused kernel from fused_kernel.py")
        except ImportError:
            print("No fused_kernel.py found. Testing reference vs head-stacked only.")
            print("To test a fused kernel, save it as benchmarks/fused_kernel.py")
            print()
            fused_model_class = None

    # --- Correctness ---
    print("=" * 70)
    print("CORRECTNESS")
    print("=" * 70)

    for name, cfg in CONFIGS.items():
        H, Dh = cfg["H"], cfg["Dh"]

        ref = ReferenceModel(H, Dh).to(device=device, dtype=dtype).eval()
        stacked = HeadStackedModel(H, Dh).to(device=device, dtype=dtype).eval()

        # Copy weights so comparison is fair
        stacked.weight.data.copy_(ref.weight.data)

        # Reference vs head-stacked (sanity check)
        max_d, mean_d = test_correctness(stacked, ref, cfg, device, dtype)
        status = "PASS" if max_d < 1e-3 else "FAIL"
        print(f"  {name:25s} head-stacked vs ref:  max={max_d:.2e} mean={mean_d:.2e}  [{status}]")

        # Fused vs reference
        if fused_model_class is not None:
            if fused_init_args:
                fused = fused_model_class(*fused_init_args).to(device=device, dtype=dtype).eval()
            else:
                fused = fused_model_class(H, Dh).to(device=device, dtype=dtype).eval()

            # Try to match weights
            if hasattr(fused, 'weight') and hasattr(ref, 'weight'):
                try:
                    fused.weight.data.copy_(ref.weight.data)
                except RuntimeError:
                    pass

            max_d, mean_d = test_correctness(fused, ref, cfg, device, dtype)
            status = "PASS" if max_d < 1e-2 else "WARN" if max_d < 1e-1 else "FAIL"
            print(f"  {name:25s} fused vs ref:         max={max_d:.2e} mean={mean_d:.2e}  [{status}]")

    print()

    # --- Throughput ---
    print("=" * 70)
    print("THROUGHPUT (median ms, lower is better)")
    print("=" * 70)

    # Focus on decode configs for benchmarking
    bench_configs = {k: v for k, v in CONFIGS.items() if "decode" in k}

    print(f"  {'Config':25s} {'Reference':>12s} {'HeadStacked':>12s}", end="")
    if fused_model_class is not None:
        print(f" {'Fused':>12s} {'Speedup':>8s}", end="")
    print()
    print("  " + "-" * 75)

    for name, cfg in bench_configs.items():
        H, Dh = cfg["H"], cfg["Dh"]

        ref = ReferenceModel(H, Dh).to(device=device, dtype=dtype).eval()
        stacked = HeadStackedModel(H, Dh).to(device=device, dtype=dtype).eval()

        med_ref, _, _ = benchmark(ref, cfg, device, dtype)
        med_stk, _, _ = benchmark(stacked, cfg, device, dtype)

        print(f"  {name:25s} {med_ref:10.3f}ms {med_stk:10.3f}ms", end="")

        if fused_model_class is not None:
            if fused_init_args:
                fused = fused_model_class(*fused_init_args).to(device=device, dtype=dtype).eval()
            else:
                fused = fused_model_class(H, Dh).to(device=device, dtype=dtype).eval()

            med_fus, p10, p90 = benchmark(fused, cfg, device, dtype)
            speedup = med_stk / med_fus if med_fus > 0 else float('inf')
            print(f" {med_fus:10.3f}ms {speedup:7.2f}x", end="")

        print(f"  ({cfg['desc']})")

    print()

    # --- Memory ---
    print("=" * 70)
    print("PEAK MEMORY")
    print("=" * 70)

    cfg = CONFIGS["decode_7B"]
    H, Dh = cfg["H"], cfg["Dh"]

    for label, ModelClass in [("Reference", ReferenceModel), ("HeadStacked", HeadStackedModel)]:
        model = ModelClass(H, Dh).to(device=device, dtype=dtype).eval()
        inputs = make_inputs(cfg, device, dtype)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  {label:20s}: {peak:.1f} MB")

    if fused_model_class is not None:
        if fused_init_args:
            fused = fused_model_class(*fused_init_args).to(device=device, dtype=dtype).eval()
        else:
            fused = fused_model_class(H, Dh).to(device=device, dtype=dtype).eval()
        torch.cuda.reset_peak_memory_stats()
        inputs = make_inputs(cfg, device, dtype)
        with torch.no_grad():
            fused(*inputs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  {'Fused':20s}: {peak:.1f} MB")


if __name__ == "__main__":
    run_all()
