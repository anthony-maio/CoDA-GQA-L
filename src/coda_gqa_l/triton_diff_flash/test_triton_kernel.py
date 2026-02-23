#!/usr/bin/env python3
"""Correctness tests for the fused differential FlashAttention Triton kernel.

Tests the kernel against a reference implementation (two PyTorch SDPA calls
+ epilogue) across decode, prefill, GQA, and dtype configurations.

Usage:
    python kernels/triton_diff_flash/test_triton_kernel.py
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# Import from package (subpackage of coda_gqa_l)
from coda_gqa_l.triton_diff_flash import diff_flash_attn, is_available, diagnose


# ---------------------------------------------------------------------------
# Reference: two SDPA calls + differential epilogue
# ---------------------------------------------------------------------------

def reference_diff_attn(q, q_noise, k, v, lam, rms_weight=None, eps=1e-6,
                        is_causal=False, sm_scale=None):
    """Known-correct reference: two separate PyTorch SDPA calls."""
    B, H, Lq, Dh = q.shape
    _, Hkv, Lk, _ = k.shape
    groups = H // Hkv

    if groups > 1:
        k_exp = k[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
        v_exp = v[:, :, None, :, :].expand(B, Hkv, groups, Lk, Dh).reshape(B, H, Lk, Dh)
    else:
        k_exp, v_exp = k, v

    if sm_scale is not None:
        # PyTorch SDPA uses 1/sqrt(Dh) by default; we match the custom scale
        # by prescaling Q and passing scale=1 (not directly supported).
        # Easier: just use default and trust it matches our kernel's default.
        pass

    out_sig = F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=None, is_causal=is_causal,
    )
    out_noise = F.scaled_dot_product_attention(
        q_noise, k_exp, v_exp, attn_mask=None, is_causal=is_causal,
    )

    out = out_sig - lam * out_noise

    if rms_weight is not None:
        var = out.pow(2).mean(dim=-1, keepdim=True)
        out = out * torch.rsqrt(var + eps) * rms_weight

    return out


# ---------------------------------------------------------------------------
# Configs (same shapes as benchmarks/test_fused_kernel.py)
# ---------------------------------------------------------------------------

CONFIGS = {
    "decode_7B":          dict(B=1, H=32, Hkv=32, Lq=1,   Lk=384, Dh=128),
    "decode_70B":         dict(B=1, H=64, Hkv=8,  Lq=1,   Lk=384, Dh=128),
    "decode_eve":         dict(B=1, H=8,  Hkv=8,  Lq=1,   Lk=384, Dh=64),
    "prefill_7B":         dict(B=1, H=32, Hkv=32, Lq=256, Lk=640, Dh=128),
    "prefill_small_block":dict(B=1, H=32, Hkv=32, Lq=32,  Lk=416, Dh=128),
    "batched_decode":     dict(B=4, H=32, Hkv=32, Lq=1,   Lk=384, Dh=128),
}

# GQA configs: vary the ratio H/Hkv
GQA_CONFIGS = {
    "gqa_2":  dict(B=1, H=8,  Hkv=4, Lq=16, Lk=128, Dh=64),
    "gqa_4":  dict(B=1, H=16, Hkv=4, Lq=16, Lk=128, Dh=64),
    "gqa_8":  dict(B=1, H=32, Hkv=4, Lq=16, Lk=128, Dh=64),
}

# Edge cases
EDGE_CONFIGS = {
    "square":      dict(B=1, H=8, Hkv=8, Lq=64,  Lk=64,  Dh=64),
    "tiny_kv":     dict(B=1, H=8, Hkv=8, Lq=1,   Lk=16,  Dh=64),
    "long_prefix": dict(B=1, H=8, Hkv=8, Lq=8,   Lk=512, Dh=64),
}


def make_inputs(cfg, device, dtype):
    B, H, Hkv = cfg["B"], cfg["H"], cfg["Hkv"]
    Lq, Lk, Dh = cfg["Lq"], cfg["Lk"], cfg["Dh"]
    q = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    q_noise = torch.randn(B, H, Lq, Dh, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, Lk, Dh, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, Lk, Dh, device=device, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, H, Lq, 1, device=device, dtype=dtype))
    return q, q_noise, k, v, lam


def run_test(name, cfg, device, dtype, is_causal, apply_rms, tol_max, tol_mean):
    """Run a single correctness test. Returns (passed, max_diff, mean_diff)."""
    q, q_noise, k, v, lam = make_inputs(cfg, device, dtype)
    Dh = cfg["Dh"]

    rms_weight = None
    if apply_rms:
        rms_weight = torch.ones(Dh, device=device, dtype=dtype)

    with torch.no_grad():
        out_ref = reference_diff_attn(
            q, q_noise, k, v, lam,
            rms_weight=rms_weight, is_causal=is_causal,
        )
        out_fused = diff_flash_attn(
            q, q_noise, k, v, lam,
            rms_weight=rms_weight, is_causal=is_causal,
        )

    diff = (out_ref - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = max_diff < tol_max and mean_diff < tol_mean
    return passed, max_diff, mean_diff


def main():
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Kernel status: {diagnose()}")
    print()

    if not is_available():
        print("Triton kernel not available, running fallback-only tests...")
        # Still test the fallback path
        print()

    # Tolerance per dtype — online softmax in fp32 accumulators with bf16/fp16
    # inputs naturally gives ~2-5e-2 max diff vs PyTorch SDPA (which may use
    # different precision pathways internally). These are realistic bounds.
    tols = {
        torch.bfloat16: (6e-2, 1e-2),
        torch.float16:  (2e-2, 5e-3),
        torch.float32:  (1e-4, 1e-5),
    }

    total = 0
    passed_count = 0
    failed = []

    # Main configs: bf16, non-causal + causal, with and without RMSNorm
    print("=" * 80)
    print("MAIN CONFIGS")
    print("=" * 80)
    for name, cfg in CONFIGS.items():
        for dtype, (tol_max, tol_mean) in tols.items():
            dtype_name = str(dtype).split(".")[-1]
            for is_causal in [False, True]:
                for apply_rms in [False, True]:
                    tag = f"{name}/{dtype_name}/causal={is_causal}/rms={apply_rms}"
                    ok, mx, mn = run_test(name, cfg, device, dtype, is_causal, apply_rms, tol_max, tol_mean)
                    total += 1
                    if ok:
                        passed_count += 1
                        print(f"  PASS {tag:55s} max={mx:.2e} mean={mn:.2e}")
                    else:
                        failed.append(tag)
                        print(f"  FAIL {tag:55s} max={mx:.2e} mean={mn:.2e}")

    # GQA configs (bf16 only, causal + non-causal)
    print()
    print("=" * 80)
    print("GQA CONFIGS")
    print("=" * 80)
    dtype = torch.bfloat16
    tol_max, tol_mean = tols[dtype]
    for name, cfg in GQA_CONFIGS.items():
        for is_causal in [False, True]:
            tag = f"{name}/bf16/causal={is_causal}"
            ok, mx, mn = run_test(name, cfg, device, dtype, is_causal, False, tol_max, tol_mean)
            total += 1
            if ok:
                passed_count += 1
                print(f"  PASS {tag:55s} max={mx:.2e} mean={mn:.2e}")
            else:
                failed.append(tag)
                print(f"  FAIL {tag:55s} max={mx:.2e} mean={mn:.2e}")

    # Edge cases (bf16 only)
    print()
    print("=" * 80)
    print("EDGE CASES")
    print("=" * 80)
    for name, cfg in EDGE_CONFIGS.items():
        for is_causal in [False, True]:
            tag = f"{name}/bf16/causal={is_causal}"
            ok, mx, mn = run_test(name, cfg, device, dtype, is_causal, True, tol_max, tol_mean)
            total += 1
            if ok:
                passed_count += 1
                print(f"  PASS {tag:55s} max={mx:.2e} mean={mn:.2e}")
            else:
                failed.append(tag)
                print(f"  FAIL {tag:55s} max={mx:.2e} mean={mn:.2e}")

    # Summary
    print()
    print("=" * 80)
    print(f"RESULTS: {passed_count}/{total} passed")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
