#!/usr/bin/env python3
"""Empirical proof: is_causal=True uses UPPER-LEFT alignment for rectangular attention.

This script demonstrates the PyTorch SDPA causal alignment behavior that
affects any KV-cache prefill where Lq < Lk.

The core issue:
  - is_causal=True creates an upper-left aligned triangular mask
  - For KV-cache prefill, we need LOWER-RIGHT alignment (all prefix visible)
  - Using is_causal=True with rectangular attention silently computes wrong results

Run:
  python benchmarks/test_sdpa_causal.py

This script requires no GPU and no model weights. Pure PyTorch.
"""

import sys

import torch
import torch.nn.functional as F


def make_upper_left_mask(Lq: int, Lk: int) -> torch.Tensor:
    """Upper-left causal: query i sees keys 0..i. Standard tril."""
    return torch.tril(torch.ones(Lq, Lk, dtype=torch.bool), diagonal=0)


def make_lower_right_mask(Lq: int, Lk: int) -> torch.Tensor:
    """Lower-right causal: query i sees keys 0..(Lk-Lq+i).
    This is what KV-cache prefill needs: all prefix keys visible,
    plus causal within the current block.
    """
    offset = Lk - Lq
    return torch.tril(torch.ones(Lq, Lk, dtype=torch.bool), diagonal=offset)


def make_prefix_causal_mask(Lprev: int, Lq: int) -> torch.Tensor:
    """Explicit prefix-causal: all prefix visible + causal within block.
    This is what CoDA-GQA-L's _get_causal_mask produces.
    """
    prefix = torch.ones(Lq, Lprev, dtype=torch.bool)
    block = torch.tril(torch.ones(Lq, Lq, dtype=torch.bool), diagonal=0)
    return torch.cat([prefix, block], dim=1)


def sdpa_with_mask(q, k, v, mask):
    """SDPA with explicit boolean mask (True = attend)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)


def main():
    torch.manual_seed(42)
    print("=" * 70)
    print("SDPA Causal Alignment Test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    print()

    # --- Check causal_lower_right availability ---
    has_causal_lower_right = False
    try:
        from torch.nn.attention.bias import causal_lower_right
        has_causal_lower_right = True
        print("torch.nn.attention.bias.causal_lower_right: AVAILABLE")
    except ImportError:
        print("torch.nn.attention.bias.causal_lower_right: NOT AVAILABLE")

    try:
        from torch.nn.attention.bias import CausalBias
        print("torch.nn.attention.bias.CausalBias: AVAILABLE")
    except ImportError:
        print("torch.nn.attention.bias.CausalBias: NOT AVAILABLE")

    # --- Version info ---
    major, minor = torch.__version__.split(".")[:2]
    torch_version = (int(major), int(minor))
    print(f"\nParsed version: {torch_version}")
    print()

    # Version history:
    # - PyTorch < 2.5:  is_causal=True only worked for square (Lq==Lk).
    #                   Rectangular was undefined / error in some backends.
    # - PyTorch 2.5+:   is_causal=True explicitly defined as UPPER-LEFT for non-square.
    # - PyTorch 2.6+:   causal_lower_right added in torch.nn.attention.bias
    #                   (prototype API, subject to change).
    # - PyTorch 2.10+:  causal_lower_right documented in stable API.
    #
    # CONCLUSION: You can NEVER safely use is_causal=True when Lq != Lk
    # for KV-cache style prefill. You need either:
    #   1. causal_lower_right (torch >= 2.6)
    #   2. Explicit boolean/float mask (any version)

    print("-" * 70)
    print("TEST 1: Square attention (Lq == Lk) -- is_causal=True is fine")
    print("-" * 70)

    B, H, Dh = 1, 1, 16
    Lq = Lk = 8

    q = torch.randn(B, H, Lq, Dh)
    k = torch.randn(B, H, Lk, Dh)
    v = torch.randn(B, H, Lk, Dh)

    out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ul = sdpa_with_mask(q, k, v, make_upper_left_mask(Lq, Lk))
    out_lr = sdpa_with_mask(q, k, v, make_lower_right_mask(Lq, Lk))

    diff_ul = (out_causal - out_ul).abs().max().item()
    diff_lr = (out_causal - out_lr).abs().max().item()

    print(f"  Lq={Lq}, Lk={Lk}")
    print(f"  is_causal vs upper-left mask: max diff = {diff_ul:.2e}")
    print(f"  is_causal vs lower-right mask: max diff = {diff_lr:.2e}")
    print(f"  (For square, upper-left == lower-right, so both should match)")
    assert diff_ul < 1e-5, f"Square: is_causal should match upper-left, got {diff_ul}"
    assert diff_lr < 1e-5, f"Square: is_causal should match lower-right, got {diff_lr}"
    print("  PASS")
    print()

    print("-" * 70)
    print("TEST 2: Rectangular attention (Lq=4, Lk=12) -- the critical case")
    print("-" * 70)
    print("  Simulates KV-cache prefill: 8 prefix keys + 4 block keys")
    print("  Correct behavior: each query sees ALL prefix + causal in block")
    print()

    Lq, Lk = 4, 12
    Lprev = Lk - Lq  # 8 prefix keys

    q = torch.randn(B, H, Lq, Dh)
    k = torch.randn(B, H, Lk, Dh)
    v = torch.randn(B, H, Lk, Dh)

    out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ul = sdpa_with_mask(q, k, v, make_upper_left_mask(Lq, Lk))
    out_lr = sdpa_with_mask(q, k, v, make_lower_right_mask(Lq, Lk))
    out_prefix = sdpa_with_mask(q, k, v, make_prefix_causal_mask(Lprev, Lq))

    diff_vs_ul = (out_causal - out_ul).abs().max().item()
    diff_vs_lr = (out_causal - out_lr).abs().max().item()
    diff_lr_vs_prefix = (out_lr - out_prefix).abs().max().item()

    print(f"  is_causal matches upper-left:  max diff = {diff_vs_ul:.2e}")
    print(f"  is_causal matches lower-right: max diff = {diff_vs_lr:.2e}")
    print(f"  lower-right == prefix-causal:  max diff = {diff_lr_vs_prefix:.2e}")
    print()

    # Show the masks visually
    print("  Upper-left mask (what is_causal=True gives you):")
    ul = make_upper_left_mask(Lq, Lk)
    for i in range(Lq):
        row = "".join("1" if ul[i, j] else "." for j in range(Lk))
        visible = ul[i].sum().item()
        print(f"    q[{i}]: {row}  (sees {visible}/{Lk} keys)")
    print()

    print("  Lower-right / prefix-causal mask (what KV-cache prefill NEEDS):")
    lr = make_lower_right_mask(Lq, Lk)
    for i in range(Lq):
        row = "".join("1" if lr[i, j] else "." for j in range(Lk))
        visible = lr[i].sum().item()
        print(f"    q[{i}]: {row}  (sees {visible}/{Lk} keys)")
    print()

    if diff_vs_ul < 1e-5:
        print("  CONFIRMED: is_causal=True uses UPPER-LEFT alignment")
        print("  -> Query 0 sees only key 0 (WRONG: should see all 8 prefix + key 0)")
        print("  -> This is a SILENT correctness bug for KV-cache prefill!")
    elif diff_vs_lr < 1e-5:
        print("  SURPRISE: is_causal=True uses LOWER-RIGHT alignment on this version")
        print("  -> This would be correct for KV-cache prefill")
        print("  -> But you still shouldn't rely on this -- use explicit masks")
    else:
        print("  UNEXPECTED: is_causal matches neither upper-left nor lower-right")
        print(f"  -> diff vs UL: {diff_vs_ul}, diff vs LR: {diff_vs_lr}")

    assert diff_lr_vs_prefix < 1e-5, "lower-right and prefix-causal should be identical"
    print()

    print("-" * 70)
    print("TEST 3: Numerical impact -- how wrong is upper-left?")
    print("-" * 70)

    output_diff = (out_lr - out_ul).abs()
    print(f"  lower-right vs upper-left output difference:")
    print(f"    max:  {output_diff.max().item():.4f}")
    print(f"    mean: {output_diff.mean().item():.4f}")
    print(f"  This is how wrong your attention output is with is_causal=True")
    print()

    # Larger scale test
    Lq_big, Lk_big = 32, 416  # block_size=32, prefix=384 (like CoDA W+Me+Ms)
    q_big = torch.randn(B, H, Lq_big, Dh)
    k_big = torch.randn(B, H, Lk_big, Dh)
    v_big = torch.randn(B, H, Lk_big, Dh)

    out_ul_big = sdpa_with_mask(q_big, k_big, v_big, make_upper_left_mask(Lq_big, Lk_big))
    out_lr_big = sdpa_with_mask(q_big, k_big, v_big, make_lower_right_mask(Lq_big, Lk_big))
    big_diff = (out_lr_big - out_ul_big).abs()

    print(f"  At CoDA-GQA-L scale (block=32, prefix=384):")
    print(f"    max diff:  {big_diff.max().item():.4f}")
    print(f"    mean diff: {big_diff.mean().item():.4f}")
    print(f"    Query 0 with upper-left sees {make_upper_left_mask(Lq_big, Lk_big)[0].sum().item()}/416 keys")
    print(f"    Query 0 with lower-right sees {make_lower_right_mask(Lq_big, Lk_big)[0].sum().item()}/416 keys")
    print()

    # --- Test causal_lower_right if available ---
    if has_causal_lower_right:
        print("-" * 70)
        print("TEST 4: causal_lower_right (torch.nn.attention.bias)")
        print("-" * 70)

        from torch.nn.attention.bias import causal_lower_right as clr

        bias = clr(Lq, Lk)
        out_clr = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)
        diff_clr_vs_lr = (out_clr - out_lr).abs().max().item()
        diff_clr_vs_prefix = (out_clr - out_prefix).abs().max().item()

        print(f"  causal_lower_right vs manual lower-right: {diff_clr_vs_lr:.2e}")
        print(f"  causal_lower_right vs prefix-causal:      {diff_clr_vs_prefix:.2e}")

        if diff_clr_vs_lr < 1e-5:
            print("  CONFIRMED: causal_lower_right produces correct KV-cache alignment")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("PyTorch version history for causal alignment:")
    print("  < 2.5:  is_causal=True with Lq!=Lk was undefined/error")
    print("  2.5+:   is_causal=True defined as UPPER-LEFT (wrong for KV-cache)")
    print("  2.6+:   causal_lower_right added (prototype API)")
    print("  2.10+:  causal_lower_right in stable docs")
    print()
    print("Safe options for any PyTorch version:")
    print("  1. Explicit bool mask: prefix(all-True) + tril(block)")
    print("     Pro: works everywhere. Con: may force Math SDPA backend.")
    print("  2. causal_lower_right (torch >= 2.6)")
    print("     Pro: may unlock Flash/MemEfficient. Con: version-gated.")
    print("  3. is_causal=True ONLY when Lq == Lk (square attention)")
    print()
    print(f"CoDA-GQA-L current code: uses option 1 + option 3 (correct).")
    print(f"  - Square (no prefix): is_causal=True")
    print(f"  - Rectangular (prefix): explicit _get_causal_mask()")
    print()

    if has_causal_lower_right:
        print(f"This PyTorch ({torch.__version__}) supports causal_lower_right.")
        print("Could upgrade to option 2 for potential Flash/MemEfficient speedup.")
    else:
        print(f"This PyTorch ({torch.__version__}) does NOT have causal_lower_right.")
        print("Explicit mask fallback is the correct approach.")


if __name__ == "__main__":
    main()
