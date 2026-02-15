# needle_demo.py
# Demonstrates needle-in-haystack retention with bounded KV cache.
#
# A "needle" token (high write-gate score) is placed early in a long sequence.
# After chunked prefill, we verify the needle's key is retained in memory
# despite the sequence being much longer than the cache window.
#
# Run:
#   python examples/needle_demo.py
#   RUN_LONG=1 python examples/needle_demo.py

from __future__ import annotations

import os

import torch
import torch.nn.functional as tF

from coda_gqa_l import CoDAGQALandmarkPerf2
from coda_gqa_l.primitives import apply_rope


def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


@torch.no_grad()
def run():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    B = 1
    D = 512
    H = 8
    Hkv = 2
    W = 128
    Me = 32
    Ms = 32

    model = CoDAGQALandmarkPerf2(
        embed_dim=D,
        num_heads=H,
        num_kv_heads=Hkv,
        window=W,
        num_landmarks_exact=Me,
        num_landmarks_summary=Ms,
        write_policy="gated",
        write_gate_threshold_exact=0.10,
        write_gate_threshold_summary=0.05,
    ).to(device=device, dtype=dtype)

    # Configure the write gate to strongly depend on a chosen direction u.
    # Tokens aligned with u get high gate scores; anti-aligned get low.
    u = torch.randn(D, device=device, dtype=torch.float32)
    u = u / (u.norm() + 1e-6)
    model.write_proj.weight.data.copy_(u.view(1, D).to(dtype=model.write_proj.weight.dtype))
    model.write_proj.bias.data.zero_()

    # Length schedule
    run_long = os.environ.get("RUN_LONG", "0") == "1"

    if device == "cpu" and not run_long:
        lens = [256, 1024]
        block_size = 128
    else:
        lens = [256, 1024, 4096]
        if run_long or device == "cuda":
            lens.append(16384)
        block_size = 256

    needle_pos = 64

    print("device:", device, "dtype:", dtype)
    print(f"Config: W={W}, Me={Me}, Ms={Ms}, block_size={block_size}, needle_pos={needle_pos}")

    capped_bytes = model.cache_bytes(batch_size=B, dtype=dtype)
    print("Capped cache bytes (per layer):", _fmt_bytes(capped_bytes))

    Dh = D // H
    bytes_per = torch.tensor([], dtype=dtype).element_size()

    for L in lens:
        # Background tokens: anti-aligned with u (low gate).
        # Needle token: aligned with u (high gate).
        noise = torch.randn(B, L, D, device=device, dtype=torch.float32) * 0.01
        x = (-5.0 * u.view(1, 1, D) + noise).to(dtype=dtype)
        x[:, needle_pos, :] = (5.0 * u + noise[0, needle_pos]).to(dtype=dtype)

        full_bytes = 2 * B * Hkv * L * Dh * bytes_per

        state = model.init_state(batch_size=B, device=torch.device(device), dtype=dtype)
        _, state = model.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=False)

        # Get the needle's key with RoPE applied at its position.
        # perf2 stores RoPE'd keys, so we must RoPE the reference key too.
        needle_x = x[:, needle_pos : needle_pos + 1, :]
        k_need, _ = model._project_kv_raw(needle_x)
        cos_r, sin_r = model.rope(seq_len=1, offset=needle_pos, device=k_need.device, dtype=k_need.dtype)
        k_need = apply_rope(k_need, cos_r, sin_r)
        k_need = k_need[:, :, 0, :]  # (B, Hkv, Dh)

        # Check recent segment (first W slots of buffer).
        n_recent = state.recent_filled
        k_recent = state.k_buf[:, :, :n_recent, :]
        allowed_recent = state.allowed[:, :n_recent]  # (B, n_recent)

        # Check exact segment (W:W+Me slots).
        k_exact = state.k_buf[:, :, W : W + Me, :]
        allowed_exact = state.allowed[:, W : W + Me]  # (B, Me)

        def max_cos(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            """Max cosine similarity between a (B,Hkv,Dh) and b (B,Hkv,N,Dh)."""
            a_n = tF.normalize(a.float(), dim=-1, eps=1e-6)
            b_n = tF.normalize(b.float(), dim=-1, eps=1e-6)
            sims = torch.einsum("bhd,bhnd->bhn", a_n, b_n)
            if mask is not None:
                # mask is (B, N) -> expand to (B, Hkv, N)
                sims = sims.masked_fill(~mask.unsqueeze(1), -1e9)
            return sims.max(dim=-1).values.max(dim=-1).values

        max_recent = max_cos(k_need, k_recent, allowed_recent) if n_recent > 0 else torch.full((B,), -1.0, device=device)
        max_exact = max_cos(k_need, k_exact, allowed_exact) if Me > 0 else torch.full((B,), -1.0, device=device)

        retained = (max_exact > 0.999) | (max_recent > 0.999)

        print(f"\nL = {L}")
        print(f"  KV(full) per layer:  {_fmt_bytes(full_bytes)}")
        print(f"  KV(capped) per layer: {_fmt_bytes(capped_bytes)}")
        print(f"  max cosine to recent: {float(max_recent.item()):.6f}")
        print(f"  max cosine to exact:  {float(max_exact.item()):.6f}")
        print(f"  needle retained:      {bool(retained.item())}")


if __name__ == "__main__":
    run()
