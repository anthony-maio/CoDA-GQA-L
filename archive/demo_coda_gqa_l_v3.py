
# demo_coda_gqa_l_v3.py
#
# Run:
#   python demo_coda_gqa_l_v3.py
#   RUN_LONG=1 python demo_coda_gqa_l_v3.py
#
# Notes:
#   - On CPU, SDPA can be slow for large L; by default we keep lengths small.
#   - On CUDA, larger lengths are included automatically.

from __future__ import annotations

import os
import torch

from coda_gqa_l_v3 import CoDAGQALandmarkV3


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

    model = CoDAGQALandmarkV3(
        embed_dim=D,
        num_heads=H,
        num_kv_heads=Hkv,
        window=W,
        num_landmarks_exact=Me,
        num_landmarks_summary=Ms,
        write_policy="gated",
        write_gate_threshold_exact=0.10,
        write_gate_threshold_summary=0.05,
        mask_unused_memory=True,
        mem_rope_exact="pos",
        mem_rope_summary="mean_pos",
    ).to(device=device, dtype=dtype)

    # Configure the write gate to strongly depend on a chosen direction u
    u = torch.randn(D, device=device, dtype=torch.float32)
    u = u / (u.norm() + 1e-6)
    model.write_proj.weight.data.copy_(u.view(1, D).to(device=device, dtype=model.write_proj.weight.dtype))
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
    print("SDPA enable_gqa supported:", getattr(__import__("coda_gqa_l_v3"), "_SDPA_ENABLE_GQA", None))

    capped_bytes = model.cache_bytes(batch_size=B, dtype=dtype)
    print("\nCapped cache bytes (per layer):", _fmt_bytes(capped_bytes))

    Dh = D // H
    bytes_per = torch.tensor([], dtype=dtype).element_size()

    for L in lens:
        # Background tokens -> low gate; needle -> high gate.
        noise = torch.randn(B, L, D, device=device, dtype=torch.float32) * 0.01
        x = (-5.0 * u.view(1, 1, D) + noise).to(dtype=dtype)
        x[:, needle_pos, :] = (5.0 * u + noise[0, needle_pos]).to(dtype=dtype)

        full_bytes = 2 * B * Hkv * L * Dh * bytes_per

        state = model.init_state(batch_size=B, device=torch.device(device), dtype=dtype)

        _, state = model.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=False)

        # Needle raw key
        needle_x = x[:, needle_pos:needle_pos+1, :]
        k_need, _ = model._project_kv_raw(needle_x)
        k_need = k_need[:, :, 0, :]

        # Stored raw keys: recent + exact memory
        k_recent, _, _, _ = model._get_recent_in_order(state)
        k_exact = state.mem_k_exact
        used_exact = state.mem_used_exact

        def max_cos(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            a_n = torch.nn.functional.normalize(a, dim=-1, eps=1e-6)
            b_n = torch.nn.functional.normalize(b, dim=-1, eps=1e-6)
            sims = torch.einsum("bhd,bhnd->bhn", a_n, b_n)
            if mask is not None:
                sims = sims.masked_fill(~mask, -1e9)
            return sims.max(dim=-1).values.max(dim=-1).values

        max_recent = max_cos(k_need, k_recent) if k_recent.size(2) > 0 else torch.full((B,), -1.0, device=device)
        max_exact = max_cos(k_need, k_exact, used_exact) if k_exact.size(2) > 0 else torch.full((B,), -1.0, device=device)

        retained = (max_exact > 0.999) | (max_recent > 0.999)

        print("\nL =", L)
        print("KV(full) per layer:", _fmt_bytes(full_bytes))
        print("KV(capped) per layer:", _fmt_bytes(capped_bytes))
        print("max cosine to recent:", float(max_recent.item()))
        print("max cosine to exact :", float(max_exact.item()))
        print("needle retained     :", bool(retained.item()))


if __name__ == "__main__":
    run()
