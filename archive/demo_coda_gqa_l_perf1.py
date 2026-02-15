# demo_coda_gqa_l_perf1.py
# Micro-benchmark + sanity checks for CoDA-GQA-L Performance Iteration 1.

import os
import time

import torch

from coda_gqa_l_v3 import CoDAGQALandmarkV3
from coda_gqa_l_perf1 import CoDAGQALandmarkPerf1


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, device: torch.device, iters: int = 10, warmup: int = 3):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


@torch.no_grad()
def bench_prefill(mod, device, dtype, B, L, D, block_size):
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    state = mod.init_state(batch_size=B, device=device, dtype=dtype)

    def run():
        _y, _ = mod.prefill_chunked(x, state, block_size=block_size, write_cache=True, return_outputs=False)

    dt = time_call(run, device, iters=5, warmup=1)
    return dt


@torch.no_grad()
def bench_decode_steps(mod, device, dtype, B, steps, D):
    state = mod.init_state(batch_size=B, device=device, dtype=dtype)
    x = torch.randn(B, 1, D, device=device, dtype=dtype)

    # warmup by filling the window
    warm = min(steps, getattr(mod, "window", 128) + 4)
    for _ in range(warm):
        _y, state = mod.step(x, state)

    def run():
        nonlocal state
        _y, state = mod.step(x, state)

    dt = time_call(run, device, iters=50, warmup=5)
    return dt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer bf16 on cuda; fall back to fp16
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    B = int(os.environ.get("BATCH", "1"))
    D = int(os.environ.get("DIM", "512"))
    H = int(os.environ.get("HEADS", "8"))
    Hkv = int(os.environ.get("KV_HEADS", "2"))
    W = int(os.environ.get("W", "128"))
    Me = int(os.environ.get("ME", "32"))
    Ms = int(os.environ.get("MS", "32"))
    block_size = int(os.environ.get("BLOCK", "256"))

    lengths = [256, 1024, 4096]
    if os.environ.get("RUN_LONG", "0") == "1":
        lengths.append(16384)

    # Instantiate modules
    m_v3 = CoDAGQALandmarkV3(
        embed_dim=D,
        num_heads=H,
        num_kv_heads=Hkv,
        window=W,
        num_landmarks_exact=Me,
        num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    m_p1 = CoDAGQALandmarkPerf1(
        embed_dim=D,
        num_heads=H,
        num_kv_heads=Hkv,
        window=W,
        num_landmarks_exact=Me,
        num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)

    m_v3.eval()
    m_p1.eval()

    print(f"device={device} dtype={dtype}  B={B} D={D} H={H} Hkv={Hkv} W={W} Me={Me} Ms={Ms} block={block_size}")
    print()

    # Cache bytes
    print("Per-layer cache bytes:")
    print(f"  v3 : {human_bytes(m_v3.cache_bytes(batch_size=B, dtype=dtype))}")
    print(f"  p1 : {human_bytes(m_p1.cache_bytes(batch_size=B, dtype=dtype))}")
    print()

    # Prefill benchmark
    print("Prefill (chunked) avg seconds per call:")
    for L in lengths:
        dt_v3 = bench_prefill(m_v3, device, dtype, B, L, D, block_size)
        dt_p1 = bench_prefill(m_p1, device, dtype, B, L, D, block_size)
        print(f"  L={L:6d}: v3={dt_v3:.6f}s   p1={dt_p1:.6f}s   speedup={dt_v3/dt_p1:.2f}x")
    print()

    # Decode benchmark
    print("Decode step avg seconds per token (post-warmup):")
    dt_v3 = bench_decode_steps(m_v3, device, dtype, B, steps=1000, D=D)
    dt_p1 = bench_decode_steps(m_p1, device, dtype, B, steps=1000, D=D)
    print(f"  v3={dt_v3:.6f}s   p1={dt_p1:.6f}s   speedup={dt_v3/dt_p1:.2f}x")


if __name__ == "__main__":
    main()
