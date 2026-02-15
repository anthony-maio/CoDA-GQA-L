# bench.py
# Micro-benchmark for CoDA-GQA-L (prefill + decode timing).
#
# Run:
#   python benchmarks/bench.py
#   RUN_LONG=1 python benchmarks/bench.py
#
# Environment variables:
#   BATCH, DIM, HEADS, KV_HEADS, W, ME, MS, BLOCK, RUN_LONG

import os
import time

import torch

from coda_gqa_l import CoDAGQALandmarkPerf2


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
    warm = min(steps, mod.window + 4)
    for _ in range(warm):
        _y, state = mod.step(x, state)

    def run():
        nonlocal state
        _y, state = mod.step(x, state)

    dt = time_call(run, device, iters=50, warmup=5)
    return dt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = CoDAGQALandmarkPerf2(
        embed_dim=D,
        num_heads=H,
        num_kv_heads=Hkv,
        window=W,
        num_landmarks_exact=Me,
        num_landmarks_summary=Ms,
    ).to(device=device, dtype=dtype)
    model.eval()

    print(f"device={device} dtype={dtype}  B={B} D={D} H={H} Hkv={Hkv} W={W} Me={Me} Ms={Ms} block={block_size}")
    print(f"Per-layer cache bytes: {human_bytes(model.cache_bytes(batch_size=B, dtype=dtype))}")

    Dh = D // H
    bytes_per = torch.tensor([], dtype=dtype).element_size()

    print()
    print("Prefill (chunked) avg seconds per call:")
    for L in lengths:
        full_kv_bytes = 2 * B * Hkv * L * Dh * bytes_per
        dt = bench_prefill(model, device, dtype, B, L, D, block_size)
        print(f"  L={L:6d}: {dt:.6f}s   (full KV would be {human_bytes(full_kv_bytes)})")
    print()

    print("Decode step avg seconds per token (post-warmup):")
    dt = bench_decode_steps(model, device, dtype, B, steps=1000, D=D)
    print(f"  {dt:.6f}s")


if __name__ == "__main__":
    main()
