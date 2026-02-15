"""Unified benchmark suite for CoDA-GQA-L attention configurations.

Measures prefill throughput, decode throughput, KV cache size, and peak VRAM
across multiple attention configurations and writes JSON results to results/.

Usage:
    python benchmarks/run_suite.py                                  # all configs
    python benchmarks/run_suite.py --configs tiny-cache,medium-cache  # subset
    python benchmarks/run_suite.py --batch 4                         # override B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from coda_gqa_l import BaselineGQA, CoDAGQA, CoDAGQALandmarkPerf2

# ---------------------------------------------------------------------------
# Helpers (adapted from bench.py)
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sync(device: torch.device) -> None:
    """Synchronize CUDA device if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], Any],
    device: torch.device,
    iters: int = 10,
    warmup: int = 3,
) -> float:
    """Time a callable, returning average seconds per invocation."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------

# Each entry: (config_name, attention_class, constructor kwargs, extra metadata)
# The constructor kwargs do NOT include embed_dim/num_heads/num_kv_heads; those
# are injected from CLI defaults so they stay consistent across configs.

BOUNDED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiny-cache": {
        "class": "CoDAGQALandmarkPerf2",
        "kwargs": {"window": 128, "num_landmarks_exact": 32, "num_landmarks_summary": 32},
    },
    "medium-cache": {
        "class": "CoDAGQALandmarkPerf2",
        "kwargs": {"window": 256, "num_landmarks_exact": 64, "num_landmarks_summary": 64},
    },
    "window-only": {
        "class": "CoDAGQALandmarkPerf2",
        "kwargs": {"window": 256, "num_landmarks_exact": 0, "num_landmarks_summary": 0},
    },
}

UNBOUNDED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "class": "BaselineGQA",
        "kwargs": {},
    },
    "coda-unbounded": {
        "class": "CoDAGQA",
        "kwargs": {},
    },
}

ALL_CONFIGS: Dict[str, Dict[str, Any]] = {**UNBOUNDED_CONFIGS, **BOUNDED_CONFIGS}

CLASS_MAP = {
    "BaselineGQA": BaselineGQA,
    "CoDAGQA": CoDAGQA,
    "CoDAGQALandmarkPerf2": CoDAGQALandmarkPerf2,
}


def build_model(
    config_name: str,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Instantiate an attention module from the config matrix."""
    cfg = ALL_CONFIGS[config_name]
    cls = CLASS_MAP[cfg["class"]]
    kwargs = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        **cfg["kwargs"],
    }
    model = cls(**kwargs).to(device=device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


@torch.no_grad()
def bench_prefill_bounded(
    model: CoDAGQALandmarkPerf2,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    L: int,
    D: int,
    block_size: int = 256,
) -> float:
    """Return prefill time (seconds) for a bounded-memory model."""
    x = torch.randn(B, L, D, device=device, dtype=dtype)

    def run() -> None:
        state = model.init_state(batch_size=B, device=device, dtype=dtype)
        model.prefill_chunked(
            x, state, block_size=block_size, write_cache=True, return_outputs=False,
        )

    return time_call(run, device, iters=5, warmup=2)


@torch.no_grad()
def bench_prefill_unbounded(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    L: int,
    D: int,
) -> float:
    """Return prefill time (seconds) for an unbounded model (BaselineGQA / CoDAGQA)."""
    x = torch.randn(B, L, D, device=device, dtype=dtype)

    def run() -> None:
        model(x, is_causal=True, past_key_value=None)

    return time_call(run, device, iters=5, warmup=2)


@torch.no_grad()
def bench_decode_bounded(
    model: CoDAGQALandmarkPerf2,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    D: int,
    steps: int = 200,
) -> float:
    """Return per-step decode time (seconds) for a bounded-memory model."""
    state = model.init_state(batch_size=B, device=device, dtype=dtype)
    x = torch.randn(B, 1, D, device=device, dtype=dtype)

    # Warm up state by filling the window.
    warm = min(steps, model.window + 4)
    for _ in range(warm):
        _, state = model.step(x, state)

    def run() -> None:
        nonlocal state
        _, state = model.step(x, state)

    return time_call(run, device, iters=50, warmup=5)


@torch.no_grad()
def bench_decode_unbounded(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    D: int,
    steps: int = 200,
) -> float:
    """Return per-step decode time (seconds) for an unbounded model.

    We first do a prefill of `steps` tokens to build the KV cache, then
    measure incremental decode steps.
    """
    # Build a KV cache by prefilling.
    x_prefill = torch.randn(B, steps, D, device=device, dtype=dtype)
    _, kv = model(x_prefill, is_causal=True, past_key_value=None)

    x_step = torch.randn(B, 1, D, device=device, dtype=dtype)

    # Use a mutable container so the closure can update kv.
    cache = [kv]

    def run() -> None:
        _, cache[0] = model(x_step, is_causal=True, past_key_value=cache[0])

    return time_call(run, device, iters=50, warmup=5)


def compute_kv_cache_bytes_unbounded(
    B: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype,
) -> int:
    """Compute KV cache size for unbounded models at a given sequence length."""
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return 2 * B * num_kv_heads * seq_len * head_dim * bytes_per


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def gather_system_info(dtype: torch.dtype) -> Dict[str, str]:
    """Collect system metadata for the results JSON."""
    gpu_name = "cpu"
    cuda_version = "n/a"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "n/a"
    return {
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------


def run_config(
    config_name: str,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    batch: int,
    prefill_lengths: List[int],
    decode_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    results_dir: Path,
) -> Dict[str, Any]:
    """Benchmark a single config and return the result dict."""
    cfg = ALL_CONFIGS[config_name]
    is_bounded = cfg["class"] == "CoDAGQALandmarkPerf2"
    head_dim = embed_dim // num_heads

    print(f"\n{'='*60}")
    print(f"Config: {config_name}  ({cfg['class']})")
    print(f"{'='*60}")

    model = build_model(config_name, embed_dim, num_heads, num_kv_heads, device, dtype)

    # Reset peak memory tracking.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # -- Prefill --
    prefill_tok_per_sec: Dict[str, float] = {}
    for L in prefill_lengths:
        print(f"  Prefill L={L} ...", end=" ", flush=True)
        if is_bounded:
            dt = bench_prefill_bounded(model, device, dtype, batch, L, embed_dim)
        else:
            dt = bench_prefill_unbounded(model, device, dtype, batch, L, embed_dim)
        tps = (batch * L) / dt
        prefill_tok_per_sec[str(L)] = round(tps, 1)
        print(f"{tps:,.0f} tok/s  ({dt*1000:.2f} ms)")

    # -- Decode --
    print(f"  Decode (B={batch}, steps={decode_steps}) ...", end=" ", flush=True)
    if is_bounded:
        dt = bench_decode_bounded(model, device, dtype, batch, embed_dim, steps=decode_steps)
    else:
        dt = bench_decode_unbounded(model, device, dtype, batch, embed_dim, steps=decode_steps)
    decode_tps = round(batch / dt, 1)
    print(f"{decode_tps:,.0f} tok/s  ({dt*1e6:.1f} us/step)")

    # -- KV cache bytes --
    if is_bounded:
        kv_bytes = model.cache_bytes(batch_size=batch, dtype=dtype)
    else:
        # For unbounded, report at the largest prefill length tested.
        max_L = max(prefill_lengths)
        kv_bytes = compute_kv_cache_bytes_unbounded(
            batch, num_kv_heads, head_dim, max_L, dtype,
        )
    print(f"  KV cache: {human_bytes(kv_bytes)}")

    # -- Peak VRAM --
    peak_vram: Optional[int] = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device)
        print(f"  Peak VRAM: {human_bytes(peak_vram)}")

    # -- Build config metadata --
    config_meta: Dict[str, Any] = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
    }
    if is_bounded:
        config_meta["window"] = cfg["kwargs"]["window"]
        config_meta["Me"] = cfg["kwargs"]["num_landmarks_exact"]
        config_meta["Ms"] = cfg["kwargs"]["num_landmarks_summary"]

    result = {
        "config_name": config_name,
        "attention_type": cfg["class"],
        "system": gather_system_info(dtype),
        "config": config_meta,
        "results": {
            "prefill_tokens_per_sec": prefill_tok_per_sec,
            "decode_tokens_per_sec": decode_tps,
            "kv_cache_bytes": kv_bytes,
            "peak_vram_bytes": peak_vram,
        },
    }

    # Write JSON.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{config_name}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  -> Saved {out_path}")

    # Free GPU memory between configs.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoDA-GQA-L benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated list of config names to run (default: all). "
             f"Available: {', '.join(ALL_CONFIGS.keys())}",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding dim (default: 512)")
    parser.add_argument("--num-heads", type=int, default=8, help="Query heads (default: 8)")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="KV heads (default: 2)")
    parser.add_argument(
        "--prefill-lengths",
        type=str,
        default="512,2048,4096",
        help="Comma-separated prefill lengths (default: 512,2048,4096)",
    )
    parser.add_argument("--decode-steps", type=int, default=200, help="Decode warmup steps (default: 200)")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for JSON results (default: results/ next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device and dtype selection (same logic as bench.py).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # Resolve configs to run.
    if args.configs is not None:
        config_names = [c.strip() for c in args.configs.split(",")]
        for name in config_names:
            if name not in ALL_CONFIGS:
                print(f"ERROR: Unknown config '{name}'. Available: {', '.join(ALL_CONFIGS.keys())}")
                sys.exit(1)
    else:
        config_names = list(ALL_CONFIGS.keys())

    prefill_lengths = [int(x.strip()) for x in args.prefill_lengths.split(",")]

    # Resolve results directory.
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("CoDA-GQA-L Benchmark Suite")
    print(f"  Device: {device}")
    print(f"  Dtype:  {dtype}")
    print(f"  Batch:  {args.batch}")
    print(f"  D={args.embed_dim}, H={args.num_heads}, Hkv={args.num_kv_heads}")
    print(f"  Prefill lengths: {prefill_lengths}")
    print(f"  Decode steps: {args.decode_steps}")
    print(f"  Configs: {config_names}")
    print(f"  Results dir: {results_dir}")

    all_results: List[Dict[str, Any]] = []
    for name in config_names:
        result = run_config(
            config_name=name,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            batch=args.batch,
            prefill_lengths=prefill_lengths,
            decode_steps=args.decode_steps,
            device=device,
            dtype=dtype,
            results_dir=results_dir,
        )
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_results)} config(s) benchmarked.")
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
