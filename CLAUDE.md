# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoDA-GQA-L is a research attention mechanism combining two innovations:

1. **CoDA-GQA** (Constrained Orthogonal Differential Attention + Grouped-Query Attention): Sharpens attention by computing signal and inhibitory streams from the same query projection, subtracting the noise stream weighted by a learned gate lambda. The inhibitory query is produced via a per-head orthogonal rotation of the signal query (no second Wq).

2. **Bounded KV Memory (-L suffix)**: Replaces O(L) KV cache with O(W + Me + Ms) per layer using three segments:
   - **Recent window** (W): ring buffer of exact recent tokens
   - **Exact landmark bank** (Me): novelty-filtered LRU cache of important evicted tokens
   - **Summary landmark bank** (Ms): EMA prototypes compressing older context

## File Layout

```
src/coda_gqa_l/
  __init__.py        Public API exports
  attention.py       CoDAGQALandmarkPerf2 (facade: projections, SDPA, prefill, decode)
  memory_banks.py    MemoryBankMixin (exact/summary bank updates, scratch buffers, ring helpers)
  state.py           CoDAGQALandmarkStatePerf2 dataclass (KV buffers, ring metadata, norm caches)
  primitives.py      HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
  baseline.py        CoDAGQA (unbounded diff-attn), BaselineGQA (standard GQA)
  triton_bank_routing.py  Fused exact-bank routing kernel (97x, Makora-evaluated)
  triton_diff_flash.py    Fused differential FlashAttention kernel
  triton_summary_bank.py  Fused summary bank update kernel (31x, Makora-generated)
  eve_adapter.py     EveCoDAAdapter (drop-in replacement for Eve-2 CausalSelfAttention)
  qwen3_adapter.py   Qwen3CoDAAdapter (drop-in for Qwen3 attention, rope_theta=1M)

benchmarks/
  bench.py              Prefill + decode timing (quick single-config)
  run_suite.py           Full benchmark suite (5 configs, JSON output to results/)
  render_tables.py       Reads results/*.json, renders markdown comparison tables
  forward_check_generic.py  Weight transfer validation for Llama-family models
  eval_eve.py            Eve-2 integration benchmarks (forward-check, perplexity, needle)
notebooks/
  h100_experiments.ipynb  Full paper experiments: fine-tune, PPL, needle, benchmarks
  colab_qwen3_coda.ipynb  Colab Pro+ pipeline: Qwen3-4B + CoDA train & HF publish
results/                 Benchmark output JSONs + tables.md (gitignored)
paper/
  paper_outline.md       Full paper outline with method/experiment sections
  claims.md              Maps each paper claim to evidence artifacts
examples/needle_demo.py  Needle-in-haystack retention demo
docs/                    Design deep-dives (deep_dive_perf2.md, deep_dive_v3.md)
archive/                 Previous iterations (v3, perf1) for reference only
```

CoDAGQALandmarkPerf2 inherits from MemoryBankMixin via mixin pattern, keeping
bank methods as `self.*` with zero behavioral change from the pre-split code.

## Running

```bash
pip install -e .

# Quick single-config benchmark
python benchmarks/bench.py
RUN_LONG=1 python benchmarks/bench.py

# Full benchmark suite (5 configs, JSON output)
python benchmarks/run_suite.py
python benchmarks/run_suite.py --configs tiny-cache,medium-cache
python benchmarks/render_tables.py

# Eve-2 integration benchmarks (requires transformers, datasets)
python benchmarks/eval_eve.py --experiment forward-check
python benchmarks/eval_eve.py --experiment perplexity
python benchmarks/eval_eve.py --experiment needle
python benchmarks/eval_eve.py --experiment all

# Needle demo
python examples/needle_demo.py

# Tests
python -m pytest tests/ -v
```

Requires PyTorch >= 2.0. CUDA with bf16 recommended; falls back to fp32 on CPU.

## Imports

```python
# External usage
from coda_gqa_l import CoDAGQALandmarkPerf2, CoDAGQALandmarkStatePerf2
from coda_gqa_l import CoDAGQA, BaselineGQA
from coda_gqa_l import EveCoDAAdapter
from coda_gqa_l import Qwen3CoDAAdapter

# Internal (within package)
from .primitives import HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
from .state import CoDAGQALandmarkStatePerf2
from .memory_banks import MemoryBankMixin
from .eve_adapter import EveCoDAAdapter
from .qwen3_adapter import Qwen3CoDAAdapter
```

## Architecture: Attention Flow

```
x -> q_proj -> q
          \-> RoPE(q, pos) -> q_roped
                          \-> R(theta) -> q_noise

x -> k_proj -> k_raw -> RoPE(k, pos) -> k (stored in cache)
x -> v_proj -> v (stored in cache)

SDPA( cat([q_roped, q_noise], dim=head), k_buf, v_buf )
  -> split -> out_sig, out_noise
  -> out_sig - lambda * out_noise
  -> HeadwiseRMSNorm
  -> o_proj -> y
```

## Architecture: Memory Update (on eviction from recent window)

```
evicted token (k, v, gate) from ring buffer
  |
  +-> write gate check (gate >= threshold?)
  |
  +-> Exact bank: V-routing cosine similarity (on values, not keys)
  |     novel? -> insert (free slot or replace LRU)
  |     hit?   -> optionally refresh, update LRU timestamp
  |
  +-> Summary bank: V-routing cosine similarity -> best matching slot
        EMA update: mem += eta_eff * (token - mem)
        eta_eff = sigmoid(eta_logit) * gate
        first insert with eta=1 (fast warmup)
```

V-routing uses Values for similarity because Keys have RoPE applied,
making identical tokens at different positions appear dissimilar.

## Buffer Layout

`k_buf` / `v_buf` shape: `(B, Hkv, W + Me + Ms, Dh)`

```
slots:  [0 .. W-1]  [W .. W+Me-1]  [W+Me .. W+Me+Ms-1]
         recent       exact bank     summary bank
```

`allowed` mask `(B, Lbuf)` tracks which slots are valid.

## Key Design Decisions

- **V-routing**: Memory bank cosine similarity uses Values (not Keys) for position-invariant semantic matching. Keys have RoPE applied, making their similarity position-dependent; Values are RoPE-free and preserve pure semantic content for deduplication (exact bank) and EMA blending (summary bank). Norm caches: `_exact_v_norm`, `_sum_v_norm`.
- **Dense packing for FlashAttention**: When B==1 and there are invalid prefix slots, valid slots are dense-packed before SDPA, eliminating the boolean `allowed` mask. This unlocks FlashAttention/MemEfficient backends instead of falling back to Math kernel. For batched (B>1), falls back to explicit masks.
- **RoPE-at-write**: Keys are RoPE'd before storage. Eliminates O(Lbuf) key rotations per decode step. V-routing ensures this doesn't break semantic matching.
- **Single SDPA call via head-stacking**: `q_cat = [q ; q_noise]` along head dim. Reduces kernel launches vs two separate calls.
- **Vectorized block memory updates**: `scatter_reduce`/`scatter_add` with winner-take-all for exact bank. No Python loops.
- **GQA compatibility**: KV stored in KV-head space. `repeat_kv` expands heads for SDPA (unlocks MemEfficient/Flash backends).
- **`detach_evicted=True`**: Evicted tokens are detached by default (inference-friendly).
- **Winner-take-all in exact block update**: Uses float32 for tie-breaking to survive bf16/fp16 precision loss.

## Typical Hyperparameter Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| W (window) | 128-512 | Recent context size |
| Me (exact) | 16-128 | Needle retention capacity |
| Ms (summary) | 16-128 | Background compression capacity |
| block_size (prefill) | 128-1024 | Smaller = closer to streaming semantics |
| write_gate_threshold_exact | 0.05-0.2 | |
| write_gate_threshold_summary | 0.02-0.1 | |
| summary_eta_logit | -3 to -1 | eta ~ 0.05-0.27 |

## Metrics (Optional)

Enable with `collect_metrics=True` in `CoDAGQALandmarkPerf2`. Counters accumulate in `state.metrics` dict:
- `exact_hits`, `exact_inserts`, `exact_overwrites`, `exact_fill_ratio`
- `summary_updates`, `summary_inserts`, `summary_fill_ratio`
- `tokens_gated_out`, `total_evictions`

Zero overhead when disabled (`state.metrics is None`). Reset with `model.reset_metrics(state)`.

## Eve-2 Integration

`EveCoDAAdapter` wraps CoDA-GQA-L as a drop-in for Eve's `CausalSelfAttention`:
```python
adapter = EveCoDAAdapter.from_eve_attention(eve_block.attn, bounded=False)
eve_block.attn = adapter  # Block.forward(x, freqs_cis) works unchanged
```
- Splits fused QKV weights, reconstructs biased Linear layers
- `freqs_cis` accepted but ignored (CoDA manages its own RoPE)
- `bounded=False`: CoDAGQA (unbounded, for training)
- `bounded=True`: CoDAGQALandmarkPerf2 (bounded, for inference)

## Qwen3 Integration

`Qwen3CoDAAdapter` wraps CoDA-GQA-L as a drop-in for Qwen3's attention:
```python
adapters = Qwen3CoDAAdapter.swap_qwen3_layers(model, bounded=False)
```
- 1:1 weight mapping (same Q/K/V/O layout as Llama)
- `rope_theta=1,000,000` (auto-detected from model config)
- Contiguous-half RoPE (`rope_interleaved=False`)
- Inherits from `LlamaCoDAAdapter` (thin wrapper with Qwen3 defaults)
- Tested with Qwen3-4B (32 Q heads, 8 KV heads, head_dim=128, 36 layers)

Training notebook: `notebooks/colab_qwen3_coda.ipynb` (Colab Pro+ A100 40GB)

## Known Gaps

- Fused differential attention kernel exists (`triton_diff_flash.py`) but not yet integrated into attention.py (needs bf16 eval)
- No distributed cache sharding (tensor parallel, sequence parallel)
- No quantized KV storage
- Bounded inference in eval_eve.py requires layer-by-layer state threading (TODO)
