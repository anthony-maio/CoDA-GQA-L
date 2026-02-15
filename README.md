# CoDA-GQA-L

**Constrained Orthogonal Differential Attention with Grouped-Query Attention and Landmark Memory**

CoDA-GQA-L is a bounded-memory attention mechanism that achieves O(1) per-token KV cache growth while preserving long-range retrieval capability. It combines two independent innovations into a single attention layer:

1. **CoDA-GQA** (Constrained Orthogonal Differential Attention + Grouped-Query Attention): sharpens attention by computing signal and inhibitory streams from a single query projection. The inhibitory query is produced by a learned per-head orthogonal rotation of the signal query, eliminating the need for a second `W_q` matrix. The two streams are subtracted with a learned per-token gate `lambda`, producing a differential output that suppresses diffuse attention mass over weakly-relevant tokens.

2. **Bounded KV Memory (-L suffix)**: replaces the standard O(L) KV cache with a fixed-size buffer of size O(W + M_e + M_s) per layer, composed of three segments -- a recent window (exact ring buffer), an exact landmark bank (novelty-filtered LRU cache for needle-like tokens), and a summary landmark bank (EMA prototypes that compress older context). As sequence length grows, KV memory remains constant while the mechanism selectively retains important tokens and compresses background information.

Together, these components provide an attention layer that can process arbitrarily long sequences with bounded memory, improved attention sharpness, and no additional KV projections beyond standard GQA.


## Key Innovation

Standard transformers cache all past keys and values, producing O(L) memory growth per layer. Approaches like sliding window attention cap memory but discard distant context entirely. Retrieval-augmented methods require external indices. CoDA-GQA-L takes a different approach: it maintains a fixed-size buffer and uses learned importance gating to decide **which** evicted tokens are preserved as exact snapshots and which are compressed into summary prototypes.

The differential attention component addresses a separate problem. In standard softmax attention over long contexts, many weakly-relevant tokens compete for probability mass, diluting attention to the tokens that matter. By computing a second attention stream from an orthogonally-rotated query and subtracting it (scaled by a learned gate), the mechanism cancels shared "noise" patterns while preserving signal-specific patterns. Critically, this requires no second key-value projection -- the same KV cache serves both streams, and the inhibitory query is derived from the signal query via a parameter-efficient rotation.

The combination is natural: differential attention improves retrieval quality, and bounded memory ensures the cost of that retrieval stays constant. The write gate that controls memory admission is itself learned, allowing the model to develop its own importance criterion during training.


## Architecture

### Attention Flow

The differential attention mechanism uses a single query projection with a learned orthogonal rotation to produce the inhibitory stream:

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

The signal and inhibitory queries are stacked along the head dimension and processed in a **single SDPA call**, reducing kernel launch overhead compared to two separate attention operations. The orthogonal rotation `R(theta)` is parameterized by per-head angle vectors and preserves query norms, acting as a constrained "different view" of the same query subspace.

### Memory Update (on eviction from recent window)

When the recent window is full and a new token arrives, the oldest token is evicted and routed to the landmark banks according to a learned write gate:

```
evicted token (k, v, gate) from ring buffer
  |
  +-> write gate check (gate >= threshold?)
  |
  +-> Exact bank: cosine similarity to used slots
  |     novel? -> insert (free slot or replace LRU)
  |     hit?   -> optionally refresh, update LRU timestamp
  |
  +-> Summary bank: cosine similarity -> best matching slot
        EMA update: mem += eta_eff * (token - mem)
        eta_eff = sigmoid(eta_logit) * gate
        first insert with eta=1 (fast warmup)
```

The write gate `g = sigmoid(W_g * x + b_g)` is a learned scalar that determines whether an evicted token is important enough to update long-term memory. Separate thresholds control admission to the exact and summary banks.


## Memory Bound

Standard transformer attention caches all past keys and values, resulting in O(L) memory growth per layer where L is the sequence length. CoDA-GQA-L replaces this with a fixed-size buffer:

```
k_buf / v_buf shape: (B, H_kv, W + M_e + M_s, D_h)

slots:  [0 .. W-1]  [W .. W+Me-1]  [W+Me .. W+Me+Ms-1]
         recent       exact bank     summary bank
```

An `allowed` mask of shape `(B, L_buf)` tracks which slots contain valid data. The total buffer length `L_buf = W + M_e + M_s` is fixed at model construction time, independent of sequence length. For typical configurations (W=256, M_e=64, M_s=64), the per-layer cache is 384 slots regardless of whether the model has processed 1K or 100K tokens.

**Per-layer cache size** (bytes):

```
2 * B * H_kv * L_buf * D_h * bytes_per_element  +  metadata
```

The `cache_bytes()` method on the module returns the exact byte count for a given batch size and dtype.


## Installation

Requires Python >= 3.9 and PyTorch >= 2.0. CUDA with bf16 is recommended; the implementation falls back to fp32 on CPU.

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```


## Quick Start

```python
import torch
from coda_gqa_l import CoDAGQALandmarkPerf2, CoDAGQALandmarkStatePerf2

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

# Create a single attention layer
model = CoDAGQALandmarkPerf2(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,          # GQA: 4 query heads share each KV head
    window=128,              # Recent context (exact ring buffer)
    num_landmarks_exact=32,  # Needle retention slots (novelty-filtered LRU)
    num_landmarks_summary=32,# Compression slots (EMA prototypes)
).to(device=device, dtype=dtype)
model.eval()

B = 1  # batch size

# Initialize per-layer state
state = model.init_state(batch_size=B, device=device, dtype=dtype)

# --- Prefill: ingest a prompt ---
prompt = torch.randn(B, 1024, 512, device=device, dtype=dtype)
y_prefill, state = model.prefill_chunked(
    prompt, state, block_size=256, write_cache=True
)
# y_prefill: (B, 1024, 512) -- attention output for each prompt token

# --- Decode: generate tokens one at a time ---
for step in range(100):
    x_t = torch.randn(B, 1, 512, device=device, dtype=dtype)
    y_t, state = model.step(x_t, state)
    # y_t: (B, 1, 512)

# Cache size is constant regardless of how many tokens have been processed
print(f"Cache bytes: {model.cache_bytes(batch_size=B, dtype=dtype):,}")
# state.pos tracks the total number of tokens written
print(f"Tokens processed: {state.pos}")
```


## Constructor Parameters

The full constructor signature of `CoDAGQALandmarkPerf2`:

```python
CoDAGQALandmarkPerf2(
    embed_dim: int,              # Model dimension
    num_heads: int,              # Number of query heads
    num_kv_heads: int,           # Number of KV heads (GQA)
    window: int,                 # Recent window size W
    num_landmarks_exact: int,    # Exact bank size Me
    num_landmarks_summary: int,  # Summary bank size Ms

    # Differential attention
    rope_base: float = 10_000.0,
    lambda_init_bias: float = -6.0,   # Initial lambda near zero (sigmoid(-6) ~ 0.002)
    theta_init: float = pi/2,         # Initial orthogonal rotation angle

    # Write policy
    write_policy: str = "gated",              # "gated" or "none"
    write_gate_init_bias: float = -2.0,
    write_gate_threshold_exact: float = 0.10,
    write_gate_threshold_summary: float = 0.05,

    # Exact bank behavior
    exact_match_threshold: float = 0.90,
    exact_novelty_threshold: float = 0.70,
    exact_refresh_on_hit: bool = False,
    exact_candidates_per_block: int = 8,

    # Summary bank behavior
    summary_eta_init_logit: float = -3.0,     # eta ~ sigmoid(-3) ~ 0.05
    summary_candidates_per_block: int = 64,
    summary_overwrite_on_insert: bool = True,

    # Memory initialization and masking
    mem_init: str = "random_normal",  # "random_normal" or "zeros"
    mask_unused_memory: bool = True,
    detach_evicted: bool = True,      # Detach evicted tokens (inference-friendly)
)
```


## Hyperparameter Guide

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `window` (W) | 128 -- 512 | Recent context size. Larger windows retain more exact recent tokens but increase per-layer memory. |
| `num_landmarks_exact` (M_e) | 16 -- 128 | Needle retention capacity. Determines how many distinct important tokens can be preserved from older context. |
| `num_landmarks_summary` (M_s) | 16 -- 128 | Background compression capacity. Number of EMA prototype slots for compressing non-landmark context. |
| `block_size` (prefill) | 128 -- 1024 | Prefill chunk size. Smaller blocks produce memory states closer to streaming (decode) semantics but increase overhead. |
| `write_gate_threshold_exact` | 0.05 -- 0.2 | Minimum gate value for an evicted token to be considered for the exact bank. |
| `write_gate_threshold_summary` | 0.02 -- 0.1 | Minimum gate value for an evicted token to update the summary bank. |
| `summary_eta_init_logit` | -3 to -1 | Controls the EMA learning rate for summary slots: `eta = sigmoid(logit)`, so -3 gives ~0.05 and -1 gives ~0.27. |


## Benchmarks

Prefill and decode timing benchmarks are provided in `benchmarks/bench.py`. The benchmark measures chunked prefill throughput at various sequence lengths and per-token decode latency at steady state (after the recent window is full).

```bash
# Default configuration (B=1, D=512, H=8, Hkv=2, W=128, Me=32, Ms=32)
python benchmarks/bench.py

# Include longer sequences (16K tokens)
RUN_LONG=1 python benchmarks/bench.py

# Custom configuration via environment variables
BATCH=1 DIM=512 HEADS=8 KV_HEADS=2 W=128 ME=32 MS=32 BLOCK=256 python benchmarks/bench.py
```

The benchmark reports per-layer cache bytes and compares against what the full (unbounded) KV cache would require at each sequence length.

A needle-in-haystack retention demo is also provided:

```bash
python examples/needle_demo.py
RUN_LONG=1 python examples/needle_demo.py
```

This demo places a high-gate-score "needle" token at position 64 in a long sequence, runs chunked prefill, and verifies that the needle's key is retained in the exact landmark bank (measured by cosine similarity > 0.999) despite the sequence being much longer than the cache window.


## Testing

The test suite covers correctness, determinism, edge configurations, and structural invariants:

```bash
python -m pytest tests/ -v
```

Test modules:

- **`test_correctness.py`** -- Output shape, cache size accounting, prefill/decode equivalence, baseline comparison
- **`test_determinism.py`** -- Reproducibility across runs with fixed seeds
- **`test_edge_configs.py`** -- Zero-sized banks, window=1, policy overrides, initialization variants, prefill boundary conditions
- **`test_invariants.py`** -- Buffer layout, allowed-mask consistency, position tracking, ring buffer state, memory bank population, cache byte accounting


## Project Structure

```
src/coda_gqa_l/
  __init__.py         Public API exports
  attention.py        CoDAGQALandmarkPerf2 (facade: projections, SDPA, prefill, decode)
  memory_banks.py     MemoryBankMixin (exact/summary bank updates, scratch buffers, ring helpers)
  state.py            CoDAGQALandmarkStatePerf2 dataclass (KV buffers, ring metadata, norm caches)
  primitives.py       HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
  baseline.py         CoDAGQA (unbounded diff-attn), BaselineGQA (standard GQA)

benchmarks/bench.py       Prefill + decode timing benchmark
examples/needle_demo.py   Needle-in-haystack retention demo
tests/                    49 tests across 4 modules
docs/                     Design deep-dives (deep_dive_perf2.md, deep_dive_v3.md)
archive/                  Previous iterations (v3, perf1) for reference only
```

The main module `CoDAGQALandmarkPerf2` inherits from `MemoryBankMixin` via mixin pattern, keeping memory bank update logic separate from attention and SDPA computation while maintaining the full `self.*` interface.


## Key Design Decisions

- **RoPE-at-write.** Keys are rotated with RoPE before storage in the cache. This eliminates O(L_buf) key rotations per decode step at the cost of making memory routing decisions position-sensitive. If position-free memory matching is required, a separate content key projection or raw key storage (as in the v3 iteration) would be needed.

- **Single SDPA call via head-stacking.** The signal query `q` and inhibitory query `q_noise` are concatenated along the head dimension, so both attention streams are computed in a single `scaled_dot_product_attention` call. This reduces kernel launch overhead compared to two separate calls, though the total FLOPs remain approximately 2x that of standard attention.

- **Vectorized block memory updates.** During chunked prefill, evicted tokens are updated into the exact and summary banks using `scatter_reduce` and `scatter_add` with winner-take-all selection. This avoids Python loops over evicted tokens and enables efficient batch processing.

- **GQA compatibility.** Keys and values are stored in KV-head space. The `repeat_kv` function uses `expand` + `reshape` (zero-copy) to match query heads for SDPA, which unlocks FlashAttention and MemoryEfficient backends. The `enable_gqa=True` SDPA flag is intentionally avoided because it forces fallback to the Math kernel in current PyTorch versions.

- **Detached eviction.** By default (`detach_evicted=True`), tokens evicted from the recent window are detached from the computation graph before being written to landmark banks. This makes the module inference-friendly. Setting `detach_evicted=False` preserves gradients through eviction for training.

- **Winner-take-all in float32.** When multiple evicted tokens compete for the same exact bank slot during block updates, tie-breaking uses float32 arithmetic to avoid precision loss in bf16/fp16.

- **Scratch buffer reuse.** Memory bank updates reuse lazily-allocated scratch tensors keyed by `(B, H_kv, D_h, device, dtype)`, eliminating per-block allocations during prefill.

- **V-routing with cached norms.** Cosine similarity routing in both banks uses Values (not Keys) for position-invariant semantic matching, with incrementally-updated normalized value caches (`_exact_v_norm`, `_sum_v_norm`) rather than re-normalizing the full bank on every update. Keys have RoPE applied at write time, making their cosine similarity position-dependent; Values are RoPE-free and preserve pure semantic content for deduplication and EMA blending.


## Baselines

The package includes two unbounded attention baselines for comparison and ablation:

- **`CoDAGQA`**: Differential attention with GQA but no memory bounding. Uses the same orthogonal rotation and lambda gating mechanism with standard O(L) KV cache growth.
- **`BaselineGQA`**: Standard grouped-query attention with RoPE. Single SDPA call, O(L) KV cache.

```python
from coda_gqa_l import CoDAGQA, BaselineGQA

# Unbounded differential attention
coda = CoDAGQA(embed_dim=512, num_heads=8, num_kv_heads=2)
y, kv_cache = coda(x, is_causal=True)

# Standard GQA baseline
gqa = BaselineGQA(embed_dim=512, num_heads=8, num_kv_heads=2)
y, kv_cache = gqa(x, is_causal=True)
```


## Design Deep-Dives

- [docs/deep_dive_perf2.md](docs/deep_dive_perf2.md) -- Current iteration: full architecture specification, memory layout, complexity analysis, failure modes, and practical tuning guidance
- [docs/deep_dive_v3.md](docs/deep_dive_v3.md) -- Earlier iteration with raw-key storage (historical reference)


## Known Limitations

- **No fused differential attention kernel.** Compute is still approximately 2x that of standard attention (two streams via head-stacking). A custom CUDA kernel could fuse the signal-noise subtraction with the softmax.
- **No end-to-end training or finetuning demo.** The current release provides the attention module and inference-path benchmarks. Integration into a full transformer stack with training loop is left to downstream users.
- **No distributed cache sharding.** Tensor parallelism and sequence parallelism for the bounded KV buffer are not implemented.
- **No quantized KV storage.** Keys and values are stored in full precision (bf16/fp16/fp32). INT8 or INT4 KV quantization could further reduce memory.


## Citation

If you use CoDA-GQA-L in your research, please cite:

```bibtex
@misc{coda_gqa_l_2025,
  title   = {{CoDA-GQA-L}: Constrained Orthogonal Differential Attention with
             Grouped-Query Attention and Landmark Memory},
  author  = {},
  year    = {2025},
  url     = {},
  note    = {Software available from \url{}}
}
```


## License

MIT
