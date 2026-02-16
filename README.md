# CoDA-GQA-L

**Bounded-memory differential attention that actually works.**

```bash
pip install coda-gqa-l
```

A 70B model serving 128K context burns 160 GB on KV cache alone. CoDA-GQA-L does it in 136 MB -- a fixed-size buffer that doesn't grow no matter how long the input is.

| Context | Standard KV (70B, 80 layers) | CoDA-GQA-L | Compression |
|---------|------------------------------|------------|-------------|
| 2K      | 2.56 GB                      | 136 MB     | 18.8x       |
| 32K     | 40 GB                        | 136 MB     | 294x        |
| 128K    | 160 GB                       | 136 MB     | **1,176x**  |

The mechanism replaces the O(L) KV cache with three fixed-size segments per layer:

- **Recent window** (W=256) -- ring buffer of exact recent tokens, FIFO eviction
- **Exact landmark bank** (Me=64) -- novelty-filtered LRU cache for important tokens a learned write gate decides are worth keeping
- **Summary landmark bank** (Ms=64) -- EMA prototypes compressing older context into semantic cluster centroids

384 slots per layer, always, whether you processed 2K or 128K tokens.

The bounded state is serializable. `torch.save()` it, load it a week later, query it without re-reading the original document. At 7B scale each state is 54 MB. We call this pattern **stateful neural databases**.

## How it works

Three ideas, briefly:

**Differential attention via orthogonal rotation.** Builds on Microsoft's Diff Transformer (Ye et al., 2024) but drops the second Wq projection. A learned per-head rotation produces the noise query from the signal query. Signal minus gated noise, one SDPA call (head-stacked), HeadwiseRMSNorm on the output.

**Value-routing.** Keys have RoPE baked in and their cosine similarity is position-dependent -- same word at position 100 vs 5000 looks orthogonal in key-space. Memory banks route on Values instead (RoPE-free, pure semantic content). This is what makes deduplication and EMA blending actually work.

**Two-phase training.** Phase 1 trains unbounded differential attention. Phase 2 switches to bounded cache with gradient flow through evictions so the write gate learns what to keep. Without Phase 2, bounded eval is catastrophic (PPL 5.62 to 2,464 on Mistral 7B).

## Quick start

```python
import torch
from coda_gqa_l import CoDAGQALandmarkPerf2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

attn = CoDAGQALandmarkPerf2(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,          # GQA: 4x head sharing
    window=256,
    num_landmarks_exact=64,
    num_landmarks_summary=64,
).to(device=device, dtype=dtype).eval()

state = attn.init_state(batch_size=1, device=device, dtype=dtype)

# Prefill
prompt = torch.randn(1, 2048, 512, device=device, dtype=dtype)
y, state = attn.prefill_chunked(prompt, state, block_size=256)

# Decode -- memory stays constant
for _ in range(1000):
    x_t = torch.randn(1, 1, 512, device=device, dtype=dtype)
    y_t, state = attn.step(x_t, state)

print(f"Tokens seen: {state.pos}")        # 3048
print(f"Cache size: {attn.cache_bytes(1, dtype):,} bytes")  # constant
```

### Stateful neural databases

```python
# Ingest a document
state = attn.init_state(batch_size=1, device="cuda", dtype=torch.bfloat16)
_, state = attn.prefill_chunked(document_embeddings, state, block_size=256)

# Save -- fixed size regardless of document length
torch.save(state, "document_42.pt")

# Later: load and query without re-reading the document
state = torch.load("document_42.pt")
answer, state = attn.step(query_embedding, state)
```

At 7B scale (32 layers): 54 MB per document state. 100 documents = 5.4 GB. Route queries to the right state on demand -- no vector database, no chunk-and-retrieve.

## Drop-in model adapters

### Llama family (Llama 2/3, Mistral, SmolLM)

```python
from coda_gqa_l import LlamaCoDAAdapter

adapter = LlamaCoDAAdapter.from_llama_attention(
    llama_block.self_attn,
    bounded=False,  # unbounded for training, True for inference
)
llama_block.self_attn = adapter
```

### Eve-2 MoE

```python
from coda_gqa_l import EveCoDAAdapter

adapter = EveCoDAAdapter.from_eve_attention(eve_block.attn, bounded=True)
eve_block.attn = adapter
```

## Training

Two-phase protocol. Phase 1 teaches differential attention with full context. Phase 2 switches to bounded cache so the model adapts to limited memory.

```bash
# Smoke test (~5 min)
python benchmarks/train_coda.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --max-steps 200 --bounded-steps 100 --bounded-config medium

# Mistral 7B (~6 hours on H100)
python benchmarks/train_coda.py \
    --model mistralai/Mistral-7B-v0.3 \
    --max-steps 2000 --bounded-steps 2000 \
    --bounded-lr-scale 0.5 --bounded-block-size 128 \
    --no-detach-evicted --batch-size 1 --grad-accum 8
```

**Bounded configs:**

| Config | Window | Exact | Summary | Total slots |
|--------|--------|-------|---------|-------------|
| tiny   | 128    | 32    | 32      | 192         |
| medium | 256    | 64    | 64      | 384         |
| large  | 512    | 128   | 128     | 768         |

### Training results (SmolLM2-135M on H200 NVL)

| Phase | Steps | PPL start | PPL end | Throughput |
|-------|-------|-----------|---------|------------|
| Phase 1 (unbounded) | 2000 | 70.0 | 22.0 | 36K tok/s |
| Phase 2 (bounded, medium) | 2000 | 35.75 | 31.12 | 1.8K tok/s |

41.5% PPL gap for 5.3x context compression. Phase 2 trains with `detach_evicted=False` so gradients flow through bank updates.

## Benchmarks

All numbers from H200 NVL, bf16, standalone attention modules.

```bash
python benchmarks/run_suite.py                          # 5 configs, JSON output
python benchmarks/run_suite.py --embed-dim 4096 \       # 7B scale
    --num-heads 32 --num-kv-heads 8
python benchmarks/render_tables.py                      # markdown tables
```

### Memory (per layer, medium-cache W=256 Me=64 Ms=64)

| Scale | Standard KV | CoDA bounded | Compression |
|-------|-------------|--------------|-------------|
| 7B (D=4096, H=32, Hkv=8)  | 32.0 MB | 1.7 MB | 18.8x |
| 70B (D=8192, H=64, Hkv=8) | 32.0 MB | 1.7 MB | 18.8x |

Across all layers at 128K context:

| Model | Standard KV total | CoDA total | Compression |
|-------|-------------------|------------|-------------|
| 7B (32 layers)  | 64 GB  | 54 MB  | **1,185x** |
| 70B (80 layers) | 160 GB | 136 MB | **1,176x** |

### Throughput (70B scale, tokens/sec)

| Config | Prefill 2K | Prefill 8K | Decode | Peak VRAM |
|--------|----------:|----------:|-------:|----------:|
| Baseline GQA       | 1,336,349 | 966,464 | 4,676 | 1.1 GB   |
| CoDA unbounded     | 889,203   | 598,314 | 2,914 | 1.6 GB   |
| CoDA medium-cache  | 149,832   | 153,716 | 1,753 | 568.6 MB |
| CoDA window-only   | 359,417   | 356,026 | 1,773 | 546.0 MB |

Bounded prefill throughput is flat (~150K tok/s) regardless of sequence length. Baseline drops 28% from 2K to 8K. Bounded VRAM is 1.9x lower.

## Architecture

### Attention flow

```
x -> q_proj -> RoPE(q) -> q_signal
                        \-> R(theta) -> q_noise

x -> k_proj -> RoPE(k) -> k  (stored in buffer)
x -> v_proj -> v             (stored in buffer)

SDPA([q_signal; q_noise], k_buf, v_buf)
  -> split -> out_sig, out_noise
  -> out_sig - lambda * out_noise
  -> HeadwiseRMSNorm -> o_proj -> output
```

### Buffer layout

```
k_buf / v_buf: (B, Hkv, W + Me + Ms, Dh)

Slots:  [0..W-1]      [W..W+Me-1]      [W+Me..W+Me+Ms-1]
         recent ring    exact bank       summary bank
```

On eviction from the ring:
1. Write gate check -- is this token worth remembering?
2. Exact bank: V-routing cosine similarity. Novel? Insert. Duplicate? Update LRU.
3. Summary bank: V-routing cosine similarity. EMA blend into best-matching prototype.

### Dense packing

For B=1, valid slots are dense-packed before SDPA. This avoids the boolean mask that forces PyTorch into the slow Math backend, unlocking FlashAttention/MemEfficient kernels.

## Correctness

Bounded is mathematically identical to unbounded when W >= L (no evictions):

| W | L | Max diff |
|---|---|----------|
| 512 | 512 | 1.2e-7 |
| 1024 | 1024 | 1.5e-7 |
| 2048 | 2048 | 1.8e-7 |

60 tests covering correctness, determinism, edge configs, invariants, and backward pass safety.

```bash
python -m pytest tests/ -v
```

## What's in the repo

```
src/coda_gqa_l/
  attention.py       CoDAGQALandmarkPerf2 (main module)
  memory_banks.py    Exact/summary bank updates (mixin)
  state.py           KV buffer + ring metadata dataclass
  primitives.py      RoPE, RMSNorm, GQA utils
  baseline.py        Unbounded CoDAGQA + standard GQA
  llama_adapter.py   Drop-in for Llama/Mistral/SmolLM
  eve_adapter.py     Drop-in for Eve-2 MoE

benchmarks/
  train_coda.py      Two-phase training pipeline
  run_suite.py       Perf benchmarks (5 configs, JSON output)
  eval_eve.py        Eve-2 integration eval
  bench.py           Quick single-config timing

tests/               60 tests
```

## Metrics

Optional instrumentation for understanding bank behavior:

```python
attn = CoDAGQALandmarkPerf2(..., collect_metrics=True)
state = attn.init_state(...)

# After processing...
print(state.metrics)
# {'exact_hits': 42, 'exact_inserts': 18, 'exact_fill_ratio': 0.28,
#  'summary_updates': 156, 'summary_fill_ratio': 1.0,
#  'tokens_gated_out': 89, 'total_evictions': 768}
```

Zero overhead when disabled (default).

## Limitations

- 2x attention FLOPs from dual-stream differential attention. No fused Triton kernel yet.
- Fine-tuning required. Cold-swap doesn't work -- differential attention reshapes activations.
- Phase 2 training is ~20x slower than Phase 1 (gradient flow through bank updates).
- ~30-45% PPL gap between bounded and unbounded. Real information loss from context compression.
- No distributed cache sharding. No quantized KV storage.

## Links

- Paper: [Zenodo]
- Trained weights: [HuggingFace]
- Technical deep-dive: [HuggingFace article]

## Citation

```bibtex
@software{coda_gqa_l_2026,
  title  = {CoDA-GQA-L: Bounded-Memory Differential Attention
            with Value-Routed Landmark Banks},
  author = {Maio, Anthony},
  year   = {2026},
}
```

## License

MIT
