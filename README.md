# CoDA-GQA-L

**Bounded-memory differential attention that actually works.**

```bash
pip install coda-gqa-l
```

A 70B model serving 128K context burns 160 GB on KV cache alone. CoDA-GQA-L does it in 120 MB -- a fixed-size buffer that doesn't grow no matter how long the input is.

| Context | Standard KV (70B, 80 layers) | CoDA-GQA-L | Compression |
|---------|------------------------------|------------|-------------|
| 2K      | 2.56 GB                      | 120 MB     | 21.3x       |
| 32K     | 40 GB                        | 120 MB     | 341x        |
| 128K    | 160 GB                       | 120 MB     | **1,365x**  |

The mechanism replaces the O(L) KV cache with three fixed-size segments per layer:

- **Recent window** (W=256) -- ring buffer of exact recent tokens, FIFO eviction
- **Exact landmark bank** (Me=64) -- novelty-filtered LRU cache for important tokens a learned write gate decides are worth keeping
- **Summary landmark bank** (Ms=64) -- EMA prototypes compressing older context into semantic cluster centroids

384 slots per layer, always, whether you processed 2K or 128K tokens.

The bounded state is serializable. `torch.save()` it, load it a week later, query it without re-reading the original document. At 7B scale each state is 48 MB. We call this pattern **stateful neural databases**.

## How it works

Three ideas, briefly:

**Differential attention via orthogonal rotation.** Builds on Microsoft's Diff Transformer (Ye et al., 2024) but drops the second Wq projection. A learned per-head rotation produces the noise query from the signal query. Signal minus gated noise, one SDPA call (head-stacked), HeadwiseRMSNorm on the output. A 2x2 factorial ablation shows CoDA adds zero overhead unbounded but reduces the bounded penalty by 5.7x.

**Value-routing.** Keys have RoPE baked in and their cosine similarity is position-dependent -- same word at position 100 vs 5000 looks orthogonal in key-space. Memory banks route on Values instead (RoPE-free, pure semantic content). This is what makes deduplication and EMA blending actually work.

**Two-phase training.** Phase 1 trains unbounded differential attention. Phase 2 switches to bounded cache with gradient flow through evictions so the write gate learns what to keep. Without Phase 2, bounded eval is catastrophic (PPL 5.75 to 2,464 on Mistral 7B).

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

At 7B scale (32 layers): 48 MB per document state. 100 documents = 4.8 GB. Route queries to the right state on demand -- no vector database, no chunk-and-retrieve.

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

# Mistral 7B (~1.6 hours on H200)
python benchmarks/train_coda.py \
    --model mistralai/Mistral-7B-v0.3 \
    --max-steps 2000 --bounded-steps 600 \
    --bounded-lr-scale 0.5 --bounded-block-size 128 \
    --no-detach-evicted --batch-size 1 --grad-accum 8
```

**Bounded configs:**

| Config | Window | Exact | Summary | Total slots | KV Cache/Layer |
|--------|--------|-------|---------|-------------|----------------|
| tiny   | 128    | 32    | 32      | 192         | 108.9 KB       |
| medium | 256    | 64    | 64      | 384         | 217.9 KB       |
| large  | 512    | 128   | 128     | 768         | 3.0 MB         |

### Training results (Mistral-7B-v0.3 on H200 NVL)

| Phase | Steps | PPL start | PPL end | Throughput |
|-------|-------|-----------|---------|------------|
| Phase 1 (unbounded) | 2,000 | 23.50 | 5.75 | ~4,950 tok/s |
| Phase 2 (bounded, medium) | 600 | 27.88 | 6.31 | ~2,000 tok/s |

**23.5% PPL overhead** (+1.13 PPL) vs. the 4.81 baseline for **9.5x memory compression**. Total training time: ~1.6 hours on H200.

Phase 2 trains with `detach_evicted=False` so gradients flow through bank updates.

### Context-length scaling

Context-length degradation is remarkably flat between 1K and 4K:

| Context | Bounded PPL | vs 2K |
|---------|-------------|-------|
| 512     | 6.36        | +7.1% |
| 1,024   | 6.09        | +2.5% |
| 2,048   | 5.94        | ---   |
| 4,096   | 5.95        | +0.2% |
| 8,192   | 6.87        | +15.7% |

The model was trained at seq_len=8,192. Bounded PPL at 2K (5.94) and 4K (5.95) is nearly identical.

### Differential attention ablation (2x2 factorial)

A 2x2 factorial ablation isolates the interaction between differential attention and bounded memory. Both methods achieve identical unbounded PPL, but CoDA's bounded penalty is 5.7x smaller:

| Method | Unbounded PPL | Bounded PPL | Bounded Penalty |
|--------|---------------|-------------|-----------------|
| Standard GQA (no diff. attn) | 5.75 | 6.84 | +1.09 |
| CoDA (differential attn)     | 5.75 | 5.94 | +0.19 |

**Interaction effect: +0.90 PPL. Penalty reduction: 5.7x.** Differential attention adds zero overhead unbounded but reduces the information loss from context compression by nearly 6x. The two innovations are designed to work together.

### Dynamic bank expansion

Memory banks can be expanded at inference time (64 to 128 slots per bank) without retraining:

| Context | Fixed PPL (384 slots) | Expanded PPL (512 slots) | Improvement |
|---------|-----------------------|--------------------------|-------------|
| 2,048   | 5.94                  | 5.94                     | 0.0%        |
| 4,096   | 5.95                  | 5.96                     | -0.2%       |
| 8,192   | 6.87                  | 6.80                     | +1.0%       |

Marginal benefit at 8K. A model trained *with* expansion (medium-expand config) achieves 5.81 PPL.

## Benchmarks

All numbers from H200 NVL, bf16.

```bash
python benchmarks/run_suite.py                          # 5 configs, JSON output
python benchmarks/run_suite.py --embed-dim 4096 \       # 7B scale
    --num-heads 32 --num-kv-heads 8
python benchmarks/render_tables.py                      # markdown tables
```

### Memory (Mistral-7B, per layer, medium-cache W=256 Me=64 Ms=64)

| Seq Length | Standard KV | CoDA bounded | Compression |
|------------|-------------|--------------|-------------|
| 512        | 2.0 MB      | 1.5 MB       | 1.3x        |
| 1,024      | 4.0 MB      | 1.5 MB       | 2.7x        |
| 2,048      | 8.0 MB      | 1.5 MB       | 5.3x        |
| 4,096      | 16.0 MB     | 1.5 MB       | 10.7x       |
| 8,192      | 32.0 MB     | 1.5 MB       | 21.3x       |

Across all layers:

| Scenario | Standard KV | CoDA total | Compression |
|----------|-------------|------------|-------------|
| 7B, 2K ctx (32 layers)    | 512 MB  | 48 MB  | 10.7x       |
| 7B, 128K ctx (32 layers)  | 32 GB   | 48 MB  | **682x**    |
| 70B, 2K ctx (80 layers)   | 2.56 GB | 120 MB | 21.3x       |
| 70B, 128K ctx (80 layers) | 160 GB  | 120 MB | **1,365x**  |

### Throughput (D=512 test model, H200 NVL)

| Config | Prefill @4096 | Decode | KV Cache | Peak VRAM |
|--------|-------------:|-------:|---------:|----------:|
| Baseline GQA       | 15.2M tok/s | 5,504 tok/s | 2.0 MB  | 59.8 MB |
| CoDA unbounded     | 8.0M tok/s  | 3,449 tok/s | 2.0 MB  | 75.9 MB |
| CoDA medium-cache  | 210K tok/s  | 2,283 tok/s | 218 KB  | 52.0 MB |
| CoDA window-only   | 664K tok/s  | 2,298 tok/s | 129 KB  | 51.9 MB |
| CoDA tiny-cache    | 206K tok/s  | 833 tok/s   | 109 KB  | 51.9 MB |

Bounded prefill throughput is flat regardless of sequence length. Bounded VRAM is lower than unbounded (52 MB vs 76 MB) because the smaller KV cache more than offsets differential attention overhead.

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
  attention.py            CoDAGQALandmarkPerf2 (main module)
  memory_banks.py         Exact/summary bank updates (mixin)
  state.py                KV buffer + ring metadata dataclass
  primitives.py           RoPE, RMSNorm, GQA utils
  baseline.py             Unbounded CoDAGQA + standard GQA
  llama_adapter.py        Drop-in for Llama/Mistral/SmolLM
  eve_adapter.py          Drop-in for Eve-2 MoE
  triton_diff_flash/      Fused differential FlashAttention kernel
  triton_bank_routing/    Fused exact-bank routing kernel

benchmarks/
  train_coda.py           Two-phase training pipeline
  run_suite.py            Perf benchmarks (5 configs, JSON output)
  eval_llm.py             Full-model perplexity evaluation
  run_ablation_h100.sh    Differential attention ablation

tests/                    60 tests
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

## Custom Triton kernels

Two fused kernels address throughput bottlenecks (verified on H200 NVL with Triton 3.4.0):

- **`triton_diff_flash`** -- Fused differential FlashAttention forward kernel. Single HBM pass computes both signal and noise attention with online softmax, applies the differential epilogue and optional HeadwiseRMSNorm in-register.
- **`triton_bank_routing`** -- Fused exact-bank routing kernel replacing ~15 PyTorch micro-kernel launches (matmul, mean, max, masked_fill, topk, cumsum, clamp, gather, scatter, where) with a single deterministic GPU kernel.

Install with: `pip install coda-gqa-l[triton]`

## Limitations

- 2x attention FLOPs from dual-stream differential attention. Fused Triton forward kernel partially addresses this; backward pass kernel is future work.
- Fine-tuning required. Cold-swap doesn't work -- differential attention reshapes activations (cold-swap PPL: 2,464).
- +23.5% PPL gap between bounded and unbounded at 7B scale. Real information loss from context compression.
- Configuration sensitivity: the bounded config used at inference must match training (large-cache trained with medium yields worse PPL).
- No distributed cache sharding. No quantized KV storage.

## Links

- Paper: [arXiv / Preprint](https://github.com/anthony-maio/CoDA-GQA-L/tree/main/paper)
- Trained weights: [anthonym21/Mistral-7B-v0.3-CoDA-GQA-L](https://huggingface.co/anthonym21/Mistral-7B-v0.3-CoDA-GQA-L)
- PyPI: `pip install coda-gqa-l`

## Citation

```bibtex
@software{coda_gqa_l_2026,
  title  = {CoDA-GQA-L: Constrained Orthogonal Differential Attention
            with Grouped-Query Value-Routed Landmark Banks},
  author = {Maio, Anthony},
  year   = {2026},
}
```

## License

MIT
