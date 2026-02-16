# CoDA-GQA-L

**Bounded-memory differential attention that actually works.**

Standard transformers burn O(L) memory per layer caching keys and values. At 128K context, that's gigabytes of KV cache just sitting in VRAM. Sliding window attention caps memory but throws away everything outside the window. CoDA-GQA-L takes a different approach: it keeps a fixed-size buffer and *learns* what to remember.

The mechanism combines three ideas:

1. **Differential attention via orthogonal rotation** -- instead of a second Wq projection (like Microsoft's Diff Transformer), we rotate the signal query to get the noise query. Same sharpening effect, half the query parameters. A learned per-token gate `lambda` controls how much noise to subtract.

2. **Dual memory banks** -- when tokens fall off the recent window, they're routed to either an *exact bank* (LRU cache for important/unique tokens) or a *summary bank* (EMA prototypes that compress repeated patterns). A learned write gate decides what's worth keeping.

3. **Value-routing** -- memory bank matching uses Values, not Keys, because Keys have RoPE baked in and their cosine similarity is position-dependent. Same word at position 100 vs 5000 looks completely different in key-space. Values are position-free, so deduplication and EMA blending actually work.

The result: O(W + Me + Ms) memory per layer, regardless of sequence length. For typical configs that's ~100-200KB per layer vs megabytes for unbounded.

## What's Here

This repo is the full implementation -- attention module, training pipeline, benchmark suite, and model adapters. It's research code that's been beaten into shape with 56 tests.

```
src/coda_gqa_l/
  attention.py       The main module (CoDAGQALandmarkPerf2)
  memory_banks.py    Exact/summary bank update logic (mixin)
  state.py           KV buffer + ring metadata dataclass
  primitives.py      RoPE, RMSNorm, GQA utils
  baseline.py        Unbounded CoDAGQA + standard GQA baselines
  llama_adapter.py   Drop-in for Llama/Mistral/SmolLM models
  eve_adapter.py     Drop-in for Eve-2 MoE

benchmarks/
  train_coda.py      Two-phase training pipeline (the important one)
  run_suite.py       Perf benchmarks across configs
  eval_eve.py        Eve-2 integration eval
  bench.py           Quick single-config timing

tests/               56 tests (correctness, determinism, edge cases, invariants)
```

## Installation

```bash
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

Requires PyTorch >= 2.0. CUDA + bf16 recommended; works on CPU in fp32.

## Quick Start

```python
import torch
from coda_gqa_l import CoDAGQALandmarkPerf2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

model = CoDAGQALandmarkPerf2(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,          # GQA: 4x head sharing
    window=256,              # Recent ring buffer
    num_landmarks_exact=64,  # Exact landmark slots
    num_landmarks_summary=64,# EMA summary slots
).to(device=device, dtype=dtype).eval()

B = 1
state = model.init_state(batch_size=B, device=device, dtype=dtype)

# Prefill a prompt
prompt = torch.randn(B, 2048, 512, device=device, dtype=dtype)
y, state = model.prefill_chunked(prompt, state, block_size=256)

# Decode tokens one at a time -- memory stays constant
for _ in range(1000):
    x_t = torch.randn(B, 1, 512, device=device, dtype=dtype)
    y_t, state = model.step(x_t, state)

print(f"Cache: {model.cache_bytes(batch_size=B, dtype=dtype):,} bytes")
print(f"Tokens processed: {state.pos}")  # 3048, cache didn't grow
```

## Training: Two-Phase Protocol

This is the key finding. You can't just swap in bounded attention and expect it to work -- the memory banks need to learn what to keep.

**The problem**: Training with unbounded attention (Phase 1) then evaluating with bounded produces catastrophic perplexity. On Mistral 7B: baseline PPL 4.81, CoDA unbounded 5.62, CoDA bounded (cold-swap) **2464**. The memory banks are untrained, so they provide zero useful context.

**The fix**: Phase 2. After Phase 1 teaches the model differential attention with full context, Phase 2 switches to bounded KV cache so the model adapts its attention patterns to limited memory.

```bash
# Quick smoke test (~5 min on a GPU)
python benchmarks/train_coda.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --max-steps 200 --bounded-steps 100 --bounded-config medium

# Mistral 7B on H100 (~6 hours total)
python benchmarks/train_coda.py \
    --model mistralai/Mistral-7B-v0.3 \
    --max-steps 2000 --bounded-steps 1000 \
    --bounded-config medium \
    --batch-size 2 --grad-accum 4

# Llama 3.1 8B
python benchmarks/train_coda.py \
    --model meta-llama/Llama-3.1-8B \
    --max-steps 3000 --bounded-steps 2000 \
    --bounded-config large \
    --batch-size 1 --grad-accum 8
```

**Bounded configs:**

| Config | Window | Exact | Summary | Total Slots |
|--------|--------|-------|---------|-------------|
| tiny   | 128    | 32    | 32      | 192         |
| medium | 256    | 64    | 64      | 384         |
| large  | 512    | 128   | 128     | 768         |

Phase 2 uses 0.1x the Phase 1 learning rate and trains the model to work within the bounded cache. The write gate and EMA parameters start at sensible defaults and adapt from there.

## Drop-In Model Adapters

### Llama Family (Llama 2/3, Mistral, SmolLM, etc.)

```python
from coda_gqa_l import LlamaCoDAAdapter

# Weight transfer is 1:1 (Llama already uses separate Q/K/V/O)
adapter = LlamaCoDAAdapter.from_llama_attention(
    llama_block.self_attn,
    bounded=False,  # unbounded for training, True for inference
)
llama_block.self_attn = adapter
```

### Eve-2 MoE

```python
from coda_gqa_l import EveCoDAAdapter

adapter = EveCoDAAdapter.from_eve_attention(
    eve_block.attn,
    bounded=False,
)
eve_block.attn = adapter
```

Eve uses a fused QKV projection, so the adapter splits the weights automatically.

## Architecture Details

### Attention Flow

```
x -> q_proj -> RoPE(q) -> q_signal
                       \-> R(theta) -> q_noise    # orthogonal rotation

x -> k_proj -> RoPE(k) -> k  (stored in buffer)
x -> v_proj -> v             (stored in buffer)

# Single SDPA call with head-stacked queries
SDPA([q_signal; q_noise], k_buf, v_buf)
  -> split -> out_sig, out_noise
  -> out_sig - lambda * out_noise
  -> HeadwiseRMSNorm -> o_proj -> output
```

Signal and noise queries are stacked along the head dimension for one SDPA call instead of two. The orthogonal rotation `R(theta)` has `H * Dh/2` learnable angles per layer.

### Memory Buffer Layout

```
k_buf / v_buf: (B, Hkv, W + Me + Ms, Dh)

Slots:  [0..W-1]      [W..W+Me-1]      [W+Me..W+Me+Ms-1]
         recent ring    exact bank       summary bank
```

When the ring buffer is full, the oldest token gets evicted and routed:
- **Write gate check**: `g = sigmoid(W_g * x)` -- is this token worth remembering?
- **Exact bank**: Cosine similarity on Values. Novel? Insert. Duplicate? Update LRU.
- **Summary bank**: Cosine similarity on Values. EMA blend into best-matching prototype.

### Dense Packing

For B=1, valid slots are dense-packed before SDPA. This avoids the boolean attention mask that forces PyTorch into the slow Math SDPA backend, unlocking FlashAttention/MemEfficient kernels.

## Benchmarks

```bash
# Quick single-config
python benchmarks/bench.py

# Full suite (5 configs, JSON output)
python benchmarks/run_suite.py
python benchmarks/render_tables.py

# Include long sequences
RUN_LONG=1 python benchmarks/bench.py
```

### Memory

| Config | KV Cache / Layer | vs Unbounded @ L=2048 |
|--------|------------------|-----------------------|
| tiny (W=128, Me=32, Ms=32) | 96.9 KB | **~20x smaller** |
| medium (W=256, Me=64, Ms=64) | 193.9 KB | **~10x smaller** |
| unbounded baseline | ~2.0 MB | 1x |

## Correctness

The bounded code path is **mathematically identical** to unbounded when window >= sequence length (no evictions). Verified to float32 precision:

| W | L | Max Diff |
|---|---|----------|
| 512 | 512 | 1.2e-7 |
| 1024 | 1024 | 1.5e-7 |
| 2048 | 2048 | 1.8e-7 |

All quality degradation at smaller windows comes from context loss, not bugs.

```bash
python -m pytest tests/ -v  # 56 tests
```

## Known Limitations

- **2x attention FLOPs**: Two streams (signal + noise). A fused Triton kernel would help but doesn't exist yet.
- **Fine-tuning required**: Cold-swap doesn't work. The differential mechanism reshapes activations enough to break pre-trained representations without training.
- **No gradient through banks**: Evicted tokens are detached (`detach_evicted=True`). The model learns to work with bounded memory indirectly through attention output quality.
- **No distributed/quantized KV**: No tensor parallel, no INT8 keys. Future work.

## Baselines

```python
from coda_gqa_l import CoDAGQA, BaselineGQA

# Unbounded differential attention (same mechanism, O(L) cache)
coda = CoDAGQA(embed_dim=512, num_heads=8, num_kv_heads=2)
y, kv = coda(x, is_causal=True)

# Standard GQA (no differential, O(L) cache)
gqa = BaselineGQA(embed_dim=512, num_heads=8, num_kv_heads=2)
y, kv = gqa(x, is_causal=True)
```

## License

MIT
