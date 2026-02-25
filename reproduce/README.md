# Reproducing CoDA-GQA-L Results

This directory contains everything needed to reproduce every experiment in the paper.

## Contents

| File | Description |
|------|-------------|
| `reproduce_results.ipynb` | Clean notebook reproducing all results (recommended) |
| `run_all.sh` | One-command shell script for all experiments |
| `configs.json` | Machine-readable experiment configurations |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
# From repo root
pip install -e .
pip install -r reproduce/requirements.txt

# Option A: Run the notebook
jupyter notebook reproduce/reproduce_results.ipynb

# Option B: Run everything from the command line
bash reproduce/run_all.sh                     # eval-only (~30 min)
TRAIN=1 bash reproduce/run_all.sh             # with training (~6 hours)
```

## Pre-trained Checkpoint (Skip Training)

Our trained Mistral-7B checkpoint is available on HuggingFace:

```python
from huggingface_hub import hf_hub_download
adapter_path = hf_hub_download("anthonym21/Mistral-7B-v0.3-CoDA-GQA-L", "coda_adapters.pt")
```

Using the published checkpoint, you can reproduce all eval results (Tables 1-3, needle, ablations) without any GPU training time.

## Hardware Requirements

| Experiment | VRAM | Time | Notes |
|-----------|------|------|-------|
| Smoke test (SmolLM2-135M) | 8GB | 5 min | Any GPU |
| Cold-swap PPL | 24GB+ | 30 min | No training |
| Trained model eval | 24GB+ | 30 min | Uses published checkpoint |
| Throughput benchmarks | Any | 2 min | Standalone test model |
| Needle-in-haystack | Any | 1 min | Standalone test model |
| Two-phase training | 40GB+ | 3-6 hrs | H100/H200 recommended |
| Diff attn ablation | 40GB+ | ~4 hrs | H100/H200 recommended |

**Verified hardware**: NVIDIA H200 NVL (140GB), CUDA 12.8, PyTorch 2.8, Triton 3.4.

## Expected Results

### Perplexity (WikiText-2, Mistral-7B)

| Configuration | PPL | vs Baseline | KV Cache |
|--------------|----:|----------:|-------:|
| Baseline | 4.81 | -- | O(L) |
| CoDA unbounded | 5.38 | +11.7% | O(L) |
| Bounded medium | 5.94 | +23.5% | 218KB |
| Bounded tiny | 6.31 | +31.2% | 109KB |
| Window-only | 6.22 | +29.3% | 129KB |

### Context-Length Scaling (bounded medium, 8K-trained)

| Context | 512 | 1024 | 2048 | 4096 | 8192 |
|---------|----:|-----:|-----:|-----:|-----:|
| PPL | 6.36 | 6.09 | 5.94 | 5.95 | 6.87 |

### Needle-in-Haystack

100% retention at 256, 1K, 4K, and 16K tokens (cosine similarity >= 0.999).

### Differential Attention Ablation

| Model | Bounded PPL |
|-------|------------:|
| GQA + bounded (no diff attn) | 5.81 |
| CoDA + bounded (diff attn) | 5.56 |

Differential attention improves bounded PPL by 4.3%.

---

## Experiment Details

### 1. Smoke test (~5 min, any GPU)

Verify weight transfer correctness on SmolLM2-135M:

```bash
python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M \
    --experiment forward-check --dtype fp32
```

### 2. Cold-swap perplexity (no training)

Measures structural overhead of swapping attention layers without fine-tuning.
Use `--dtype fp32` for cold-swap (bf16 rounding compounds through 32 layers):

```bash
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity --dtype fp32
```

### 3. Two-phase training

**Phase 1** (unbounded, 2000 steps): teaches differential attention.
**Phase 2** (bounded, 600 steps): adapts to fixed KV cache.

Training at `--seq-len 8192` is critical. Models trained at 2048 show catastrophic PPL blowup at longer contexts.

```bash
# Full training from scratch (~3-6 hours, 40GB+ VRAM)
python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
    --max-steps 2000 --bounded-steps 600 --bounded-config medium \
    --seq-len 8192 --batch-size 1 --grad-accum 8 \
    --head-norm-mode identity --dtype bf16 \
    --output-dir results/training

# Resume Phase 2 from published Phase 1 checkpoint
python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
    --adapter-weights path/to/coda_adapters.pt \
    --max-steps 0 --bounded-steps 600 --bounded-config medium \
    --seq-len 8192 --batch-size 1 --grad-accum 8 \
    --head-norm-mode identity --dtype bf16 \
    --output-dir results/phase2_only
```

Note: gradient checkpointing is automatically disabled for Phase 2 (in-place state ops are incompatible).

### 4. Trained model evaluation

```bash
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity \
    --adapter-weights path/to/coda_adapters.pt \
    --head-norm-mode identity --dtype bf16
```

### 5. Throughput & KV cache memory

Standalone benchmark (no trained model needed):

```bash
python benchmarks/run_suite.py
python benchmarks/render_tables.py
```

### 6. Needle-in-haystack retention

```bash
python examples/needle_demo.py
RUN_LONG=1 python examples/needle_demo.py  # includes L=16384
```

### 7. Ablation: differential attention

Train two Mistral-7B configs head-to-head:

```bash
# GQA + bounded (no differential attention)
python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
    --max-steps 0 --bounded-steps 1200 --no-differential \
    --bounded-config medium --seq-len 2048 \
    --batch-size 1 --grad-accum 8 --head-norm-mode identity \
    --output-dir results/ablation_gqa

# CoDA + bounded (full differential attention)
python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
    --max-steps 1200 --bounded-steps 1200 \
    --bounded-config medium --seq-len 2048 \
    --batch-size 1 --grad-accum 8 --head-norm-mode identity \
    --output-dir results/ablation_coda
```

Or use the provided script: `bash benchmarks/run_ablation_h100.sh`

### 8. Ablation: memory bank configurations

```bash
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity \
    --bounded-configs window-only,tiny,medium,large \
    --adapter-weights path/to/coda_adapters.pt \
    --head-norm-mode identity --dtype bf16
```

## Script Reference

| Script | What it does | Time |
|--------|-------------|------|
| `benchmarks/eval_llm.py` | PPL eval (baseline, cold-swap, bounded, trained) | 10-30 min |
| `benchmarks/train_coda.py` | Two-phase training (unbounded + bounded) | 1-6 hours |
| `benchmarks/run_suite.py` | Throughput + memory benchmarks (5 configs) | ~2 min |
| `benchmarks/render_tables.py` | Render results/*.json as markdown tables | seconds |
| `benchmarks/forward_check_generic.py` | Weight transfer validation | ~1 min |
| `benchmarks/run_ablation_h100.sh` | GQA vs CoDA ablation (H100) | ~4 hours |
| `examples/needle_demo.py` | Needle-in-haystack retention | ~1 min |
| `examples/bounded_generate.py` | Bounded text generation with metrics | ~1 min |

## Output Format

All eval scripts write JSON results to `results/`. Each JSON includes system metadata (GPU, CUDA version, PyTorch version, dtype, timestamp) alongside the experiment results. Use `benchmarks/render_tables.py` to produce markdown comparison tables from these JSONs.
