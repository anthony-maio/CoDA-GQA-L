# Reproducing CoDA-GQA-L results

This directory contains instructions and scripts to reproduce every experiment in the paper. All commands assume you're at the repo root.

## Setup

```bash
pip install -e .
pip install transformers datasets accelerate huggingface_hub
```

Hardware: 8+ GB VRAM for SmolLM2-135M experiments, 40+ GB for Mistral-7B (H100 or A100 recommended).

## Quick smoke test (~5 min, any GPU)

Verify that CoDA weight transfer works and training runs without errors:

```bash
# Forward equivalence: proves weight mapping is correct
python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M \
    --experiment forward-check --dtype fp32

# Short training run
python benchmarks/train_coda.py --model HuggingFaceTB/SmolLM2-135M \
    --max-steps 200 --eval-every 100
```

## Table 1: Cold-swap perplexity

No training needed. Swaps attention layers and measures PPL degradation.
Use `--dtype fp32` for accurate cold-swap results (bf16 rounding compounds through 30+ layers).

```bash
# SmolLM2-135M (~10 min)
python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M \
    --experiment perplexity --dtype fp32

# Mistral-7B-v0.3 (~30 min, needs 24+ GB VRAM)
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity --dtype fp32
```

Results saved to `results/` as JSON with baseline, unbounded, and bounded PPL.

## Table 2: Fine-tuned perplexity

Two-phase training: Phase 1 (unbounded) teaches differential attention, Phase 2 (bounded) adapts to limited KV cache.

```bash
# Train Mistral-7B (~3-6 hours on H100)
python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
    --max-steps 2000 --bounded-steps 1000 --bounded-config medium \
    --batch-size 2 --grad-accum 4 --output-dir results/mistral7b_train

# Evaluate trained model
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity \
    --adapter-weights results/mistral7b_train/best \
    --head-norm-mode identity --dtype bf16
```

Training configs: `--freeze attention` (default, ~20% params trainable), `--freeze coda-only` (~50K params, fastest).

## Table 3: Throughput and KV cache memory

Standalone benchmark -- no trained model needed. Compares 5 attention configs:

```bash
python benchmarks/run_suite.py
python benchmarks/render_tables.py
```

Configs: baseline GQA, CoDA unbounded, tiny-cache (W=128/Me=32/Ms=32), medium-cache (W=256/Me=64/Ms=64), window-only (W=256/Me=0/Ms=0).

## Needle-in-haystack retention

Verifies that important tokens survive eviction from the recent window via the exact memory bank:

```bash
python examples/needle_demo.py
RUN_LONG=1 python examples/needle_demo.py  # includes L=16384
```

## Ablation: Differential attention contribution

The key ablation question: does differential attention actually help, or would standard GQA with the same bounded memory banks achieve similar PPL?

This script trains two Mistral-7B configs head-to-head on an H100:
1. Standard GQA + bounded cache (no differential attention, `--no-differential`)
2. CoDA + bounded cache (with differential attention)

```bash
bash benchmarks/run_ablation_h100.sh
```

See [benchmarks/run_ablation_h100.sh](../benchmarks/run_ablation_h100.sh) for full details. Designed for RunPod `/workspace`. Results go to `/workspace/ablation_results/`.

## Ablation: Memory bank configurations

Sweep bounded configs to measure the contribution of exact and summary banks:

```bash
python benchmarks/eval_llm.py --model mistralai/Mistral-7B-v0.3 \
    --experiment perplexity \
    --bounded-configs window-only,tiny,medium,large \
    --adapter-weights results/mistral7b_train/best --dtype bf16
```

Configs: window-only (no banks), tiny (W=128+Me=32+Ms=32), medium (W=256+Me=64+Ms=64), large (W=512+Me=128+Ms=128).

## Memory bank metrics

Collect hit rates, fill ratios, and eviction counts during bounded generation:

```bash
python examples/bounded_generate.py --collect-metrics --max-new-tokens 500
```

## Eve-2 integration (optional)

Benchmarks against the Eve-2 MoE architecture (272M params, D=512, H=8):

```bash
python benchmarks/eval_eve.py --experiment forward-check
python benchmarks/eval_eve.py --experiment perplexity
python benchmarks/eval_eve.py --experiment needle
python benchmarks/eval_eve.py --experiment all
```

## One-command runner

Run all non-training experiments with a single command:

```bash
bash reproduce/run_all.sh
```

To include training (adds several hours of GPU time):

```bash
TRAIN=1 bash reproduce/run_all.sh
```

Override the model:

```bash
MODEL=HuggingFaceTB/SmolLM2-135M bash reproduce/run_all.sh
```

## Script reference

| Script | What it does | Time |
|--------|-------------|------|
| `benchmarks/eval_llm.py` | PPL eval (baseline, cold-swap, bounded, trained) | 10-30 min |
| `benchmarks/train_coda.py` | Two-phase training (unbounded + bounded) | 1-6 hours |
| `benchmarks/run_suite.py` | Throughput + memory benchmarks (5 configs) | ~2 min |
| `benchmarks/render_tables.py` | Render results/*.json as markdown tables | seconds |
| `benchmarks/forward_check_generic.py` | Weight transfer validation | ~1 min |
| `benchmarks/run_ablation_h100.sh` | GQA vs CoDA ablation (H100) | ~4 hours |
| `benchmarks/eval_eve.py` | Eve-2 integration benchmarks | 10-30 min |
| `examples/needle_demo.py` | Needle-in-haystack retention | ~1 min |
| `examples/bounded_generate.py` | Bounded text generation with metrics | ~1 min |

## Output format

All eval scripts write JSON results to `results/`. Each JSON includes system metadata (GPU, CUDA version, PyTorch version, dtype, timestamp) alongside the experiment results. Use `benchmarks/render_tables.py` to produce markdown comparison tables from these JSONs.
