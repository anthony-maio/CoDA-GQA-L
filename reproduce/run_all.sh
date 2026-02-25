#!/bin/bash
# =============================================================================
# CoDA-GQA-L: Full reproduction pipeline
# =============================================================================
# Runs all non-training experiments. Add TRAIN=1 for the full pipeline
# including fine-tuning (adds several hours of GPU time).
#
# Usage:
#   bash reproduce/run_all.sh                          # eval-only (~30 min)
#   TRAIN=1 bash reproduce/run_all.sh                  # with training (~4 hours)
#   MODEL=HuggingFaceTB/SmolLM2-135M bash reproduce/run_all.sh  # small model
#
# Results go to results/reproduce_<timestamp>/
# =============================================================================

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/reproduce_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
SMOKE_MODEL="${SMOKE_MODEL:-HuggingFaceTB/SmolLM2-135M}"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "CoDA-GQA-L Reproduction"
echo "============================================================"
echo "  Results:     $RESULTS_DIR"
echo "  Model:       $MODEL"
echo "  Smoke model: $SMOKE_MODEL"
echo "  Training:    ${TRAIN:-0}"
echo ""

# Record environment
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU: CPU only')
" | tee "$RESULTS_DIR/environment.txt"

# --- Step 1: Forward equivalence (smoke test) ---
echo ""
echo "--- Step 1: Forward equivalence check ---"
python benchmarks/eval_llm.py --model "$SMOKE_MODEL" \
    --experiment forward-check --dtype fp32 --results-dir "$RESULTS_DIR"

# --- Step 2: Cold-swap perplexity ---
echo ""
echo "--- Step 2: Cold-swap perplexity ---"
python benchmarks/eval_llm.py --model "$MODEL" \
    --experiment perplexity --dtype fp32 --results-dir "$RESULTS_DIR"

# --- Step 3: Throughput benchmarks ---
echo ""
echo "--- Step 3: Throughput benchmarks ---"
python benchmarks/run_suite.py --results-dir "$RESULTS_DIR"

# --- Step 4: Needle retention ---
echo ""
echo "--- Step 4: Needle retention ---"
python examples/needle_demo.py 2>&1 | tee "$RESULTS_DIR/needle_demo.log"

# --- Step 5: Training + trained eval (opt-in) ---
# --- Step 5: Download published checkpoint OR train from scratch ---
ADAPTER_PATH="${ADAPTER_PATH:-}"

if [[ "${TRAIN:-0}" == "1" ]]; then
    echo ""
    echo "--- Step 5: Training (Phase 1 + Phase 2, seq_len=8192) ---"
    python benchmarks/train_coda.py --model "$MODEL" \
        --max-steps 2000 --bounded-steps 600 --bounded-config medium \
        --seq-len 8192 --batch-size 1 --grad-accum 8 \
        --head-norm-mode identity --dtype bf16 \
        --output-dir "$RESULTS_DIR/training"

    ADAPTER_PATH="$RESULTS_DIR/training/phase2/best/coda_adapters.pt"
else
    # Download published checkpoint
    echo ""
    echo "--- Step 5: Downloading published checkpoint ---"
    ADAPTER_PATH=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('anthonym21/Mistral-7B-v0.3-CoDA-GQA-L', 'coda_adapters.pt'))
")
    echo "  Checkpoint: $ADAPTER_PATH"
fi

if [[ -n "$ADAPTER_PATH" && -f "$ADAPTER_PATH" ]]; then
    echo ""
    echo "--- Step 6: Trained model evaluation ---"
    python benchmarks/eval_llm.py --model "$MODEL" --experiment perplexity \
        --adapter-weights "$ADAPTER_PATH" \
        --head-norm-mode identity --dtype bf16 --results-dir "$RESULTS_DIR"

    echo ""
    echo "--- Step 7: Memory bank config ablation ---"
    python benchmarks/eval_llm.py --model "$MODEL" --experiment perplexity \
        --bounded-configs window-only,tiny,medium,large \
        --adapter-weights "$ADAPTER_PATH" \
        --head-norm-mode identity --dtype bf16 --results-dir "$RESULTS_DIR"
fi

# --- Render tables ---
echo ""
echo "--- Rendering tables ---"
python benchmarks/render_tables.py --results-dir "$RESULTS_DIR" || true

echo ""
echo "============================================================"
echo "Done. Results saved to: $RESULTS_DIR"
echo "============================================================"
ls -la "$RESULTS_DIR/"
