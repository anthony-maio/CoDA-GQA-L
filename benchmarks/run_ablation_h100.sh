#!/bin/bash
# =============================================================================
# CoDA-GQA-L Ablation Study: Differential Attention Contribution
# =============================================================================
# Run this on H100/RunPod (/workspace).
#
# This script answers: "Does differential attention help the bounded cache,
# or does standard GQA with the same memory banks achieve similar PPL?"
#
# It runs two training configs and collects benchmark data:
#   1. GQA + bounded cache (no differential attention) -- the ablation
#   2. CoDA + bounded cache (differential attention)  -- the baseline
#   3. Bounded generation with --collect-metrics
#   4. Full benchmark suite (throughput tables)
#
# Results are saved to /workspace/ablation_results/
#
# Usage:
#   bash benchmarks/run_ablation_h100.sh
#
# Prerequisites: NVIDIA GPU with >= 40GB VRAM (H100, A100, etc.)
# =============================================================================

set -e

RESULTS_DIR="/workspace/ablation_results"
REPO_DIR="/workspace/CoDA-GQA-L"

echo "============================================================"
echo "CoDA-GQA-L Ablation Study"
echo "============================================================"
echo "Results directory: $RESULTS_DIR"
echo ""

# --- Setup ---
mkdir -p "$RESULTS_DIR"

# Clone or update repo
if [ -d "$REPO_DIR" ]; then
    echo "Updating existing repo..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    cd /workspace && git clone https://github.com/anthony-maio/CoDA-GQA-L.git
fi
cd "$REPO_DIR"

# Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install -e . 2>&1 | tail -1
pip install transformers accelerate datasets huggingface_hub 2>&1 | tail -1

# Record environment
echo ""
echo "--- Environment ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" | tee "$RESULTS_DIR/environment.txt"

# =============================================================================
# Ablation 1: GQA + bounded cache (NO differential attention)
# =============================================================================
echo ""
echo "============================================================"
echo "ABLATION 1: Standard GQA + Bounded Cache (no differential)"
echo "============================================================"
echo "This tests bounded memory banks without differential attention."
echo "Phase 1 skipped (--max-steps 0), Phase 2 only (1200 steps)."
echo ""

python benchmarks/train_coda.py \
    --model mistralai/Mistral-7B-v0.3 \
    --max-steps 0 \
    --bounded-steps 1200 \
    --bounded-config medium \
    --no-differential \
    --batch-size 1 \
    --grad-accum 8 \
    --seq-len 2048 \
    --lr 5e-5 \
    --lr-coda 1e-3 \
    --bounded-lr-scale 0.5 \
    --eval-every 200 \
    --save-every 400 \
    --gradient-checkpointing \
    --output-dir "$RESULTS_DIR/gqa_bounded" \
    2>&1 | tee "$RESULTS_DIR/ablation1_gqa_bounded.log"

# =============================================================================
# Ablation 2: CoDA + bounded cache (WITH differential attention)
# =============================================================================
echo ""
echo "============================================================"
echo "ABLATION 2: CoDA + Bounded Cache (with differential)"
echo "============================================================"
echo "This is the full CoDA-GQA-L pipeline for comparison."
echo "Phase 1: 1200 steps unbounded, Phase 2: 1200 steps bounded."
echo ""

python benchmarks/train_coda.py \
    --model mistralai/Mistral-7B-v0.3 \
    --max-steps 1200 \
    --bounded-steps 1200 \
    --bounded-config medium \
    --batch-size 1 \
    --grad-accum 8 \
    --seq-len 2048 \
    --lr 5e-5 \
    --lr-coda 1e-3 \
    --bounded-lr-scale 0.5 \
    --eval-every 200 \
    --save-every 400 \
    --gradient-checkpointing \
    --output-dir "$RESULTS_DIR/coda_bounded" \
    2>&1 | tee "$RESULTS_DIR/ablation2_coda_bounded.log"

# =============================================================================
# Collect results
# =============================================================================
echo ""
echo "============================================================"
echo "Collecting results..."
echo "============================================================"

python -c "
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')

def get_final_ppl(run_dir):
    log_path = run_dir / 'training_log.json'
    if not log_path.exists():
        # Check phase2 subdir
        log_path = run_dir / 'phase2' / 'training_log.json'
    if not log_path.exists():
        return None, None
    log = json.loads(log_path.read_text())
    eval_ppls = [e['eval_ppl'] for e in log if 'eval_ppl' in e]
    bounded_ppls = [e['bounded_ppl'] for e in log if 'bounded_ppl' in e]
    best_eval = min(eval_ppls) if eval_ppls else None
    best_bounded = min(bounded_ppls) if bounded_ppls else None
    return best_eval, best_bounded

# Parse results
gqa_eval, gqa_bounded = get_final_ppl(results_dir / 'gqa_bounded')
coda_eval, coda_bounded = get_final_ppl(results_dir / 'coda_bounded')

print()
print('=' * 60)
print('ABLATION RESULTS: Differential Attention Contribution')
print('=' * 60)
print()
print(f'{'Config':<35} {'Best Eval PPL':>15} {'Best Bounded PPL':>18}')
print('-' * 70)
if gqa_eval or gqa_bounded:
    print(f'{'GQA + bounded (no diff attn)':<35} {str(gqa_eval or \"N/A\"):>15} {str(gqa_bounded or \"N/A\"):>18}')
if coda_eval or coda_bounded:
    print(f'{'CoDA + bounded (diff attn)':<35} {str(coda_eval or \"N/A\"):>15} {str(coda_bounded or \"N/A\"):>18}')
print()

if gqa_bounded and coda_bounded:
    delta = gqa_bounded - coda_bounded
    pct = (delta / coda_bounded) * 100
    if delta > 0:
        print(f'Differential attention improves bounded PPL by {delta:.2f} ({pct:.1f}%)')
    elif delta < 0:
        print(f'Standard GQA is better by {abs(delta):.2f} ({abs(pct):.1f}%)')
    else:
        print('No difference between configs.')

# Save structured results
summary = {
    'gqa_bounded': {'best_eval_ppl': gqa_eval, 'best_bounded_ppl': gqa_bounded},
    'coda_bounded': {'best_eval_ppl': coda_eval, 'best_bounded_ppl': coda_bounded},
}
(results_dir / 'ablation_summary.json').write_text(json.dumps(summary, indent=2))
print(f'\nResults saved to {results_dir / \"ablation_summary.json\"}')
" 2>&1 | tee -a "$RESULTS_DIR/summary.txt"

# =============================================================================
# Additional benchmarks (if time permits)
# =============================================================================
echo ""
echo "============================================================"
echo "Running bounded generation with memory bank metrics..."
echo "============================================================"

python examples/bounded_generate.py \
    --collect-metrics \
    --max-new-tokens 500 \
    2>&1 | tee "$RESULTS_DIR/bounded_generate_metrics.log"

echo ""
echo "============================================================"
echo "Running benchmark suite..."
echo "============================================================"

python benchmarks/run_suite.py \
    2>&1 | tee "$RESULTS_DIR/benchmark_suite.log"

echo ""
echo "============================================================"
echo "All done! Results in: $RESULTS_DIR"
echo "============================================================"
ls -la "$RESULTS_DIR/"
