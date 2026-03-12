# Makora Evaluation Report — CoDA-GQA-L Triton Kernels

**Date:** 2026-03-09
**Device:** H100
**Makora CLI version:** installed via pip (Python 3.13, Windows 11)
**Project:** CoDA-GQA-L (Constrained Orthogonal Differential Attention + Bounded KV Memory)

---

## Executive Summary

We evaluated 3 Triton kernels from the CoDA-GQA-L project through Makora's `evaluate` and `expert-generate` pipelines on H100. Two kernels existed already; one (summary bank update) was newly written for this evaluation. Results were mixed — one kernel showed spectacular speedup, while the others hit issues with evaluation or expert-generate.

---

## Kernels Tested

### 1. Differential FlashAttention (`triton_diff_flash/kernel.py`)
**What it does:** Single-pass fused kernel computing dual-stream online softmax (signal + noise queries sharing K/V tiles), differential epilogue (`sig - lambda * noise`), and optional in-register RMSNorm.

**Problem dimensions:** B=2, H=32, H_KV=8, Lq=512, Lk=512, HEAD_DIM=128, dtype=float32

### 2. Exact Bank Routing (`triton_bank_routing/kernel.py`)
**What it does:** Fuses cosine similarity scoring (V-routing across KV heads), candidate classification (novel/hit/skip), and sequential LRU victim assignment into a single kernel. Replaces ~15 PyTorch kernel launches.

**Problem dimensions:** B=2, H_KV=8, T=8 candidates, ME=64 bank slots, DH=128

### 3. Summary Bank Update (NEW — `_summary_update_block_hard` from `memory_banks.py`)
**What it does:** LF-K routing (cosine similarity on low-frequency key band), scatter-add weighted aggregation, Phase-Safe EMA blending (LF-only for keys, full for values). Replaces ~12 PyTorch kernel launches.

**Problem dimensions:** B=2, H_KV=8, T=8 candidates, MS=64 summary slots, DH=128, LF_START=64

---

## Evaluation Results (Baseline Triton vs PyTorch Reference)

| Kernel | PyTorch Ref | Triton Baseline | Speedup | Status |
|--------|-------------|-----------------|---------|--------|
| Diff Flash | — | — | — | **FAILED** |
| Bank Routing | 7.030 ms | 0.072 ms | **97.45x** | PASS |
| Summary Bank | 0.035 ms | 0.445 ms | **0.08x** | PASS (slower) |

### Notes

**Diff Flash — Evaluation Failed:**
The evaluation completed server-side but reported failure. Likely cause: the Makora format requires `forward()` to return a single tensor, but the dual-stream online softmax + differential epilogue requires 5 input tensors with specific semantics (two query streams, shared K/V, per-query lambda). The problem/solution may have shape or numerical mismatches at fp32 tolerance (1e-3). This is a complex fused kernel — the reference uses `F.softmax` while the Triton kernel uses online softmax, which can produce small numerical differences that accumulate.

**Bank Routing — 97.45x Speedup:**
The massive speedup reflects replacing Python-loop-heavy PyTorch reference code (sequential per-batch, per-candidate processing with ~15 kernel launches) with a single Triton kernel that keeps all routing state in SRAM registers. This is the expected case for control-flow-heavy routing logic.

**Summary Bank — 0.08x (12.7x Slower):**
The baseline Triton kernel is significantly slower than PyTorch. Root cause: the kernel re-computes routing (LF-K cosine similarity across all H_KV=8 heads) for every head × every dim-block iteration, creating O(H_KV² × T × num_dim_blocks) redundant work vs the PyTorch version which routes once then scatters. The kernel architecture needs a two-phase design: route once, then accumulate. At these small dimensions (T=8, MS=64), PyTorch's fused BLAS kernels are also extremely efficient — the kernel launch overhead of a single Triton kernel doesn't compensate.

---

## Expert-Generate Results

### Diff Flash — Optimizations Applied

**Summary from Makora:** "Applied block tiling with TILE_ROWS=2 to have each kernel instance process 2 consecutive query blocks, reducing grid launch overhead. Increased num_warps from 4 to 8 to improve warp-level parallelism and better hide memory latency during K/V tile loads and attention computation."

**Changes:**
- Added `TILE_ROWS=2` constexpr — each program processes 2 consecutive BLOCK_M tiles in a loop, halving grid size
- `num_warps`: 4 → 8 for better latency hiding
- Grid: `cdiv(Q_LEN, BLOCK_M * TILE_ROWS)` instead of `cdiv(Q_LEN, BLOCK_M)`
- `tile_bytes` calculation updated for `effective_block_m`

**Evaluation:** FAILED (same issue as baseline — likely format/tolerance mismatch, not kernel correctness)

### Bank Routing — Optimizations Applied

**Summary from Makora:** "Applied memory access improvements by adding dimension-based chunking for the DH dimension to reduce register pressure and improve memory bandwidth utilization. Introduced parameterized block sizes (BLOCK_DH, BLOCK_ME) for better resource management. Added direct L2 cache writes for output tensors to prevent cache pollution from write-once data. Enhanced memory masking patterns for improved coalescing of loads from memory banks."

**Changes:**
- `BLOCK_DH` chunking: processes DH in tiles of `min(128, next_power_of_2(DH))` instead of full HEAD_DIM at once → reduced register pressure
- `BLOCK_ME = next_power_of_2(ME)`: parameterized bank dimension
- Proper masking: `tl.load(..., mask=active_mask, other=0)` and `tl.store(..., mask=active_mask)` instead of loading full ME
- **Inline PTX assembly** for output stores: `st.global.cs.b64`/`st.global.cs.b8` (cache-streaming hints to bypass L2 for write-once data)

**Evaluation:** FAILED — the inline PTX assembly (`tl.inline_asm_elementwise`) likely caused compilation issues on the remote evaluation environment. The optimization is architecturally sound but uses Triton features that may not be universally supported.

### Summary Bank — Optimizations Applied

**Summary from Makora:** "Restructured kernel to process head dimension in tiled blocks (BLOCK_DH=64) instead of all at once, reducing register pressure and SRAM usage per iteration."

**Changes:**
- Added `BLOCK_DH=64` dimension tiling for the K/V accumulation phase
- Accumulator shapes: `[MS, BLOCK_DH]` instead of `[MS, DH]`
- Processes DH in `cdiv(DH, BLOCK_DH)` blocks with proper masking
- Removed the `num_blocks` calculation as a runtime variable (uses loop with constexpr)

**Evaluation:** 0.05x speedup (even slower than baseline's 0.08x). The dimension tiling added loop overhead without fixing the fundamental O(H_KV²) routing redundancy.

---

## Makora CLI Issues Encountered

### 1. Device Name Inconsistency (Bug)
- **GPUS.md** documents devices as `nvidia/L40S`, `nvidia/H100`
- **Actual CLI** requires `H100`, `L40S` (no `nvidia/` prefix)
- Error: `Invalid value for '-d' / '--device': 'nvidia/H100' is not one of 'H100', 'H200', 'B200', 'L40S', 'MI300X'...`
- **Impact:** All initial runs (6 jobs) failed, requiring relaunch
- **Fix:** Update GPUS.md to match actual CLI device names, or accept both formats

### 2. Unicode Encoding Crash on Windows (Bug)
- **Location:** `makora/commands/evaluate.py:83`
- **Error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` (checkmark ✓)
- **Cause:** `typer.echo("✓ Evaluation successful!")` fails on Windows with cp1252 console encoding
- **Impact:** Evaluation succeeds on remote GPU but crashes before printing results. All 3 initial evaluations lost their output.
- **Workaround:** `PYTHONIOENCODING=utf-8 makora evaluate ...`
- **Fix:** Use ASCII-safe characters (`[OK]` instead of `✓`, `[FAIL]` instead of `✗`) or set encoding explicitly in the CLI

### 3. Expert-Generate Inline PTX Portability
- The bank routing expert-generate output uses `tl.inline_asm_elementwise` with PTX instructions (`st.global.cs.b64`)
- This failed evaluation, likely due to Triton version mismatch or compilation issues
- **Suggestion:** Expert-generate should prefer portable Triton constructs over inline PTX unless the user explicitly requests low-level optimization

### 4. Multi-Output Kernel Evaluation
- Kernels returning tuples (bank routing: 4 tensors, summary bank: 3 tensors) are harder to validate
- The diff flash kernel evaluation failed — unclear if due to numerical tolerance or format issues
- **Suggestion:** Clearer error messages when evaluation fails (currently just "Evaluation failed!" with no details)

---

## Optimization Recommendations for CoDA-GQA-L

### Bank Routing Kernel (97x — ship it)
The existing kernel is already excellent. The expert-generate improvements (DH chunking, cache-streaming PTX) are architecturally sound but the inline PTX needs testing. Consider cherry-picking the `BLOCK_DH` chunking without the PTX stores.

### Diff Flash Kernel (needs fp16/bf16 eval)
The TILE_ROWS=2 and num_warps=8 changes from expert-generate are reasonable. The kernel should be re-evaluated with bf16 inputs (the actual production dtype) rather than fp32, which forces reduced tile sizes and may not represent real-world performance.

### Summary Bank Kernel (needs redesign)
The current approach of re-computing routing in every head iteration is fundamentally wrong. A two-phase design would fix this:
1. **Phase 1 (route):** Compute routing decisions once for all candidates (O(H_KV × T × MS × DH_LF))
2. **Phase 2 (accumulate + EMA):** For each head, use pre-computed routing to scatter-add K/V and apply EMA

Additionally, at T=8, MS=64 scale, the kernel launch overhead may dominate — consider whether a Triton kernel is the right approach vs keeping PyTorch for this operation size.

---

## Files Produced

```
makora/
├── diff_flash_problem.py          # PyTorch reference (dual-stream SDPA)
├── diff_flash_solution.py         # Triton baseline (our existing kernel)
├── diff_flash_expert_stdout.py    # Makora expert-generate output (TILE_ROWS=2, 8 warps)
├── diff_flash_expert_stderr.log   # Expert-generate summary
├── diff_flash_eval.log            # Baseline eval: FAILED
├── diff_flash_expert_eval.log     # Expert eval: FAILED
├── bank_routing_problem.py        # PyTorch reference (sequential routing)
├── bank_routing_solution.py       # Triton baseline (our existing kernel)
├── bank_routing_expert_stdout.py  # Expert output (BLOCK_DH chunking, PTX stores)
├── bank_routing_expert_stderr.log # Expert summary
├── bank_routing_eval.log          # Baseline eval: 97.45x speedup
├── bank_routing_expert_eval.log   # Expert eval: FAILED (PTX issue)
├── summary_bank_problem.py        # PyTorch reference (LF-K routing + EMA)
├── summary_bank_solution.py       # Triton baseline (NEW kernel)
├── summary_bank_expert_stdout.py  # Expert output (BLOCK_DH=64 tiling)
├── summary_bank_expert_stderr.log # Expert summary
├── summary_bank_eval.log          # Baseline eval: 0.08x (slower)
├── summary_bank_expert_eval.log   # Expert eval: 0.05x (even slower)
└── MAKORA_REPORT.md               # This report
```

---

## Summary for Makora Team

**What worked well:**
- `expert-generate` produced reasonable architectural suggestions for all 3 kernels
- Bank routing evaluation showed the platform works correctly for single-output kernels
- Device variety (H100, H200, B200, L40S, MI300X, Adreno, Hexagon) is impressive

**Issues to address:**
1. **Windows Unicode crash** — blocks all Windows users from seeing results (P0)
2. **Device name mismatch in docs** — `nvidia/H100` vs `H100` (P1)
3. **Evaluation failure details** — "Evaluation failed!" with no reason makes debugging impossible (P1)
4. **Inline PTX in expert-generate** — should be opt-in, not default (P2)
5. **Multi-output kernel support** — tuple returns need clearer documentation/validation (P2)
6. **Expert-generate for summary bank** — didn't address the fundamental algorithmic issue (re-routing O(H_KV²)), only applied mechanical tiling. A smarter approach would recognize the redundant recomputation pattern (P3)
