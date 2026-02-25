# Changelog

## [1.2.0] - 2026-02-25

### Added
- **Reproducibility notebook** (`reproduce/reproduce_results.ipynb`): Complete 2x2 factorial ablation with matched training conditions, replacing the earlier simple bounded-only comparison.
- **Phase 2 OOM auto-detection**: Automatically detects GPU VRAM and falls back to `seq_len=4096` on GPUs with <120 GB (e.g., H100 80 GB), preventing out-of-memory failures during bounded training.

### Changed
- **Differential attention ablation upgraded to 2x2 factorial design**: Both GQA and CoDA now trained with identical budgets (2,000 unbounded + 600 bounded steps). Results show identical unbounded PPL (5.75) with a 5.7x bounded penalty reduction (CoDA +0.19 vs GQA +1.09), demonstrating genuine synergy rather than the previously reported 4.3% additive improvement.
- Ablation script reference updated from `run_ablation_h100.sh` to `reproduce/reproduce_results.ipynb` Section 8.

### Key Results Update
- GQA unbounded: 5.75 PPL | GQA bounded: 6.84 PPL (penalty: +1.09)
- CoDA unbounded: 5.75 PPL | CoDA bounded: 5.94 PPL (penalty: +0.19)
- Interaction effect: +0.90 PPL | Penalty reduction factor: **5.7x**
- Identical unbounded baselines confirm zero overhead from differential attention

## [1.1.0] - 2026-02-19

### Added
- **Custom Triton kernels**: `triton_diff_flash` (fused differential FlashAttention forward) and `triton_bank_routing` (fused exact-bank routing replacing ~15 PyTorch launches). Both verified on H200 NVL with Triton 3.4.0.
- **Dynamic bank expansion**: Inference-time expansion from 64 to 128 slots per bank without retraining. Provides +1.0% improvement at 8K context.
- **`LlamaCoDAAdapter.swap_llama_layers()` classmethod**: Convenience method for swapping all attention layers in a Llama-family model at once.
- **`--adapter-weights` flag** for `train_coda.py`: Enables resuming Phase 2 training from Phase 1 checkpoint.
- **`--no-differential` ablation flag**: Train with standard GQA + bounded banks (no differential attention) for controlled ablation studies.
- **`eval_llm.py`**: Full-model perplexity evaluation across configurations.
- **`run_ablation_h100.sh`**: Differential attention ablation script for H100/H200.
- **PTX fallback for Blackwell+ GPUs** (sm_120).

### Fixed
- Autograd: `clone()` without `detach()` to preserve cross-chunk gradients during Phase 2.
- Autograd: `detach+clone` state buffers between SDPA and writes to prevent in-place modification errors.
- Gradient checkpointing incompatibility with bounded Phase 2 training.
- Triton kernel type mismatch (`any_used` int1 vs int32).
- `novel_keep` UnboundLocalError when Triton path is active.
- Checkpoint param counting for nested state dicts.
- Triton kernel package discovery and registration in setuptools.
- Winner-take-all scatter routing replaced with deterministic assignment.
- CPU-GPU sync issues in bank update path.

### Changed
- Triton kernels moved into `coda_gqa_l` as proper subpackages (from external `kernels/` directory).
- Prefill block size increased to 1024 with projections hoisted out of chunk loop.
- Stacked K/V duplication replaced with two SDPA calls sharing K/V tensors.
- Bounded attention uses `causal_lower_right` for prefill FlashAttention (B==1).

### Training Results (Mistral-7B-v0.3)
- Phase 1 (unbounded, 2,000 steps): PPL 23.50 -> 5.75
- Phase 2 (bounded medium, 600 steps): PPL 27.88 -> 6.31
- Bounded PPL overhead: +23.5% vs. baseline (4.81)
- 100% needle-in-haystack retention at all lengths up to 16K
- 4.3% PPL improvement from differential attention over plain GQA in bounded regime
- Total training time: ~1.6 hours on H200 NVL

## [1.0.0] - 2026-02-16

### Added
- Initial public release.
- `CoDAGQALandmarkPerf2`: Bounded-memory differential attention module.
- `CoDAGQA` / `BaselineGQA`: Unbounded attention baselines.
- `LlamaCoDAAdapter` / `EveCoDAAdapter`: Drop-in adapters for Llama and Eve model families.
- Two-phase training pipeline (`train_coda.py`).
- 56 passing tests covering correctness, determinism, edge configs, invariants, and backward pass.
- WikiText-103 benchmarks on SmolLM2-135M.
- GitHub Actions CI/CD with automated PyPI publishing.
