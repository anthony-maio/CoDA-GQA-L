"""Evaluate Eve-2 models with optional CoDA-GQA-L attention swapping.

Loads Eve-2 (anthonym21/Eve-2-MoE-IT-272M) from HuggingFace, optionally
replaces its attention layers with CoDA-GQA-L via EveCoDAAdapter, and runs
evaluation experiments: forward equivalence checking, WikiText-2 perplexity,
and needle-in-haystack passkey retrieval.

Usage:
    python benchmarks/eval_eve.py --experiment forward-check
    python benchmarks/eval_eve.py --experiment perplexity
    python benchmarks/eval_eve.py --experiment needle
    python benchmarks/eval_eve.py --experiment all
    python benchmarks/eval_eve.py --experiment perplexity --no-swap
    python benchmarks/eval_eve.py --experiment perplexity --model anthonym21/Eve-2-MoE-IT-272M

Requires: transformers, datasets, torch >= 2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

_MISSING: List[str] = []

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    _MISSING.append("transformers")

try:
    from datasets import load_dataset
except ImportError:
    _MISSING.append("datasets")

if _MISSING:
    print(
        f"ERROR: Missing required packages: {', '.join(_MISSING)}\n"
        f"Install them with:\n"
        f"  pip install {' '.join(_MISSING)}"
    )
    sys.exit(1)

from coda_gqa_l import EveCoDAAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "anthonym21/Eve-2-MoE-IT-272M"

# Eve-2 architecture constants
EVE_N_EMBD = 512
EVE_N_HEAD = 8
EVE_HEAD_DIM = 64
EVE_N_LAYER = 12
EVE_HKV = 8  # MHA: Hkv == H
EVE_VOCAB_SIZE = 50304
EVE_BLOCK_SIZE = 2048

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def get_device_dtype() -> Tuple[torch.device, torch.dtype]:
    """Select device and dtype following project conventions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def save_results(results: Dict[str, Any], experiment: str, results_dir: Path) -> Path:
    """Save results dict to JSON and return the file path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"eve_{experiment}_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return out_path


def gather_system_info(device: torch.device, dtype: torch.dtype) -> Dict[str, str]:
    """Collect system metadata for the results JSON."""
    gpu_name = "cpu"
    cuda_version = "n/a"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "n/a"
    return {
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_eve_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, Any]:
    """Load the Eve-2 model and tokenizer from HuggingFace.

    Returns (model, tokenizer).  The model is moved to the specified device
    and dtype, set to eval mode, and gradient computation is disabled.
    """
    print(f"Loading model: {model_name}")
    print(f"  Device: {device}, Dtype: {dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as e:
        print(f"  Warning: AutoTokenizer failed ({e}), trying GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure the tokenizer has a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Make sure 'transformers' is installed and the model is accessible.")
        sys.exit(1)

    model = model.to(device=device)
    model.eval()
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def get_model_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run model forward and extract logits, handling various output formats.

    Eve-2 models may return raw tensors, tuples, or HuggingFace
    CausalLMOutput objects.  This helper normalizes to a plain tensor.
    """
    output = model(input_ids)

    # HuggingFace CausalLMOutput
    if hasattr(output, "logits"):
        return output.logits

    # Raw tensor
    if isinstance(output, torch.Tensor):
        return output

    # Tuple (logits, ...)
    if isinstance(output, (tuple, list)):
        return output[0]

    raise ValueError(
        f"Unexpected model output type: {type(output)}. "
        f"Cannot extract logits."
    )


# ---------------------------------------------------------------------------
# Attention swapping
# ---------------------------------------------------------------------------


def swap_attention_layers(
    model: nn.Module,
    bounded: bool = False,
    **coda_kwargs,
) -> nn.Module:
    """Replace all CausalSelfAttention layers with EveCoDAAdapter in-place.

    Args:
        model: The Eve-2 model (must have model.transformer.h structure or
            model.model.transformer.h for AutoModelForCausalLM wrappers).
        bounded: Whether to use bounded-memory mode.
        **coda_kwargs: Additional kwargs for EveCoDAAdapter (window, etc.)

    Returns:
        The model (modified in-place).
    """
    # Navigate to the block list.  AutoModelForCausalLM wraps the underlying
    # model, so the blocks may be at model.transformer.h or model.model.transformer.h.
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        blocks = model.model.transformer.h
    else:
        # Try to find blocks by inspecting named modules.
        for name, module in model.named_modules():
            if name.endswith(".h") and isinstance(module, nn.ModuleList):
                blocks = module
                break

    if blocks is None:
        raise RuntimeError(
            "Could not locate transformer blocks (model.transformer.h). "
            "The model structure may differ from expected Eve-2 architecture."
        )

    n_swapped = 0
    for i, block in enumerate(blocks):
        if hasattr(block, "attn"):
            original_attn = block.attn
            try:
                adapter = EveCoDAAdapter.from_eve_attention(
                    original_attn, bounded=bounded, **coda_kwargs
                )
                # Move to the same device/dtype as the original.
                device = next(original_attn.parameters()).device
                dtype = next(original_attn.parameters()).dtype
                adapter = adapter.to(device=device, dtype=dtype)
                block.attn = adapter
                n_swapped += 1
            except Exception as e:
                print(f"  Warning: Failed to swap layer {i}: {e}")

    print(f"  Swapped {n_swapped}/{len(blocks)} attention layers")
    return model


# ---------------------------------------------------------------------------
# Experiment 1: Forward Equivalence Check
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_forward_check(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int = 64,
) -> Dict[str, Any]:
    """Compare original Eve-2 logits vs CoDA-swapped logits.

    Loads the model twice (original and swapped) and feeds the same input
    through both, reporting the max and mean absolute difference in logits.
    With lambda initialized near zero, the differential component should be
    nearly off, producing very similar outputs.
    """
    print("\n" + "=" * 60)
    print("Experiment: Forward Equivalence Check")
    print("=" * 60)

    # Load original model.
    model_orig, tokenizer = load_eve_model(model_name, device, dtype)

    # Create a test input.
    torch.manual_seed(42)
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy, stars were born from clouds of hydrogen gas."
    )
    input_ids = tokenizer(
        test_text, return_tensors="pt", truncation=True, max_length=seq_len
    ).input_ids.to(device)
    actual_len = input_ids.size(1)
    print(f"\n  Test input: {actual_len} tokens")

    # Get original logits.
    print("  Computing original model logits...")
    logits_original = get_model_logits(model_orig, input_ids)
    print(f"  Original logits shape: {logits_original.shape}")

    # Deepcopy the model and swap attention layers.
    print("  Creating swapped model (CoDA unbounded)...")
    model_swapped = copy.deepcopy(model_orig)
    swap_attention_layers(model_swapped, bounded=False)

    # Get swapped logits.
    print("  Computing swapped model logits...")
    logits_swapped = get_model_logits(model_swapped, input_ids)

    # Compare.
    diff = (logits_original.float() - logits_swapped.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()

    # Check per-layer adapter outputs individually.
    layer_results = []
    blocks = None
    if hasattr(model_swapped, "transformer"):
        blocks = model_swapped.transformer.h
    elif hasattr(model_swapped, "model") and hasattr(model_swapped.model, "transformer"):
        blocks = model_swapped.model.transformer.h

    if blocks is not None:
        x_test = torch.randn(1, seq_len, EVE_N_EMBD, device=device, dtype=dtype)
        for i, block in enumerate(blocks):
            adapter = block.attn
            if isinstance(adapter, EveCoDAAdapter):
                y = adapter(x_test)
                info = {
                    "layer": i,
                    "output_mean": y.mean().item(),
                    "output_std": y.std().item(),
                    "has_nan": bool(y.isnan().any()),
                    "has_inf": bool(y.isinf().any()),
                }
                layer_results.append(info)
                print(
                    f"  Layer {i}: mean={info['output_mean']:.6f} "
                    f"std={info['output_std']:.6f} "
                    f"nan={info['has_nan']} inf={info['has_inf']}"
                )

    # Report.
    print(f"\n  Logit comparison (original vs CoDA-swapped):")
    print(f"    Max absolute difference:    {max_diff:.8f}")
    print(f"    Mean absolute difference:   {mean_diff:.8f}")
    print(f"    Median absolute difference: {median_diff:.8f}")

    if max_diff < 0.01:
        print(f"    Status: EXCELLENT - near-identical outputs")
    elif max_diff < 0.1:
        print(f"    Status: GOOD - small differences (expected with lambda~0.0025)")
    elif max_diff < 1.0:
        print(f"    Status: ACCEPTABLE - moderate differences")
    else:
        print(f"    Status: WARNING - large differences, investigate weight transfer")

    results = {
        "experiment": "forward-check",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "seq_len": actual_len,
            "bounded": False,
        },
        "results": {
            "max_logit_diff": max_diff,
            "mean_logit_diff": mean_diff,
            "median_logit_diff": median_diff,
            "original_logits_shape": list(logits_original.shape),
            "layer_checks": layer_results,
        },
    }

    # Cleanup.
    del model_orig, model_swapped
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Experiment 2: WikiText-2 Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 2048,
    stride: int = 512,
    max_tokens: Optional[int] = None,
) -> float:
    """Compute perplexity on WikiText-2 test set using sliding-window evaluation.

    Args:
        model: The causal LM model.
        tokenizer: Tokenizer for encoding text.
        device: Target device.
        dtype: Model dtype.
        max_length: Maximum context window per chunk.
        stride: Step size between chunks (non-overlapping portion counted).
        max_tokens: If set, limit the total number of tokens evaluated
            (useful for quick testing).

    Returns:
        Perplexity value (float).
    """
    print("  Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"  ERROR: Failed to load WikiText-2: {e}")
        print("  Make sure the 'datasets' package is installed and you have internet access.")
        return float("nan")

    # Concatenate all text.
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]

    print(f"  Total tokens: {seq_len:,}")

    nlls = []
    n_tokens = 0
    n_chunks = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end]
        target_len = end - begin

        logits = get_model_logits(model, chunk)

        # Shift for causal LM: predict next token from current.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Only count the non-overlapping portion (stride tokens) except
        # for the first chunk where we count everything.
        if begin > 0:
            # The first (max_length - stride) tokens in this chunk overlap
            # with the previous chunk; skip them.
            skip = max_length - stride - 1
            if skip > 0 and skip < token_losses.size(0):
                token_losses = token_losses[skip:]

        nlls.append(token_losses.sum())
        n_tokens += token_losses.numel()
        n_chunks += 1

        if n_chunks % 20 == 0:
            running_ppl = torch.exp(torch.stack(nlls).sum() / n_tokens).item()
            print(f"    Chunk {n_chunks}: {n_tokens:,} tokens evaluated, running PPL={running_ppl:.2f}")

        if end >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / n_tokens).item()

    print(f"  Final: {n_tokens:,} tokens, {n_chunks} chunks, PPL={ppl:.2f}")
    return ppl


@torch.no_grad()
def run_perplexity(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    no_swap: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run WikiText-2 perplexity evaluation.

    Compares:
    1. Original Eve model (baseline)
    2. Eve with CoDA-GQA-L unbounded (weight-swapped, no retraining)

    Bounded mode evaluation is left as TODO -- it requires layer-by-layer
    state management that cannot be done through the standard model.forward().
    """
    print("\n" + "=" * 60)
    print("Experiment: WikiText-2 Perplexity")
    print("=" * 60)

    model, tokenizer = load_eve_model(model_name, device, dtype)
    ppl_results: Dict[str, Any] = {}

    # --- Baseline perplexity ---
    print("\n  [1/2] Baseline (original attention)")
    ppl_baseline = compute_perplexity(
        model, tokenizer, device, dtype, max_tokens=max_tokens
    )
    ppl_results["baseline"] = ppl_baseline
    print(f"  Baseline PPL: {ppl_baseline:.2f}")

    if not no_swap:
        # --- CoDA-GQA unbounded perplexity ---
        print("\n  [2/2] CoDA-GQA unbounded (weight-swapped, no retraining)")
        swap_attention_layers(model, bounded=False)
        ppl_coda = compute_perplexity(
            model, tokenizer, device, dtype, max_tokens=max_tokens
        )
        ppl_results["coda_unbounded"] = ppl_coda
        print(f"  CoDA unbounded PPL: {ppl_coda:.2f}")

        if ppl_baseline > 0 and not (ppl_baseline != ppl_baseline):  # not NaN
            diff_pct = ((ppl_coda - ppl_baseline) / ppl_baseline) * 100
            print(f"  PPL change: {diff_pct:+.2f}%")
            ppl_results["ppl_change_pct"] = diff_pct

        # TODO: Bounded mode perplexity evaluation.
        # This requires processing the input through the model layer by layer,
        # managing CoDAGQALandmarkStatePerf2 per-layer state, which cannot be
        # done through the standard model.forward() call.  A custom inference
        # loop would be needed:
        #
        # for layer_idx, block in enumerate(blocks):
        #     x = block.ln_1(x)
        #     adapter = block.attn  # EveCoDAAdapter (bounded=True)
        #     adapter.init_inference_state(...)
        #     x_attn = adapter(x, freqs_cis=None)
        #     x = x + x_attn
        #     x = x + block.mlp(block.ln_2(x))
        #
        # This is left for future implementation.
        ppl_results["coda_bounded"] = "TODO: requires custom layer-by-layer inference loop"
    else:
        print("\n  [2/2] Skipped (--no-swap)")

    results = {
        "experiment": "perplexity",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "max_length": EVE_BLOCK_SIZE,
            "stride": 512,
            "max_tokens": max_tokens,
            "no_swap": no_swap,
        },
        "results": ppl_results,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Needle-in-Haystack Retrieval
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_needle(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    depths: Optional[List[int]] = None,
    no_swap: bool = False,
) -> Dict[str, Any]:
    """Run passkey retrieval test at various context depths.

    Plants a random passkey (5-digit number) in filler text and checks
    whether the model can retrieve it from the final position.  Tests at
    25%, 50%, and 75% depth within the context.
    """
    print("\n" + "=" * 60)
    print("Experiment: Needle-in-Haystack Retrieval")
    print("=" * 60)

    if depths is None:
        depths = [256, 512, 1024, 2048]

    model, tokenizer = load_eve_model(model_name, device, dtype)

    def _run_needle_suite(
        model: nn.Module,
        label: str,
    ) -> List[Dict[str, Any]]:
        """Run the needle test suite with the given model."""
        print(f"\n  [{label}]")
        results = []
        random.seed(42)

        filler_sentence = "The quick brown fox jumps over the lazy dog. "

        for depth in depths:
            passkey = random.randint(10000, 99999)

            needle_text = (
                f"The secret passkey is {passkey}. Remember this number."
            )

            # Build filler text to approximately fill the target depth in tokens.
            # Rough estimate: 1 word ~ 1.3 tokens for GPT-2 tokenizer.
            n_filler_repeats = max(1, (depth * 4) // len(filler_sentence.split()))
            filler = filler_sentence * n_filler_repeats

            for position_pct in [0.25, 0.50, 0.75]:
                char_pos = int(len(filler) * position_pct)
                context = (
                    filler[:char_pos]
                    + " " + needle_text + " "
                    + filler[char_pos:]
                )
                prompt = (
                    context
                    + "\n\nWhat is the secret passkey mentioned earlier? The passkey is"
                )

                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=EVE_BLOCK_SIZE,
                ).input_ids.to(device)

                actual_tokens = input_ids.size(1)

                logits = get_model_logits(model, input_ids)
                next_token_logits = logits[:, -1, :]

                # Get top-5 predictions.
                top5_ids = next_token_logits.topk(5, dim=-1).indices[0]
                top5_tokens = [tokenizer.decode(tid.unsqueeze(0)) for tid in top5_ids]
                predicted_first = top5_tokens[0].strip()

                # Check if the passkey appears in the generated tokens.
                passkey_str = str(passkey)
                top5_text = " ".join(top5_tokens)
                correct = (
                    passkey_str in top5_text
                    or passkey_str[:3] in predicted_first
                )

                trial = {
                    "target_depth": depth,
                    "actual_tokens": actual_tokens,
                    "position_pct": position_pct,
                    "passkey": passkey,
                    "predicted_first": predicted_first,
                    "top5_tokens": top5_tokens,
                    "correct": correct,
                }
                results.append(trial)
                status = "HIT" if correct else "MISS"
                print(
                    f"    depth={depth}, pos={position_pct:.0%}, "
                    f"tokens={actual_tokens}, passkey={passkey}, "
                    f"predicted='{predicted_first}', [{status}]"
                )

        return results

    all_results: Dict[str, Any] = {}

    # --- Baseline ---
    baseline_trials = _run_needle_suite(model, "Baseline (original attention)")
    baseline_correct = sum(1 for t in baseline_trials if t["correct"])
    print(f"  Baseline accuracy: {baseline_correct}/{len(baseline_trials)}")
    all_results["baseline"] = {
        "trials": baseline_trials,
        "accuracy": baseline_correct / len(baseline_trials) if baseline_trials else 0,
    }

    if not no_swap:
        # --- CoDA-GQA unbounded ---
        swap_attention_layers(model, bounded=False)
        coda_trials = _run_needle_suite(model, "CoDA-GQA unbounded (weight-swapped)")
        coda_correct = sum(1 for t in coda_trials if t["correct"])
        print(f"  CoDA unbounded accuracy: {coda_correct}/{len(coda_trials)}")
        all_results["coda_unbounded"] = {
            "trials": coda_trials,
            "accuracy": coda_correct / len(coda_trials) if coda_trials else 0,
        }
    else:
        print("\n  CoDA swap skipped (--no-swap)")

    results = {
        "experiment": "needle",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "depths": depths,
            "positions": [0.25, 0.50, 0.75],
            "max_length": EVE_BLOCK_SIZE,
            "no_swap": no_swap,
        },
        "results": all_results,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# KV Cache Memory Comparison Table
# ---------------------------------------------------------------------------


def print_memory_table() -> None:
    """Print analytical KV cache memory comparison table.

    Computes memory requirements for standard unbounded KV cache vs
    CoDA-GQA-L bounded cache at various sequence lengths.  No model
    needed -- this is purely arithmetic.

    Eve-2: MHA with Hkv=8, Dh=64, bf16 (2 bytes per element).
    """
    print("\n" + "=" * 60)
    print("KV Cache Memory Comparison (per layer, bf16)")
    print("=" * 60)

    Hkv = EVE_HKV
    Dh = EVE_HEAD_DIM
    bytes_per = 2  # bf16

    # Bounded configs to compare.
    bounded_configs = [
        {"W": 128, "Me": 32, "Ms": 32},
        {"W": 256, "Me": 64, "Ms": 64},
        {"W": 512, "Me": 128, "Ms": 128},
    ]

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    # Header.
    header = f"{'Seq Len':>10} | {'Standard KV':>14}"
    for cfg in bounded_configs:
        label = f"W={cfg['W']},Me={cfg['Me']},Ms={cfg['Ms']}"
        header += f" | {label:>24}"
    print(header)
    print("-" * len(header))

    for L in seq_lengths:
        # Standard: 2 tensors (K, V) x Hkv x L x Dh x bytes_per
        std_bytes = 2 * Hkv * L * Dh * bytes_per
        row = f"{L:>10} | {human_bytes(std_bytes):>14}"

        for cfg in bounded_configs:
            total_slots = cfg["W"] + cfg["Me"] + cfg["Ms"]
            coda_bytes = 2 * Hkv * total_slots * Dh * bytes_per
            compression = std_bytes / coda_bytes if coda_bytes > 0 else float("inf")
            cell = f"{human_bytes(coda_bytes)} ({compression:.0f}x)"
            row += f" | {cell:>24}"

        print(row)

    # Total for all 12 layers.
    print(f"\n  Multiply by {EVE_N_LAYER} layers for total model KV cache.")
    for cfg in bounded_configs:
        total_slots = cfg["W"] + cfg["Me"] + cfg["Ms"]
        per_layer = 2 * Hkv * total_slots * Dh * bytes_per
        total = per_layer * EVE_N_LAYER
        print(
            f"  W={cfg['W']}, Me={cfg['Me']}, Ms={cfg['Ms']}: "
            f"{human_bytes(per_layer)}/layer x {EVE_N_LAYER} = {human_bytes(total)} total"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Eve-2 models with optional CoDA-GQA-L attention swapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/eval_eve.py --experiment forward-check
  python benchmarks/eval_eve.py --experiment perplexity
  python benchmarks/eval_eve.py --experiment needle
  python benchmarks/eval_eve.py --experiment all
  python benchmarks/eval_eve.py --experiment perplexity --no-swap
  python benchmarks/eval_eve.py --experiment perplexity --model anthonym21/Eve-2-MoE-IT-272M
  python benchmarks/eval_eve.py --experiment perplexity --max-tokens 10000
""",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["forward-check", "perplexity", "needle", "all"],
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Only run baseline (no CoDA attention swapping).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limit tokens for perplexity evaluation (useful for quick testing).",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default=None,
        help="Comma-separated list of needle depths (default: 256,512,1024,2048).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for JSON results (default: results/ at project root).",
    )
    parser.add_argument(
        "--no-memory-table",
        action="store_true",
        help="Skip printing the KV cache memory comparison table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device, dtype = get_device_dtype()

    # Resolve results directory.
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    print("CoDA-GQA-L Eve-2 Evaluation")
    print(f"  Device: {device}")
    print(f"  Dtype:  {dtype}")
    print(f"  Model:  {args.model}")
    print(f"  Results dir: {results_dir}")

    # Print memory comparison table unless suppressed.
    if not args.no_memory_table:
        print_memory_table()

    experiments = (
        ["forward-check", "perplexity", "needle"]
        if args.experiment == "all"
        else [args.experiment]
    )

    all_results: Dict[str, Any] = {}

    for exp in experiments:
        try:
            if exp == "forward-check":
                result = run_forward_check(args.model, device, dtype)
            elif exp == "perplexity":
                result = run_perplexity(
                    args.model, device, dtype,
                    no_swap=args.no_swap,
                    max_tokens=args.max_tokens,
                )
            elif exp == "needle":
                depths = None
                if args.depths:
                    depths = [int(d.strip()) for d in args.depths.split(",")]
                result = run_needle(
                    args.model, device, dtype,
                    depths=depths,
                    no_swap=args.no_swap,
                )
            else:
                print(f"Unknown experiment: {exp}")
                continue

            all_results[exp] = result

            # Save individual result.
            out_path = save_results(result, exp, results_dir)
            print(f"\n  Results saved to {out_path}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n  ERROR in experiment '{exp}': {e}")
            import traceback
            traceback.print_exc()
            all_results[exp] = {"error": str(e)}

    # Save combined results if running all experiments.
    if args.experiment == "all" and all_results:
        combined = {
            "experiments": list(all_results.keys()),
            "model": args.model,
            "system": gather_system_info(device, dtype),
            "results": all_results,
        }
        out_path = save_results(combined, "all", results_dir)
        print(f"\n  Combined results saved to {out_path}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
