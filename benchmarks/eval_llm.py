"""Evaluate any HuggingFace causal LM with CoDA-GQA-L attention swapping.

Works with Llama-family models (SmolLM, SmolLM2, Llama 2/3, Mistral, etc.)
and Eve-family models.  Detects model architecture automatically and uses
the appropriate adapter (LlamaCoDAAdapter or EveCoDAAdapter).

Cold-swap design (no retraining):
  - HeadwiseRMSNorm bypassed (head_norm_mode="identity")
  - lambda_proj.weight zeroed, bias=-20 -> lambda ≈ 0 (near-identity)
  - theta_init=0 -> noise query = signal query
  - write_proj zeroed (bounded mode)

IMPORTANT: Cold-swap evaluation should use float32 (--dtype fp32) for accurate
results. In bf16, the two SDPA calls in CoDA's differential attention produce
slightly different rounding than the single SDPA in the original model. These
1-ULP per-layer differences compound through 30+ layers, causing artificial PPL
degradation that does NOT reflect algorithmic quality. Float32 gives exact
PPL match (verified: 10.83 vs 10.83 at L=2048 on SmolLM2-135M).

Usage:
    # SmolLM2-135M (fast, good for testing)
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity

    # SmolLM2-360M (GQA: H=15, Hkv=5)
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-360M --experiment perplexity

    # Llama 3.2 1B
    python benchmarks/eval_llm.py --model meta-llama/Llama-3.2-1B --experiment perplexity

    # Quick test (5k tokens)
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity --max-tokens 5000

    # Baseline only (no swap)
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity --no-swap

    # Force bf16 (faster but cold-swap PPL will show numerical artifacts)
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity --dtype bf16

    # All experiments
    python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment all

Requires: transformers, datasets, torch >= 2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

from coda_gqa_l import LlamaCoDAAdapter

# Also import Eve adapter for Eve-family models.
try:
    from coda_gqa_l import EveCoDAAdapter
except ImportError:
    EveCoDAAdapter = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def get_device_dtype(dtype_str: str = "fp32") -> Tuple[torch.device, torch.dtype]:
    """Return (device, dtype) based on the requested dtype string.

    Args:
        dtype_str: One of "auto", "fp32", "bf16", "fp16".
            "auto" selects bf16 on CUDA, fp32 on CPU.
            "fp32" always uses float32 (recommended for cold-swap accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype_str == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    elif dtype_str == "fp32":
        dtype = torch.float32
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    elif dtype_str == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str!r}. Use auto/fp32/bf16/fp16.")
    return device, dtype


def save_results(results: Dict[str, Any], experiment: str, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use model short name in filename.
    model_name = results.get("model", "unknown").replace("/", "_")
    out_path = results_dir / f"{model_name}_{experiment}_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return out_path


def gather_system_info(device: torch.device, dtype: torch.dtype) -> Dict[str, str]:
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


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Load a HuggingFace causal LM model and tokenizer.

    Returns (model, tokenizer, model_info) where model_info contains
    architecture details extracted from the config.
    """
    print(f"Loading model: {model_name}")
    print(f"  Device: {device}, Dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device=device)
    model.eval()

    # Extract model info.
    model_info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", None) or getattr(config, "n_embd", None),
        "num_heads": getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None),
        "num_kv_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": None,
        "num_layers": getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    # Infer head_dim.
    if model_info["hidden_size"] and model_info["num_heads"]:
        model_info["head_dim"] = model_info["hidden_size"] // model_info["num_heads"]

    # If num_kv_heads is None, assume MHA.
    if model_info["num_kv_heads"] is None:
        model_info["num_kv_heads"] = model_info["num_heads"]

    print(f"  Model type: {model_info['model_type']}")
    print(f"  Parameters: {model_info['n_params']:,}")
    print(f"  Architecture: D={model_info['hidden_size']}, "
          f"H={model_info['num_heads']}, Hkv={model_info['num_kv_heads']}, "
          f"Dh={model_info['head_dim']}, L={model_info['num_layers']}")

    return model, tokenizer, model_info


def get_model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run model forward and extract logits."""
    output = model(input_ids)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise ValueError(f"Unexpected model output type: {type(output)}")


# ---------------------------------------------------------------------------
# Attention layer detection and swapping
# ---------------------------------------------------------------------------


def _detect_arch_type(model: nn.Module) -> str:
    """Detect whether this is a Llama-family or Eve-family model."""
    # Check for Llama-family: model.model.layers[i].self_attn
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    # Check for Eve-family: model.transformer.h[i].attn
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "eve"
    # Check wrapped: model.model.transformer.h
    if hasattr(model, "model") and hasattr(getattr(model.model, "transformer", None), "h"):
        return "eve"
    return "unknown"


def _get_attention_layers(model: nn.Module, arch_type: str):
    """Return (blocks_list, attn_attr_name) for the given architecture."""
    if arch_type == "llama":
        return model.model.layers, "self_attn"
    elif arch_type == "eve":
        if hasattr(model, "transformer"):
            return model.transformer.h, "attn"
        return model.model.transformer.h, "attn"
    raise ValueError(f"Unknown architecture type: {arch_type}")


def swap_attention_layers(
    model: nn.Module,
    model_info: Dict[str, Any],
    bounded: bool = False,
    head_norm_mode: str = "identity",
    theta_init: float = 0.0,
    cold_swap: bool = True,
    **coda_kwargs,
) -> Tuple[nn.Module, List]:
    """Replace attention layers with CoDA adapters.

    Args:
        cold_swap: If True (default), zero out newly initialized parameters
            (lambda_proj.weight, write_proj.weight if bounded) so the CoDA
            layer acts as near-identity on top of the original attention.
            This prevents random init of untrained CoDA-specific parameters
            from perturbing the pre-trained model outputs.

    Returns (model, adapters_list).
    """
    arch_type = _detect_arch_type(model)
    print(f"  Architecture: {arch_type}")

    blocks, attn_attr = _get_attention_layers(model, arch_type)

    n_swapped = 0
    adapters = []

    for i, block in enumerate(blocks):
        original_attn = getattr(block, attn_attr)
        try:
            if arch_type == "llama":
                adapter = LlamaCoDAAdapter.from_llama_attention(
                    original_attn,
                    bounded=bounded,
                    head_norm_mode=head_norm_mode,
                    theta_init=theta_init,
                    **coda_kwargs,
                )
            elif arch_type == "eve" and EveCoDAAdapter is not None:
                adapter = EveCoDAAdapter.from_eve_attention(
                    original_attn,
                    bounded=bounded,
                    head_norm_mode=head_norm_mode,
                    theta_init=theta_init,
                    **coda_kwargs,
                )
            else:
                print(f"  Warning: No adapter for {arch_type}, skipping layer {i}")
                continue

            device = next(original_attn.parameters()).device
            dtype = next(original_attn.parameters()).dtype
            adapter = adapter.to(device=device, dtype=dtype)

            # Cold-swap: zero out randomly initialized CoDA parameters so
            # the differential mechanism is near-identity (lambda ≈ 0, theta = 0).
            if cold_swap:
                with torch.no_grad():
                    coda = adapter.coda
                    # Zero lambda_proj weights -> lambda = sigmoid(bias) = constant
                    coda.lambda_proj.weight.zero_()
                    # Ensure bias is very negative -> lambda ≈ 0
                    coda.lambda_proj.bias.fill_(-20.0)
                    # Zero write gate weights if present (bounded mode)
                    if hasattr(coda, "write_proj"):
                        coda.write_proj.weight.zero_()

            setattr(block, attn_attr, adapter)
            adapters.append(adapter)
            n_swapped += 1
        except Exception as e:
            print(f"  Warning: Failed to swap layer {i}: {e}")

    print(f"  Swapped {n_swapped}/{len(blocks)} attention layers"
          f"{' (cold-swap)' if cold_swap else ''}")
    return model, adapters


def load_adapter_weights(
    adapters: List,
    checkpoint_path: str,
) -> None:
    """Load trained adapter weights from a checkpoint file.

    Checkpoint format (from train_coda.py):
        {"layer_0": adapter_state_dict, "layer_1": ..., ...}

    Uses strict=False to allow loading unbounded (CoDAGQA) weights into
    bounded (CoDAGQALandmarkPerf2) adapters. Shared params (q/k/v/o projections,
    lambda, theta, head_norm) are loaded; bounded-only params (write_proj,
    summary_eta_logit) keep their default initialization.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "coda_adapters.pt"

    print(f"  Loading adapter weights from {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)

    loaded = 0
    for i, adapter in enumerate(adapters):
        key = f"layer_{i}"
        if key in state:
            missing, unexpected = adapter.load_state_dict(state[key], strict=False)
            if unexpected:
                print(f"    Layer {i}: {len(unexpected)} unexpected keys (ignored)")
            if missing:
                # Expected when loading unbounded weights into bounded adapter.
                bounded_only = [k for k in missing if any(
                    s in k for s in ("write_proj", "summary_eta", "exact_", "summary_")
                )]
                other_missing = [k for k in missing if k not in bounded_only]
                if other_missing:
                    print(f"    Layer {i}: WARNING missing keys: {other_missing}")
            loaded += 1
        else:
            print(f"    Warning: no weights for layer {i}")

    print(f"  Loaded weights for {loaded}/{len(adapters)} layers")


# ---------------------------------------------------------------------------
# Experiment: WikiText-2 Perplexity
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
    """Compute perplexity on WikiText-2 using sliding-window evaluation."""
    print("  Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"  ERROR: Failed to load WikiText-2: {e}")
        return float("nan")

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
    t0 = time.time()

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end]

        logits = get_model_logits(model, chunk)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if begin > 0:
            skip = max_length - stride - 1
            if skip > 0 and skip < token_losses.size(0):
                token_losses = token_losses[skip:]

        nlls.append(token_losses.sum())
        n_tokens += token_losses.numel()
        n_chunks += 1

        if n_chunks % 20 == 0:
            running_ppl = torch.exp(torch.stack(nlls).sum() / n_tokens).item()
            elapsed = time.time() - t0
            tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
            print(f"    Chunk {n_chunks}: {n_tokens:,} tokens, "
                  f"PPL={running_ppl:.2f}, {tok_per_sec:.0f} tok/s")

        if end >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / n_tokens).item()
    elapsed = time.time() - t0

    print(f"  Final: {n_tokens:,} tokens, {n_chunks} chunks, "
          f"PPL={ppl:.2f}, {elapsed:.1f}s")
    return ppl


@torch.no_grad()
def compute_perplexity_bounded(
    model_orig: nn.Module,
    model_info: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    window: int = 256,
    Me: int = 64,
    Ms: int = 64,
    block_size: int = 256,
    chunk_size: int = 512,
    max_tokens: Optional[int] = None,
    head_norm_mode: str = "identity",
    adapter_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute perplexity with bounded CoDA attention (streaming).

    When adapter_weights is provided, loads trained CoDA weights into the
    bounded adapters instead of using cold-swap zeroing.

    Returns a dict with PPL and diagnostics.
    """
    trained = adapter_weights is not None
    model_bounded = copy.deepcopy(model_orig)

    model_bounded, adapters = swap_attention_layers(
        model_bounded, model_info, bounded=True,
        head_norm_mode=head_norm_mode,
        cold_swap=not trained,
        window=window, num_landmarks_exact=Me,
        num_landmarks_summary=Ms, block_size=block_size,
    )
    if trained:
        load_adapter_weights(adapters, adapter_weights)

    if not adapters:
        return {"ppl": float("nan"), "error": "no layers swapped"}

    total_slots = window + Me + Ms
    cache_per_layer = adapters[0].cache_bytes(batch_size=1, dtype=dtype)
    n_layers = len(adapters)
    print(f"    Config: W={window}, Me={Me}, Ms={Ms} ({total_slots} slots/layer)")
    print(f"    Cache: {human_bytes(cache_per_layer)}/layer, "
          f"{human_bytes(cache_per_layer * n_layers)} total")

    # Load dataset.
    print("    Loading WikiText-2...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"    ERROR: Failed to load WikiText-2: {e}")
        return {"ppl": float("nan"), "error": str(e)}

    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]

    print(f"    Total tokens: {seq_len:,}")

    nlls: List[torch.Tensor] = []
    n_tokens = 0
    n_chunks = 0
    t0 = time.time()

    # Process as non-overlapping chunks.  Each chunk of size C produces C-1
    # loss tokens (shifted next-token prediction).  No overlap avoids
    # double-writing tokens to the bounded KV cache.
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        if end - start < 2:
            break  # Need at least 2 tokens for a loss token.
        chunk = input_ids[:, start:end]

        logits = get_model_logits(model_bounded, chunk)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        nlls.append(token_losses.sum())
        n_tokens += token_losses.numel()
        n_chunks += 1

        if n_chunks % 20 == 0:
            running_ppl = torch.exp(torch.stack(nlls).sum() / n_tokens).item()
            elapsed = time.time() - t0
            tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
            print(f"      Chunk {n_chunks}: {n_tokens:,} tokens, "
                  f"PPL={running_ppl:.2f}, {tok_per_sec:.0f} tok/s")

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / n_tokens).item()
    elapsed = time.time() - t0
    print(f"    Final: {n_tokens:,} tokens, {n_chunks} chunks, "
          f"PPL={ppl:.2f}, {elapsed:.1f}s")

    # Check Phase-Safe invariant from layer 0.
    phase_safe_ok = True
    hf_max = 0.0
    if adapters:
        st = adapters[0].get_state()
        if st is not None:
            lf = adapters[0].coda.lf_start
            s_off = window + Me
            hf_max = st.k_buf[:, :, s_off:, :lf].abs().max().item()
            phase_safe_ok = hf_max < 1e-6
            print(f"    Phase-Safe: HF max abs = {hf_max:.8f} "
                  f"{'OK' if phase_safe_ok else 'VIOLATION'}")

    result = {
        "ppl": ppl,
        "n_tokens": n_tokens,
        "n_chunks": n_chunks,
        "elapsed_s": elapsed,
        "cache_bytes_per_layer": cache_per_layer,
        "cache_bytes_total": cache_per_layer * n_layers,
        "phase_safe_ok": phase_safe_ok,
        "phase_safe_hf_max": hf_max,
    }

    del model_bounded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# Bounded configs to sweep.
BOUNDED_CONFIGS = {
    "tiny": {"window": 128, "Me": 32, "Ms": 32},
    "medium": {"window": 256, "Me": 64, "Ms": 64},
    "large": {"window": 512, "Me": 128, "Ms": 128},
    "window-only": {"window": 256, "Me": 0, "Ms": 0},
}


@torch.no_grad()
def run_perplexity(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    no_swap: bool = False,
    max_tokens: Optional[int] = None,
    bounded_configs: Optional[List[str]] = None,
    head_norm_mode: str = "identity",
    adapter_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Run WikiText-2 perplexity evaluation.

    Compares:
    1. Original model (baseline)
    2. CoDA-GQA-L unbounded (cold-swap, head_norm bypassed)
    3. CoDA-GQA-L bounded at various cache sizes (streaming)
    """
    print("\n" + "=" * 60)
    print("Experiment: WikiText-2 Perplexity")
    print("=" * 60)

    model, tokenizer, model_info = load_model(model_name, device, dtype)

    # Detect max context length.
    max_length = model_info.get("max_position_embeddings", 2048) or 2048
    max_length = min(max_length, 2048)  # Cap for memory reasons.

    ppl_results: Dict[str, Any] = {"model_info": model_info}

    # --- Baseline ---
    print(f"\n  [1] Baseline (original attention)")
    ppl_baseline = compute_perplexity(
        model, tokenizer, device, dtype,
        max_length=max_length, max_tokens=max_tokens,
    )
    ppl_results["baseline"] = ppl_baseline
    print(f"  Baseline PPL: {ppl_baseline:.2f}")

    if not no_swap:
        # --- CoDA-GQA unbounded ---
        trained = adapter_weights is not None
        label = "trained" if trained else "cold-swap"
        print(f"\n  [2] CoDA-GQA unbounded ({label}, head_norm={head_norm_mode})")
        model_ub = copy.deepcopy(model)
        model_ub, ub_adapters = swap_attention_layers(
            model_ub, model_info, bounded=False,
            head_norm_mode=head_norm_mode,
            cold_swap=not trained,
        )
        if trained:
            load_adapter_weights(ub_adapters, adapter_weights)
        ppl_coda = compute_perplexity(
            model_ub, tokenizer, device, dtype,
            max_length=max_length, max_tokens=max_tokens,
        )
        ppl_results["coda_unbounded"] = ppl_coda
        print(f"  CoDA unbounded PPL: {ppl_coda:.2f}")

        if ppl_baseline > 0 and ppl_baseline == ppl_baseline:  # not NaN
            diff_pct = ((ppl_coda - ppl_baseline) / ppl_baseline) * 100
            print(f"  PPL change: {diff_pct:+.2f}%")
            ppl_results["unbounded_change_pct"] = diff_pct

        del model_ub
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- CoDA-GQA bounded ---
        configs_to_run = bounded_configs or list(BOUNDED_CONFIGS.keys())
        ppl_results["bounded"] = {}

        step = 3
        for cfg_name in configs_to_run:
            if cfg_name not in BOUNDED_CONFIGS:
                print(f"\n  [{step}] Unknown bounded config: {cfg_name}, skipping")
                step += 1
                continue

            cfg = BOUNDED_CONFIGS[cfg_name]
            print(f"\n  [{step}] CoDA-GQA bounded '{cfg_name}' "
                  f"(W={cfg['window']}, Me={cfg['Me']}, Ms={cfg['Ms']})")

            bounded_result = compute_perplexity_bounded(
                model, model_info, tokenizer, device, dtype,
                window=cfg["window"], Me=cfg["Me"], Ms=cfg["Ms"],
                max_tokens=max_tokens, head_norm_mode=head_norm_mode,
                adapter_weights=adapter_weights,
            )
            ppl_results["bounded"][cfg_name] = {
                **bounded_result,
                "config": cfg,
                "total_slots": cfg["window"] + cfg["Me"] + cfg["Ms"],
            }

            ppl_bounded = bounded_result["ppl"]
            if ppl_baseline > 0 and ppl_baseline == ppl_baseline:
                diff = ((ppl_bounded - ppl_baseline) / ppl_baseline) * 100
                ppl_results["bounded"][cfg_name]["change_pct"] = diff
                print(f"  '{cfg_name}' PPL change vs baseline: {diff:+.2f}%")

            step += 1
    else:
        print("\n  [2+] Skipped (--no-swap)")

    results = {
        "experiment": "perplexity",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "max_length": max_length,
            "stride": 512,
            "max_tokens": max_tokens,
            "no_swap": no_swap,
            "head_norm_mode": head_norm_mode,
            "adapter_weights": adapter_weights,
        },
        "results": ppl_results,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Experiment: Forward Equivalence Check
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_forward_check(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    head_norm_mode: str = "identity",
) -> Dict[str, Any]:
    """Compare original model logits vs CoDA-swapped logits.

    With lambda~0.0025 and head_norm bypassed, outputs should be nearly
    identical to the original model.
    """
    print("\n" + "=" * 60)
    print("Experiment: Forward Equivalence Check")
    print("=" * 60)

    model_orig, tokenizer, model_info = load_model(model_name, device, dtype)

    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy, stars were born from clouds of hydrogen gas."
    )
    input_ids = tokenizer(
        test_text, return_tensors="pt", truncation=True, max_length=128,
    ).input_ids.to(device)
    actual_len = input_ids.size(1)
    print(f"\n  Test input: {actual_len} tokens")

    # Original logits.
    print("  Computing original model logits...")
    logits_original = get_model_logits(model_orig, input_ids)

    # Swapped logits.
    print(f"  Creating swapped model (head_norm={head_norm_mode})...")
    model_swapped = copy.deepcopy(model_orig)
    model_swapped, _ = swap_attention_layers(
        model_swapped, model_info, bounded=False,
        head_norm_mode=head_norm_mode,
    )

    print("  Computing swapped model logits...")
    logits_swapped = get_model_logits(model_swapped, input_ids)

    # Compare.
    diff = (logits_original.float() - logits_swapped.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n  Logit comparison:")
    print(f"    Max absolute difference:  {max_diff:.6f}")
    print(f"    Mean absolute difference: {mean_diff:.6f}")

    if max_diff < 0.1:
        print(f"    Status: EXCELLENT")
    elif max_diff < 1.0:
        print(f"    Status: GOOD")
    elif max_diff < 10.0:
        print(f"    Status: ACCEPTABLE")
    else:
        print(f"    Status: WARNING - large differences")

    results = {
        "experiment": "forward-check",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "seq_len": actual_len,
            "head_norm_mode": head_norm_mode,
        },
        "results": {
            "max_logit_diff": max_diff,
            "mean_logit_diff": mean_diff,
            "model_info": model_info,
        },
    }

    del model_orig, model_swapped
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# KV Cache Memory Comparison
# ---------------------------------------------------------------------------


def print_memory_table(model_info: Dict[str, Any]) -> None:
    """Print KV cache memory comparison table for the given model."""
    print("\n" + "=" * 60)
    print("KV Cache Memory Comparison (per layer)")
    print("=" * 60)

    Hkv = model_info["num_kv_heads"]
    Dh = model_info["head_dim"]
    n_layers = model_info["num_layers"]
    bytes_per = 2  # bf16

    bounded_configs = [
        {"W": 128, "Me": 32, "Ms": 32},
        {"W": 256, "Me": 64, "Ms": 64},
        {"W": 512, "Me": 128, "Ms": 128},
    ]

    seq_lengths = [512, 1024, 2048, 4096, 8192]

    header = f"{'Seq Len':>10} | {'Standard KV':>14}"
    for cfg in bounded_configs:
        label = f"W={cfg['W']},Me={cfg['Me']},Ms={cfg['Ms']}"
        header += f" | {label:>24}"
    print(header)
    print("-" * len(header))

    for L in seq_lengths:
        std_bytes = 2 * Hkv * L * Dh * bytes_per
        row = f"{L:>10} | {human_bytes(std_bytes):>14}"

        for cfg in bounded_configs:
            total_slots = cfg["W"] + cfg["Me"] + cfg["Ms"]
            coda_bytes = 2 * Hkv * total_slots * Dh * bytes_per
            compression = std_bytes / coda_bytes if coda_bytes > 0 else float("inf")
            cell = f"{human_bytes(coda_bytes)} ({compression:.0f}x)"
            row += f" | {cell:>24}"

        print(row)

    print(f"\n  Multiply by {n_layers} layers for total model KV cache.")
    for cfg in bounded_configs:
        total_slots = cfg["W"] + cfg["Me"] + cfg["Ms"]
        per_layer = 2 * Hkv * total_slots * Dh * bytes_per
        total = per_layer * n_layers
        print(
            f"  W={cfg['W']}, Me={cfg['Me']}, Ms={cfg['Ms']}: "
            f"{human_bytes(per_layer)}/layer x {n_layers} = {human_bytes(total)} total"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HuggingFace LMs with CoDA-GQA-L attention swapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-360M --experiment perplexity
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment forward-check
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment all
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity --max-tokens 5000
  python benchmarks/eval_llm.py --model HuggingFaceTB/SmolLM2-135M --experiment perplexity --no-swap
""",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["forward-check", "perplexity", "all"],
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model name (default: HuggingFaceTB/SmolLM2-135M)",
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
        help="Limit tokens for perplexity evaluation.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for JSON results (default: results/).",
    )
    parser.add_argument(
        "--bounded-configs",
        type=str,
        default=None,
        help=f"Comma-separated bounded configs to evaluate. "
             f"Available: {', '.join(BOUNDED_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--head-norm-mode",
        type=str,
        default="identity",
        choices=["identity", "full"],
        help="Head norm mode: 'identity' for cold-swap (default), 'full' for trained CoDA.",
    )
    parser.add_argument(
        "--no-memory-table",
        action="store_true",
        help="Skip KV cache memory comparison table.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Data type: fp32 (default, accurate cold-swap), bf16, fp16, or auto.",
    )
    parser.add_argument(
        "--adapter-weights",
        type=str,
        default=None,
        help="Path to trained adapter checkpoint (coda_adapters.pt). "
             "When provided, loads trained CoDA weights instead of cold-swap zeroing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, dtype = get_device_dtype(args.dtype)

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    print("CoDA-GQA-L LLM Evaluation")
    print(f"  Device: {device}")
    print(f"  Dtype:  {dtype}")
    print(f"  Model:  {args.model}")
    print(f"  Head norm: {args.head_norm_mode}")
    if args.adapter_weights:
        print(f"  Adapter weights: {args.adapter_weights}")
    print(f"  Results dir: {results_dir}")

    experiments = (
        ["forward-check", "perplexity"]
        if args.experiment == "all"
        else [args.experiment]
    )

    all_results: Dict[str, Any] = {}

    for exp in experiments:
        try:
            if exp == "forward-check":
                result = run_forward_check(
                    args.model, device, dtype,
                    head_norm_mode=args.head_norm_mode,
                )
            elif exp == "perplexity":
                bc = None
                if args.bounded_configs:
                    bc = [c.strip() for c in args.bounded_configs.split(",")]
                result = run_perplexity(
                    args.model, device, dtype,
                    no_swap=args.no_swap,
                    max_tokens=args.max_tokens,
                    bounded_configs=bc,
                    head_norm_mode=args.head_norm_mode,
                    adapter_weights=args.adapter_weights,
                )

                # Print memory table after perplexity.
                if not args.no_memory_table:
                    model_info = result["results"].get("model_info", {})
                    if model_info.get("num_kv_heads") and model_info.get("head_dim"):
                        print_memory_table(model_info)
            else:
                print(f"Unknown experiment: {exp}")
                continue

            all_results[exp] = result
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
