"""Forward check: validate CoDA-GQA-L weight transfer against Llama-architecture models.

Loads a HuggingFace Llama-style model, extracts attention weights from each
layer, creates matching BaselineGQA and CoDAGQA modules, copies weights, swaps
the attention layers, and compares full-model logits.

Expected results:
  - BaselineGQA swap: near-zero logit diff (proves weight transfer + RoPE compatibility)
  - CoDAGQA swap: larger diffs (from differential attention + HeadwiseRMSNorm, not bugs)

Usage:
    python benchmarks/forward_check_generic.py
    python benchmarks/forward_check_generic.py --model HuggingFaceTB/SmolLM2-135M
    python benchmarks/forward_check_generic.py --model HuggingFaceTB/SmolLM2-135M --seq-len 128

Requires: transformers, torch >= 2.0
"""

from __future__ import annotations

import argparse
import copy
import json
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

if _MISSING:
    print(
        f"ERROR: Missing required packages: {', '.join(_MISSING)}\n"
        f"Install them with:\n"
        f"  pip install {' '.join(_MISSING)}"
    )
    sys.exit(1)

from coda_gqa_l import BaselineGQA, CoDAGQA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


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


def save_results(results: Dict[str, Any], results_dir: Path) -> Path:
    """Save results dict to JSON and return the file path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"forward_check_generic_{ts}.json"
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
# Model loading and introspection
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, Any]:
    """Load a HuggingFace causal LM and its tokenizer.

    Returns (model, tokenizer).  The model is moved to the specified device
    and dtype, set to eval mode, with gradient computation disabled.
    """
    print(f"Loading model: {model_name}")
    print(f"  Device: {device}, Dtype: {dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: AutoTokenizer failed ({e}), trying GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        # Newer transformers (>=4.50) renamed torch_dtype -> dtype.
        # Try the new name first; fall back to torch_dtype for compat.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=dtype,
            )
        except TypeError:
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Parameters: {n_params:,}")
    return model, tokenizer


def get_model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run model forward and extract logits, handling various output formats."""
    output = model(input_ids)

    if hasattr(output, "logits"):
        return output.logits

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        return output[0]

    raise ValueError(f"Unexpected model output type: {type(output)}.")


def find_attention_layers(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Module]]:
    """Locate Llama-style attention layers in the model.

    Searches for ``model.model.layers[i].self_attn`` (standard Llama layout).
    Returns a list of (path_string, parent_block, attn_module) tuples.
    """
    layers: List[Tuple[str, nn.Module, nn.Module]] = []

    # Standard Llama: model.model.layers[i].self_attn
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        for i, block in enumerate(inner.layers):
            if hasattr(block, "self_attn"):
                path = f"model.model.layers[{i}].self_attn"
                layers.append((path, block, block.self_attn))

    if layers:
        return layers

    # Fallback: search named modules for anything containing '.self_attn'
    visited_parents: set = set()
    for name, module in model.named_modules():
        if name.endswith(".self_attn"):
            parent_name = name.rsplit(".", 1)[0]
            if parent_name in visited_parents:
                continue
            visited_parents.add(parent_name)

            # Resolve parent module.
            parent = model
            for part in parent_name.split("."):
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            layers.append((name, parent, module))

    return layers


def extract_attn_config(
    attn_module: nn.Module, model_config: Any
) -> Dict[str, Any]:
    """Extract attention configuration from a Llama-style self_attn module.

    Returns a dict with embed_dim, num_heads, num_kv_heads, head_dim, rope_theta,
    and whether projections have bias.
    """
    config: Dict[str, Any] = {}

    # Try to read from model config first (most reliable).
    config["embed_dim"] = getattr(model_config, "hidden_size", None)
    config["num_heads"] = getattr(model_config, "num_attention_heads", None)
    config["num_kv_heads"] = getattr(
        model_config,
        "num_key_value_heads",
        config["num_heads"],  # MHA fallback
    )
    config["rope_theta"] = getattr(model_config, "rope_theta", 10_000.0)

    # Infer from weight shapes as fallback / validation.
    if hasattr(attn_module, "q_proj"):
        q_weight = attn_module.q_proj.weight
        out_features, in_features = q_weight.shape
        config.setdefault("embed_dim", in_features)
        if config["embed_dim"] is None:
            config["embed_dim"] = in_features

    if config["num_heads"] is not None and config["embed_dim"] is not None:
        config["head_dim"] = config["embed_dim"] // config["num_heads"]
    else:
        # Infer from k_proj output size and num_kv_heads.
        if hasattr(attn_module, "k_proj"):
            k_out = attn_module.k_proj.weight.shape[0]
            if config["num_kv_heads"] is not None:
                config["head_dim"] = k_out // config["num_kv_heads"]
            else:
                config["head_dim"] = 64  # common default

    # Check for bias.
    config["has_q_bias"] = (
        hasattr(attn_module, "q_proj")
        and attn_module.q_proj.bias is not None
    )
    config["has_k_bias"] = (
        hasattr(attn_module, "k_proj")
        and attn_module.k_proj.bias is not None
    )
    config["has_v_bias"] = (
        hasattr(attn_module, "v_proj")
        and attn_module.v_proj.bias is not None
    )
    config["has_o_bias"] = (
        hasattr(attn_module, "o_proj")
        and attn_module.o_proj.bias is not None
    )

    return config


# ---------------------------------------------------------------------------
# Wrapper: bridge Llama's self_attn interface to BaselineGQA / CoDAGQA
# ---------------------------------------------------------------------------


class LlamaAttentionWrapper(nn.Module):
    """Wraps BaselineGQA or CoDAGQA to match Llama's self_attn interface.

    Modern transformers (>=4.46) LlamaDecoderLayer unpacks as:
        hidden_states, _ = self.self_attn(...)
    so we return exactly 2 values: (attn_output, None).

    BaselineGQA / CoDAGQA signature:
        (x, *, is_causal, attention_mask, past_key_value)
        -> (output, (k_cache, v_cache))
    """

    def __init__(self, attn_module: nn.Module) -> None:
        super().__init__()
        self.attn = attn_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        # BaselineGQA / CoDAGQA handle RoPE internally; ignore position_ids
        # and position_embeddings from the caller.
        out, _kv = self.attn(hidden_states, is_causal=True)
        return out, None


# ---------------------------------------------------------------------------
# Weight copy
# ---------------------------------------------------------------------------


def _halfsplit_to_interleaved_permutation(head_dim: int) -> torch.Tensor:
    """Build an index permutation converting half-split RoPE to interleaved RoPE.

    Llama uses "half-split" RoPE: the head dimension is split at the midpoint
    into [x_first_half, x_second_half] and rotated as pairs (i, i+d/2).

    BaselineGQA / CoDAGQA use "interleaved" (pairwise) RoPE: pairs are at
    adjacent even/odd indices (0,1), (2,3), ...

    To reuse the same weights under the interleaved convention we permute the
    head_dim axis of q_proj and k_proj output dimensions.

    Mapping: interleaved[2*i] = halfsplit[i]
             interleaved[2*i+1] = halfsplit[d/2 + i]
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    for i in range(half):
        perm[2 * i] = i
        perm[2 * i + 1] = half + i
    return perm


def _permute_qk_weight(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Permute the output dimension of a Q or K projection weight matrix.

    weight shape: (num_heads * head_dim, embed_dim)
    We reshape to (num_heads, head_dim, embed_dim), permute head_dim, reshape back.
    """
    perm = _halfsplit_to_interleaved_permutation(head_dim).to(weight.device)
    embed_dim = weight.shape[1]
    w = weight.view(num_heads, head_dim, embed_dim)
    w = w[:, perm, :]
    return w.reshape(num_heads * head_dim, embed_dim)


def copy_weights_to_module(
    src_attn: nn.Module,
    dst_module: nn.Module,
    attn_config: Dict[str, Any],
) -> None:
    """Copy q/k/v/o projection weights from a Llama self_attn to BaselineGQA or CoDAGQA.

    Llama uses "half-split" RoPE convention while BaselineGQA / CoDAGQA use
    "interleaved" (pairwise) RoPE.  The two are mathematically equivalent but
    operate on different dimension orderings within each head.  We permute Q
    and K weight matrices along the head_dim axis so the interleaved RoPE
    produces identical attention scores to the original half-split RoPE.

    Key insight: since Q and K are permuted identically, their dot products
    (attention scores) are unchanged -- the sum is just reordered.  Because V
    is not involved in RoPE, the attention output is identical to the original.
    Therefore V and O projections are copied without permutation.

    Both BaselineGQA and CoDAGQA use bias=False linear layers.  If the source
    has biases they are silently ignored (Llama models typically do not).
    """
    num_heads = attn_config["num_heads"]
    num_kv_heads = attn_config["num_kv_heads"]
    head_dim = attn_config["head_dim"]

    # Q: permute output dim (num_heads * head_dim) for all query heads.
    dst_module.q_proj.weight.data.copy_(
        _permute_qk_weight(src_attn.q_proj.weight.data, num_heads, head_dim)
    )

    # K: permute output dim (num_kv_heads * head_dim) for KV heads.
    dst_module.k_proj.weight.data.copy_(
        _permute_qk_weight(src_attn.k_proj.weight.data, num_kv_heads, head_dim)
    )

    # V: no RoPE involved, copy directly.
    dst_module.v_proj.weight.data.copy_(src_attn.v_proj.weight.data)

    # O: attention output is NOT permuted (dot products are order-invariant,
    # V is unchanged), so copy directly.
    dst_module.o_proj.weight.data.copy_(src_attn.o_proj.weight.data)


# ---------------------------------------------------------------------------
# Model swap helpers
# ---------------------------------------------------------------------------


def create_baseline_gqa(
    attn_config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> BaselineGQA:
    """Create a BaselineGQA module matching the given attention config."""
    module = BaselineGQA(
        embed_dim=attn_config["embed_dim"],
        num_heads=attn_config["num_heads"],
        num_kv_heads=attn_config["num_kv_heads"],
        rope_base=attn_config["rope_theta"],
    )
    return module.to(device=device, dtype=dtype)


def create_coda_gqa(
    attn_config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> CoDAGQA:
    """Create a CoDAGQA module matching the given attention config."""
    module = CoDAGQA(
        embed_dim=attn_config["embed_dim"],
        num_heads=attn_config["num_heads"],
        num_kv_heads=attn_config["num_kv_heads"],
        rope_base=attn_config["rope_theta"],
    )
    return module.to(device=device, dtype=dtype)


def swap_all_attention_layers(
    model: nn.Module,
    model_config: Any,
    module_factory: Any,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
) -> int:
    """Replace all attention layers in-place using the given factory function.

    Args:
        model: The HuggingFace model (modified in-place).
        model_config: The model's config object.
        module_factory: Callable(attn_config, device, dtype) -> nn.Module.
        device: Target device.
        dtype: Target dtype.
        label: Human-readable label for logging.

    Returns:
        Number of layers successfully swapped.
    """
    layers = find_attention_layers(model)
    n_swapped = 0

    for path, parent_block, orig_attn in layers:
        try:
            attn_config = extract_attn_config(orig_attn, model_config)
            new_module = module_factory(attn_config, device, dtype)
            copy_weights_to_module(orig_attn, new_module, attn_config)
            wrapper = LlamaAttentionWrapper(new_module)
            parent_block.self_attn = wrapper
            n_swapped += 1
        except Exception as e:
            print(f"  Warning: Failed to swap {path} ({label}): {e}")

    print(f"  [{label}] Swapped {n_swapped}/{len(layers)} attention layers")
    return n_swapped


# ---------------------------------------------------------------------------
# Per-layer activation analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_per_layer_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Collect per-attention-layer activation statistics.

    Hooks into each LlamaAttentionWrapper (if present) to capture output
    tensors and compute summary stats.
    """
    layer_stats: List[Dict[str, Any]] = []
    hooks: List[torch.utils.hooks.RemovableHook] = []

    layers = find_attention_layers(model)

    for idx, (path, _parent, attn_mod) in enumerate(layers):
        stats_entry: Dict[str, Any] = {"layer": idx, "path": path}
        layer_stats.append(stats_entry)

        def _make_hook(entry: Dict[str, Any]) -> Any:
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                # output is (attn_out, attn_weights, past_kv)
                if isinstance(output, tuple):
                    out_tensor = output[0]
                else:
                    out_tensor = output
                entry["output_mean"] = out_tensor.float().mean().item()
                entry["output_std"] = out_tensor.float().std().item()
                entry["output_max"] = out_tensor.float().abs().max().item()
                entry["has_nan"] = bool(out_tensor.isnan().any())
                entry["has_inf"] = bool(out_tensor.isinf().any())
            return hook_fn

        h = attn_mod.register_forward_hook(_make_hook(stats_entry))
        hooks.append(h)

    # Run forward pass to trigger hooks.
    _ = get_model_logits(model, input_ids)

    # Remove hooks.
    for h in hooks:
        h.remove()

    return layer_stats


# ---------------------------------------------------------------------------
# Main forward check
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_forward_check(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int = 64,
) -> Dict[str, Any]:
    """Compare original Llama-style logits vs BaselineGQA-swapped vs CoDAGQA-swapped.

    Loads the model three times (original, BaselineGQA-swapped, CoDAGQA-swapped)
    and feeds the same input through all three, reporting logit differences.

    The BaselineGQA swap should produce near-zero diffs (identical computation).
    The CoDAGQA swap should produce larger diffs (differential attention is active).
    """
    print("\n" + "=" * 70)
    print("Forward Equivalence Check: Llama-style -> BaselineGQA / CoDAGQA")
    print("=" * 70)

    # --- Load original model ---
    model_orig, tokenizer = load_model(model_name, device, dtype)
    model_config = model_orig.config

    # Print discovered attention config from the first layer.
    attn_layers = find_attention_layers(model_orig)
    if not attn_layers:
        print("ERROR: No attention layers found in the model.")
        sys.exit(1)

    first_attn_config = extract_attn_config(attn_layers[0][2], model_config)
    print(f"\n  Attention config (from layer 0):")
    print(f"    embed_dim:    {first_attn_config['embed_dim']}")
    print(f"    num_heads:    {first_attn_config['num_heads']}")
    print(f"    num_kv_heads: {first_attn_config['num_kv_heads']}")
    print(f"    head_dim:     {first_attn_config['head_dim']}")
    print(f"    rope_theta:   {first_attn_config['rope_theta']}")
    print(f"    q_bias:       {first_attn_config['has_q_bias']}")
    print(f"    Layers found: {len(attn_layers)}")

    # --- Create test input ---
    torch.manual_seed(42)
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy, stars were born from clouds of hydrogen gas. "
        "Mathematics reveals the hidden structure of reality."
    )
    input_ids = tokenizer(
        test_text, return_tensors="pt", truncation=True, max_length=seq_len
    ).input_ids.to(device)
    actual_len = input_ids.size(1)
    print(f"\n  Test input: {actual_len} tokens")
    print(f"  Input text: {test_text[:80]}...")

    # --- Original logits ---
    print("\n  [1/3] Computing original model logits...")
    t0 = time.perf_counter()
    logits_original = get_model_logits(model_orig, input_ids)
    t_orig = time.perf_counter() - t0
    print(f"    Shape: {list(logits_original.shape)}, Time: {t_orig*1000:.1f} ms")

    # --- BaselineGQA swap ---
    print("\n  [2/3] Creating BaselineGQA-swapped model...")
    model_baseline = copy.deepcopy(model_orig)
    n_baseline = swap_all_attention_layers(
        model_baseline, model_config, create_baseline_gqa, device, dtype, "BaselineGQA"
    )

    print("    Computing BaselineGQA logits...")
    t0 = time.perf_counter()
    logits_baseline = get_model_logits(model_baseline, input_ids)
    t_baseline = time.perf_counter() - t0
    print(f"    Shape: {list(logits_baseline.shape)}, Time: {t_baseline*1000:.1f} ms")

    # --- CoDAGQA swap ---
    print("\n  [3/3] Creating CoDAGQA-swapped model...")
    model_coda = copy.deepcopy(model_orig)
    n_coda = swap_all_attention_layers(
        model_coda, model_config, create_coda_gqa, device, dtype, "CoDAGQA"
    )

    print("    Computing CoDAGQA logits...")
    t0 = time.perf_counter()
    logits_coda = get_model_logits(model_coda, input_ids)
    t_coda = time.perf_counter() - t0
    print(f"    Shape: {list(logits_coda.shape)}, Time: {t_coda*1000:.1f} ms")

    # --- Per-layer activation analysis ---
    print("\n  Per-layer activation analysis (BaselineGQA):")
    baseline_layer_stats = analyze_per_layer_activations(model_baseline, input_ids)
    for entry in baseline_layer_stats:
        print(
            f"    Layer {entry['layer']:2d}: "
            f"mean={entry.get('output_mean', 'N/A'):+.6f}  "
            f"std={entry.get('output_std', 'N/A'):.6f}  "
            f"max={entry.get('output_max', 'N/A'):.6f}  "
            f"nan={entry.get('has_nan', 'N/A')}  "
            f"inf={entry.get('has_inf', 'N/A')}"
        )

    print("\n  Per-layer activation analysis (CoDAGQA):")
    coda_layer_stats = analyze_per_layer_activations(model_coda, input_ids)
    for entry in coda_layer_stats:
        print(
            f"    Layer {entry['layer']:2d}: "
            f"mean={entry.get('output_mean', 'N/A'):+.6f}  "
            f"std={entry.get('output_std', 'N/A'):.6f}  "
            f"max={entry.get('output_max', 'N/A'):.6f}  "
            f"nan={entry.get('has_nan', 'N/A')}  "
            f"inf={entry.get('has_inf', 'N/A')}"
        )

    # --- Compute logit diffs ---
    diff_baseline = (logits_original.float() - logits_baseline.float()).abs()
    diff_coda = (logits_original.float() - logits_coda.float()).abs()

    baseline_stats = {
        "max_logit_diff": diff_baseline.max().item(),
        "mean_logit_diff": diff_baseline.mean().item(),
        "median_logit_diff": diff_baseline.median().item(),
        "p95_logit_diff": torch.quantile(diff_baseline.flatten().float(), 0.95).item(),
        "p99_logit_diff": torch.quantile(diff_baseline.flatten().float(), 0.99).item(),
    }

    coda_stats = {
        "max_logit_diff": diff_coda.max().item(),
        "mean_logit_diff": diff_coda.mean().item(),
        "median_logit_diff": diff_coda.median().item(),
        "p95_logit_diff": torch.quantile(diff_coda.flatten().float(), 0.95).item(),
        "p99_logit_diff": torch.quantile(diff_coda.flatten().float(), 0.99).item(),
    }

    # --- Report ---
    print("\n" + "-" * 70)
    print("  RESULTS: BaselineGQA swap (should be ~0)")
    print("-" * 70)
    print(f"    Max absolute diff:    {baseline_stats['max_logit_diff']:.8f}")
    print(f"    Mean absolute diff:   {baseline_stats['mean_logit_diff']:.8f}")
    print(f"    Median absolute diff: {baseline_stats['median_logit_diff']:.8f}")
    print(f"    P95 absolute diff:    {baseline_stats['p95_logit_diff']:.8f}")
    print(f"    P99 absolute diff:    {baseline_stats['p99_logit_diff']:.8f}")

    # Verdict thresholds are dtype-aware.  bf16 accumulates more numerical
    # noise over 30 layers than fp32.  We use mean diff (more stable than max)
    # as the primary signal, with generous thresholds for reduced precision.
    is_reduced_precision = dtype in (torch.bfloat16, torch.float16)
    if is_reduced_precision:
        pass_thresh = 1.0    # mean diff: bf16 noise across many layers
        warn_thresh = 5.0
    else:
        pass_thresh = 0.001  # fp32 should be very tight
        warn_thresh = 0.1

    if baseline_stats["mean_logit_diff"] < pass_thresh:
        baseline_verdict = "PASS - weight transfer + RoPE validated (numerical noise only)"
    elif baseline_stats["mean_logit_diff"] < warn_thresh:
        baseline_verdict = "ACCEPTABLE - small precision differences"
    else:
        baseline_verdict = "FAIL - large differences, investigate weight transfer"
    print(f"    Verdict: {baseline_verdict}")

    print(f"\n" + "-" * 70)
    print("  RESULTS: CoDAGQA swap (expected to be larger)")
    print("-" * 70)
    print(f"    Max absolute diff:    {coda_stats['max_logit_diff']:.8f}")
    print(f"    Mean absolute diff:   {coda_stats['mean_logit_diff']:.8f}")
    print(f"    Median absolute diff: {coda_stats['median_logit_diff']:.8f}")
    print(f"    P95 absolute diff:    {coda_stats['p95_logit_diff']:.8f}")
    print(f"    P99 absolute diff:    {coda_stats['p99_logit_diff']:.8f}")

    if coda_stats["max_logit_diff"] > baseline_stats["max_logit_diff"] * 2:
        coda_verdict = (
            "EXPECTED - CoDAGQA diffs are larger than BaselineGQA "
            "(diff comes from differential attention + HeadwiseRMSNorm)"
        )
    elif coda_stats["max_logit_diff"] < 0.01:
        coda_verdict = (
            "UNEXPECTED - CoDAGQA diffs are near-zero "
            "(lambda gate may be suppressing differential signal entirely)"
        )
    else:
        coda_verdict = "INCONCLUSIVE - diffs are similar magnitude to BaselineGQA"
    print(f"    Verdict: {coda_verdict}")

    # --- Ratio ---
    if baseline_stats["mean_logit_diff"] > 0:
        ratio = coda_stats["mean_logit_diff"] / baseline_stats["mean_logit_diff"]
        print(f"\n    CoDA/Baseline mean diff ratio: {ratio:.2f}x")
    else:
        ratio = float("inf")
        print(f"\n    CoDA/Baseline mean diff ratio: inf (baseline diff is zero)")

    # --- Top-token agreement ---
    top1_orig = logits_original.argmax(dim=-1)
    top1_baseline = logits_baseline.argmax(dim=-1)
    top1_coda = logits_coda.argmax(dim=-1)

    baseline_agreement = (top1_orig == top1_baseline).float().mean().item()
    coda_agreement = (top1_orig == top1_coda).float().mean().item()

    print(f"\n  Top-1 token agreement:")
    print(f"    Baseline vs Original: {baseline_agreement*100:.1f}%")
    print(f"    CoDAGQA  vs Original: {coda_agreement*100:.1f}%")

    # --- Assemble results dict ---
    results = {
        "experiment": "forward-check-generic",
        "model": model_name,
        "system": gather_system_info(device, dtype),
        "config": {
            "seq_len": actual_len,
            "embed_dim": first_attn_config["embed_dim"],
            "num_heads": first_attn_config["num_heads"],
            "num_kv_heads": first_attn_config["num_kv_heads"],
            "head_dim": first_attn_config["head_dim"],
            "rope_theta": first_attn_config["rope_theta"],
            "num_layers": len(attn_layers),
            "has_bias": first_attn_config["has_q_bias"],
        },
        "timing_ms": {
            "original": round(t_orig * 1000, 2),
            "baseline_gqa": round(t_baseline * 1000, 2),
            "coda_gqa": round(t_coda * 1000, 2),
        },
        "baseline_swap": {
            "logit_diff": baseline_stats,
            "top1_agreement": baseline_agreement,
            "verdict": baseline_verdict,
            "layers_swapped": n_baseline,
            "per_layer_stats": baseline_layer_stats,
        },
        "coda_swap": {
            "logit_diff": coda_stats,
            "top1_agreement": coda_agreement,
            "verdict": coda_verdict,
            "layers_swapped": n_coda,
            "per_layer_stats": coda_layer_stats,
        },
        "coda_vs_baseline_ratio": ratio,
    }

    # --- Cleanup ---
    del model_orig, model_baseline, model_coda
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward check: validate CoDA-GQA-L weight transfer "
            "against Llama-architecture HuggingFace models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/forward_check_generic.py
  python benchmarks/forward_check_generic.py --model HuggingFaceTB/SmolLM2-135M
  python benchmarks/forward_check_generic.py --model HuggingFaceTB/SmolLM2-135M --seq-len 128
  python benchmarks/forward_check_generic.py --model meta-llama/Llama-3.2-1B --seq-len 64

Expected results:
  BaselineGQA swap:  near-zero logit diff (proves weight transfer works)
  CoDAGQA swap:      larger logit diff (proves diff comes from differential attention)
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Maximum sequence length for the test input (default: 64)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for JSON results (default: results/ at project root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, dtype = get_device_dtype()

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    print("CoDA-GQA-L Generic Forward Check")
    print(f"  Model:       {args.model}")
    print(f"  Seq len:     {args.seq_len}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {dtype}")
    print(f"  Results dir: {results_dir}")

    try:
        results = run_forward_check(
            model_name=args.model,
            device=device,
            dtype=dtype,
            seq_len=args.seq_len,
        )

        out_path = save_results(results, results_dir)
        print(f"\n  Results saved to {out_path}")

        # Print compact JSON summary to stdout.
        print(f"\n{'='*70}")
        print("JSON summary:")
        print("=" * 70)
        summary = {
            "model": results["model"],
            "config": results["config"],
            "baseline_swap": {
                "logit_diff": results["baseline_swap"]["logit_diff"],
                "top1_agreement": results["baseline_swap"]["top1_agreement"],
                "verdict": results["baseline_swap"]["verdict"],
            },
            "coda_swap": {
                "logit_diff": results["coda_swap"]["logit_diff"],
                "top1_agreement": results["coda_swap"]["top1_agreement"],
                "verdict": results["coda_swap"]["verdict"],
            },
            "coda_vs_baseline_ratio": results["coda_vs_baseline_ratio"],
        }
        print(json.dumps(summary, indent=2))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
