"""Fine-tune a Llama-family model with CoDA-GQA-L attention.

Two-phase training:
  Phase 1 (unbounded): Teaches the model to use CoDA's differential attention
    (lambda, theta, head_norm) with full KV context.
  Phase 2 (bounded, optional): Switches to bounded KV cache so the model
    learns to work within O(W+Me+Ms) memory. This is where the write gate,
    memory banks, and EMA parameters actually get used.

Training flow:
  1. Load pre-trained model, swap attention layers to CoDA-GQA (unbounded)
  2. Freeze non-attention params (configurable)
  3. Phase 1: Train unbounded with next-token prediction
  4. Phase 2: Switch to bounded, continue training (model adapts to limited context)
  5. Evaluate: unbounded PPL + bounded PPL

Usage:
    # Quick test on SmolLM2-135M (~5 min on GPU)
    python benchmarks/train_coda.py --model HuggingFaceTB/SmolLM2-135M \
        --max-steps 200 --eval-every 100

    # With Phase 2 bounded training (the key experiment)
    python benchmarks/train_coda.py --model HuggingFaceTB/SmolLM2-135M \
        --max-steps 200 --bounded-steps 100 --bounded-config medium

    # Mistral 7B on H100 (~6 hours with Phase 2)
    python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
        --max-steps 2000 --bounded-steps 1000 --bounded-config medium \
        --batch-size 2 --grad-accum 4

    # Llama 3.1 8B (full pipeline)
    python benchmarks/train_coda.py --model meta-llama/Llama-3.1-8B \
        --max-steps 3000 --bounded-steps 2000 --bounded-config large \
        --batch-size 1 --grad-accum 8

    # Train CoDA params only (fastest, smallest memory)
    python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \
        --freeze coda-only --lr-coda 1e-3

Requires: transformers, datasets, torch >= 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    print(f"Missing: {', '.join(_MISSING)}. pip install {' '.join(_MISSING)}")
    sys.exit(1)

from coda_gqa_l import CoDAGQALandmarkPerf2, LlamaCoDAAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_bytes(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _token_cache_dir() -> Path:
    """Persistent cache directory that survives CWD changes.

    Priority: $CODA_TOKEN_CACHE > /workspace/.token_cache (RunPod) >
    script-relative .token_cache.
    """
    env = os.environ.get("CODA_TOKEN_CACHE")
    if env:
        return Path(env)
    # RunPod /workspace is persistent across restarts
    if Path("/workspace").is_dir():
        return Path("/workspace/.token_cache")
    # Fall back to script directory (not CWD)
    return Path(__file__).resolve().parent / ".token_cache"


def _token_cache_path(model_name: str, dataset_name: str, seq_len: int, split: str = "train") -> Path:
    """Deterministic cache path for tokenized data."""
    import hashlib
    key = f"{model_name}|{dataset_name}|{seq_len}|{split}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    safe_name = model_name.replace("/", "_")
    return _token_cache_dir() / f"{safe_name}_{dataset_name}_{split}_{seq_len}_{h}.pt"


def load_train_tokens(
    tokenizer, seq_len: int, dataset_name: str = "wikitext-103",
    model_name: str = "",
) -> torch.Tensor:
    """Load and chunk training data. Returns tensor (N, seq_len).

    Caches tokenized chunks to disk so subsequent runs skip the ~10 min
    tokenization step for large datasets like wikitext-103.
    """
    print(f"  Dataset: {dataset_name}")

    cache = _token_cache_path(model_name, dataset_name, seq_len, "train")
    if cache.exists():
        print(f"  Loading cached tokens from {cache}")
        chunks = torch.load(cache, map_location="cpu", weights_only=True)
        print(f"  {chunks.shape[0]:,} chunks x {seq_len} = {chunks.numel():,} tokens")
        return chunks

    if dataset_name in ("wikitext-103", "wikitext-2"):
        config_name = (
            "wikitext-103-raw-v1" if "103" in dataset_name
            else "wikitext-2-raw-v1"
        )
        ds = load_dataset("wikitext", config_name, split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    elif Path(dataset_name).is_file():
        text = Path(dataset_name).read_text(encoding="utf-8")
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Use 'wikitext-103', 'wikitext-2', or a path to a .txt file."
        )

    print(f"  Tokenizing ({len(text):,} chars)...")
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    n = len(tokens) // seq_len
    chunks = tokens[: n * seq_len].view(n, seq_len)
    print(f"  {n:,} chunks x {seq_len} = {n * seq_len:,} tokens")

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chunks, cache)
    print(f"  Cached tokens to {cache} ({cache.stat().st_size / 1024 / 1024:.0f} MB)")

    return chunks


def load_eval_tokens(
    tokenizer, max_tokens: int = 50_000, model_name: str = "",
) -> torch.Tensor:
    """Load WikiText-2 test tokens for evaluation.

    Caches to disk like load_train_tokens.
    """
    cache = _token_cache_path(model_name, "wikitext-2-eval", max_tokens, "test")
    if cache.exists():
        print(f"  Loading cached eval tokens from {cache}")
        return torch.load(cache, map_location="cpu", weights_only=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, cache)
    print(f"  Cached eval tokens to {cache}")

    return tokens


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def swap_attention(
    model: nn.Module,
    head_norm_mode: str = "identity",
    theta_init: float = 0.0,
    lambda_init_bias: float = -6.0,
    bounded: bool = False,
    window: int = 256,
    num_landmarks_exact: int = 64,
    num_landmarks_summary: int = 64,
    block_size: int = 256,
) -> Tuple[nn.Module, List[LlamaCoDAAdapter]]:
    """Replace Llama-family attention with CoDA adapters.

    Default initialization is near-identity so the model starts close to its
    pre-trained PPL and the differential mechanism "wakes up" gradually:

    - theta_init=0: noise query = signal query initially. Training moves theta
      toward an angle that decorrelates signal from noise.
    - head_norm_mode="identity": bypasses RMSNorm to preserve output scale.
      Using "full" would normalize the output to unit RMS, destroying the
      pre-trained scale and causing catastrophic initial PPL.
    - lambda_proj default init: bias=-6 (lambda~0.0025), random weights.
      Small enough to not disrupt outputs, large enough for gradients to flow.
    """
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError("Model must be Llama-family (model.model.layers)")

    blocks = model.model.layers
    adapters: List[LlamaCoDAAdapter] = []

    for block in blocks:
        attn = block.self_attn
        adapter = LlamaCoDAAdapter.from_llama_attention(
            attn,
            bounded=bounded,
            head_norm_mode=head_norm_mode,
            theta_init=theta_init,
            lambda_init_bias=lambda_init_bias,
            rope_interleaved=False,  # Llama convention
            window=window,
            num_landmarks_exact=num_landmarks_exact,
            num_landmarks_summary=num_landmarks_summary,
            block_size=block_size,
        )
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        adapter = adapter.to(device=device, dtype=dtype)
        block.self_attn = adapter
        adapters.append(adapter)

    mode = "bounded" if bounded else "unbounded"
    print(f"  Swapped {len(adapters)}/{len(blocks)} attention layers "
          f"({mode}, theta={theta_init:.2f}, head_norm={head_norm_mode})")
    if bounded:
        print(f"  Bounded config: W={window}, Me={num_landmarks_exact}, "
              f"Ms={num_landmarks_summary}, block_size={block_size}")
    return model, adapters


def switch_to_bounded(
    model: nn.Module,
    adapters_ub: List[LlamaCoDAAdapter],
    device: torch.device,
    dtype: torch.dtype,
    *,
    window: int = 256,
    num_landmarks_exact: int = 64,
    num_landmarks_summary: int = 64,
    max_landmarks_exact: Optional[int] = None,
    max_landmarks_summary: Optional[int] = None,
    block_size: int = 256,
    detach_evicted: bool = True,
) -> Tuple[nn.Module, List[LlamaCoDAAdapter]]:
    """Switch unbounded adapters to bounded, copying all trained weights.

    Creates new bounded LlamaCoDAAdapters, copies shared weights (Q/K/V/O
    projections, lambda_proj, theta, head_norm) from the unbounded adapters.
    Bounded-only params (write_proj, summary_eta_logit) start at their defaults.

    Args:
        max_landmarks_exact: Maximum exact bank slots for dynamic expansion.
            None (default) means no expansion (fixed at num_landmarks_exact).
        max_landmarks_summary: Maximum summary bank slots for dynamic expansion.
            None (default) means no expansion (fixed at num_landmarks_summary).
        detach_evicted: If True (default), evicted tokens are detached before
            writing to memory banks, blocking gradients through the bank update
            path.  Set False to allow write_proj and summary_eta_logit to
            receive gradients during training (experimental -- may increase
            memory and cause gradient instability).

    Returns (model, bounded_adapters).
    """
    blocks = model.model.layers
    adapters_b: List[LlamaCoDAAdapter] = []

    for i, (block, adapter_ub) in enumerate(zip(blocks, adapters_ub)):
        coda_ub = adapter_ub.coda

        expand_kwargs = {}
        if max_landmarks_exact is not None:
            expand_kwargs["max_landmarks_exact"] = max_landmarks_exact
        if max_landmarks_summary is not None:
            expand_kwargs["max_landmarks_summary"] = max_landmarks_summary

        adapter_b = LlamaCoDAAdapter(
            hidden_size=adapter_ub.hidden_size,
            num_heads=adapter_ub.num_heads,
            num_kv_heads=adapter_ub.num_kv_heads,
            head_dim=adapter_ub.head_dim,
            bounded=True,
            rope_theta=10_000.0,
            head_norm_mode=coda_ub.head_norm_mode,
            rope_interleaved=coda_ub.rope_interleaved,
            window=window,
            num_landmarks_exact=num_landmarks_exact,
            num_landmarks_summary=num_landmarks_summary,
            block_size=block_size,
            detach_evicted=detach_evicted,
            **expand_kwargs,
        )
        adapter_b = adapter_b.to(device=device, dtype=dtype)
        coda_b = adapter_b.coda

        # Transfer shared weights.
        with torch.no_grad():
            coda_b.q_proj.load_state_dict(coda_ub.q_proj.state_dict())
            coda_b.k_proj.load_state_dict(coda_ub.k_proj.state_dict())
            coda_b.v_proj.load_state_dict(coda_ub.v_proj.state_dict())
            coda_b.o_proj.load_state_dict(coda_ub.o_proj.state_dict())
            coda_b.lambda_proj.load_state_dict(coda_ub.lambda_proj.state_dict())
            coda_b.theta.copy_(coda_ub.theta)
            if (
                hasattr(coda_b, "head_norm")
                and hasattr(coda_ub, "head_norm")
                and not isinstance(coda_ub.head_norm, nn.Identity)
            ):
                coda_b.head_norm.load_state_dict(coda_ub.head_norm.state_dict())

        block.self_attn = adapter_b
        adapters_b.append(adapter_b)

    print(f"  Switched {len(adapters_b)} layers: unbounded -> bounded")
    expand_str = ""
    if max_landmarks_exact is not None and max_landmarks_exact > num_landmarks_exact:
        expand_str += f", max_Me={max_landmarks_exact}"
    if max_landmarks_summary is not None and max_landmarks_summary > num_landmarks_summary:
        expand_str += f", max_Ms={max_landmarks_summary}"
    print(f"  Config: W={window}, Me={num_landmarks_exact}, "
          f"Ms={num_landmarks_summary}, block_size={block_size}{expand_str}")
    return model, adapters_b


def reset_adapter_states(adapters: List[LlamaCoDAAdapter]) -> None:
    """Reset inference state for all bounded adapters (call per training batch)."""
    for a in adapters:
        a.reset_state()


def freeze_params(
    model: nn.Module,
    adapters: List[LlamaCoDAAdapter],
    freeze: str,
) -> None:
    """Freeze parameters based on strategy."""
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    coda_param_names = {"lambda_proj", "theta", "head_norm"}

    if freeze == "coda-only":
        for adapter in adapters:
            for name, p in adapter.named_parameters():
                if any(cn in name for cn in coda_param_names):
                    p.requires_grad = True
    elif freeze == "attention":
        for adapter in adapters:
            for p in adapter.parameters():
                p.requires_grad = True
    elif freeze == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze strategy: {freeze!r}")

    n_train = count_params(model, trainable_only=True)
    n_total = count_params(model)
    print(f"  Trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.1f}%)")


def freeze_differential(adapters: List[LlamaCoDAAdapter]) -> None:
    """Freeze theta and lambda_proj so differential attention is fully disabled.

    Used for ablation: with theta=0 (no rotation) and lambda≈0 (no noise
    subtraction), the model behaves as standard GQA. Freezing prevents
    these params from learning during training, isolating the contribution
    of the bounded memory banks alone.
    """
    n_frozen = 0
    for adapter in adapters:
        for name, p in adapter.named_parameters():
            if "theta" in name or "lambda_proj" in name:
                p.requires_grad = False
                n_frozen += 1
    print(f"  Ablation: froze {n_frozen} differential params (theta + lambda_proj)")


def setup_optimizer(
    model: nn.Module,
    adapters: List[LlamaCoDAAdapter],
    lr: float,
    lr_coda: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create AdamW with separate LR for CoDA params vs projections."""
    coda_names = {"lambda_proj", "theta", "head_norm"}
    adapter_param_ids = set()
    for adapter in adapters:
        for p in adapter.parameters():
            adapter_param_ids.add(id(p))

    coda_params: List[nn.Parameter] = []
    proj_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in adapter_param_ids:
            if any(cn in name for cn in coda_names):
                coda_params.append(p)
            else:
                proj_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if coda_params:
        groups.append({"params": coda_params, "lr": lr_coda, "weight_decay": 0.0})
        print(f"    CoDA params: {sum(p.numel() for p in coda_params):,} @ lr={lr_coda}")
    if proj_params:
        groups.append({"params": proj_params, "lr": lr, "weight_decay": weight_decay})
        print(f"    Proj params: {sum(p.numel() for p in proj_params):,} @ lr={lr}")
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
        print(f"    Other params: {sum(p.numel() for p in other_params):,} @ lr={lr}")

    return torch.optim.AdamW(groups)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    eval_tokens: torch.Tensor,
    device: torch.device,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute perplexity on pre-tokenized eval data (unbounded models only).

    Uses overlapping sliding window for efficient PPL estimation.
    NOT suitable for bounded/stateful models -- use eval_ppl_sequential instead.
    """
    was_training = model.training
    model.eval()

    input_ids = eval_tokens.unsqueeze(0).to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    n_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end]
        outputs = model(chunk, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        if begin > 0:
            skip = max_length - stride - 1
            if 0 < skip < loss.size(0):
                loss = loss[skip:]

        nlls.append(loss.sum())
        n_tokens += loss.numel()
        if end >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens).item()
    if was_training:
        model.train()
    return ppl


@torch.no_grad()
def eval_ppl_sequential(
    model: nn.Module,
    adapters: List[LlamaCoDAAdapter],
    eval_tokens: torch.Tensor,
    device: torch.device,
    chunk_size: int = 2048,
    max_tokens: int = 0,
) -> float:
    """Evaluate perplexity with non-overlapping sequential chunks.

    Each chunk gets a **fresh state** (matching training behavior) so that
    positions never exceed the model's trained context length.  Without this,
    ``state.pos`` accumulates across chunks, producing RoPE embeddings far
    beyond the trained range and inflating PPL to nonsense values (e.g. 2400
    instead of ~65).

    Long-range retention across chunks is tested separately via the needle
    benchmark, not PPL.
    """
    was_training = model.training
    model.eval()

    tokens = eval_tokens[:max_tokens] if max_tokens > 0 else eval_tokens
    input_ids = tokens.unsqueeze(0).to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    n_tokens = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        if end - start < 2:
            break

        # Fresh state per chunk: positions start at 0, banks are empty.
        # This matches the training forward path (which also allocates fresh
        # state per call) and keeps RoPE positions within the model's trained
        # context window.
        reset_adapter_states(adapters)

        chunk = input_ids[:, start:end]
        outputs = model(chunk, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        nlls.append(loss.sum())
        n_tokens += loss.numel()

    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens).item()

    # Clean up.
    reset_adapter_states(adapters)

    if was_training:
        model.train()
    return ppl


@torch.no_grad()
def eval_bounded_ppl(
    model: nn.Module,
    adapters: List[LlamaCoDAAdapter],
    eval_tokens: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    window: int = 256,
    Me: int = 64,
    Ms: int = 64,
    block_size: int = 256,
    chunk_size: int = 2048,
    max_tokens: int = 20_000,
) -> float:
    """Evaluate with bounded CoDA attention by transferring trained weights.

    For Phase 1 (unbounded adapters): Creates bounded adapters, copies trained
    Q/K/V/O + CoDA weights. Bounded-only params (write gate, EMA eta) use
    their default initialization.

    For Phase 2 (bounded adapters): Evaluates the model directly since adapters
    are already bounded with trained params.
    """
    is_already_bounded = adapters and adapters[0].bounded

    if is_already_bounded:
        # Phase 2: model already has bounded adapters with trained params.
        # Just use eval_ppl_sequential directly.
        return eval_ppl_sequential(
            model, adapters, eval_tokens, device,
            chunk_size=chunk_size,
            max_tokens=max_tokens,
        )

    # Phase 1: create bounded copies from unbounded adapters.
    import copy

    model_b = copy.deepcopy(model)
    model_b.eval()

    bounded_adapters: List[LlamaCoDAAdapter] = []

    # Replace each unbounded adapter with a bounded one using trained weights.
    for i, block in enumerate(model_b.model.layers):
        adapter_ub = adapters[i]
        coda_ub = adapter_ub.coda

        adapter_b = LlamaCoDAAdapter(
            hidden_size=adapter_ub.hidden_size,
            num_heads=adapter_ub.num_heads,
            num_kv_heads=adapter_ub.num_kv_heads,
            head_dim=adapter_ub.head_dim,
            bounded=True,
            rope_theta=coda_ub.rope.inv_freq.numel() and 10_000.0,
            window=window,
            num_landmarks_exact=Me,
            num_landmarks_summary=Ms,
            block_size=block_size,
            head_norm_mode=coda_ub.head_norm_mode,
            rope_interleaved=coda_ub.rope_interleaved,
        )
        adapter_b = adapter_b.to(device=device, dtype=dtype)
        coda_b = adapter_b.coda

        # Transfer shared weights: projections + CoDA params.
        with torch.no_grad():
            coda_b.q_proj.load_state_dict(coda_ub.q_proj.state_dict())
            coda_b.k_proj.load_state_dict(coda_ub.k_proj.state_dict())
            coda_b.v_proj.load_state_dict(coda_ub.v_proj.state_dict())
            coda_b.o_proj.load_state_dict(coda_ub.o_proj.state_dict())
            coda_b.lambda_proj.load_state_dict(coda_ub.lambda_proj.state_dict())
            coda_b.theta.copy_(coda_ub.theta)
            if (
                hasattr(coda_b, "head_norm")
                and hasattr(coda_ub, "head_norm")
                and not isinstance(coda_ub.head_norm, nn.Identity)
            ):
                coda_b.head_norm.load_state_dict(coda_ub.head_norm.state_dict())

        block.self_attn = adapter_b
        bounded_adapters.append(adapter_b)

    # Streaming PPL on a subset of eval data.
    ppl = eval_ppl_sequential(
        model_b, bounded_adapters, eval_tokens, device,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )

    del model_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ppl


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def save_checkpoint(
    adapters: List[LlamaCoDAAdapter],
    path: Path,
    step: int,
    log: Optional[List[Dict]] = None,
) -> None:
    """Save adapter weights."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {f"layer_{i}": a.state_dict() for i, a in enumerate(adapters)}
    torch.save(state, path / "coda_adapters.pt")
    meta = {"step": step}
    if log:
        meta["recent_log"] = log[-5:]
    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def train(
    model: nn.Module,
    adapters: List[LlamaCoDAAdapter],
    optimizer: torch.optim.Optimizer,
    train_chunks: torch.Tensor,
    eval_tokens: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    warmup_steps: int,
    eval_every: int,
    save_every: int,
    output_dir: Path,
    max_grad_norm: float = 1.0,
    eval_bounded: bool = True,
    eval_bounded_tokens: int = 20_000,
) -> List[Dict[str, Any]]:
    """Main training loop."""
    model.train()

    # Cosine schedule with linear warmup.
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Shuffle training data.
    n_chunks = len(train_chunks)
    perm = torch.randperm(n_chunks)
    chunk_idx = 0

    def get_batch() -> torch.Tensor:
        nonlocal chunk_idx, perm
        if chunk_idx + batch_size > n_chunks:
            perm = torch.randperm(n_chunks)
            chunk_idx = 0
        batch = train_chunks[perm[chunk_idx : chunk_idx + batch_size]].to(device)
        chunk_idx += batch_size
        return batch

    # Detect if we're in bounded training mode (adapters have bounded=True).
    is_bounded_training = adapters and adapters[0].bounded

    log: List[Dict[str, Any]] = []
    best_ppl = float("inf")
    t0 = time.time()
    accum_loss = 0.0
    tokens_total = 0

    # Initial eval.
    if eval_tokens is not None:
        if is_bounded_training:
            ppl0 = eval_ppl_sequential(
                model, adapters, eval_tokens, device,
                chunk_size=seq_len, max_tokens=eval_bounded_tokens,
            )
            print(f"  Step 0: eval PPL (bounded) = {ppl0:.2f}")
        else:
            ppl0 = eval_ppl(model, eval_tokens, device, max_length=seq_len)
            print(f"  Step 0: eval PPL = {ppl0:.2f}")
        log.append({"step": 0, "eval_ppl": round(ppl0, 2)})
        best_ppl = ppl0

    optimizer.zero_grad()
    print_every = max(grad_accum * 5, grad_accum)

    for step in range(1, max_steps + 1):
        input_ids = get_batch()

        # In bounded mode, reset state before each forward pass so each
        # training sequence starts with a fresh KV buffer.
        if is_bounded_training:
            reset_adapter_states(adapters)

        # Forward + loss.
        with torch.amp.autocast(
            device.type, dtype=dtype, enabled=(dtype != torch.float32),
        ):
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
            loss_scaled = loss / grad_accum

        loss_scaled.backward()
        accum_loss += loss.item()
        tokens_total += input_ids.numel()

        # Optimizer step with gradient accumulation.
        if step % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = accum_loss / grad_accum
            elapsed = time.time() - t0
            tok_s = tokens_total / elapsed if elapsed > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            entry = {
                "step": step,
                "loss": round(avg_loss, 4),
                "ppl_train": round(math.exp(min(avg_loss, 20)), 2),
                "grad_norm": round(gn, 4),
                "lr": lr_now,
            }
            log.append(entry)

            if step % print_every == 0:
                print(
                    f"  Step {step:>6}: loss={avg_loss:.4f}  "
                    f"ppl={entry['ppl_train']:.1f}  "
                    f"gnorm={gn:.3f}  lr={lr_now:.2e}  "
                    f"{tok_s:.0f} tok/s"
                )

            accum_loss = 0.0

        # Periodic eval.
        if eval_every and step % eval_every == 0 and eval_tokens is not None:
            if is_bounded_training:
                ppl = eval_ppl_sequential(
                    model, adapters, eval_tokens, device,
                    chunk_size=seq_len, max_tokens=eval_bounded_tokens,
                )
                label = "bounded"
            else:
                ppl = eval_ppl(model, eval_tokens, device, max_length=seq_len)
                label = "unbounded"
            marker = "best" if ppl < best_ppl else f"+{ppl - best_ppl:.2f}"
            print(f"  Step {step}: eval PPL ({label}) = {ppl:.2f} ({marker})")
            log.append({"step": step, "eval_ppl": round(ppl, 2)})
            if ppl < best_ppl:
                best_ppl = ppl
                save_checkpoint(adapters, output_dir / "best", step, log)

        # Periodic save.
        if save_every and step % save_every == 0:
            save_checkpoint(adapters, output_dir / f"step_{step}", step, log)

    # Final eval.
    if eval_tokens is not None:
        if is_bounded_training:
            # Phase 2: model already has bounded adapters, eval directly.
            final_ppl = eval_ppl_sequential(
                model, adapters, eval_tokens, device,
                chunk_size=seq_len, max_tokens=eval_bounded_tokens,
            )
            print(f"\n  Final eval PPL (bounded): {final_ppl:.2f}")
            log.append({
                "step": max_steps,
                "eval_ppl": round(final_ppl, 2),
                "bounded_ppl": round(final_ppl, 2),
                "bounded_config": "in-place",
                "final": True,
            })
        else:
            # Phase 1: unbounded eval + optional bounded projection.
            final_ppl = eval_ppl(model, eval_tokens, device, max_length=seq_len)
            print(f"\n  Final eval PPL (unbounded): {final_ppl:.2f}")
            log.append({"step": max_steps, "eval_ppl": round(final_ppl, 2), "final": True})

            if eval_bounded:
                print(f"\n  Evaluating bounded PPL (W=256, Me=64, Ms=64)...")
                bounded_ppl = eval_bounded_ppl(
                    model, adapters, eval_tokens, device, dtype,
                    max_tokens=eval_bounded_tokens,
                    chunk_size=seq_len,
                )
                print(f"  Bounded PPL: {bounded_ppl:.2f}")
                log.append({
                    "step": max_steps,
                    "bounded_ppl": round(bounded_ppl, 2),
                    "bounded_config": "medium",
                })

    # Save final checkpoint + log.
    save_checkpoint(adapters, output_dir / "final", max_steps, log)

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n  Training complete: {max_steps} steps, {elapsed:.0f}s, "
          f"{tokens_total:,} tokens processed")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"  Peak VRAM: {human_bytes(peak)}")

    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


BOUNDED_CONFIGS = {
    "tiny": {"window": 128, "Me": 32, "Ms": 32},
    "medium": {"window": 256, "Me": 64, "Ms": 64},
    "medium-expand": {"window": 256, "Me": 64, "Ms": 64,
                      "max_Me": 128, "max_Ms": 128},
    "large": {"window": 512, "Me": 128, "Ms": 128},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Llama-family model with CoDA-GQA-L attention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (unbounded only)
  python benchmarks/train_coda.py --model HuggingFaceTB/SmolLM2-135M --max-steps 200

  # With Phase 2 bounded training (the key experiment)
  python benchmarks/train_coda.py --model HuggingFaceTB/SmolLM2-135M \\
      --max-steps 200 --bounded-steps 100 --bounded-config medium

  # Mistral 7B full pipeline on H100
  python benchmarks/train_coda.py --model mistralai/Mistral-7B-v0.3 \\
      --max-steps 2000 --bounded-steps 1000 --bounded-config medium \\
      --batch-size 2 --grad-accum 4

  # Llama 3.1 8B (memory-constrained)
  python benchmarks/train_coda.py --model meta-llama/Llama-3.1-8B \\
      --max-steps 3000 --bounded-steps 2000 --bounded-config large \\
      --batch-size 1 --grad-accum 8
""",
    )
    # Model & data.
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument(
        "--dataset", type=str, default="wikitext-103",
        help="Training data: wikitext-103, wikitext-2, or path to .txt file.",
    )

    # Training.
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-5, help="LR for projections.")
    p.add_argument("--lr-coda", type=float, default=1e-3, help="LR for CoDA params.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--freeze", type=str, default="attention",
        choices=["coda-only", "attention", "all"],
        help="What to train: coda-only (~50K params), attention (~30%%), all.",
    )

    # Eval & save.
    p.add_argument("--eval-every", type=int, default=200,
                    help="Eval every N steps (0 to disable).")
    p.add_argument("--save-every", type=int, default=500,
                    help="Checkpoint every N steps (0 to disable).")
    p.add_argument("--eval-tokens", type=int, default=50_000,
                    help="Max tokens for periodic eval.")
    p.add_argument("--eval-bounded-tokens", type=int, default=20_000,
                    help="Max tokens for final bounded eval.")
    p.add_argument("--no-eval-bounded", action="store_true",
                    help="Skip bounded eval at end of training.")

    # CoDA initialization.
    p.add_argument("--head-norm-mode", type=str, default="identity",
                    choices=["identity", "full"],
                    help="Head norm: 'identity' preserves pre-trained scale (default), "
                         "'full' normalizes to unit RMS.")
    p.add_argument("--theta-init", type=float, default=0.0,
                    help="Initial theta for pairwise rotation (0 = identity, pi/2 = orthogonal).")

    # Phase 2: Bounded training.
    p.add_argument("--bounded-steps", type=int, default=0,
                    help="Number of Phase 2 bounded training steps (0 = skip). "
                         "After Phase 1 unbounded training completes, switches to "
                         "bounded attention and continues training so the model "
                         "learns to work within the bounded KV cache.")
    p.add_argument("--bounded-config", type=str, default="medium",
                    choices=list(BOUNDED_CONFIGS.keys()),
                    help="Bounded cache config for Phase 2 training.")
    p.add_argument("--bounded-lr-scale", type=float, default=0.1,
                    help="LR scale factor for Phase 2 (relative to Phase 1 final LR).")
    p.add_argument("--bounded-block-size", type=int, default=256,
                    help="Block size for bounded prefill during training.")
    p.add_argument("--no-detach-evicted", action="store_true",
                    help="Allow gradients through memory bank updates in Phase 2. "
                         "Enables write_proj and summary_eta_logit to receive gradients. "
                         "Experimental: may increase memory usage.")

    # Ablation.
    p.add_argument("--no-differential", action="store_true",
                    help="Ablation: disable differential attention entirely. "
                         "Sets theta=0 and lambda=0 (frozen), so the model uses "
                         "standard GQA. Memory banks still operate normally. "
                         "Use with --max-steps 0 --bounded-steps N to test "
                         "bounded cache without differential attention.")

    # Hardware.
    p.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp32"])
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")

    # Resume.
    p.add_argument("--adapter-weights", type=str, default=None,
                    help="Path to a Phase 1 checkpoint (coda_adapters.pt) to load "
                         "before training.  Use with --max-steps 0 --bounded-steps N "
                         "to skip Phase 1 and start Phase 2 directly from pre-trained "
                         "unbounded weights.")

    # Output.
    p.add_argument("--output-dir", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_short = args.model.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"{model_short}_{ts}"

    eff_batch = args.batch_size * args.grad_accum * args.seq_len

    print("=" * 60)
    print("CoDA-GQA-L Training")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Device:    {device}, dtype: {dtype}")
    print(f"  Phase 1:   {args.max_steps} steps (unbounded)")
    if args.bounded_steps > 0:
        bcfg = BOUNDED_CONFIGS[args.bounded_config]
        expand_info = ""
        if bcfg.get("max_Me"):
            expand_info += f", max_Me={bcfg['max_Me']}"
        if bcfg.get("max_Ms"):
            expand_info += f", max_Ms={bcfg['max_Ms']}"
        print(f"  Phase 2:   {args.bounded_steps} steps (bounded, "
              f"W={bcfg['window']}, Me={bcfg['Me']}, Ms={bcfg['Ms']}"
              f"{expand_info})")
    print(f"  Batch:     {args.batch_size} x {args.grad_accum} accum x {args.seq_len} seq")
    print(f"  Eff batch: {eff_batch:,} tokens/update")
    print(f"  Freeze:    {args.freeze}")
    if args.no_differential:
        print(f"  Ablation:  --no-differential (standard GQA, no theta/lambda)")
    print(f"  Output:    {output_dir}")

    # --- Load model ---
    print(f"\n--- Model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype,
    ).to(device)

    total_params = count_params(model)
    print(f"  Params: {total_params:,}")

    # --- Swap attention ---
    lambda_init_bias = -100.0 if args.no_differential else -6.0
    model, adapters = swap_attention(
        model,
        head_norm_mode=args.head_norm_mode,
        theta_init=0.0 if args.no_differential else args.theta_init,
        lambda_init_bias=lambda_init_bias,
    )

    # --- Load pre-trained adapter weights (optional) ---
    if args.adapter_weights is not None:
        ckpt_path = Path(args.adapter_weights)
        print(f"  Loading adapter weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        loaded = 0
        for i, adapter in enumerate(adapters):
            key = f"layer_{i}"
            if key in state:
                result = adapter.load_state_dict(state[key], strict=False)
                loaded += 1
                if i == 0 and result.missing_keys:
                    print(f"  Note: missing keys (expected for Phase 1→2): "
                          f"{result.missing_keys}")
        del state
        print(f"  Loaded weights for {loaded}/{len(adapters)} layers")

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.config.use_cache = False
        print("  Gradient checkpointing: enabled")

    # --- Freeze + optimizer ---
    print(f"\n--- Optimizer ---")
    freeze_params(model, adapters, args.freeze)
    if args.no_differential:
        freeze_differential(adapters)
    optimizer = setup_optimizer(
        model, adapters, args.lr, args.lr_coda, args.weight_decay,
    )

    # --- Data ---
    print(f"\n--- Data ---")
    train_chunks = load_train_tokens(
        tokenizer, args.seq_len, args.dataset, model_name=args.model,
    )
    eval_tokens = (
        load_eval_tokens(tokenizer, args.eval_tokens, model_name=args.model)
        if args.eval_every > 0
        else None
    )
    if eval_tokens is not None:
        print(f"  Eval: {len(eval_tokens):,} tokens (WikiText-2 test)")

    # --- Save config ---
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "total_params": total_params,
        "trainable_params": count_params(model, trainable_only=True),
        "n_adapters": len(adapters),
        "eff_batch_tokens": eff_batch,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8",
    )

    # --- Phase 1: Unbounded Training ---
    print(f"\n--- Phase 1: Unbounded Training ({args.max_steps} steps) ---")
    log = train(
        model, adapters, optimizer, train_chunks, eval_tokens,
        device, dtype,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=output_dir,
        eval_bounded=not args.no_eval_bounded,
        eval_bounded_tokens=args.eval_bounded_tokens,
    )

    # --- Phase 2: Bounded Training ---
    if args.bounded_steps > 0:
        bcfg = BOUNDED_CONFIGS[args.bounded_config]
        print(f"\n--- Phase 2: Bounded Training ({args.bounded_steps} steps) ---")
        print(f"  Switching to bounded attention...")

        # Save Phase 1 checkpoint before switching.
        save_checkpoint(adapters, output_dir / "phase1_final", args.max_steps, log)

        # Switch all adapters from unbounded to bounded.
        detach = not args.no_detach_evicted
        if not detach:
            print("  detach_evicted=False: gradients flow through memory bank updates")
        model, adapters_b = switch_to_bounded(
            model, adapters, device, dtype,
            window=bcfg["window"],
            num_landmarks_exact=bcfg["Me"],
            num_landmarks_summary=bcfg["Ms"],
            max_landmarks_exact=bcfg.get("max_Me"),
            max_landmarks_summary=bcfg.get("max_Ms"),
            block_size=args.bounded_block_size,
            detach_evicted=detach,
        )

        # Gradient checkpointing must be disabled for Phase 2 (always).
        # The bounded forward accumulates variable-size landmark bank state
        # per sequence: the exact bank grows by a data-dependent number of
        # tokens per chunk (novelty-filtered, float-sensitive threshold).
        # GC recomputes the forward during backward; borderline tokens flip
        # in/out of the bank due to float non-determinism, yielding different
        # tensor shapes → CheckpointError.  Both reentrant and non-reentrant
        # modes fail.  H100/A100 (80 GB) have enough VRAM for Phase 2 without
        # GC; peak is ~40-50 GB at batch=1 seq=2048.
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if args.gradient_checkpointing:
            print("  Gradient checkpointing: disabled for Phase 2 "
                  "(incompatible with bounded landmark state)")
        if not detach:
            print("  (Also required for --no-detach-evicted: "
                  "in-place graph ops cannot be replayed)")

        # Set up optimizer for Phase 2: all adapter params trainable,
        # with lower LR to avoid catastrophic forgetting.
        print(f"\n--- Phase 2 Optimizer ---")
        freeze_params(model, adapters_b, args.freeze)
        if args.no_differential:
            freeze_differential(adapters_b)
        lr_p2 = args.lr * args.bounded_lr_scale
        lr_coda_p2 = args.lr_coda * args.bounded_lr_scale
        optimizer_p2 = setup_optimizer(
            model, adapters_b, lr_p2, lr_coda_p2, args.weight_decay,
        )

        # Phase 2 eval before training (bounded PPL baseline).
        if eval_tokens is not None:
            ppl_p2_start = eval_ppl_sequential(
                model, adapters_b, eval_tokens, device,
                chunk_size=args.seq_len,
                max_tokens=args.eval_bounded_tokens,
            )
            print(f"  Phase 2 start bounded PPL: {ppl_p2_start:.2f}")
            log.append({
                "step": args.max_steps,
                "phase": 2,
                "bounded_ppl_start": round(ppl_p2_start, 2),
            })

        log_p2 = train(
            model, adapters_b, optimizer_p2, train_chunks, eval_tokens,
            device, dtype,
            max_steps=args.bounded_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_len=args.seq_len,
            warmup_steps=min(50, args.bounded_steps // 10),
            eval_every=args.eval_every,
            save_every=args.save_every,
            output_dir=output_dir / "phase2",
            eval_bounded=True,
            eval_bounded_tokens=args.eval_bounded_tokens,
        )

        # Merge logs with phase annotation.
        for entry in log_p2:
            entry["phase"] = 2
            if "step" in entry:
                entry["step"] += args.max_steps  # offset for total step count
        log.extend(log_p2)

        # Use bounded adapters for final summary.
        adapters = adapters_b

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Output: {output_dir}")

    losses = [e["loss"] for e in log if "loss" in e]
    ppls = [e["eval_ppl"] for e in log if "eval_ppl" in e]
    bounded = [e for e in log if "bounded_ppl" in e]

    if losses:
        print(f"  Train loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if ppls:
        print(f"  Eval PPL:   {ppls[0]:.2f} -> {ppls[-1]:.2f}")
    if bounded:
        print(f"  Bounded PPL: {bounded[-1]['bounded_ppl']:.2f}")
    if args.bounded_steps > 0:
        print(f"  Phase 2:     {args.bounded_steps} steps ({args.bounded_config})")


if __name__ == "__main__":
    main()
