#!/usr/bin/env python3
"""Bounded-mode generation with CoDA-GQA-L memory banks.

Unlike unbounded mode (which works with HF's model.generate()), bounded
mode needs a manual generation loop because CoDA-GQA-L manages its own
KV cache internally via three memory segments:

  - Recent window (W=256): ring buffer of latest tokens
  - Exact landmark bank (Me=64): novelty-filtered LRU cache
  - Summary landmark bank (Ms=64): EMA compressed prototypes

Total cache: 384 slots/layer (constant) vs O(L) unbounded growth.

HF's generate() can't drive this because it either re-processes the full
sequence each step (use_cache=False) or manages its own KV cache
(use_cache=True). CoDA needs single-token inputs during decode so each
adapter's internal state machine routes through step() correctly.

Usage:
    python examples/bounded_generate.py
    python examples/bounded_generate.py --prompt "Once upon a time"
    python examples/bounded_generate.py --max-new-tokens 200 --collect-metrics
    python examples/bounded_generate.py -W 128 -Me 32 -Ms 32  # smaller cache
"""

import argparse
import time

import torch

# These may not be installed -- fail with clear message
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("pip install transformers accelerate")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("pip install huggingface_hub")

from coda_gqa_l import LlamaCoDAAdapter


def sample_next_token(logits, temperature=0.8, top_k=50, top_p=0.9):
    """Sample a single token from logits with temperature + top-k + top-p."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        # Keep at least one token
        remove[..., 0] = False
        # Map removal mask back to original indices
        remove_orig = remove.scatter(dim=-1, index=sorted_idx, src=remove)
        logits = logits.masked_fill(remove_orig, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    p = argparse.ArgumentParser(
        description="Bounded CoDA-GQA-L generation (manual decode loop)"
    )
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("-W", "--window", type=int, default=256)
    p.add_argument("-Me", "--num-exact", type=int, default=64)
    p.add_argument("-Ms", "--num-summary", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--collect-metrics", action="store_true",
                    help="Track memory bank hit/insert/eviction counters")
    p.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    p.add_argument("--coda-repo", default="anthonym21/Mistral-7B-v0.3-CoDA-GQA-L")
    p.add_argument("--adapters-file", default="coda_adapters.pt",
                    help="Filename of adapter weights inside --coda-repo")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    total_slots = args.window + args.num_exact + args.num_summary

    # ------------------------------------------------------------------
    # 1. Load tokenizer (use base model -- CoDA repo's tokenizer config is broken)
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ------------------------------------------------------------------
    # 2. Load base model
    # ------------------------------------------------------------------
    print(f"Loading {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
    )
    num_layers = len(model.model.layers)

    # ------------------------------------------------------------------
    # 3. Swap attention layers -> bounded CoDA adapters
    # ------------------------------------------------------------------
    print(f"Installing bounded CoDA-GQA-L attention "
          f"(W={args.window}, Me={args.num_exact}, Ms={args.num_summary}, "
          f"total={total_slots} slots/layer)...")

    for layer in model.model.layers:
        adapter = LlamaCoDAAdapter.from_llama_attention(
            layer.self_attn,
            bounded=True,
            window=args.window,
            num_landmarks_exact=args.num_exact,
            num_landmarks_summary=args.num_summary,
            block_size=args.block_size,
            collect_metrics=args.collect_metrics,
        )
        layer.self_attn = adapter

    # ------------------------------------------------------------------
    # 4. Load trained CoDA weights (strict=False: bounded-only params
    #    like write_proj and summary_eta_logit use default init)
    # ------------------------------------------------------------------
    print(f"Loading trained adapters from {args.coda_repo}/{args.adapters_file}...")
    adapters_path = hf_hub_download(args.coda_repo, args.adapters_file)
    ckpt = torch.load(adapters_path, map_location="cpu", weights_only=True)

    loaded, skipped = 0, 0
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt:
            missing, unexpected = layer.self_attn.load_state_dict(
                ckpt[key], strict=False,
            )
            loaded += 1
            if missing:
                skipped += len(missing)
        else:
            print(f"  WARNING: no checkpoint entry for layer {i}")

    print(f"  Loaded {loaded}/{num_layers} layers "
          f"({skipped} bounded-only params kept at default init)")

    # CRITICAL: eval() must come AFTER adapter installation.
    # New nn.Modules default to training=True. In training mode,
    # _forward_bounded allocates a fresh empty state every call
    # (for gradient-checkpointing safety), so decode tokens have
    # zero context. eval() switches to the persistent stateful path.
    model.eval()

    # ------------------------------------------------------------------
    # 5. Report cache budget
    # ------------------------------------------------------------------
    adapter0 = model.model.layers[0].self_attn
    cache_per_layer = adapter0.cache_bytes(batch_size=1, dtype=dtype)
    cache_total = cache_per_layer * num_layers

    # Rough estimate of what unbounded would use at 4096 tokens
    # (2 buffers K+V * num_kv_heads * head_dim * seq_len * bytes_per_element)
    hkv = adapter0.num_kv_heads
    dh = adapter0.head_dim
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    unbounded_4k = num_layers * 2 * hkv * dh * 4096 * elem_bytes

    print(f"\nBounded KV cache budget:")
    print(f"  {total_slots} slots/layer x {num_layers} layers")
    print(f"  Per layer:  {human_bytes(cache_per_layer)}")
    print(f"  Total:      {human_bytes(cache_total)}")
    print(f"  Unbounded at 4096 tokens would be: {human_bytes(unbounded_4k)}")
    print(f"  Savings:    {unbounded_4k / cache_total:.1f}x smaller")

    # ------------------------------------------------------------------
    # 6. Reset all adapter states (fresh memory banks)
    # ------------------------------------------------------------------
    for layer in model.model.layers:
        layer.self_attn.reset_state()

    # ------------------------------------------------------------------
    # 7. Tokenize prompt
    # ------------------------------------------------------------------
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)  # first device in device_map
    prompt_len = input_ids.shape[1]

    print(f"\nPrompt ({prompt_len} tokens): \"{args.prompt}\"")
    print(f"Generating up to {args.max_new_tokens} tokens...\n")

    # ------------------------------------------------------------------
    # 8. Prefill (each adapter sees L > 1 -> prefill_chunked)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    # ------------------------------------------------------------------
    # 9. Decode loop (each adapter sees L == 1 -> step)
    # ------------------------------------------------------------------
    generated_tokens = []
    t_decode_start = time.perf_counter()

    for step_i in range(args.max_new_tokens):
        next_token = sample_next_token(
            logits, args.temperature, args.top_k, args.top_p,
        )
        tok_id = next_token.item()
        generated_tokens.append(tok_id)

        # Stop on EOS
        if tok_id == tokenizer.eos_token_id:
            break

        # Single-token forward: adapters route through step() internally
        with torch.no_grad():
            outputs = model(input_ids=next_token, use_cache=False)
        logits = outputs.logits[:, -1, :]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start
    num_generated = len(generated_tokens)

    # ------------------------------------------------------------------
    # 10. Print results
    # ------------------------------------------------------------------
    full_ids = torch.cat([
        input_ids[0].cpu(),
        torch.tensor(generated_tokens, dtype=torch.long),
    ])
    text = tokenizer.decode(full_ids, skip_special_tokens=True)

    print("=" * 70)
    print(text)
    print("=" * 70)

    print(f"\nPrefill:  {prompt_len} tokens in {t_prefill:.2f}s "
          f"({prompt_len / t_prefill:.0f} tok/s)")
    print(f"Decode:   {num_generated} tokens in {t_decode:.2f}s "
          f"({num_generated / max(t_decode, 1e-9):.1f} tok/s)")
    print(f"Cache:    {total_slots} slots/layer "
          f"({human_bytes(cache_total)} total, "
          f"{unbounded_4k / cache_total:.1f}x smaller than unbounded)")

    # ------------------------------------------------------------------
    # 11. Memory bank metrics (if enabled)
    # ------------------------------------------------------------------
    if args.collect_metrics:
        print(f"\n--- Memory bank metrics (per layer) ---")
        for i, layer in enumerate(model.model.layers):
            state = layer.self_attn.get_state()
            if state is None or state.metrics is None:
                continue
            m = state.metrics
            print(f"\n  Layer {i}:")
            print(f"    Exact bank:   {m.get('exact_inserts', 0)} inserts, "
                  f"{m.get('exact_hits', 0)} hits, "
                  f"{m.get('exact_overwrites', 0)} overwrites, "
                  f"fill {m.get('exact_fill_ratio', 0):.1%}")
            print(f"    Summary bank: {m.get('summary_inserts', 0)} inserts, "
                  f"{m.get('summary_updates', 0)} updates, "
                  f"fill {m.get('summary_fill_ratio', 0):.1%}")
            print(f"    Gated out:    {m.get('tokens_gated_out', 0)} tokens "
                  f"(below write threshold)")
            print(f"    Evictions:    {m.get('total_evictions', 0)} "
                  f"from recent window")

    # ------------------------------------------------------------------
    # 12. VRAM
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"\nPeak VRAM: {human_bytes(peak)}")


if __name__ == "__main__":
    main()
