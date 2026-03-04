# qwen3_adapter.py
# Drop-in adapter for replacing Qwen3 attention with CoDA-GQA-L.
#
# Qwen3 uses the same separate Q/K/V/O projection layout as Llama, so weight
# mapping is 1:1.  Key differences from Llama defaults:
#   - rope_theta = 1,000,000 (vs 10,000 for Llama 2, 500,000 for Llama 3)
#   - Contiguous-half RoPE (same as Llama)
#   - GQA with 4:1 ratio (32 Q heads, 8 KV heads for Qwen3-4B)
#   - head_dim = 128 (same as Llama 3 / Mistral)
#
# This is a thin wrapper around LlamaCoDAAdapter with Qwen3-appropriate
# defaults.  The from_qwen3_attention() classmethod extracts config from
# Qwen3's attention module and handles any Qwen-specific attribute names.

from __future__ import annotations

import math
from typing import List, Optional

import torch.nn as nn

from .llama_adapter import LlamaCoDAAdapter


class Qwen3CoDAAdapter(LlamaCoDAAdapter):
    """Drop-in replacement for Qwen3 attention using CoDA-GQA-L.

    Inherits from LlamaCoDAAdapter since Qwen3 shares the same attention
    layout (separate Q/K/V/O projections, GQA, contiguous-half RoPE).
    Overrides default rope_theta to match Qwen3's 1,000,000.

    Supports two modes controlled by the ``bounded`` flag:

    * **Training (unbounded, bounded=False)** -- Uses ``CoDAGQA`` with standard
      causal attention and O(L) KV cache.

    * **Inference (bounded, bounded=True)** -- Uses ``CoDAGQALandmarkPerf2``
      with bounded KV cache: O(W + Me + Ms) per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bounded: bool = False,
        rope_theta: float = 1_000_000.0,  # Qwen3 default
        lambda_init_bias: float = -6.0,
        theta_init: float = math.pi / 2,
        eps: float = 1e-6,
        head_norm_mode: str = "identity",
        rope_interleaved: bool = False,
        window: int = 256,
        num_landmarks_exact: int = 64,
        num_landmarks_summary: int = 64,
        block_size: int = 1024,
        **coda_kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bounded=bounded,
            rope_theta=rope_theta,
            lambda_init_bias=lambda_init_bias,
            theta_init=theta_init,
            eps=eps,
            head_norm_mode=head_norm_mode,
            rope_interleaved=rope_interleaved,
            window=window,
            num_landmarks_exact=num_landmarks_exact,
            num_landmarks_summary=num_landmarks_summary,
            block_size=block_size,
            **coda_kwargs,
        )

    @classmethod
    def from_qwen3_attention(
        cls,
        qwen3_attn: nn.Module,
        bounded: bool = False,
        head_norm_mode: str = "identity",
        rope_interleaved: bool = False,
        **coda_kwargs,
    ) -> "Qwen3CoDAAdapter":
        """Create adapter by copying weights from a Qwen3 attention module.

        Qwen3 uses separate Q/K/V/O projections like Llama, so weight
        transfer is 1:1.  This method handles Qwen3-specific attribute names
        and defaults (rope_theta=1,000,000).

        Args:
            qwen3_attn: A Qwen3 attention module (Qwen3Attention) with
                q_proj, k_proj, v_proj, o_proj attributes.
            bounded: Whether to use bounded-memory inference mode.
            head_norm_mode: "identity" for cold-swap (default), "full" for
                HeadwiseRMSNorm (requires training).
            rope_interleaved: RoPE dimension pairing convention.  Qwen3 uses
                contiguous-half (False), same as Llama.
            **coda_kwargs: Forwarded to the constructor (window, etc.).
        """
        # Extract dimensions -- Qwen3 uses the same attribute names as Llama.
        hidden_size = qwen3_attn.q_proj.in_features
        num_heads = getattr(qwen3_attn, "num_heads", None)
        num_kv_heads = getattr(qwen3_attn, "num_key_value_heads", None)
        head_dim = getattr(qwen3_attn, "head_dim", None)

        # Fallback: infer from projection shapes.
        if head_dim is None:
            q_out = qwen3_attn.q_proj.out_features
            if num_heads is not None:
                head_dim = q_out // num_heads
            else:
                for hd in [128, 64, 96, 256]:
                    if q_out % hd == 0:
                        head_dim = hd
                        break
                if head_dim is None:
                    raise ValueError(f"Cannot infer head_dim from q_proj shape {q_out}")

        if num_heads is None:
            num_heads = qwen3_attn.q_proj.out_features // head_dim

        if num_kv_heads is None:
            num_kv_heads = qwen3_attn.k_proj.out_features // head_dim

        # Get RoPE theta from config (Qwen3 default: 1,000,000).
        rope_theta = 1_000_000.0
        config = getattr(qwen3_attn, "config", None)
        if config is not None:
            rope_theta = getattr(config, "rope_theta", rope_theta)

        adapter = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bounded=bounded,
            rope_theta=rope_theta,
            head_norm_mode=head_norm_mode,
            rope_interleaved=rope_interleaved,
            **coda_kwargs,
        )

        # --- Copy weights (1:1 mapping) ---
        coda = adapter.coda

        coda.q_proj.weight.data.copy_(qwen3_attn.q_proj.weight.data)
        if qwen3_attn.q_proj.bias is not None:
            if coda.q_proj.bias is None:
                coda.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
                coda.q_proj.weight.data.copy_(qwen3_attn.q_proj.weight.data)
            coda.q_proj.bias.data.copy_(qwen3_attn.q_proj.bias.data)

        coda.k_proj.weight.data.copy_(qwen3_attn.k_proj.weight.data)
        if qwen3_attn.k_proj.bias is not None:
            if coda.k_proj.bias is None:
                coda.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
                coda.k_proj.weight.data.copy_(qwen3_attn.k_proj.weight.data)
            coda.k_proj.bias.data.copy_(qwen3_attn.k_proj.bias.data)

        coda.v_proj.weight.data.copy_(qwen3_attn.v_proj.weight.data)
        if qwen3_attn.v_proj.bias is not None:
            if coda.v_proj.bias is None:
                coda.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
                coda.v_proj.weight.data.copy_(qwen3_attn.v_proj.weight.data)
            coda.v_proj.bias.data.copy_(qwen3_attn.v_proj.bias.data)

        coda.o_proj.weight.data.copy_(qwen3_attn.o_proj.weight.data)
        if qwen3_attn.o_proj.bias is not None:
            if coda.o_proj.bias is None:
                coda.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
                coda.o_proj.weight.data.copy_(qwen3_attn.o_proj.weight.data)
            coda.o_proj.bias.data.copy_(qwen3_attn.o_proj.bias.data)

        return adapter

    @classmethod
    def swap_qwen3_layers(
        cls,
        model: nn.Module,
        bounded: bool = False,
        head_norm_mode: str = "identity",
        rope_interleaved: bool = False,
        **coda_kwargs,
    ) -> List["Qwen3CoDAAdapter"]:
        """Replace all attention layers in a Qwen3 model with CoDA adapters.

        Iterates over ``model.model.layers``, calls :meth:`from_qwen3_attention`
        on each ``self_attn``, and replaces it in-place.

        Args:
            model: A HuggingFace Qwen3 model (Qwen3ForCausalLM).
            bounded: Whether to use bounded-memory mode.
            head_norm_mode: Forwarded to :meth:`from_qwen3_attention`.
            rope_interleaved: Forwarded to :meth:`from_qwen3_attention`.
            **coda_kwargs: Forwarded to the adapter constructor (window,
                num_landmarks_exact, num_landmarks_summary, block_size, etc.).

        Returns:
            List of the created :class:`Qwen3CoDAAdapter` instances (one per layer).
        """
        blocks = model.model.layers
        adapters: List[Qwen3CoDAAdapter] = []

        for block in blocks:
            adapter = cls.from_qwen3_attention(
                block.self_attn,
                bounded=bounded,
                head_norm_mode=head_norm_mode,
                rope_interleaved=rope_interleaved,
                **coda_kwargs,
            )
            block.self_attn = adapter
            adapters.append(adapter)

        return adapters
