# llama_adapter.py
# Drop-in adapter for replacing Llama-family attention with CoDA-GQA-L.
#
# Works with any HuggingFace model using the LlamaForCausalLM architecture,
# including SmolLM, SmolLM2, Llama 2, Llama 3, Mistral, and other variants
# that use separate Q/K/V/O projections with optional GQA.
#
# Weight mapping is 1:1 since Llama already uses separate projections:
#   q_proj -> q_proj, k_proj -> k_proj, v_proj -> v_proj, o_proj -> o_proj
#
# For cold-swap evaluation (no retraining), use head_norm_mode="identity"
# to bypass HeadwiseRMSNorm which would destroy the pre-trained output scale.

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import CoDAGQALandmarkPerf2
from .baseline import CoDAGQA
from .state import CoDAGQALandmarkStatePerf2


class LlamaCoDAAdapter(nn.Module):
    """Drop-in replacement for Llama-family attention using CoDA-GQA-L.

    Supports two modes controlled by the ``bounded`` flag:

    * **Training (unbounded, bounded=False)** -- Uses ``CoDAGQA`` with standard
      causal attention and O(L) KV cache.

    * **Inference (bounded, bounded=True)** -- Uses ``CoDAGQALandmarkPerf2``
      with bounded KV cache: O(W + Me + Ms) per layer.

    The forward signature is compatible with Llama's attention module:
    ``forward(hidden_states, ...) -> (attn_output, ...)``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bounded: bool = False,
        rope_theta: float = 10_000.0,
        # CoDA differential attention parameters
        lambda_init_bias: float = -6.0,
        theta_init: float = math.pi / 2,
        eps: float = 1e-6,
        # WARNING: These defaults (rope_interleaved=False, head_norm_mode=
        # "identity") differ from bare CoDAGQA and CoDAGQALandmarkPerf2
        # (rope_interleaved=True, head_norm_mode="full"). When transferring
        # weights trained with bare CoDAGQA into LlamaCoDAAdapter, you MUST
        # explicitly pass rope_interleaved and head_norm_mode to match the
        # training configuration, otherwise outputs will silently diverge.
        # These Llama-specific defaults are correct for cold-swap evaluation
        # of Llama-family models (contiguous-half RoPE, no HeadwiseRMSNorm).
        head_norm_mode: str = "identity",
        rope_interleaved: bool = False,
        # Bounded-mode parameters
        window: int = 256,
        num_landmarks_exact: int = 64,
        num_landmarks_summary: int = 64,
        block_size: int = 256,
        **coda_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bounded = bounded
        self.block_size = block_size

        if bounded:
            self.coda = CoDAGQALandmarkPerf2(
                embed_dim=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                window=window,
                num_landmarks_exact=num_landmarks_exact,
                num_landmarks_summary=num_landmarks_summary,
                rope_base=rope_theta,
                lambda_init_bias=lambda_init_bias,
                theta_init=theta_init,
                eps=eps,
                head_norm_mode=head_norm_mode,
                rope_interleaved=rope_interleaved,
                **coda_kwargs,
            )
        else:
            self.coda = CoDAGQA(
                embed_dim=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_base=rope_theta,
                lambda_init_bias=lambda_init_bias,
                theta_init=theta_init,
                eps=eps,
                head_norm_mode=head_norm_mode,
                rope_interleaved=rope_interleaved,
            )

        # Inference state (bounded mode only).
        self._state: Optional[CoDAGQALandmarkStatePerf2] = None

    # ------------------------------------------------------------------
    # Weight transfer from Llama-family attention
    # ------------------------------------------------------------------

    @classmethod
    def from_llama_attention(
        cls,
        llama_attn: nn.Module,
        bounded: bool = False,
        head_norm_mode: str = "identity",
        rope_interleaved: bool = False,
        **coda_kwargs,
    ) -> "LlamaCoDAAdapter":
        """Create adapter by copying weights from a Llama-family attention module.

        Llama uses separate Q/K/V/O projections, so the weight mapping is 1:1.
        Supports GQA (num_kv_heads < num_heads).

        .. warning::

            The defaults here (``rope_interleaved=False``,
            ``head_norm_mode="identity"``) are tuned for Llama-family models
            and differ from ``CoDAGQA`` / ``CoDAGQALandmarkPerf2`` defaults
            (``rope_interleaved=True``, ``head_norm_mode="full"``).  If you
            trained with bare ``CoDAGQA()`` using its defaults, you **must**
            pass ``rope_interleaved=True`` and ``head_norm_mode="full"`` here
            to match, otherwise outputs will silently diverge.

        Args:
            llama_attn: A Llama-style attention module with q_proj, k_proj,
                v_proj, o_proj attributes.
            bounded: Whether to use bounded-memory inference mode.
            head_norm_mode: "identity" for cold-swap (default), "full" for
                HeadwiseRMSNorm (requires training).
            rope_interleaved: RoPE dimension pairing convention.  False
                (default) uses the contiguous-half layout native to Llama
                models: pairs are (i, i + D/2).  True uses the interleaved
                layout: pairs are (2i, 2i+1).  Must match the convention
                used during training.
            **coda_kwargs: Forwarded to the constructor (window, etc.).
        """
        # Extract dimensions from the attention module.
        hidden_size = llama_attn.q_proj.in_features
        num_heads = getattr(llama_attn, "num_heads", None)
        num_kv_heads = getattr(llama_attn, "num_key_value_heads", None)
        head_dim = getattr(llama_attn, "head_dim", None)

        # Fallback: infer from projection shapes.
        if head_dim is None:
            q_out = llama_attn.q_proj.out_features
            if num_heads is not None:
                head_dim = q_out // num_heads
            else:
                # Common head dims: 64, 128. Try 128 first (Llama 3).
                for hd in [128, 64, 96, 256]:
                    if q_out % hd == 0:
                        head_dim = hd
                        break
                if head_dim is None:
                    raise ValueError(f"Cannot infer head_dim from q_proj shape {q_out}")

        if num_heads is None:
            num_heads = llama_attn.q_proj.out_features // head_dim

        if num_kv_heads is None:
            num_kv_heads = llama_attn.k_proj.out_features // head_dim

        # Get RoPE theta from config if available.
        rope_theta = 10_000.0
        config = getattr(llama_attn, "config", None)
        if config is not None:
            rope_theta = getattr(config, "rope_theta", rope_theta)

        # Emit a warning when the chosen settings differ from CoDAGQA /
        # CoDAGQALandmarkPerf2 defaults, which is the most common source of
        # silent numerical divergence when transferring weights.
        _CODA_DEFAULT_ROPE_INTERLEAVED = True
        _CODA_DEFAULT_HEAD_NORM_MODE = "full"
        mismatches = []
        if rope_interleaved != _CODA_DEFAULT_ROPE_INTERLEAVED:
            mismatches.append(
                f"rope_interleaved={rope_interleaved} (CoDAGQA default: "
                f"{_CODA_DEFAULT_ROPE_INTERLEAVED})"
            )
        if head_norm_mode != _CODA_DEFAULT_HEAD_NORM_MODE:
            mismatches.append(
                f"head_norm_mode={head_norm_mode!r} (CoDAGQA default: "
                f"{_CODA_DEFAULT_HEAD_NORM_MODE!r})"
            )
        if mismatches:
            warnings.warn(
                "LlamaCoDAAdapter.from_llama_attention: settings differ from "
                "CoDAGQA / CoDAGQALandmarkPerf2 defaults -- "
                + "; ".join(mismatches)
                + ". If you trained with bare CoDAGQA() using its defaults, "
                "pass rope_interleaved=True and head_norm_mode='full' to "
                "from_llama_attention() to match. Mismatched settings will "
                "produce silently wrong outputs (e.g. max_diff >> 0).",
                UserWarning,
                stacklevel=2,
            )

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

        # --- Copy weights ---
        coda = adapter.coda

        # Q projection
        coda.q_proj.weight.data.copy_(llama_attn.q_proj.weight.data)
        if llama_attn.q_proj.bias is not None:
            if coda.q_proj.bias is None:
                coda.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
                coda.q_proj.weight.data.copy_(llama_attn.q_proj.weight.data)
            coda.q_proj.bias.data.copy_(llama_attn.q_proj.bias.data)

        # K projection
        coda.k_proj.weight.data.copy_(llama_attn.k_proj.weight.data)
        if llama_attn.k_proj.bias is not None:
            if coda.k_proj.bias is None:
                coda.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
                coda.k_proj.weight.data.copy_(llama_attn.k_proj.weight.data)
            coda.k_proj.bias.data.copy_(llama_attn.k_proj.bias.data)

        # V projection
        coda.v_proj.weight.data.copy_(llama_attn.v_proj.weight.data)
        if llama_attn.v_proj.bias is not None:
            if coda.v_proj.bias is None:
                coda.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
                coda.v_proj.weight.data.copy_(llama_attn.v_proj.weight.data)
            coda.v_proj.bias.data.copy_(llama_attn.v_proj.bias.data)

        # O projection
        coda.o_proj.weight.data.copy_(llama_attn.o_proj.weight.data)
        if llama_attn.o_proj.bias is not None:
            if coda.o_proj.bias is None:
                coda.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
                coda.o_proj.weight.data.copy_(llama_attn.o_proj.weight.data)
            coda.o_proj.bias.data.copy_(llama_attn.o_proj.bias.data)

        return adapter

    # ------------------------------------------------------------------
    # Forward (compatible with Llama attention interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: object = None,
        position_ids: object = None,
        past_key_value: object = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: object = None,
        position_embeddings: object = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass matching Llama attention interface.

        CoDA manages its own RoPE and KV cache internally, so position_ids,
        past_key_value, and position_embeddings are accepted but ignored.

        Returns:
            Tuple of (attn_output,) or (attn_output, None, past_kv) depending
            on use_cache flag, matching Llama's return convention.
        """
        if not self.bounded:
            y = self._forward_unbounded(hidden_states)
        else:
            y = self._forward_bounded(hidden_states)

        # Llama's LlamaDecoderLayer unpacks as: hidden_states, _ = self.self_attn(...)
        # Always return exactly 2 values: (output, past_key_value_or_None).
        return y, past_key_value

    def _forward_unbounded(self, x: torch.Tensor) -> torch.Tensor:
        out, _kv = self.coda(x, is_causal=True)
        return out

    def _forward_bounded(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # Training: stateless forward for gradient-checkpointing safety.
        # Gradient checkpointing recomputes the forward during backward.
        # If we persist state across calls, recomputation sees already-mutated
        # buffers and produces tensors with different shapes -> CheckpointError.
        # Fix: allocate a fresh local state each forward so first pass and
        # recomputation are identical.  Uses zeros init for determinism
        # (random_normal would advance the RNG differently on recompute).
        if self.training:
            state = self.coda.init_state(
                batch_size=B, device=x.device, dtype=x.dtype,
                mem_init="zeros",
            )
            y, _ = self.coda.prefill_chunked(x, state, block_size=self.block_size)
            return y

        # Inference: stateful streaming with persistent state.
        if self._state is None:
            self._state = self.coda.init_state(
                batch_size=B, device=x.device, dtype=x.dtype,
            )

        if L == 1:
            y, self._state = self.coda.step(x, self._state)
        else:
            y, self._state = self.coda.prefill_chunked(
                x, self._state, block_size=self.block_size,
            )
        return y

    # ------------------------------------------------------------------
    # State management (bounded mode)
    # ------------------------------------------------------------------

    def init_inference_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> CoDAGQALandmarkStatePerf2:
        if not self.bounded:
            raise RuntimeError("init_inference_state() requires bounded=True")
        self._state = self.coda.init_state(
            batch_size=batch_size, device=device, dtype=dtype,
        )
        return self._state

    def get_state(self) -> Optional[CoDAGQALandmarkStatePerf2]:
        return self._state

    def reset_state(self) -> None:
        self._state = None

    def cache_bytes(self, *, batch_size: int, dtype: torch.dtype) -> int:
        if not self.bounded:
            raise RuntimeError("cache_bytes() requires bounded=True")
        return self.coda.cache_bytes(batch_size=batch_size, dtype=dtype)

    def extra_repr(self) -> str:
        mode = "bounded" if self.bounded else "unbounded"
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"mode={mode}"
        )
