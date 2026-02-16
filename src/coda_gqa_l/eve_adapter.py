# eve_adapter.py
# Drop-in adapter that lets CoDA-GQA-L attention replace Eve-2's CausalSelfAttention.
#
# Eve-2 (anthonym21/Eve-2-MoE-IT-272M) uses MHA with fused QKV, complex-valued
# RoPE (theta=10000), and bias on all projections.  CoDA uses separate Q/K/V
# projections with real-valued pairwise RoPE (theta=10000) and bias=False.
#
# The mathematical equivalence of the two RoPE representations (complex vs
# real pairwise) holds for the same theta, so the adapter lets CoDA manage its
# own RoPE and ignores Eve's `freqs_cis` argument.
#
# Bias handling: after construction the adapter replaces CoDA's bias-free
# nn.Linear layers with bias-enabled versions, copying weights and biases
# from Eve's fused projection.

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import CoDAGQALandmarkPerf2
from .baseline import CoDAGQA
from .state import CoDAGQALandmarkStatePerf2


class EveCoDAAdapter(nn.Module):
    """Drop-in replacement for Eve-2's CausalSelfAttention using CoDA attention.

    Supports two modes controlled by the ``bounded`` flag:

    * **Training (unbounded, bounded=False)** -- Uses ``CoDAGQA`` with standard
      causal attention and an O(L) KV cache, matching the semantics Eve's
      ``CausalSelfAttention`` would have during training.

    * **Inference (bounded, bounded=True)** -- Uses ``CoDAGQALandmarkPerf2``
      with a fixed-size KV cache comprising a recent window, exact landmark
      bank, and summary landmark bank.  Total memory is
      O(W + Me + Ms) per layer instead of O(L).

    The forward signature matches Eve's ``CausalSelfAttention.forward(x,
    freqs_cis=None)`` so the adapter can be swapped in without changing
    Eve's ``Block`` code.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int,
        bounded: bool = False,
        rope_theta: float = 10_000.0,
        # CoDA differential attention parameters
        lambda_init_bias: float = -6.0,
        theta_init: float = math.pi / 2,
        eps: float = 1e-6,
        # NOTE: These defaults (rope_interleaved=True, head_norm_mode="full")
        # match CoDAGQA and CoDAGQALandmarkPerf2 defaults.  They differ from
        # LlamaCoDAAdapter (rope_interleaved=False, head_norm_mode="identity").
        # Eve-2 uses complex-valued RoPE equivalent to interleaved real-valued
        # RoPE, so interleaved=True is correct for Eve models.  If transferring
        # weights to/from LlamaCoDAAdapter, ensure settings match.
        head_norm_mode: str = "full",
        rope_interleaved: bool = True,
        # Bounded-mode parameters (ignored when bounded=False)
        window: int = 256,
        num_landmarks_exact: int = 64,
        num_landmarks_summary: int = 64,
        block_size: int = 256,
        **coda_kwargs,
    ):
        """
        Args:
            n_embd: Embedding dimension (512 for Eve-2).
            n_head: Number of attention heads (8 for Eve-2).
            head_dim: Head dimension (64 for Eve-2).
            bounded: If True, use bounded-memory CoDAGQALandmarkPerf2
                for inference.  If False, use unbounded CoDAGQA for training.
            rope_theta: RoPE base frequency (10000 for both Eve-2 and CoDA
                default).
            lambda_init_bias: Initial bias for the differential lambda gate.
            theta_init: Initial angle for the orthogonal noise rotation.
            eps: Epsilon for HeadwiseRMSNorm.
            head_norm_mode: "full" for HeadwiseRMSNorm, "identity" to bypass
                (required for cold-swap evaluation without retraining).
            window: Recent window size (bounded mode only).
            num_landmarks_exact: Exact landmark bank capacity (bounded mode
                only).
            num_landmarks_summary: Summary landmark bank capacity (bounded mode
                only).
            block_size: Prefill chunk size (bounded mode only).
            **coda_kwargs: Additional keyword arguments forwarded to the
                bounded CoDAGQALandmarkPerf2 constructor (e.g.
                write_gate_threshold_exact, summary_eta_init_logit, etc.).
        """
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.bounded = bounded
        self.block_size = block_size

        # Eve is MHA: num_kv_heads == num_heads (no GQA grouping).
        num_kv_heads = n_head

        if bounded:
            self.coda = CoDAGQALandmarkPerf2(
                embed_dim=n_embd,
                num_heads=n_head,
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
                embed_dim=n_embd,
                num_heads=n_head,
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
    # Weight transfer from Eve's CausalSelfAttention
    # ------------------------------------------------------------------

    @classmethod
    def from_eve_attention(
        cls,
        eve_attn: nn.Module,
        bounded: bool = False,
        head_norm_mode: str = "full",
        rope_interleaved: bool = True,
        **coda_kwargs,
    ) -> "EveCoDAAdapter":
        """Create an adapter by copying weights from an Eve CausalSelfAttention.

        Eve's fused ``c_attn`` weight (3*n_embd, n_embd) is split into
        separate Q, K, V projections.  Biases are preserved by replacing
        CoDA's bias-free ``nn.Linear`` layers with bias-enabled versions.

        .. note::

            The defaults here (``rope_interleaved=True``,
            ``head_norm_mode="full"``) match ``CoDAGQA`` /
            ``CoDAGQALandmarkPerf2`` defaults.  Eve-2 uses complex-valued
            RoPE equivalent to interleaved real-valued RoPE, so
            ``rope_interleaved=True`` is correct for Eve models.  These
            differ from ``LlamaCoDAAdapter`` defaults
            (``rope_interleaved=False``, ``head_norm_mode="identity"``).

        Args:
            eve_attn: An instance of Eve-2's ``CausalSelfAttention``.
            bounded: Whether to use bounded-memory inference mode.
            head_norm_mode: "full" for HeadwiseRMSNorm (default, matches
                CoDAGQA), "identity" to bypass (for cold-swap evaluation
                without retraining).
            rope_interleaved: RoPE dimension pairing convention.  True
                (default) uses the interleaved layout native to Eve-2:
                pairs are (2i, 2i+1).  Must match the convention used
                during training.
            **coda_kwargs: Forwarded to the ``EveCoDAAdapter`` constructor
                (e.g. window, num_landmarks_exact, block_size, etc.).

        Returns:
            A fully initialized ``EveCoDAAdapter`` with Eve's weights.
        """
        n_head = eve_attn.n_head
        n_embd = eve_attn.n_embd
        head_dim = eve_attn.head_dim

        adapter = cls(
            n_embd=n_embd,
            n_head=n_head,
            head_dim=head_dim,
            bounded=bounded,
            head_norm_mode=head_norm_mode,
            rope_interleaved=rope_interleaved,
            **coda_kwargs,
        )

        # --- Extract weights and biases from Eve's fused QKV projection ---
        # c_attn.weight shape: (3 * n_embd, n_embd)
        # c_attn.bias shape:   (3 * n_embd,) if bias=True, else None
        fused_w = eve_attn.c_attn.weight.data
        has_attn_bias = eve_attn.c_attn.bias is not None

        q_weight = fused_w[:n_embd, :]
        k_weight = fused_w[n_embd : 2 * n_embd, :]
        v_weight = fused_w[2 * n_embd :, :]

        if has_attn_bias:
            fused_b = eve_attn.c_attn.bias.data
            q_bias = fused_b[:n_embd]
            k_bias = fused_b[n_embd : 2 * n_embd]
            v_bias = fused_b[2 * n_embd :]

        # Output projection
        o_weight = eve_attn.c_proj.weight.data
        has_proj_bias = eve_attn.c_proj.bias is not None
        if has_proj_bias:
            o_bias = eve_attn.c_proj.bias.data

        # --- Copy weights into CoDA's projection layers ---
        coda = adapter.coda

        if has_attn_bias:
            coda.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=True)
            coda.q_proj.bias.data.copy_(q_bias)
        coda.q_proj.weight.data.copy_(q_weight)

        if has_attn_bias:
            coda.k_proj = nn.Linear(n_embd, n_head * head_dim, bias=True)
            coda.k_proj.bias.data.copy_(k_bias)
        coda.k_proj.weight.data.copy_(k_weight)

        if has_attn_bias:
            coda.v_proj = nn.Linear(n_embd, n_head * head_dim, bias=True)
            coda.v_proj.bias.data.copy_(v_bias)
        coda.v_proj.weight.data.copy_(v_weight)

        if has_proj_bias:
            coda.o_proj = nn.Linear(n_embd, n_embd, bias=True)
            coda.o_proj.bias.data.copy_(o_bias)
        coda.o_proj.weight.data.copy_(o_weight)

        return adapter

    # ------------------------------------------------------------------
    # Manual weight loading (when you have the tensors directly)
    # ------------------------------------------------------------------

    def load_eve_weights(
        self,
        *,
        c_attn_weight: torch.Tensor,
        c_attn_bias: Optional[torch.Tensor] = None,
        c_proj_weight: torch.Tensor,
        c_proj_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Load Eve projection weights/biases from raw tensors.

        Useful when loading from a checkpoint dictionary without
        constructing an Eve module first.

        Args:
            c_attn_weight: Fused QKV weight ``(3 * n_embd, n_embd)``.
            c_attn_bias: Fused QKV bias ``(3 * n_embd,)``, or None if
                the original projection had no bias.
            c_proj_weight: Output projection weight ``(n_embd, n_embd)``.
            c_proj_bias: Output projection bias ``(n_embd,)``, or None.
        """
        n = self.n_embd
        coda = self.coda

        q_w = c_attn_weight[:n, :]
        k_w = c_attn_weight[n : 2 * n, :]
        v_w = c_attn_weight[2 * n :, :]

        if c_attn_bias is not None:
            q_b = c_attn_bias[:n]
            k_b = c_attn_bias[n : 2 * n]
            v_b = c_attn_bias[2 * n :]

            # Replace with biased Linear layers if not already biased.
            if coda.q_proj.bias is None:
                coda.q_proj = nn.Linear(n, self.n_head * self.head_dim, bias=True)
            if coda.k_proj.bias is None:
                coda.k_proj = nn.Linear(n, self.n_head * self.head_dim, bias=True)
            if coda.v_proj.bias is None:
                coda.v_proj = nn.Linear(n, self.n_head * self.head_dim, bias=True)
            coda.q_proj.bias.data.copy_(q_b)
            coda.k_proj.bias.data.copy_(k_b)
            coda.v_proj.bias.data.copy_(v_b)

        coda.q_proj.weight.data.copy_(q_w)
        coda.k_proj.weight.data.copy_(k_w)
        coda.v_proj.weight.data.copy_(v_w)

        if c_proj_bias is not None:
            if coda.o_proj.bias is None:
                coda.o_proj = nn.Linear(n, n, bias=True)
            coda.o_proj.bias.data.copy_(c_proj_bias)
        coda.o_proj.weight.data.copy_(c_proj_weight)

    # ------------------------------------------------------------------
    # Forward (matches Eve's CausalSelfAttention.forward signature)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: object = None,
    ) -> torch.Tensor:
        """Forward pass matching Eve's ``CausalSelfAttention.forward``.

        Args:
            x: Input tensor ``(B, T, C)`` where C == n_embd.
            freqs_cis: Eve's complex-valued RoPE frequencies.  Accepted
                for API compatibility but **ignored** -- CoDA computes its
                own real-valued pairwise RoPE internally with the same
                theta=10000.

        Returns:
            Output tensor ``(B, T, C)``.
        """
        if not self.bounded:
            return self._forward_unbounded(x)
        else:
            return self._forward_bounded(x)

    # ------------------------------------------------------------------
    # Unbounded forward (training mode via CoDAGQA)
    # ------------------------------------------------------------------

    def _forward_unbounded(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to CoDAGQA.forward() for standard causal training."""
        out, _kv = self.coda(x, is_causal=True)
        return out

    # ------------------------------------------------------------------
    # Bounded forward (inference mode via CoDAGQALandmarkPerf2)
    # ------------------------------------------------------------------

    def _forward_bounded(self, x: torch.Tensor) -> torch.Tensor:
        """Route through bounded KV cache: prefill for L>1, step for L==1."""
        B, L, C = x.shape

        # Auto-initialise state on first call.
        if self._state is None:
            self._state = self.coda.init_state(
                batch_size=B, device=x.device, dtype=x.dtype,
            )

        state = self._state

        if L == 1:
            # Autoregressive decode: single token step.
            y, self._state = self.coda.step(x, state)
        else:
            # Prefill: process in chunks to populate the bounded cache.
            y, self._state = self.coda.prefill_chunked(
                x, state, block_size=self.block_size,
            )

        return y

    # ------------------------------------------------------------------
    # State management helpers (bounded mode)
    # ------------------------------------------------------------------

    def init_inference_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CoDAGQALandmarkStatePerf2:
        """Initialize (or reinitialize) the bounded KV cache state.

        Call this before starting a new generation sequence to reset the
        memory banks.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device.
            dtype: Data type for KV buffers (e.g. ``torch.bfloat16``).

        Returns:
            The freshly created state object (also stored internally).

        Raises:
            RuntimeError: If the adapter is in unbounded mode.
        """
        if not self.bounded:
            raise RuntimeError(
                "init_inference_state() is only available in bounded mode. "
                "Construct the adapter with bounded=True."
            )
        self._state = self.coda.init_state(
            batch_size=batch_size, device=device, dtype=dtype,
        )
        return self._state

    def set_state(self, state: CoDAGQALandmarkStatePerf2) -> None:
        """Set the current inference state.

        Useful for layer-by-layer decode loops where an external driver
        manages states per layer.

        Args:
            state: A ``CoDAGQALandmarkStatePerf2`` instance (typically
                obtained from a previous ``get_state()`` call or from
                ``init_inference_state()``).
        """
        self._state = state

    def get_state(self) -> Optional[CoDAGQALandmarkStatePerf2]:
        """Return the current inference state, or ``None`` if uninitialized."""
        return self._state

    def reset_state(self) -> None:
        """Clear the inference state, forcing reinitialization on next forward."""
        self._state = None

    # ------------------------------------------------------------------
    # Convenience: cache size reporting
    # ------------------------------------------------------------------

    def cache_bytes(self, *, batch_size: int, dtype: torch.dtype) -> int:
        """Return the bounded KV cache size in bytes (bounded mode only).

        Args:
            batch_size: Number of sequences.
            dtype: Element data type.

        Returns:
            Total cache memory in bytes.

        Raises:
            RuntimeError: If the adapter is in unbounded mode.
        """
        if not self.bounded:
            raise RuntimeError(
                "cache_bytes() is only meaningful in bounded mode."
            )
        return self.coda.cache_bytes(batch_size=batch_size, dtype=dtype)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        mode = "bounded" if self.bounded else "unbounded"
        parts = [
            f"n_embd={self.n_embd}",
            f"n_head={self.n_head}",
            f"head_dim={self.head_dim}",
            f"mode={mode}",
        ]
        if self.bounded:
            parts.append(f"block_size={self.block_size}")
        return ", ".join(parts)
