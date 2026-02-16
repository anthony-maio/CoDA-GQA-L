# state.py
# Mutable inference state for CoDA-GQA-L attention.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class CoDAGQALandmarkStatePerf2:
    """Mutable state for one layer of CoDA-GQA-L during inference.

    Buffer layout: [recent W | exact Me | summary Ms]
    RoPE is already applied to keys stored in k_buf.
    Routing strategies:
      - Exact bank: V-routing (cosine similarity on Values, RoPE-free).
      - Summary bank: LF-K routing (cosine similarity on low-frequency key band).
    Summary bank keys have zero HF band (Phase-Safe EMA).
    """

    k_buf: torch.Tensor       # (B, Hkv, Lbuf, Dh)
    v_buf: torch.Tensor       # (B, Hkv, Lbuf, Dh)
    allowed: torch.Tensor     # (B, Lbuf) bool (True=slot is valid)

    # Recent ring metadata (within the first W slots)
    g_recent: torch.Tensor    # (B, W) float32 write gate per recent slot
    recent_idx: int           # next eviction index (ring) when full
    recent_filled: int        # <= W

    # Exact bank LRU metadata (within exact segment)
    mem_last_exact: torch.Tensor  # (B, Me) int64

    # Absolute position of next token to be written
    pos: int

    # Cached normalized routing targets, updated slot-wise on writes.
    # Exact bank: normalized values (V-routing, RoPE-free).
    # Summary bank: normalized LF key band (LF-K routing, position-invariant).
    _exact_v_norm: Optional[torch.Tensor] = field(default=None, repr=False)
    _sum_lf_k_norm: Optional[torch.Tensor] = field(default=None, repr=False)

    # Optional lightweight metrics counters.  None means disabled (zero overhead).
    # When enabled, a dict of string -> int/float counters tracking bank activity.
    metrics: Optional[Dict[str, Any]] = field(default=None, repr=False)
