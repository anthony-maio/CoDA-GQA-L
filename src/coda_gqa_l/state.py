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
    Routing (cosine similarity) uses values (V), not keys (K),
    because RoPE makes key similarity position-dependent.
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

    # Cached normalized memory values for cosine similarity routing.
    # Routing uses V (not K) because keys have RoPE applied, making cosine
    # similarity position-dependent.  Values are RoPE-free and preserve
    # pure semantic content for deduplication and EMA blending.
    # Updated slot-wise on writes, avoiding full F.normalize(mem_v) per block.
    _exact_v_norm: Optional[torch.Tensor] = field(default=None, repr=False)
    _sum_v_norm: Optional[torch.Tensor] = field(default=None, repr=False)

    # Optional lightweight metrics counters.  None means disabled (zero overhead).
    # When enabled, a dict of string -> int/float counters tracking bank activity.
    metrics: Optional[Dict[str, Any]] = field(default=None, repr=False)
