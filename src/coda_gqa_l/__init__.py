# CoDA-GQA-L: Constrained Orthogonal Differential Attention + GQA with Landmark Memory
"""
Public API for CoDA-GQA-L.

Core attention module (bounded KV cache):
    CoDAGQALandmarkPerf2, CoDAGQALandmarkStatePerf2

Unbounded baselines:
    CoDAGQA, BaselineGQA

Shared primitives:
    HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
"""

from .attention import CoDAGQALandmarkPerf2
from .baseline import BaselineGQA, CoDAGQA
from .eve_adapter import EveCoDAAdapter
from .memory_banks import MemoryBankMixin
from .primitives import HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
from .state import CoDAGQALandmarkStatePerf2

__all__ = [
    "CoDAGQALandmarkPerf2",
    "CoDAGQALandmarkStatePerf2",
    "CoDAGQA",
    "BaselineGQA",
    "EveCoDAAdapter",
    "HeadwiseRMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "repeat_kv",
]
