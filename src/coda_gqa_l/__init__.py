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

# Add kernels/ to sys.path so Triton kernel packages (triton_bank_routing,
# triton_diff_flash) are importable as top-level modules.  Must run before
# any submodule imports that reference these packages.
import sys as _sys
from pathlib import Path as _Path
_kernels_dir = str(_Path(__file__).resolve().parent.parent.parent / "kernels")
if _kernels_dir not in _sys.path:
    _sys.path.insert(0, _kernels_dir)

from .attention import CoDAGQALandmarkPerf2
from .baseline import BaselineGQA, CoDAGQA
from .eve_adapter import EveCoDAAdapter
from .llama_adapter import LlamaCoDAAdapter
from .memory_banks import MemoryBankMixin
from .primitives import HeadwiseRMSNorm, RotaryEmbedding, apply_rope, repeat_kv
from .state import CoDAGQALandmarkStatePerf2

__all__ = [
    "CoDAGQALandmarkPerf2",
    "CoDAGQALandmarkStatePerf2",
    "CoDAGQA",
    "BaselineGQA",
    "EveCoDAAdapter",
    "LlamaCoDAAdapter",
    "HeadwiseRMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "repeat_kv",
]
