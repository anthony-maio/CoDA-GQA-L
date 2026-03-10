# neural_db.py
# High-level "Stateful Neural Database" API for CoDA-GQA-L.
#
# Workflow:
#   1. db = NeuralDatabase("user/Qwen3-4B-CoDA-GQA-L")
#   2. result = db.ingest("long document text...")
#   3. db.save_state(result, "doc.pt")        # 61 MB on disk
#   4. db.restore_state(db.load_state("doc.pt"))
#   5. answer = db.query("What is the document about?")
#
# Each query deep-copies state before decoding so multiple queries
# against the same document are independent.

from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from .state import CoDAGQALandmarkStatePerf2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_to_cpu(state: CoDAGQALandmarkStatePerf2) -> CoDAGQALandmarkStatePerf2:
    """Move all tensor fields to CPU (for serialization)."""
    return CoDAGQALandmarkStatePerf2(
        k_buf=state.k_buf.cpu(),
        v_buf=state.v_buf.cpu(),
        allowed=state.allowed.cpu(),
        g_recent=state.g_recent.cpu(),
        recent_idx=state.recent_idx,
        recent_filled=state.recent_filled,
        mem_last_exact=state.mem_last_exact.cpu(),
        pos=state.pos,
        active_exact=state.active_exact,
        active_summary=state.active_summary,
        _exact_v_norm=state._exact_v_norm.cpu() if state._exact_v_norm is not None else None,
        _sum_lf_k_norm=state._sum_lf_k_norm.cpu() if state._sum_lf_k_norm is not None else None,
        metrics=dict(state.metrics) if state.metrics is not None else None,
    )


def _state_to_device(
    state: CoDAGQALandmarkStatePerf2,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Move all tensor fields to *device*/*dtype* in-place."""
    state.k_buf = state.k_buf.to(device=device, dtype=dtype)
    state.v_buf = state.v_buf.to(device=device, dtype=dtype)
    state.allowed = state.allowed.to(device=device)  # bool, no dtype cast
    state.g_recent = state.g_recent.to(device=device, dtype=torch.float32)
    state.mem_last_exact = state.mem_last_exact.to(device=device)
    if state._exact_v_norm is not None:
        state._exact_v_norm = state._exact_v_norm.to(device=device, dtype=torch.float32)
    if state._sum_lf_k_norm is not None:
        state._sum_lf_k_norm = state._sum_lf_k_norm.to(device=device, dtype=torch.float32)


def _deep_copy_state(state: CoDAGQALandmarkStatePerf2) -> CoDAGQALandmarkStatePerf2:
    """Deep-copy a state, cloning all tensors."""
    return CoDAGQALandmarkStatePerf2(
        k_buf=state.k_buf.clone(),
        v_buf=state.v_buf.clone(),
        allowed=state.allowed.clone(),
        g_recent=state.g_recent.clone(),
        recent_idx=state.recent_idx,
        recent_filled=state.recent_filled,
        mem_last_exact=state.mem_last_exact.clone(),
        pos=state.pos,
        active_exact=state.active_exact,
        active_summary=state.active_summary,
        _exact_v_norm=state._exact_v_norm.clone() if state._exact_v_norm is not None else None,
        _sum_lf_k_norm=state._sum_lf_k_norm.clone() if state._sum_lf_k_norm is not None else None,
        metrics=dict(state.metrics) if state.metrics is not None else None,
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample a single token from logits with temperature + top-k + top-p."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 0] = False
        remove_orig = remove.scatter(dim=-1, index=sorted_idx, src=remove)
        logits = logits.masked_fill(remove_orig, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# NeuralDatabase
# ---------------------------------------------------------------------------

class NeuralDatabase:
    """High-level API for the Stateful Neural Database pattern.

    Wraps a Qwen3 (or Llama-family) model with CoDA-GQA-L bounded adapters,
    providing document ingestion (prefill -> save state) and query answering
    (load state -> decode) in a clean interface.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        window: int = 256,
        num_landmarks_exact: int = 64,
        num_landmarks_summary: int = 64,
        block_size: int = 1024,
        collect_metrics: bool = True,
        adapters_file: Optional[str] = "coda_adapters.pt",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.window = window
        self.num_landmarks_exact = num_landmarks_exact
        self.num_landmarks_summary = num_landmarks_summary

        # Auto-detect device/dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
        )

        # Swap attention layers -> bounded CoDA adapters
        from .qwen3_adapter import Qwen3CoDAAdapter

        self.adapters: List = Qwen3CoDAAdapter.swap_qwen3_layers(
            self.model,
            bounded=True,
            window=window,
            num_landmarks_exact=num_landmarks_exact,
            num_landmarks_summary=num_landmarks_summary,
            block_size=block_size,
            collect_metrics=collect_metrics,
        )

        # Load trained adapter weights if available
        if adapters_file is not None:
            self._load_adapter_weights(adapters_file)

        # CRITICAL: eval() must come AFTER adapter installation.
        # In training mode, fresh states allocated each call.
        # In eval mode, states persist for decode loop.
        self.model.eval()

    def _load_adapter_weights(self, adapters_file: str) -> None:
        """Load trained CoDA adapter weights from the model repo."""
        try:
            from huggingface_hub import hf_hub_download

            adapters_path = hf_hub_download(self.model_id, adapters_file)
        except Exception:
            # Adapter weights not found -- cold-swap only (identity init)
            return

        ckpt = torch.load(adapters_path, map_location="cpu", weights_only=True)

        for i, layer in enumerate(self.model.model.layers):
            layer_device = next(layer.parameters()).device
            key = f"layer_{i}"
            if key in ckpt:
                layer.self_attn.load_state_dict(ckpt[key], strict=False)
            layer.self_attn = layer.self_attn.to(device=layer_device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, text: str, *, max_tokens: int = 32768) -> Dict[str, Any]:
        """Process a document through bounded attention and collect state.

        Returns dict with ``"states"`` (layer_idx -> state) and ``"metadata"``.
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Document text must be at least 10 characters.")

        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_tokens,
        )
        input_ids = input_ids.to(self.model.device)

        # Reset all adapter states
        for adapter in self.adapters:
            adapter.reset_state()

        # Prefill -- adapters route L > 1 through prefill_chunked
        with torch.no_grad():
            self.model(input_ids=input_ids, use_cache=False)

        # Collect states
        states: Dict[int, CoDAGQALandmarkStatePerf2] = {}
        metrics: Dict[int, dict] = {}
        for i, adapter in enumerate(self.adapters):
            state = adapter.get_state()
            states[i] = state
            if state is not None and state.metrics is not None:
                metrics[i] = dict(state.metrics)

        return {
            "states": states,
            "metadata": {
                "doc_length_chars": len(text),
                "num_tokens": int(input_ids.shape[1]),
                "model_id": self.model_id,
                "window": self.window,
                "num_landmarks_exact": self.num_landmarks_exact,
                "num_landmarks_summary": self.num_landmarks_summary,
                "timestamp": time.time(),
            },
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save_state(ingested: Dict[str, Any], path: str) -> int:
        """Save ingested state to a ``.pt`` file.  Returns file size in bytes."""
        save_data = {
            "states": {
                k: _state_to_cpu(v) for k, v in ingested["states"].items()
            },
            "metadata": ingested["metadata"],
        }
        torch.save(save_data, path)
        return os.path.getsize(path)

    @staticmethod
    def load_state(path: str) -> Dict[str, Any]:
        """Load a previously saved state file."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def restore_state(self, loaded: Dict[str, Any]) -> None:
        """Restore loaded state into the model's adapters."""
        states = loaded["states"]
        for i, adapter in enumerate(self.adapters):
            state = states[i]
            _state_to_device(state, self.device, self.dtype)
            adapter.set_state(state)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate an answer from the current loaded state.

        Deep-copies state before decoding so repeated queries against
        the same document are independent.
        """
        if any(a.get_state() is None for a in self.adapters):
            raise RuntimeError("No document state loaded. Call ingest() or restore_state() first.")

        # Save state snapshot (queries mutate memory banks)
        saved = {i: _deep_copy_state(a.get_state()) for i, a in enumerate(self.adapters)}

        try:
            prompt = f"\n\nQuestion: {question}\nAnswer:"
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to(self.model.device)

            with torch.no_grad():
                # Prefill question tokens into existing state
                outputs = self.model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]

                generated: List[int] = []
                for _ in range(max_new_tokens):
                    next_token = _sample_token(logits, temperature, top_k, top_p)
                    tok_id = next_token.item()
                    generated.append(tok_id)

                    if tok_id == self.tokenizer.eos_token_id:
                        break

                    outputs = self.model(input_ids=next_token, use_cache=False)
                    logits = outputs.logits[:, -1, :]

            return self.tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            # Restore original state
            for i, adapter in enumerate(self.adapters):
                adapter.set_state(saved[i])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return aggregate memory bank statistics across all layers."""
        stats: Dict[str, Any] = {
            "num_layers": len(self.adapters),
            "state_size_mb": 0.0,
            "layers": [],
        }
        for i, adapter in enumerate(self.adapters):
            state = adapter.get_state()
            if state is None:
                continue

            layer_info: Dict[str, Any] = {
                "layer": i,
                "pos": state.pos,
                "recent_filled": state.recent_filled,
                "active_exact": state.active_exact,
                "active_summary": state.active_summary,
            }
            if state.metrics:
                layer_info.update(state.metrics)

            cache_bytes = adapter.cache_bytes(batch_size=1, dtype=self.dtype)
            layer_info["cache_bytes"] = cache_bytes
            stats["state_size_mb"] += cache_bytes / (1024 * 1024)
            stats["layers"].append(layer_info)

        return stats
