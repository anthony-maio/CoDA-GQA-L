"""
CoDA-GQA-L: Stateful Neural Database Demo

Gradio app for HuggingFace Spaces. Process documents into fixed-size
neural states (~61 MB), save to disk, load later, and query without
re-reading the original document.

Usage:
    python demo/app.py                           # local dev
    MODEL_ID=user/model python demo/app.py       # custom model
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
ADAPTERS_FILE = os.environ.get("ADAPTERS_FILE", "coda_adapters.pt")
MAX_INGEST_TOKENS = int(os.environ.get("MAX_INGEST_TOKENS", "32768"))

_ndb = None
_lock = threading.Lock()


def _get_ndb():
    """Lazy-load NeuralDatabase (heavy: downloads model on first call)."""
    global _ndb
    if _ndb is None:
        from coda_gqa_l import NeuralDatabase

        _ndb = NeuralDatabase(
            MODEL_ID,
            adapters_file=ADAPTERS_FILE,
            collect_metrics=True,
        )
    return _ndb


# ---------------------------------------------------------------------------
# Tab 1: Ingest
# ---------------------------------------------------------------------------

def ingest_document(text: str, uploaded_file) -> tuple:
    """Process document text and return (status, stats_json, state_file)."""
    with _lock:
        db = _get_ndb()

        # Handle file upload
        if uploaded_file is not None:
            with open(uploaded_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        if not text or len(text.strip()) < 10:
            return (
                "Error: Please provide document text (at least 10 characters).",
                None,
                None,
            )

        try:
            t0 = time.perf_counter()
            result = db.ingest(text, max_tokens=MAX_INGEST_TOKENS)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            return f"Error during ingestion: {e}", None, None

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, prefix="neural_db_",
        )
        file_size = db.save_state(result, tmp.name)

        # Collect display stats
        stats = db.get_memory_stats()
        meta = result["metadata"]

        status = (
            f"Ingested {meta['num_tokens']:,} tokens "
            f"({meta['doc_length_chars']:,} chars) in {elapsed:.1f}s.\n"
            f"State file: {file_size / (1024 * 1024):.1f} MB"
        )

        display_stats = {
            "tokens_processed": meta["num_tokens"],
            "state_size_mb": round(stats["state_size_mb"], 1),
            "num_layers": stats["num_layers"],
            "ingest_time_sec": round(elapsed, 1),
        }

        # Aggregate bank fill from metrics if available
        metrics = result.get("metrics", {})
        if metrics:
            exact_fills = [
                m.get("exact_fill_ratio", 0) for m in metrics.values()
            ]
            summary_fills = [
                m.get("summary_fill_ratio", 0) for m in metrics.values()
            ]
            if exact_fills:
                display_stats["avg_exact_bank_fill"] = f"{sum(exact_fills) / len(exact_fills):.0%}"
            if summary_fills:
                display_stats["avg_summary_bank_fill"] = f"{sum(summary_fills) / len(summary_fills):.0%}"

        return status, display_stats, tmp.name


# ---------------------------------------------------------------------------
# Tab 2: Query
# ---------------------------------------------------------------------------

_loaded_meta: Optional[dict] = None


def load_state_file(uploaded_file) -> str:
    """Load a .pt state file into the model."""
    global _loaded_meta
    if uploaded_file is None:
        return "No file uploaded."

    with _lock:
        db = _get_ndb()
        try:
            loaded = db.load_state(uploaded_file)
        except Exception as e:
            return f"Error loading state file: {e}"

        meta = loaded.get("metadata", {})
        _loaded_meta = meta

        # Check model compatibility
        file_model = meta.get("model_id", "unknown")
        warning = ""
        if file_model != db.model_id:
            warning = (
                f"\nWarning: State was created with {file_model}, "
                f"current model is {db.model_id}."
            )

        db.restore_state(loaded)

        return (
            f"State loaded: {meta.get('num_tokens', '?'):,} tokens, "
            f"model: {meta.get('model_id', '?')}"
            f"{warning}"
        )


def answer_question(
    question: str, max_tokens: int, temperature: float,
) -> tuple:
    """Generate an answer from the loaded state."""
    if not question.strip():
        return "Please enter a question.", ""

    with _lock:
        db = _get_ndb()

        # Check state is loaded
        if any(a.get_state() is None for a in db.adapters):
            return (
                "No document state loaded. Ingest a document or upload a .pt file first.",
                "",
            )

        try:
            t0 = time.perf_counter()
            answer = db.query(
                question,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            elapsed = time.perf_counter() - t0
        except Exception as e:
            return f"Error during generation: {e}", ""

        answer_tokens = len(db.tokenizer.encode(answer))
        info = f"Generated {answer_tokens} tokens in {elapsed:.1f}s"

        return answer, info


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="CoDA-GQA-L: Stateful Neural Database",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# CoDA-GQA-L: Stateful Neural Database\n"
            "Process documents into fixed-size neural states (~61 MB). "
            "Save them. Load them later. Ask questions -- "
            "without re-reading the document."
        )

        # ── Tab 1: Ingest ────────────────────────────────────────────
        with gr.Tab("Ingest Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Document Text",
                        placeholder="Paste your document here...",
                        lines=15,
                    )
                    file_input = gr.File(
                        label="Or upload a .txt file",
                        file_types=[".txt", ".md"],
                    )
                    ingest_btn = gr.Button("Ingest Document", variant="primary")

                with gr.Column(scale=1):
                    ingest_status = gr.Textbox(
                        label="Status", interactive=False, lines=3,
                    )
                    ingest_stats = gr.JSON(label="Memory Stats")
                    state_download = gr.File(label="Download State (.pt)")

            ingest_btn.click(
                fn=ingest_document,
                inputs=[text_input, file_input],
                outputs=[ingest_status, ingest_stats, state_download],
            )

        # ── Tab 2: Query ─────────────────────────────────────────────
        with gr.Tab("Query Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    state_upload = gr.File(
                        label="Upload State File (.pt)",
                        file_types=[".pt"],
                    )
                    load_status = gr.Textbox(
                        label="State Status",
                        value="No state loaded",
                        interactive=False,
                    )
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the document...",
                        lines=3,
                    )
                    with gr.Row():
                        max_tokens_slider = gr.Slider(
                            minimum=16, maximum=512, value=256, step=16,
                            label="Max Tokens",
                        )
                        temp_slider = gr.Slider(
                            minimum=0.0, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature",
                        )
                    ask_btn = gr.Button("Ask", variant="primary")

                with gr.Column(scale=2):
                    answer_output = gr.Textbox(
                        label="Answer", interactive=False, lines=12,
                    )
                    gen_info = gr.Textbox(
                        label="Generation Info", interactive=False,
                    )

            state_upload.change(
                fn=load_state_file,
                inputs=[state_upload],
                outputs=[load_status],
            )
            ask_btn.click(
                fn=answer_question,
                inputs=[question_input, max_tokens_slider, temp_slider],
                outputs=[answer_output, gen_info],
            )

        # ── Tab 3: About ─────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown("""
## How It Works

**CoDA-GQA-L** (Constrained Orthogonal Differential Attention + GQA
with Landmark Memory) replaces standard transformer attention with
bounded-memory differential attention.

### The Stateful Neural Database Concept

1. **Ingest**: Feed a document through the model. Each attention layer
   compresses the document into a fixed-size memory state:
   - **Recent window** (W=256 slots): Ring buffer of exact recent tokens
   - **Exact landmark bank** (Me=64 slots): Novelty-filtered important tokens
   - **Summary landmark bank** (Ms=64 slots): EMA-compressed prototypes

2. **Save**: The entire model state (~61 MB for Qwen3-4B, 36 layers)
   is saved to disk as a `.pt` file.

3. **Query**: Load the state file, ask questions. The model answers
   from its compressed memory without re-reading the original document.

### Key Properties

- **Fixed memory**: State size is O(W + Me + Ms) per layer, independent
  of document length. A 1K-token and a 100K-token document produce
  the same size state file.
- **Semantic compression**: The exact bank retains "needle" facts via
  novelty filtering. The summary bank captures broad themes via EMA.
- **Query isolation**: Each query runs on a snapshot of the document
  state, so multiple questions are independent.

### Architecture

```
Document -> Tokenize -> Prefill (bounded attention) -> State
State -> Save to disk (.pt file, ~61 MB)
State -> Load from disk -> Query -> Answer
```

Built with [CoDA-GQA-L](https://github.com/anthony-maio/CoDA-GQA-L).
            """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
