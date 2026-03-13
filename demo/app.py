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
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr

try:
    from demo.library import get_library_record, list_library_records, save_library_record
except ModuleNotFoundError:
    from library import get_library_record, list_library_records, save_library_record

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
ADAPTERS_FILE = os.environ.get("ADAPTERS_FILE", "coda_adapters.pt")
MAX_INGEST_TOKENS = int(os.environ.get("MAX_INGEST_TOKENS", "32768"))
STATE_LIBRARY_DIR = Path(
    os.environ.get(
        "STATE_LIBRARY_DIR",
        str(Path(__file__).resolve().parent / "state_library"),
    )
)

APP_CSS = """
:root {
  --paper: #f5f0e6;
  --ink: #18211f;
  --accent: #8a3b12;
  --accent-soft: rgba(138, 59, 18, 0.14);
  --panel: rgba(255, 251, 244, 0.9);
  --border: rgba(24, 33, 31, 0.14);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(204, 134, 62, 0.18), transparent 28%),
    linear-gradient(135deg, #efe7d8 0%, #f9f5ec 52%, #ece4d5 100%);
  color: var(--ink);
}

.hero-shell {
  border: 1px solid var(--border);
  background: linear-gradient(160deg, rgba(255, 252, 246, 0.92), rgba(244, 236, 223, 0.88));
  border-radius: 24px;
  padding: 24px;
  box-shadow: 0 20px 60px rgba(72, 46, 26, 0.10);
  margin-bottom: 18px;
}

.hero-shell h1,
.hero-shell h2,
.hero-shell h3 {
  font-family: Georgia, "Times New Roman", serif;
}

.hero-shell code,
.hero-shell pre,
.hero-shell .mono {
  font-family: "IBM Plex Mono", "Cascadia Code", Consolas, monospace;
}

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);
  gap: 16px;
}

.hero-card {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  padding: 16px 18px;
}

.hero-kicker {
  display: inline-block;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 10px;
}

.hero-stat {
  margin: 0;
  padding: 0;
  list-style: none;
}

.hero-stat li + li {
  margin-top: 8px;
}

.hero-stat strong {
  color: var(--accent);
}

@media (max-width: 860px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
}
"""

_ndb = None
_lock = threading.Lock()
_loaded_meta: Optional[dict] = None


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


def _derive_document_label(text: str, uploaded_file, preferred_label: str) -> str:
    label = preferred_label.strip()
    if label:
        return label

    if uploaded_file:
        return Path(uploaded_file).stem[:64]

    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:64]

    return f"document-{int(time.time())}"


def _library_summary() -> list:
    records = list_library_records(STATE_LIBRARY_DIR)
    return [
        {
            "label": record["label"],
            "slug": record["slug"],
            "tokens": record.get("metadata", {}).get("num_tokens", "?"),
            "chars": record.get("metadata", {}).get("doc_length_chars", "?"),
            "model_id": record.get("metadata", {}).get("model_id", "?"),
            "state_size_mb": round(record["file_size_bytes"] / (1024 * 1024), 1),
            "created_at": record.get("created_at", ""),
        }
        for record in records
    ]


def _library_choices() -> list:
    choices = []
    for record in list_library_records(STATE_LIBRARY_DIR):
        meta = record.get("metadata", {})
        tokens = meta.get("num_tokens", "?")
        size_mb = record["file_size_bytes"] / (1024 * 1024)
        label = f"{record['label']} | {tokens} tok | {size_mb:.1f} MB"
        choices.append((label, record["slug"]))
    return choices


def _library_dropdown_update(selected_slug: Optional[str] = None):
    choices = _library_choices()
    values = {value for _, value in choices}
    if selected_slug not in values:
        selected_slug = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected_slug)


def _format_loaded_status(meta: dict, *, label: Optional[str] = None) -> str:
    db = _get_ndb()
    prefix = f"Loaded {label}: " if label else "State loaded: "
    file_model = meta.get("model_id", "unknown")
    warning = ""
    if file_model != db.model_id:
        warning = (
            f"\nWarning: state was created with {file_model}, "
            f"current model is {db.model_id}."
        )
    return (
        f"{prefix}{meta.get('num_tokens', '?'):,} tokens, "
        f"model: {file_model}"
        f"{warning}"
    )


# ---------------------------------------------------------------------------
# Tab 1: Ingest
# ---------------------------------------------------------------------------

def ingest_document(text: str, uploaded_file, document_label: str) -> tuple:
    """Process document text and return state stats plus library updates."""
    global _loaded_meta
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
                _library_summary(),
                _library_dropdown_update(),
                "No state loaded",
            )

        try:
            t0 = time.perf_counter()
            result = db.ingest(text, max_tokens=MAX_INGEST_TOKENS)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            return (
                f"Error during ingestion: {e}",
                None,
                None,
                _library_summary(),
                _library_dropdown_update(),
                "No state loaded",
            )

        label = _derive_document_label(text, uploaded_file, document_label)
        record = save_library_record(
            STATE_LIBRARY_DIR,
            label=label,
            ingested=result,
            save_state=db.save_state,
        )

        _loaded_meta = result["metadata"]
        file_size = record["file_size_bytes"]

        # Collect display stats
        stats = db.get_memory_stats()
        meta = result["metadata"]

        status = (
            f"Ingested {meta['num_tokens']:,} tokens "
            f"({meta['doc_length_chars']:,} chars) in {elapsed:.1f}s.\n"
            f"Saved as {record['label']} ({file_size / (1024 * 1024):.1f} MB state)"
        )

        display_stats = {
            "tokens_processed": meta["num_tokens"],
            "state_size_mb": round(stats["state_size_mb"], 1),
            "num_layers": stats["num_layers"],
            "ingest_time_sec": round(elapsed, 1),
            "library_slug": record["slug"],
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

        return (
            status,
            display_stats,
            record["state_path"],
            _library_summary(),
            _library_dropdown_update(record["slug"]),
            _format_loaded_status(meta, label=record["label"]),
        )


# ---------------------------------------------------------------------------
# Tab 2: Query
# ---------------------------------------------------------------------------

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
        db.restore_state(loaded)
        return _format_loaded_status(meta)


def load_library_state(selected_slug: str) -> str:
    """Load a saved state from the demo's persistent library."""
    global _loaded_meta
    if not selected_slug:
        return "No saved state selected."

    record = get_library_record(STATE_LIBRARY_DIR, selected_slug)
    if record is None:
        return "Saved state not found. Refresh the library list and try again."

    with _lock:
        db = _get_ndb()
        try:
            loaded = db.load_state(record["state_path"])
        except Exception as e:
            return f"Error loading saved state: {e}"

        meta = loaded.get("metadata", {})
        _loaded_meta = meta
        db.restore_state(loaded)
        return _format_loaded_status(meta, label=record["label"])


def refresh_library():
    """Refresh the saved-state library views."""
    return _library_summary(), _library_dropdown_update()


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
        theme=gr.themes.Base(),
        css=APP_CSS,
    ) as app:
        gr.HTML(
            """
            <section class="hero-shell">
              <div class="hero-grid">
                <div class="hero-card">
                  <div class="hero-kicker">Stateful Neural Database Demo</div>
                  <h1>CoDA-GQA-L turns a document into a fixed-size neural state.</h1>
                  <p>
                    Ingest a document once, save the bounded attention state, then query it later
                    without replaying the original text. The demo now behaves like a tiny document
                    library instead of a one-off upload form.
                  </p>
                </div>
                <div class="hero-card">
                  <div class="hero-kicker">Demo Profile</div>
                  <ul class="hero-stat">
                    <li><strong>Model</strong> <span class="mono">"""
            + MODEL_ID
            + """</span></li>
                    <li><strong>State Store</strong> <span class="mono">"""
            + str(STATE_LIBRARY_DIR)
            + """</span></li>
                    <li><strong>Bound</strong> <span class="mono">W=256 Me=64 Ms=64</span></li>
                  </ul>
                </div>
              </div>
            </section>
            """
        )

        # ── Tab 1: Ingest ────────────────────────────────────────────
        with gr.Tab("Ingest Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    document_label = gr.Textbox(
                        label="Document Label",
                        placeholder="Short name for the saved neural state",
                    )
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
                    library_records = gr.JSON(
                        label="State Library",
                        value=_library_summary(),
                    )

        # ── Tab 2: Query ─────────────────────────────────────────────
        with gr.Tab("Query Document"):
            with gr.Row():
                with gr.Column(scale=2):
                    state_upload = gr.File(
                        label="Upload State File (.pt)",
                        file_types=[".pt"],
                    )
                    library_state_select = gr.Dropdown(
                        label="Or load a saved library state",
                        choices=_library_choices(),
                        value=None,
                    )
                    with gr.Row():
                        refresh_library_btn = gr.Button("Refresh Library")
                        load_library_btn = gr.Button("Load Saved State")
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

        ingest_btn.click(
            fn=ingest_document,
            inputs=[text_input, file_input, document_label],
            outputs=[
                ingest_status,
                ingest_stats,
                state_download,
                library_records,
                library_state_select,
                load_status,
            ],
        )
        state_upload.change(
            fn=load_state_file,
            inputs=[state_upload],
            outputs=[load_status],
        )
        refresh_library_btn.click(
            fn=refresh_library,
            inputs=[],
            outputs=[library_records, library_state_select],
        )
        load_library_btn.click(
            fn=load_library_state,
            inputs=[library_state_select],
            outputs=[load_status],
        )
        ask_btn.click(
            fn=answer_question,
            inputs=[question_input, max_tokens_slider, temp_slider],
            outputs=[answer_output, gen_info],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
