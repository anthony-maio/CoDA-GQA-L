"""Render markdown comparison tables from benchmark JSON results.

Reads all results/*.json files produced by run_suite.py, groups by
config_name (taking the latest result if duplicates exist), and prints
two markdown tables: a performance comparison and a config details table.

Usage:
    python benchmarks/render_tables.py               # default results/ dir
    python benchmarks/render_tables.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def human_bytes(n: Optional[int]) -> str:
    """Format byte count as a human-readable string."""
    if n is None:
        return "n/a"
    val = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024:
            return f"{val:.1f}{unit}" if unit != "B" else f"{int(val)}{unit}"
        val /= 1024
    return f"{val:.1f}PB"


def fmt_num(n: Optional[float]) -> str:
    """Format a number with comma separators, or 'n/a' if None."""
    if n is None:
        return "n/a"
    return f"{n:,.0f}"


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load JSON results, keeping only the latest per config_name.

    Returns a dict mapping config_name -> parsed JSON dict.
    """
    if not results_dir.is_dir():
        print(f"Results directory does not exist: {results_dir}")
        return {}

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return {}

    # Group by config_name, keeping the file with the latest timestamp in its
    # system.timestamp field (or filename order as fallback).
    latest: Dict[str, Dict[str, Any]] = {}
    latest_ts: Dict[str, str] = {}

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: Skipping {path.name} ({exc})")
            continue

        name = data.get("config_name")
        if name is None:
            print(f"WARNING: Skipping {path.name} (no config_name)")
            continue

        ts = data.get("system", {}).get("timestamp", "")
        if name not in latest or ts > latest_ts.get(name, ""):
            latest[name] = data
            latest_ts[name] = ts

    return latest


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

# Desired display order (configs not in this list appear at the end, sorted).
DISPLAY_ORDER = [
    "baseline",
    "coda-unbounded",
    "tiny-cache",
    "medium-cache",
    "window-only",
]


def sorted_config_names(results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Sort config names using the preferred display order."""
    ordered: List[str] = []
    for name in DISPLAY_ORDER:
        if name in results:
            ordered.append(name)
    extras = sorted(set(results.keys()) - set(DISPLAY_ORDER))
    ordered.extend(extras)
    return ordered


def discover_prefill_lengths(results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Discover all prefill lengths present across results."""
    lengths: set[str] = set()
    for data in results.values():
        lengths.update(data.get("results", {}).get("prefill_tokens_per_sec", {}).keys())

    def _sort_key(s: str) -> int:
        try:
            return int(s)
        except ValueError:
            return 0

    return sorted(lengths, key=_sort_key)


def render_performance_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Render Table 1: performance comparison."""
    names = sorted_config_names(results)
    lengths = discover_prefill_lengths(results)

    # Build header.
    header_cols = ["Config"]
    for L in lengths:
        header_cols.append(f"Prefill {L} (tok/s)")
    header_cols.extend(["Decode (tok/s)", "KV Cache", "Peak VRAM"])

    sep_cols = ["---"]
    for _ in lengths:
        sep_cols.append("---:")
    sep_cols.extend(["---:", "---:", "---:"])

    lines: List[str] = []
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(sep_cols) + " |")

    for name in names:
        data = results[name]
        res = data.get("results", {})
        prefill = res.get("prefill_tokens_per_sec", {})

        cols = [name]
        for L in lengths:
            cols.append(fmt_num(prefill.get(L)))
        cols.append(fmt_num(res.get("decode_tokens_per_sec")))
        cols.append(human_bytes(res.get("kv_cache_bytes")))
        cols.append(human_bytes(res.get("peak_vram_bytes")))

        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines)


def render_config_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Render Table 2: config details."""
    names = sorted_config_names(results)

    header = "| Config | Attention Type | Window | Me | Ms | Total Buffer |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: |"
    lines = [header, sep]

    for name in names:
        data = results[name]
        atype = data.get("attention_type", "")
        cfg = data.get("config", {})

        window = cfg.get("window", "-")
        me = cfg.get("Me", "-")
        ms = cfg.get("Ms", "-")

        if isinstance(window, int) and isinstance(me, int) and isinstance(ms, int):
            total = window + me + ms
            total_str = str(total)
        else:
            total_str = "unbounded"

        window_str = str(window) if isinstance(window, int) else "-"
        me_str = str(me) if isinstance(me, int) else "-"
        ms_str = str(ms) if isinstance(ms, int) else "-"

        lines.append(f"| {name} | {atype} | {window_str} | {me_str} | {ms_str} | {total_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render markdown tables from CoDA-GQA-L benchmark results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing JSON result files (default: results/ next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    results = load_results(results_dir)
    if not results:
        print("No results to render.")
        sys.exit(0)

    print(f"Loaded {len(results)} config result(s) from {results_dir}\n")

    # Gather system info from the first result for the header.
    first = next(iter(results.values()))
    sys_info = first.get("system", {})
    header_lines = [
        f"# CoDA-GQA-L Benchmark Results",
        f"",
        f"- GPU: {sys_info.get('gpu', 'n/a')}",
        f"- CUDA: {sys_info.get('cuda_version', 'n/a')}",
        f"- PyTorch: {sys_info.get('torch_version', 'n/a')}",
        f"- Dtype: {sys_info.get('dtype', 'n/a')}",
        f"- Timestamp: {sys_info.get('timestamp', 'n/a')}",
        f"",
    ]

    perf_table = render_performance_table(results)
    config_table = render_config_table(results)

    sections = [
        "\n".join(header_lines),
        "## Performance Comparison\n",
        perf_table,
        "",
        "## Configuration Details\n",
        config_table,
        "",
    ]

    full_output = "\n".join(sections)

    # Print to stdout.
    print(full_output)

    # Save to file.
    out_path = results_dir / "tables.md"
    out_path.write_text(full_output, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
