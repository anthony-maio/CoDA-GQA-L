from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


PathLike = Union[str, Path]
SaveStateFn = Callable[[Dict[str, Any], str], int]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "state"


def _unique_slug(base_slug: str, directory: Path) -> str:
    slug = base_slug
    counter = 2
    while (directory / f"{slug}.pt").exists() or (directory / f"{slug}.json").exists():
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def save_library_record(
    library_dir: PathLike,
    label: str,
    ingested: Dict[str, Any],
    save_state: SaveStateFn,
) -> Dict[str, Any]:
    label = label.strip()
    if not label:
        raise ValueError("State library label must be non-empty.")

    library_path = Path(library_dir)
    library_path.mkdir(parents=True, exist_ok=True)

    slug = _unique_slug(_slugify(label), library_path)
    state_path = library_path / f"{slug}.pt"
    file_size = save_state(ingested, str(state_path))
    record = {
        "slug": slug,
        "label": label,
        "state_path": str(state_path),
        "file_size_bytes": file_size,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metadata": dict(ingested.get("metadata", {})),
    }

    meta_path = library_path / f"{slug}.json"
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def get_library_record(library_dir: PathLike, slug: str) -> Optional[Dict[str, Any]]:
    meta_path = Path(library_dir) / f"{slug}.json"
    if not meta_path.exists():
        return None
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    state_path = Path(record["state_path"])
    if not state_path.exists():
        return None
    record["file_size_bytes"] = state_path.stat().st_size
    return record


def list_library_records(library_dir: PathLike) -> List[Dict[str, Any]]:
    library_path = Path(library_dir)
    if not library_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    for meta_path in library_path.glob("*.json"):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        state_path = Path(record.get("state_path", ""))
        if not state_path.exists():
            continue

        record["file_size_bytes"] = state_path.stat().st_size
        records.append(record)

    return sorted(
        records,
        key=lambda record: (
            str(record.get("created_at", "")),
            str(record.get("slug", "")),
        ),
        reverse=True,
    )
