from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.library import list_library_records, save_library_record


def _fake_save_state(payload, path: str) -> int:
    target = Path(path)
    target.write_bytes(b"state-bytes")
    return target.stat().st_size


def test_save_library_record_writes_state_and_sidecar(tmp_path):
    ingested = {
        "metadata": {
            "model_id": "test-model",
            "num_tokens": 42,
            "doc_length_chars": 314,
        }
    }

    record = save_library_record(
        tmp_path,
        label="Doc Alpha",
        ingested=ingested,
        save_state=_fake_save_state,
    )

    state_path = Path(record["state_path"])
    meta_path = tmp_path / f"{record['slug']}.json"

    assert state_path.exists()
    assert meta_path.exists()
    assert record["file_size_bytes"] == len(b"state-bytes")
    assert record["label"] == "Doc Alpha"

    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved_meta["label"] == "Doc Alpha"
    assert saved_meta["metadata"]["model_id"] == "test-model"


def test_save_library_record_disambiguates_duplicate_labels(tmp_path):
    ingested = {"metadata": {"model_id": "test", "num_tokens": 1, "doc_length_chars": 10}}

    first = save_library_record(tmp_path, "Demo Doc", ingested, _fake_save_state)
    second = save_library_record(tmp_path, "Demo Doc", ingested, _fake_save_state)

    assert first["slug"] == "demo-doc"
    assert second["slug"] == "demo-doc-2"
    assert Path(second["state_path"]).exists()


def test_list_library_records_returns_newest_first(tmp_path):
    ingested = {"metadata": {"model_id": "test", "num_tokens": 1, "doc_length_chars": 10}}

    older = save_library_record(tmp_path, "Older", ingested, _fake_save_state)
    newer = save_library_record(tmp_path, "Newer", ingested, _fake_save_state)

    records = list_library_records(tmp_path)

    assert [record["slug"] for record in records[:2]] == [newer["slug"], older["slug"]]


def test_list_library_records_skips_orphaned_metadata(tmp_path):
    meta_path = tmp_path / "broken.json"
    meta_path.write_text(
        json.dumps(
            {
                "slug": "broken",
                "label": "Broken",
                "state_path": str(tmp_path / "missing.pt"),
                "created_at": "2026-03-13T00:00:00Z",
                "file_size_bytes": 0,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    assert list_library_records(tmp_path) == []


def test_save_library_record_requires_non_empty_label(tmp_path):
    ingested = {"metadata": {"model_id": "test", "num_tokens": 1, "doc_length_chars": 10}}

    with pytest.raises(ValueError, match="label"):
        save_library_record(tmp_path, "   ", ingested, _fake_save_state)
