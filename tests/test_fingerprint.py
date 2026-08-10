"""Fingerprint servisi testleri."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.services.fingerprint import (
    compute_file_fingerprint,
    compute_opening_fingerprint,
    compute_row_fingerprint,
)


def test_file_fingerprint_stable(tmp_path: Path):
    """Aynı dosya her zaman aynı fingerprint'i üretir."""
    path = tmp_path / "test.txt"
    path.write_text("test content", encoding="utf-8")
    fp1 = compute_file_fingerprint(path)
    fp2 = compute_file_fingerprint(path)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_file_fingerprint_different_content(tmp_path: Path):
    """Farklı içerik farklı fingerprint üretir."""
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("content a", encoding="utf-8")
    p2.write_text("content b", encoding="utf-8")
    assert compute_file_fingerprint(p1) != compute_file_fingerprint(p2)


def test_opening_fingerprint_key_order_independent():
    """Anahtar sırası değişse fingerprint aynı kalır."""
    fields_a = {"external_id": "101", "type": "pothole", "source": "app"}
    fields_b = {"source": "app", "type": "pothole", "external_id": "101"}
    assert compute_opening_fingerprint(fields_a) == compute_opening_fingerprint(fields_b)


def test_opening_fingerprint_date_format_equivalent():
    """Eşdeğer tarih formatları aynı fingerprint üretir."""
    dt = datetime(2024, 6, 15, 10, 30, 0)
    fields_a = {"external_id": "101", "created_at": dt}
    fields_b = {"external_id": "101", "created_at": dt.isoformat()}
    assert compute_opening_fingerprint(fields_a) == compute_opening_fingerprint(fields_b)


def test_opening_fingerprint_semantic_change_detected():
    """Semantik opening değişikliği fingerprint'i değiştirir."""
    fields_a = {"external_id": "101", "type": "pothole"}
    fields_b = {"external_id": "101", "type": "graffiti"}
    assert compute_opening_fingerprint(fields_a) != compute_opening_fingerprint(fields_b)


def test_row_fingerprint_outcome_change_detected():
    """Outcome değişikliği row_fingerprint'i değiştirir ama opening'i değiştirmez."""
    opening = {"external_id": "101", "type": "pothole"}
    outcome_a = {"completed_at": None, "current_status": "Open"}
    outcome_b = {"completed_at": "2024-07-01T00:00:00", "current_status": "Closed"}

    row_a = compute_row_fingerprint(opening, outcome_a)
    row_b = compute_row_fingerprint(opening, outcome_b)

    assert row_a != row_b
    assert compute_opening_fingerprint(opening) == compute_opening_fingerprint(opening)


def test_null_values_consistent():
    """None ve boş string farklı temsil edilir."""
    fields_none = {"a": None}
    fields_empty = {"a": ""}
    assert compute_opening_fingerprint(fields_none) != compute_opening_fingerprint(fields_empty)
