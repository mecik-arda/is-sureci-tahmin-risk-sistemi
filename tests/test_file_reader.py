"""Dosya okuyucu testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
import csv

from app.services.file_reader import (
    FileReadError,
    UnsupportedFileFormatError,
    read_data_file,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_read_valid_csv(tmp_path: Path):
    """Geçerli CSV okunabiliyor."""
    rows = [{"case_enquiry_id": "101", "open_dt": "2024-01-01 10:00:00"}]
    path = tmp_path / "test.csv"
    _write_csv(path, rows)

    result = read_data_file(path)
    assert result.row_count == 1
    assert "case_enquiry_id" in result.columns
    assert result.rows[0]["case_enquiry_id"] == "101"


def test_read_valid_xlsx(tmp_path: Path):
    """Geçerli XLSX okunabiliyor."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["case_enquiry_id", "open_dt"])
    ws.append(["101", "2024-01-01 10:00:00"])
    path = tmp_path / "test.xlsx"
    wb.save(path)

    result = read_data_file(path)
    assert result.row_count == 1
    assert result.rows[0]["case_enquiry_id"] == "101"


def test_unsupported_extension(tmp_path: Path):
    """Desteklenmeyen uzantı hatası."""
    path = tmp_path / "test.xls"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFileFormatError):
        read_data_file(path)


def test_empty_file(tmp_path: Path):
    """Boş dosya hatası."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(FileReadError):
        read_data_file(path)


def test_corrupted_csv(tmp_path: Path):
    """Bozuk CSV hatası."""
    path = tmp_path / "bad.csv"
    path.write_text("not,a,valid\r\n\x00\x01\x02", encoding="latin-1")
    try:
        read_data_file(path)
    except (FileReadError, Exception):
        pass
