"""Stabil fingerprint (SHA256) hesaplama.

Üç farklı fingerprint kavramı:
    1. file_fingerprint: Ham dosya byte'larının SHA256 değeri.
    2. opening_fingerprint: Immutable opening alanlarının semantik olarak
       normalize edilmiş hali.
    3. row_fingerprint: Opening + mutable outcome alanlarının normalize hali.

Fingerprint üretim kuralları:
    - Anahtarlar deterministik sırada (alfabetik).
    - Tarihler ISO-8601 formatına çevrilir.
    - Null değerler tutarlı şekilde temsil edilir.
    - UTF-8 encoding kullanılır.
    - Ham JSON string sırası veya byte düzeyindeki yazım farkı fingerprint'i
      değiştirmez.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


def compute_file_fingerprint(file_path: str | Path) -> str:
    """Dosyanın SHA256 hash değerini hesaplar."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _normalize_value(value: object) -> str:
    """Tek bir değeri fingerprint için normalize eder."""
    if value is None:
        return "<<null>>"
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if text == "":
        return "<<empty>>"
    return text


def _normalize_dict(data: dict) -> str:
    """Dict'i deterministik sırayla normalize edilmiş JSON string'ine çevirir."""
    normalized = {k: _normalize_value(v) for k, v in sorted(data.items())}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_opening_fingerprint(opening_fields: dict) -> str:
    """Immutable opening alanlarından stabil fingerprint üretir."""
    payload = _normalize_dict(opening_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_row_fingerprint(opening_fields: dict, outcome_fields: dict) -> str:
    """Opening + outcome alanlarından stabil fingerprint üretir."""
    combined = {}
    combined.update({f"opening__{k}": v for k, v in opening_fields.items()})
    combined.update({f"outcome__{k}": v for k, v in outcome_fields.items()})
    payload = _normalize_dict(combined)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
