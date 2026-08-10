"""Kanonik mapping servisi.

Ham kategorik değerleri sürümlenmiş ve deterministik mapping kullanarak
sabit snake_case kanonik kodlara dönüştürür.

Kurallar:
    - Ham değer boş/None → canonical_code = "missing"
    - Ham değer mapping kataloğunda yok → canonical_code = "unknown"
    - Mapping çalışma anında otomatik yeni kod üretmez.
    - Bilinmeyen değer caller'a bildirilir (warning için).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MappingResult:
    """Tek bir mapping işleminin sonucu."""

    canonical_code: str
    is_known: bool
    is_missing: bool


class CanonicalMapper:
    """Ham kategorik değerleri kanonik kodlara dönüştürür.

    Attributes:
        version: Mapping kataloğunun sürümü (örn. "1.0.0").
    """

    def __init__(self, mapping_path: str | Path) -> None:
        self._path = Path(mapping_path)
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        self.version: str = data["version"]
        self._columns: dict[str, dict[str, str]] = data["columns"]

    def map(self, column: str, raw_value: str | None) -> MappingResult:
        """Bir ham değeri kanonik koda dönüştürür.

        Args:
            column: Kaynak kolon adı (örn. "source", "type").
            raw_value: Ham değer.

        Returns:
            MappingResult nesnesi.
        """
        if raw_value is None or str(raw_value).strip() == "":
            return MappingResult(canonical_code="missing", is_known=False, is_missing=True)

        trimmed = str(raw_value).strip()
        col_map = self._columns.get(column, {})
        canonical = col_map.get(trimmed)

        if canonical is not None:
            return MappingResult(canonical_code=canonical, is_known=True, is_missing=False)

        return MappingResult(canonical_code="unknown", is_known=False, is_missing=False)

    def get_known_values(self, column: str) -> set[str]:
        """Bir kolon için bilinen ham değer kümesini döndürür."""
        return set(self._columns.get(column, {}).keys())
