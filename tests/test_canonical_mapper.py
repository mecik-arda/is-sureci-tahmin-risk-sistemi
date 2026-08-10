"""Kanonik mapping servisi testleri."""

from __future__ import annotations

from app.services.canonical_mapper import CanonicalMapper


def test_known_value_maps_correctly(canonical_mapper: CanonicalMapper):
    """Bilinen ham değer doğru kanonik koda dönüşüyor."""
    result = canonical_mapper.map("source", "Citizens Connect App")
    assert result.canonical_code == "citizens_connect_app"
    assert result.is_known is True
    assert result.is_missing is False


def test_empty_value_maps_to_missing(canonical_mapper: CanonicalMapper):
    """Boş değer missing olur."""
    result = canonical_mapper.map("source", "")
    assert result.canonical_code == "missing"
    assert result.is_missing is True


def test_none_value_maps_to_missing(canonical_mapper: CanonicalMapper):
    """None değeri missing olur."""
    result = canonical_mapper.map("source", None)
    assert result.canonical_code == "missing"
    assert result.is_missing is True


def test_whitespace_value_maps_to_missing(canonical_mapper: CanonicalMapper):
    """Yalnız boşluk içeren değer missing olur."""
    result = canonical_mapper.map("source", "   ")
    assert result.canonical_code == "missing"
    assert result.is_missing is True


def test_unknown_value_maps_to_unknown(canonical_mapper: CanonicalMapper):
    """Bilinmeyen değer unknown olur."""
    result = canonical_mapper.map("source", "New Unknown Channel")
    assert result.canonical_code == "unknown"
    assert result.is_known is False
    assert result.is_missing is False


def test_mapping_does_not_auto_generate(canonical_mapper: CanonicalMapper):
    """Mapping runtime'da otomatik yeni kod üretmez."""
    result = canonical_mapper.map("source", "Some Brand New Value")
    assert result.canonical_code == "unknown"


def test_mapping_version_present(canonical_mapper: CanonicalMapper):
    """Mapping sürümü kayıtlı."""
    assert canonical_mapper.version == "1.0.0"


def test_type_column_mapping(canonical_mapper: CanonicalMapper):
    """Type kolonu mapping çalışıyor."""
    result = canonical_mapper.map("type", "Abandoned Bicycle")
    assert result.canonical_code == "abandoned_bicycle"
    assert result.is_known is True


def test_neighborhood_mapping(canonical_mapper: CanonicalMapper):
    """Neighborhood kolonu mapping çalışıyor."""
    result = canonical_mapper.map("neighborhood", "Roxbury")
    assert result.canonical_code == "roxbury"
    assert result.is_known is True
