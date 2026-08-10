"""Feature schema yükleme ve dogrulama.

Sabit kaynak: ml/config/feature_schema_opening_v1.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "config" / "feature_schema_opening_v1.json"


@dataclass(frozen=True)
class FeatureSchema:
    feature_schema_version: str
    canonical_mapping_version: str
    snapshot_type: str
    categorical_features: list[str]
    numeric_features: list[str]
    feature_derivations: dict[str, Any]
    excluded_features: dict[str, str]

    @property
    def all_features(self) -> list[str]:
        return list(self.categorical_features) + list(self.numeric_features)


class SchemaValidationError(Exception):
    """Snapshot input_json feature schema'ya uymadiginda firlatilir."""


def load_feature_schema(path: Path | None = None) -> FeatureSchema:
    """Feature schema JSON dosyasini yükler."""
    resolved_path = path or DEFAULT_SCHEMA_PATH
    with open(resolved_path, encoding="utf-8") as f:
        data = json.load(f)
    return FeatureSchema(
        feature_schema_version=data["feature_schema_version"],
        canonical_mapping_version=data["canonical_mapping_version"],
        snapshot_type=data["snapshot_type"],
        categorical_features=list(data["categorical_features"]),
        numeric_features=list(data["numeric_features"]),
        feature_derivations=data.get("feature_derivations", {}),
        excluded_features=data.get("excluded_features", {}),
    )


def validate_snapshot_input(
    input_json: dict[str, Any],
    schema: FeatureSchema,
) -> None:
    """Snapshot input_json'in feature türetimi için gereken kaynak alanlari içerdigini dogrular.

    Feature DEGERI bos ('missing') ise geçerlidir.
    Feature ANAHTARI yok ise schema hatasidir ve sessiz imputasyonla gizlenmez.
    """
    required_keys = ["created_at", "deadline"] + schema.categorical_features
    missing_keys = [k for k in required_keys if k not in input_json]
    if missing_keys:
        raise SchemaValidationError(
            f"Snapshot input_json eksik feature anahtarlari: {missing_keys}"
        )
