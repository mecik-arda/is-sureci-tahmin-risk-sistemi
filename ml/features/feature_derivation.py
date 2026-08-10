"""Opening snapshot'tan 10 V1 feature türetme.

Tek feature türetim kaynagi: process_snapshots.input_json
Ikinci bir bagimsiz feature listesi olusturulmaz.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ml.features.schema_loader import FeatureSchema

CATEGORICAL_FEATURES = ["source", "subject", "reason", "type", "neighborhood"]
NUMERIC_FEATURES = ["open_month", "open_weekday", "open_hour", "is_weekend", "sla_duration_hours"]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def derive_features(
    input_json: dict[str, Any],
    schema: FeatureSchema | None = None,
) -> dict[str, Any]:
    """Opening snapshot input_json'den 10 V1 feature türetir.

    Kategorik degerler dogrudan input_json'dan okunur (kanonik kod).
    Sayisal degerler created_at ve deadline'dan türetilir.

    Negatif sla_duration_hours NaN olarak isaretlenir (veri hatasi).
    """
    created_at_str = input_json.get("created_at")
    if not created_at_str:
        raise ValueError("Snapshot input_json 'created_at' alani eksik veya bos")

    created_at = _parse_iso_datetime(created_at_str)
    if created_at is None:
        raise ValueError(f"created_at parse edilemedi: {created_at_str}")

    features: dict[str, Any] = {}

    for col in CATEGORICAL_FEATURES:
        features[col] = input_json.get(col, "missing")

    features["open_month"] = created_at.month
    features["open_weekday"] = created_at.weekday()
    features["open_hour"] = created_at.hour
    features["is_weekend"] = int(created_at.weekday() >= 5)

    deadline_str = input_json.get("deadline")
    deadline = _parse_iso_datetime(deadline_str) if deadline_str else None
    if deadline is not None:
        sla_hours = (deadline - created_at).total_seconds() / 3600
        features["sla_duration_hours"] = float(sla_hours) if sla_hours >= 0 else np.nan
    else:
        features["sla_duration_hours"] = np.nan

    return features


def validate_feature_names(schema: FeatureSchema) -> None:
    """Feature schema ile bu modülün sabit listelerinin uyumlu oldugunu dogrular."""
    expected_categorical = schema.categorical_features
    expected_numeric = schema.numeric_features
    if expected_categorical != CATEGORICAL_FEATURES:
        raise ValueError(
            f"Schema kategorik feature uyumsuzlugu: schema={expected_categorical}, "
            f"modül={CATEGORICAL_FEATURES}"
        )
    if expected_numeric != NUMERIC_FEATURES:
        raise ValueError(
            f"Schema sayisal feature uyumsuzlugu: schema={expected_numeric}, "
            f"modül={NUMERIC_FEATURES}"
        )
