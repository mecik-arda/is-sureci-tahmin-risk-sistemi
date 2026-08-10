"""Deterministik dataset fingerprint hesaplama.

Ayni DB durumu ve konfigürasyon her zaman ayni fingerprint'i üretir.
Satir siralamasindan etkilenmez.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def compute_dataset_fingerprint(
    features: pd.DataFrame,
    targets: np.ndarray | pd.Series,
    metadata: dict[str, Any],
) -> str:
    """Dataset için deterministik SHA256 fingerprint üretir.

    Fingerprint sölardan etkilenir:
        - Feature degerleri (satir sirasindan bagimsiz)
        - Target degerleri
        - feature_schema_version
        - canonical_mapping_version
        - observation_cutoff
    """
    sorted_indices = features.index.argsort()
    features_sorted = features.iloc[sorted_indices]
    targets_sorted = np.asarray(targets, dtype=np.float64)[sorted_indices]

    features_hash = hashlib.sha256(
        pd.util.hash_pandas_object(features_sorted, index=True).values.tobytes()
    ).hexdigest()

    targets_hash = hashlib.sha256(targets_sorted.tobytes()).hexdigest()

    cutoff = metadata.get("observation_cutoff")
    cutoff_str = cutoff.isoformat() if isinstance(cutoff, datetime) else str(cutoff)

    content = json.dumps({
        "features_hash": features_hash,
        "targets_hash": targets_hash,
        "feature_schema_version": metadata.get("feature_schema_version"),
        "canonical_mapping_version": metadata.get("canonical_mapping_version"),
        "observation_cutoff": cutoff_str,
        "row_count": int(len(features_sorted)),
        "feature_names": list(features_sorted.columns),
    }, sort_keys=True)

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
