"""Leakage-safe egitim dataseti olusturma.

Mimari Sözlesme (S16):
    Feature kaynagi: process_snapshots.input_json (immutable opening snapshot)
    Target kaynagi: processes outcome alanlari + observation_cutoff

CSV ve source_payload_json feature kaynagi olarak kullanilmaz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.process import Process, ProcessSnapshot
from ml.datasets.fingerprint import compute_dataset_fingerprint
from ml.datasets.target_builder import (
    OBSERVATION_CUTOFF,
    compute_is_delayed,
    compute_total_duration_hours,
)
from ml.features.feature_derivation import derive_features
from ml.features.schema_loader import (
    FeatureSchema,
    SchemaValidationError,
    validate_snapshot_input,
)

SPLIT_BOUNDARIES: dict[str, tuple[datetime, datetime]] = {
    "train": (datetime(2024, 1, 1), datetime(2024, 9, 1)),
    "validation": (datetime(2024, 9, 1), datetime(2024, 10, 1)),
    "test": (datetime(2024, 10, 1), datetime(2024, 12, 1)),
    "audit": (datetime(2024, 12, 1), datetime(2025, 1, 1)),
}


@dataclass(frozen=True)
class SplitData:
    X: pd.DataFrame
    y: np.ndarray
    external_ids: list[str]

    @property
    def row_count(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class SealedSplitInfo:
    row_count: int
    date_range: tuple[str, str] | None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_version: str
    feature_schema_version: str
    canonical_mapping_version: str
    observation_cutoff: str
    source_snapshot_type: str
    split_counts: dict[str, int]
    feature_names: list[str]
    target_policy_version: str
    created_at: str


@dataclass(frozen=True)
class ClassificationDataset:
    train: SplitData
    validation: SplitData
    test_info: SealedSplitInfo
    audit_info: SealedSplitInfo
    metadata: DatasetMetadata
    fingerprint: str


@dataclass(frozen=True)
class RegressionDataset:
    train: SplitData
    validation: SplitData
    test_info: SealedSplitInfo
    audit_info: SealedSplitInfo
    metadata: DatasetMetadata
    fingerprint: str


@dataclass
class DatasetBuildResult:
    classification: ClassificationDataset
    regression: RegressionDataset
    total_snapshots: int
    classification_excluded: int
    regression_excluded: int
    unassigned_split: int
    schema_errors: list[str]


def _assign_split(created_at: datetime) -> str | None:
    for split_name, (start, end) in SPLIT_BOUNDARIES.items():
        if start <= created_at < end:
            return split_name
    return None


def _build_sealed_info(
    created_ats: list[datetime],
    masks: np.ndarray,
) -> SealedSplitInfo:
    masked_dates = [created_ats[i] for i in range(len(created_ats)) if masks[i]]
    if not masked_dates:
        return SealedSplitInfo(row_count=0, date_range=None)
    return SealedSplitInfo(
        row_count=int(masks.sum()),
        date_range=(min(masked_dates).isoformat(), max(masked_dates).isoformat()),
    )


def _build_dataset_object(
    rows: list[dict[str, Any]],
    target_key: str,
    schema: FeatureSchema,
    observation_cutoff: datetime,
    dataset_version: str,
    target_policy_version: str,
    dataset_cls: type,
) -> Any:
    valid_rows = [
        r for r in rows
        if r[target_key] is not None and r["split"] is not None
    ]

    features_list = [r["features"] for r in valid_rows]
    external_ids = [str(r["external_id"]) for r in valid_rows]
    targets = [r[target_key] for r in valid_rows]
    splits = [r["split"] for r in valid_rows]
    created_ats = [r["created_at"] for r in valid_rows]

    if features_list:
        features_df = pd.DataFrame(features_list, index=external_ids)
    else:
        features_df = pd.DataFrame(columns=schema.all_features)
    features_df.index.name = "external_id"

    dtype = np.int64 if target_key == "is_delayed" else np.float64
    target_array = np.array(targets, dtype=dtype)

    train_mask = np.array([s == "train" for s in splits], dtype=bool)
    val_mask = np.array([s == "validation" for s in splits], dtype=bool)
    test_mask = np.array([s == "test" for s in splits], dtype=bool)
    audit_mask = np.array([s == "audit" for s in splits], dtype=bool)

    train = SplitData(
        X=features_df[train_mask],
        y=target_array[train_mask],
        external_ids=features_df[train_mask].index.tolist(),
    )
    validation = SplitData(
        X=features_df[val_mask],
        y=target_array[val_mask],
        external_ids=features_df[val_mask].index.tolist(),
    )

    test_info = _build_sealed_info(created_ats, test_mask)
    audit_info = _build_sealed_info(created_ats, audit_mask)

    meta_for_fp = {
        "feature_schema_version": schema.feature_schema_version,
        "canonical_mapping_version": schema.canonical_mapping_version,
        "observation_cutoff": observation_cutoff,
    }
    fingerprint = compute_dataset_fingerprint(features_df, target_array, meta_for_fp)

    metadata = DatasetMetadata(
        dataset_version=dataset_version,
        feature_schema_version=schema.feature_schema_version,
        canonical_mapping_version=schema.canonical_mapping_version,
        observation_cutoff=str(observation_cutoff),
        source_snapshot_type=schema.snapshot_type,
        split_counts={
            "train": train.row_count,
            "validation": validation.row_count,
            "test": test_info.row_count,
            "audit": audit_info.row_count,
        },
        feature_names=schema.all_features,
        target_policy_version=target_policy_version,
        created_at=datetime.now(UTC).isoformat(),
    )

    return dataset_cls(
        train=train,
        validation=validation,
        test_info=test_info,
        audit_info=audit_info,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def build_dataset(
    session: Session,
    schema: FeatureSchema,
    observation_cutoff: datetime = OBSERVATION_CUTOFF,
) -> DatasetBuildResult:
    """Veritabanindan leakage-safe egitim dataseti olusturur.

    Feature'lar process_snapshots.input_json'den (immutable opening snapshot),
    target'lar processes outcome alanlarindan türetilir.
    """
    stmt = (
        select(ProcessSnapshot, Process)
        .join(Process, ProcessSnapshot.process_id == Process.id)
        .where(ProcessSnapshot.snapshot_type == schema.snapshot_type)
        .where(ProcessSnapshot.feature_schema_version == schema.feature_schema_version)
    )
    results = session.execute(stmt).all()

    raw_rows: list[dict[str, Any]] = []
    schema_errors: list[str] = []

    for snapshot, process in results:
        try:
            input_json = json.loads(snapshot.input_json)
            validate_snapshot_input(input_json, schema)
            features = derive_features(input_json)
        except (json.JSONDecodeError, SchemaValidationError, ValueError) as e:
            schema_errors.append(f"external_id={process.external_id}: {e}")
            continue

        cls_target = compute_is_delayed(
            process.completed_at, process.deadline, observation_cutoff
        )
        reg_target = compute_total_duration_hours(
            process.created_at, process.completed_at
        )
        split = _assign_split(process.created_at)

        raw_rows.append({
            "external_id": process.external_id,
            "features": features,
            "created_at": process.created_at,
            "is_delayed": cls_target,
            "total_duration_hours": reg_target,
            "split": split,
        })

    total_snapshots = len(results)
    classification_excluded = sum(1 for r in raw_rows if r["is_delayed"] is None)
    regression_excluded = sum(1 for r in raw_rows if r["total_duration_hours"] is None)
    unassigned_split = sum(1 for r in raw_rows if r["split"] is None)

    cls_dataset = _build_dataset_object(
        raw_rows, "is_delayed", schema, observation_cutoff,
        "classification-v1", "is_delayed-v1", ClassificationDataset,
    )
    reg_dataset = _build_dataset_object(
        raw_rows, "total_duration_hours", schema, observation_cutoff,
        "regression-v1", "total_duration_hours-v1", RegressionDataset,
    )

    return DatasetBuildResult(
        classification=cls_dataset,
        regression=reg_dataset,
        total_snapshots=total_snapshots,
        classification_excluded=classification_excluded,
        regression_excluded=regression_excluded,
        unassigned_split=unassigned_split,
        schema_errors=schema_errors,
    )
