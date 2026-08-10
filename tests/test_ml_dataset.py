"""Dataset builder, split, fingerprint ve DB entegrasyon testleri."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.models.process import Process, ProcessSnapshot
from ml.datasets.dataset_builder import (
    SPLIT_BOUNDARIES,
    build_dataset,
)
from ml.datasets.fingerprint import compute_dataset_fingerprint
from ml.features.schema_loader import load_feature_schema


def _create_process(
    session,
    external_id,
    created_at,
    deadline=None,
    completed_at=None,
    source="src_a",
    subject="subj_a",
    reason="reason_a",
    process_type="type_a",
    neighborhood="nbh_a",
    feature_schema_version="opening-v1",
):
    process = Process(
        external_id=external_id,
        created_at=created_at,
        deadline=deadline,
        completed_at=completed_at,
        current_status="Closed" if completed_at else "Open",
        source_payload_json=json.dumps({"department": "pwdx", "queue": "q1"}),
        imported_at=datetime.now(UTC),
    )
    session.add(process)
    session.flush()

    input_json = json.dumps({
        "external_id": external_id,
        "created_at": created_at.isoformat(),
        "deadline": deadline.isoformat() if deadline else None,
        "source": source,
        "subject": subject,
        "reason": reason,
        "type": process_type,
        "neighborhood": neighborhood,
    })

    snapshot = ProcessSnapshot(
        process_id=process.id,
        snapshot_type="opening",
        snapshot_at=created_at,
        feature_schema_version=feature_schema_version,
        input_json=input_json,
        input_fingerprint="test_fp",
    )
    session.add(snapshot)
    session.flush()
    return process


class TestDatasetSource:
    def test_features_from_snapshot_not_payload(self, db_session):
        _create_process(
            db_session, "ext_1",
            created_at=datetime(2024, 3, 1, 10, 0),
            deadline=datetime(2024, 4, 1, 10, 0),
            completed_at=datetime(2024, 3, 15, 10, 0),
        )
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)

        X = result.classification.train.X
        assert "department" not in X.columns
        assert "queue" not in X.columns

    def test_no_outcome_leakage_in_features(self, db_session):
        _create_process(
            db_session, "ext_1",
            created_at=datetime(2024, 3, 1, 10, 0),
            deadline=datetime(2024, 4, 1, 10, 0),
            completed_at=datetime(2024, 3, 15, 10, 0),
        )
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)

        X = result.classification.train.X
        assert "completed_at" not in X.columns
        assert "current_status" not in X.columns
        assert "case_status" not in X.columns
        assert "on_time" not in X.columns
        assert "closure_reason" not in X.columns

    def test_exactly_10_whitelist_features(self, db_session):
        _create_process(
            db_session, "ext_1",
            created_at=datetime(2024, 3, 1, 10, 0),
            deadline=datetime(2024, 4, 1, 10, 0),
            completed_at=datetime(2024, 3, 15, 10, 0),
        )
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)

        X = result.classification.train.X
        assert set(X.columns) == set(schema.all_features)
        assert len(X.columns) == 10


class TestClassificationCohort:
    def test_closed_before_sla_included_as_0(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2024, 4, 1),
            completed_at=datetime(2024, 3, 15))
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification.train.row_count == 1
        assert result.classification.train.y[0] == 0

    def test_closed_after_sla_included_as_1(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2024, 4, 1),
            completed_at=datetime(2024, 5, 1))
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification.train.y[0] == 1

    def test_open_before_cutoff_included_as_1(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2024, 6, 1),
            completed_at=None)
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification.train.y[0] == 1

    def test_open_after_cutoff_excluded(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2025, 6, 1),
            completed_at=None)
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification_excluded == 1
        assert result.classification.train.row_count == 0
        assert result.classification.validation.row_count == 0

    def test_sla_missing_excluded_from_classification(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=None,
            completed_at=datetime(2024, 3, 15))
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification_excluded == 1


class TestRegressionCohort:
    def test_closed_included(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2024, 4, 1),
            completed_at=datetime(2024, 3, 2))
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.regression.train.row_count == 1
        assert result.regression.train.y[0] == pytest.approx(24.0)

    def test_open_excluded_from_regression(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=datetime(2024, 4, 1),
            completed_at=None)
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.regression_excluded == 1

    def test_sla_missing_but_closed_included_in_regression(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 3, 1), deadline=None,
            completed_at=datetime(2024, 3, 5))
        schema = load_feature_schema()
        result = build_dataset(db_session, schema)
        assert result.classification_excluded == 1
        assert result.regression.train.row_count == 1


class TestSplitBoundaries:
    def test_january_goes_to_train(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 1, 1, 0, 0),
            deadline=datetime(2024, 2, 1), completed_at=datetime(2024, 1, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.train.row_count == 1

    def test_august_end_goes_to_train(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 8, 31, 23, 59),
            deadline=datetime(2024, 9, 30), completed_at=datetime(2024, 9, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.train.row_count == 1

    def test_september_goes_to_validation(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 9, 1, 0, 0),
            deadline=datetime(2024, 10, 1), completed_at=datetime(2024, 9, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.validation.row_count == 1

    def test_october_goes_to_test_sealed(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 10, 1, 0, 0),
            deadline=datetime(2024, 11, 1), completed_at=datetime(2024, 10, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.test_info.row_count == 1
        assert result.classification.train.row_count == 0

    def test_december_goes_to_audit_sealed(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2024, 12, 1, 0, 0),
            deadline=datetime(2025, 1, 1), completed_at=datetime(2024, 12, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.audit_info.row_count == 1

    def test_january_2025_unassigned(self, db_session):
        _create_process(db_session, "ext_1",
            created_at=datetime(2025, 1, 1, 0, 0),
            deadline=datetime(2025, 2, 1), completed_at=datetime(2025, 1, 15))
        result = build_dataset(db_session, load_feature_schema())
        assert result.unassigned_split == 1
        assert result.classification.train.row_count == 0

    def test_all_four_splits_populated(self, db_session):
        _create_process(db_session, "t1", datetime(2024, 3, 1),
            datetime(2024, 4, 1), datetime(2024, 3, 15))
        _create_process(db_session, "v1", datetime(2024, 9, 5),
            datetime(2024, 10, 5), datetime(2024, 9, 20))
        _create_process(db_session, "te1", datetime(2024, 10, 5),
            datetime(2024, 11, 5), datetime(2024, 10, 20))
        _create_process(db_session, "a1", datetime(2024, 12, 5),
            datetime(2025, 1, 5), datetime(2024, 12, 20))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.train.row_count == 1
        assert result.classification.validation.row_count == 1
        assert result.classification.test_info.row_count == 1
        assert result.classification.audit_info.row_count == 1


class TestSchemaVersionMismatch:
    def test_wrong_schema_version_excluded(self, db_session):
        _create_process(db_session, "correct",
            datetime(2024, 3, 1), datetime(2024, 4, 1), datetime(2024, 3, 15),
            feature_schema_version="opening-v1")
        _create_process(db_session, "wrong",
            datetime(2024, 3, 1), datetime(2024, 4, 1), datetime(2024, 3, 15),
            feature_schema_version="opening_v1")
        result = build_dataset(db_session, load_feature_schema())
        assert result.total_snapshots == 1


class TestSealedSplitInfo:
    def test_test_info_has_count_but_no_data(self, db_session):
        _create_process(db_session, "te1", datetime(2024, 10, 5),
            datetime(2024, 11, 5), datetime(2024, 10, 20))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.test_info.row_count == 1
        assert result.classification.test_info.date_range is not None

    def test_audit_info_has_count_but_no_data(self, db_session):
        _create_process(db_session, "a1", datetime(2024, 12, 5),
            datetime(2025, 1, 5), datetime(2024, 12, 20))
        result = build_dataset(db_session, load_feature_schema())
        assert result.classification.audit_info.row_count == 1
        assert result.classification.audit_info.date_range is not None


class TestFingerprint:
    def test_same_data_same_fingerprint(self, db_session):
        _create_process(db_session, "ext_1", datetime(2024, 3, 1),
            datetime(2024, 4, 1), datetime(2024, 3, 15))
        r1 = build_dataset(db_session, load_feature_schema())
        r2 = build_dataset(db_session, load_feature_schema())
        assert r1.classification.fingerprint == r2.classification.fingerprint

    def test_feature_change_different_fingerprint(self, db_session):
        _create_process(db_session, "ext_1", datetime(2024, 3, 1),
            datetime(2024, 4, 1), datetime(2024, 3, 15), source="src_a")
        r1 = build_dataset(db_session, load_feature_schema())

        db_session.query(ProcessSnapshot).delete()
        db_session.query(Process).delete()
        db_session.flush()

        _create_process(db_session, "ext_1", datetime(2024, 3, 1),
            datetime(2024, 4, 1), datetime(2024, 3, 15), source="src_b")
        r2 = build_dataset(db_session, load_feature_schema())
        assert r1.classification.fingerprint != r2.classification.fingerprint

    def test_cutoff_change_different_fingerprint(self, db_session):
        from ml.datasets.target_builder import OBSERVATION_CUTOFF
        _create_process(db_session, "ext_1", datetime(2024, 3, 1),
            datetime(2024, 6, 1), completed_at=None)
        schema = load_feature_schema()
        r1 = build_dataset(db_session, schema, observation_cutoff=OBSERVATION_CUTOFF)
        r2 = build_dataset(db_session, schema, observation_cutoff=datetime(2024, 7, 1))
        assert r1.classification.fingerprint != r2.classification.fingerprint

    def test_fingerprint_function_deterministic(self):
        X = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]}, index=["x", "y"])
        y = np.array([0, 1])
        meta = {
            "feature_schema_version": "opening-v1",
            "canonical_mapping_version": "1.0.0",
            "observation_cutoff": datetime(2025, 1, 13),
        }
        fp1 = compute_dataset_fingerprint(X, y, meta)
        fp2 = compute_dataset_fingerprint(X, y, meta)
        assert fp1 == fp2

    def test_fingerprint_order_independent(self):
        meta = {
            "feature_schema_version": "opening-v1",
            "canonical_mapping_version": "1.0.0",
            "observation_cutoff": datetime(2025, 1, 13),
        }
        X1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        y1 = np.array([0, 1, 0])
        X2 = pd.DataFrame({"a": [3, 1, 2]}, index=["z", "x", "y"])
        y2 = np.array([0, 0, 1])
        fp1 = compute_dataset_fingerprint(X1, y1, meta)
        fp2 = compute_dataset_fingerprint(X2, y2, meta)
        assert fp1 == fp2

    def test_fingerprint_mapping_version_sensitive(self):
        X = pd.DataFrame({"a": [1]}, index=["x"])
        y = np.array([0])
        base = {
            "feature_schema_version": "opening-v1",
            "observation_cutoff": datetime(2025, 1, 13),
        }
        fp1 = compute_dataset_fingerprint(X, y, {**base, "canonical_mapping_version": "1.0.0"})
        fp2 = compute_dataset_fingerprint(X, y, {**base, "canonical_mapping_version": "1.1.0"})
        assert fp1 != fp2
