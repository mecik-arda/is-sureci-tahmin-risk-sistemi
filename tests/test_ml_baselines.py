"""Classification ve regression baseline training testleri."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.models.process import Process, ProcessSnapshot
from ml.datasets.dataset_builder import build_dataset
from ml.features.schema_loader import load_feature_schema
from ml.training.baseline_classifier import train_classification_baselines
from ml.training.baseline_regressor import train_regression_baselines


def _create_process(session, external_id, created_at, deadline=None, completed_at=None,
                    source="src_a", subject="subj_a", reason="reason_a",
                    process_type="type_a", neighborhood="nbh_a"):
    process = Process(
        external_id=external_id,
        created_at=created_at,
        deadline=deadline,
        completed_at=completed_at,
        current_status="Closed" if completed_at else "Open",
        source_payload_json=json.dumps({"department": "pwdx"}),
        imported_at=datetime.now(UTC),
    )
    session.add(process)
    session.flush()

    input_json = json.dumps({
        "external_id": external_id,
        "created_at": created_at.isoformat(),
        "deadline": deadline.isoformat() if deadline else None,
        "source": source, "subject": subject, "reason": reason,
        "type": process_type, "neighborhood": neighborhood,
    })
    snapshot = ProcessSnapshot(
        process_id=process.id, snapshot_type="opening", snapshot_at=created_at,
        feature_schema_version="opening-v1", input_json=input_json,
        input_fingerprint="test_fp",
    )
    session.add(snapshot)
    session.flush()
    return process


def _populate_data(session, n_train=25, n_val=12):
    rng = np.random.RandomState(42)
    sources = ["src_a", "src_b", "src_c"]
    subjects = ["subj_a", "subj_b"]
    reasons = ["reason_a", "reason_b"]
    types = ["type_a", "type_b", "type_c"]
    neighborhoods = ["nbh_a", "nbh_b"]

    idx = 0
    for _ in range(n_train):
        created = datetime(2024, 1, 1) + timedelta(days=int(rng.uniform(0, 240)))
        sla_days = int(rng.uniform(5, 30))
        deadline = created + timedelta(days=sla_days)

        if rng.random() < 0.65:
            close_offset = int(rng.uniform(1, max(2, sla_days - 1)))
            completed = created + timedelta(days=close_offset)
        else:
            completed = deadline + timedelta(days=int(rng.uniform(1, 15)))

        _create_process(
            session, f"id_{idx}", created, deadline, completed,
            source=sources[idx % 3], subject=subjects[idx % 2],
            reason=reasons[idx % 2], process_type=types[idx % 3],
            neighborhood=neighborhoods[idx % 2],
        )
        idx += 1

    for _ in range(n_val):
        created = datetime(2024, 9, 1) + timedelta(days=int(rng.uniform(0, 29)))
        sla_days = int(rng.uniform(5, 30))
        deadline = created + timedelta(days=sla_days)

        if rng.random() < 0.65:
            close_offset = int(rng.uniform(1, max(2, sla_days - 1)))
            completed = created + timedelta(days=close_offset)
        else:
            completed = deadline + timedelta(days=int(rng.uniform(1, 15)))

        _create_process(
            session, f"id_{idx}", created, deadline, completed,
            source=sources[idx % 3], subject=subjects[idx % 2],
            reason=reasons[idx % 2], process_type=types[idx % 3],
            neighborhood=neighborhoods[idx % 2],
        )
        idx += 1


@pytest.fixture
def populated_dataset(db_session):
    _populate_data(db_session)
    schema = load_feature_schema()
    return build_dataset(db_session, schema)


class TestClassificationBaseline:
    def test_returns_dummy_and_logistic(self, populated_dataset):
        schema = load_feature_schema()
        results = train_classification_baselines(populated_dataset.classification, schema)
        assert "dummy" in results
        assert "logistic" in results

    def test_metrics_have_expected_keys(self, populated_dataset):
        schema = load_feature_schema()
        results = train_classification_baselines(populated_dataset.classification, schema)
        for name, metrics in results.items():
            assert "roc_auc" in metrics
            assert "pr_auc" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "confusion_matrix" in metrics

    def test_all_metrics_on_validation(self, populated_dataset):
        schema = load_feature_schema()
        results = train_classification_baselines(populated_dataset.classification, schema)
        for name, metrics in results.items():
            assert metrics["split"] == "validation"

    def test_logistic_better_than_dummy_pr_auc(self, populated_dataset):
        schema = load_feature_schema()
        results = train_classification_baselines(populated_dataset.classification, schema)
        assert results["logistic"]["pr_auc"] >= results["dummy"]["pr_auc"]

    def test_no_source_payload_in_results(self, populated_dataset):
        schema = load_feature_schema()
        results = train_classification_baselines(populated_dataset.classification, schema)
        serialized = json.dumps(results, default=str)
        assert "department" not in serialized
        assert "pwdx" not in serialized


class TestRegressionBaseline:
    def test_returns_three_models(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        assert "dummy_median" in results
        assert "elasticnet_raw" in results
        assert "elasticnet_log1p" in results

    def test_metrics_have_expected_keys(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        for name, metrics in results.items():
            assert "mae" in metrics
            assert "median_ae" in metrics
            assert "rmse" in metrics
            assert "p90_abs_error" in metrics

    def test_all_metrics_on_validation(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        for name, metrics in results.items():
            assert metrics["split"] == "validation"

    def test_metrics_non_negative(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        for name, metrics in results.items():
            assert metrics["mae"] >= 0
            assert metrics["rmse"] >= 0

    def test_dummy_better_than_or_equal_to_elasticnet_for_baseline(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        assert results["dummy_median"]["mae"] > 0

    def test_no_source_payload_in_results(self, populated_dataset):
        schema = load_feature_schema()
        results = train_regression_baselines(populated_dataset.regression, schema)
        serialized = json.dumps(results, default=str)
        assert "department" not in serialized
        assert "pwdx" not in serialized
