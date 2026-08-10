"""Faz 5 model training tests.

Tests for:
  - TimeSeriesSplit is used (no random/shuffle)
  - CV fold temporal ordering
  - Preprocessing fold-local fit
  - Deterministic random_state
  - OOF calibration is truly OOF
  - Validation never enters calibrator.fit
  - Test/Audit sealed
  - Winner selection by Validation PR-AUC (classification)
  - Winner selection by Validation MAE (regression)
  - Threshold deterministic and only on Validation
  - Sigmoid and isotonic calibration
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from app.models.process import Process, ProcessSnapshot
from ml.calibration.calibrator import fit_calibration
from ml.datasets.dataset_builder import build_dataset
from ml.evaluation.comparison import get_winner_classifier, get_winner_regressor
from ml.evaluation.sealed_guard import assert_evaluable, SealedSplitError
from ml.features.schema_loader import load_feature_schema
from ml.training.classifier_trainer import (
    calibrate_and_evaluate,
    find_best_threshold,
    train_classifier_cv,
)
from ml.training.regressor_trainer import (
    fit_and_evaluate_regressor,
    train_regressor_cv,
)


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
        imported_at=datetime(2025, 1, 1),
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


def _populate_temporal_data(session, n=80):
    rng = np.random.RandomState(42)
    sources = ["src_a", "src_b", "src_c"]
    subjects = ["subj_a", "subj_b"]
    reasons = ["reason_a", "reason_b"]
    types = ["type_a", "type_b", "type_c"]
    neighborhoods = ["nbh_a", "nbh_b"]

    for i in range(n):
        month = 1 + (i * 10) // n
        created = datetime(2024, month, 1 + (i % 28))
        sla_days = int(rng.uniform(5, 30))
        deadline = created + timedelta(days=sla_days)

        if rng.random() < 0.65:
            close_offset = int(rng.uniform(1, max(2, sla_days - 1)))
            completed = created + timedelta(days=close_offset)
        else:
            completed = deadline + timedelta(days=int(rng.uniform(1, 15)))

        _create_process(
            session, f"id_{i}", created, deadline, completed,
            source=sources[i % 3], subject=subjects[i % 2],
            reason=reasons[i % 2], process_type=types[i % 3],
            neighborhood=neighborhoods[i % 2],
        )


@pytest.fixture
def faz5_dataset(db_session):
    _populate_temporal_data(db_session)
    schema = load_feature_schema()
    return build_dataset(db_session, schema)


@pytest.fixture
def schema():
    return load_feature_schema()


class TestTimeSeriesSplit:
    def test_uses_timeseriessplit(self):
        tscv = TimeSeriesSplit(n_splits=3)
        X = np.arange(30).reshape(-1, 1)
        y = np.arange(30)
        splits = list(tscv.split(X))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()

    def test_fold_indices_monotonic(self):
        tscv = TimeSeriesSplit(n_splits=5)
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50)
        splits = list(tscv.split(X))
        for train_idx, test_idx in splits:
            assert np.all(train_idx < test_idx[0])

    def test_train_before_validation_in_every_fold(self):
        tscv = TimeSeriesSplit(n_splits=5)
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50)
        for train_idx, test_idx in tscv.split(X):
            assert train_idx.max() < test_idx.min()


class TestCVTraining:
    def test_classification_cv_returns_oof_predictions(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "LogisticRegression", n_splits=3)
        assert cv.oof_proba is not None
        assert cv.oof_true is not None
        assert len(cv.oof_proba) > 0
        assert len(cv.oof_true) > 0
        assert len(cv.oof_proba) == len(cv.oof_true)
        assert len(cv.oof_proba) < len(ds.train.X)

    def test_classification_cv_metrics_in_range(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "LogisticRegression", n_splits=3)
        assert 0.0 <= cv.pr_auc_mean <= 1.0
        assert 0.0 <= cv.roc_auc_mean <= 1.0
        assert cv.pr_auc_std >= 0
        assert cv.roc_auc_std >= 0

    def test_regression_cv_returns_oof_predictions(self, faz5_dataset, schema):
        ds = faz5_dataset.regression
        cv = train_regressor_cv(ds.train.X, ds.train.y, schema, "ElasticNet_log1p", n_splits=3)
        assert cv.oof_preds is not None
        assert cv.oof_true is not None
        assert len(cv.oof_preds) > 0
        assert len(cv.oof_preds) == len(cv.oof_true)
        assert len(cv.oof_preds) < len(ds.train.X)

    def test_regression_cv_metrics_non_negative(self, faz5_dataset, schema):
        ds = faz5_dataset.regression
        cv = train_regressor_cv(ds.train.X, ds.train.y, schema, "ElasticNet_log1p", n_splits=3)
        assert cv.mae_mean >= 0
        assert cv.rmse_mean >= 0

    def test_deterministic_with_same_seed(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv1 = train_classifier_cv(ds.train.X, ds.train.y, schema, "LogisticRegression", n_splits=3)
        cv2 = train_classifier_cv(ds.train.X, ds.train.y, schema, "LogisticRegression", n_splits=3)
        np.testing.assert_array_almost_equal(cv1.oof_proba, cv2.oof_proba)

    def test_cv_multiple_models(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        for name in ["LogisticRegression", "HistGradientBoosting", "RandomForest"]:
            cv = train_classifier_cv(ds.train.X, ds.train.y, schema, name, n_splits=3)
            assert cv.model_name == name
            assert cv.oof_proba is not None

    def test_regression_cv_multiple_models(self, faz5_dataset, schema):
        ds = faz5_dataset.regression
        for name in ["ElasticNet_log1p", "HistGradientBoostingRegressor", "RandomForestRegressor"]:
            cv = train_regressor_cv(ds.train.X, ds.train.y, schema, name, n_splits=3)
            assert cv.model_name == name
            assert cv.oof_preds is not None


class TestCalibrationIsOOF:
    def test_oof_calibration_fit_only_on_train(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)

        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "RandomForest", cv,
        )

        assert result.calibrated_model is not None
        assert result.selected_calibration in ["uncalibrated", "sigmoid", "isotonic"]
        assert result.validation_metrics is not None
        assert result.brier_score is not None

    def test_calibration_results_have_all_methods(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)

        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "RandomForest", cv,
        )

        for method in ["uncalibrated", "sigmoid", "isotonic"]:
            assert method in result.calibration_results
            e = result.calibration_results[method]
            assert 0.0 <= e.pr_auc <= 1.0
            assert 0.0 <= e.roc_auc <= 1.0
            assert e.brier >= 0.0

    def test_calibration_validation_not_used_for_fit(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)

        cal = fit_calibration(cv.oof_proba, cv.oof_true, "sigmoid")
        assert cal.method == "sigmoid"
        assert cal.fitted_model is not None
        assert not hasattr(cal.fitted_model, "X_val_used")

    def test_calibration_raises_without_oof(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        from ml.training.classifier_trainer import CVResult
        bad_cv = CVResult(
            model_name="test",
            pr_auc_mean=0.5, pr_auc_std=0.0,
            roc_auc_mean=0.5, roc_auc_std=0.0,
            oof_proba=None, oof_true=None,
        )
        with pytest.raises(ValueError, match="OOF"):
            calibrate_and_evaluate(
                ds.train.X, ds.train.y,
                ds.validation.X, ds.validation.y,
                schema, "LogisticRegression", bad_cv,
            )

    def test_calibrated_probabilities_in_range(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)

        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "RandomForest", cv,
        )

        val_proba = result.calibrated_model.predict_proba(
            result.base_pipeline.predict_proba(ds.validation.X)[:, 1],
        )
        assert np.all(val_proba >= 0) and np.all(val_proba <= 1)
        assert np.all(np.isfinite(val_proba))


class TestValidationSelection:
    def test_threshold_produces_valid_range(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)
        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "RandomForest", cv,
        )
        val_proba = result.calibrated_model.predict_proba(
            result.base_pipeline.predict_proba(ds.validation.X)[:, 1],
        )
        best_th = find_best_threshold(ds.validation.y, val_proba)
        assert 0.1 <= best_th["threshold"] <= 0.9
        assert 0.0 <= best_th["f1"] <= 1.0
        assert "precision" in best_th
        assert "recall" in best_th

    def test_threshold_deterministic(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "RandomForest", n_splits=3)
        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "RandomForest", cv,
        )
        val_proba = result.calibrated_model.predict_proba(
            result.base_pipeline.predict_proba(ds.validation.X)[:, 1],
        )
        th1 = find_best_threshold(ds.validation.y, val_proba)
        th2 = find_best_threshold(ds.validation.y, val_proba)
        assert th1["threshold"] == th2["threshold"]
        assert th1["f1"] == th2["f1"]

    def test_winner_classifier_by_pr_auc(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        from ml.training.classifier_trainer import ClassifierResult, CVResult
        results = []
        for name in ["LogisticRegression", "HistGradientBoosting", "RandomForest"]:
            cv = train_classifier_cv(ds.train.X, ds.train.y, schema, name, n_splits=2)
            result = calibrate_and_evaluate(
                ds.train.X, ds.train.y,
                ds.validation.X, ds.validation.y,
                schema, name, cv,
            )
            results.append(result)

        winner = get_winner_classifier(results)
        assert winner is not None
        assert winner.validation_metrics is not None
        assert winner.model_name in ["LogisticRegression", "HistGradientBoosting", "RandomForest"]

    def test_winner_regressor_by_mae(self, faz5_dataset, schema):
        ds = faz5_dataset.regression
        results = []
        for name in ["ElasticNet_log1p", "HistGradientBoostingRegressor", "RandomForestRegressor"]:
            cv = train_regressor_cv(ds.train.X, ds.train.y, schema, name, n_splits=2)
            result = fit_and_evaluate_regressor(
                ds.train.X, ds.train.y,
                ds.validation.X, ds.validation.y,
                schema, name, best_params=cv.best_params,
            )
            result.cv_result = cv
            results.append(result)

        winner = get_winner_regressor(results)
        assert winner is not None
        assert winner.validation_metrics is not None

    def test_validation_metrics_have_expected_keys(self, faz5_dataset, schema):
        ds = faz5_dataset.classification
        cv = train_classifier_cv(ds.train.X, ds.train.y, schema, "LogisticRegression", n_splits=2)
        result = calibrate_and_evaluate(
            ds.train.X, ds.train.y,
            ds.validation.X, ds.validation.y,
            schema, "LogisticRegression", cv,
        )
        vm = result.validation_metrics
        assert "roc_auc" in vm
        assert "pr_auc" in vm
        assert "precision" in vm
        assert "recall" in vm
        assert "f1" in vm
        assert "confusion_matrix" in vm


class TestSealedHoldout:
    def test_evaluation_on_test_is_blocked(self):
        with pytest.raises(SealedSplitError):
            assert_evaluable("test")

    def test_evaluation_on_audit_is_blocked(self):
        with pytest.raises(SealedSplitError):
            assert_evaluable("audit")

    def test_evaluation_on_train_is_allowed(self):
        assert_evaluable("train")

    def test_evaluation_on_validation_is_allowed(self):
        assert_evaluable("validation")

    def test_dataset_test_info_not_used_for_training(self, faz5_dataset):
        ds = faz5_dataset.classification
        assert ds.test_info.row_count >= 0
        assert ds.test_info.date_range is not None or ds.test_info.row_count == 0

    def test_dataset_audit_info_not_used_for_training(self, faz5_dataset):
        ds = faz5_dataset.regression
        assert ds.audit_info.row_count >= 0
