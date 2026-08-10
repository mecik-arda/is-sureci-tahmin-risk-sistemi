"""Sealed guard ve metrik testleri."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from ml.evaluation.metrics import evaluate_classification, evaluate_regression
from ml.evaluation.sealed_guard import (
    ALLOWED_EVAL_SPLITS,
    SEALED_SPLITS,
    SealedSplitError,
    assert_evaluable,
)


class TestSealedGuard:
    def test_train_is_evaluable(self):
        assert_evaluable("train")

    def test_validation_is_evaluable(self):
        assert_evaluable("validation")

    def test_test_is_sealed(self):
        with pytest.raises(SealedSplitError, match="test"):
            assert_evaluable("test")

    def test_audit_is_sealed(self):
        with pytest.raises(SealedSplitError, match="audit"):
            assert_evaluable("audit")

    def test_unknown_split_raises(self):
        with pytest.raises(SealedSplitError, match="Bilinmeyen"):
            assert_evaluable("production")

    def test_allowed_and_sealed_disjoint(self):
        assert ALLOWED_EVAL_SPLITS.isdisjoint(SEALED_SPLITS)


class TestClassificationMetrics:
    def _make_data(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_pred = y_true.copy()
        y_pred[:10] = 1 - y_pred[:10]
        y_proba = rng.uniform(0.01, 0.99, size=100).astype(np.float64)
        return y_true, y_pred, y_proba

    def test_returns_all_metrics(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba, "validation")
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix", "row_count"]:
            assert key in result

    def test_row_count_correct(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba, "validation")
        assert result["row_count"] == 100

    def test_works_on_train(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba, "train")
        assert result["split"] == "train"

    def test_raises_on_test(self):
        y_true, y_pred, y_proba = self._make_data()
        with pytest.raises(SealedSplitError):
            evaluate_classification(y_true, y_pred, y_proba, "test")

    def test_raises_on_audit(self):
        y_true, y_pred, y_proba = self._make_data()
        with pytest.raises(SealedSplitError):
            evaluate_classification(y_true, y_pred, y_proba, "audit")

    def test_confusion_matrix_shape(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba, "validation")
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestRegressionMetrics:
    def _make_data(self):
        rng = np.random.RandomState(42)
        y_true = rng.uniform(1, 100, size=100)
        y_pred = y_true + rng.normal(0, 5, size=100)
        return y_true, y_pred

    def test_returns_all_metrics(self):
        y_true, y_pred = self._make_data()
        result = evaluate_regression(y_true, y_pred, "validation")
        for key in ["mae", "median_ae", "rmse", "p90_abs_error", "row_count"]:
            assert key in result

    def test_metrics_are_non_negative(self):
        y_true, y_pred = self._make_data()
        result = evaluate_regression(y_true, y_pred, "validation")
        assert result["mae"] >= 0
        assert result["median_ae"] >= 0
        assert result["rmse"] >= 0
        assert result["p90_abs_error"] >= 0

    def test_perfect_prediction_zero_error(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = evaluate_regression(y, y.copy(), "validation")
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0

    def test_raises_on_test(self):
        y_true, y_pred = self._make_data()
        with pytest.raises(SealedSplitError):
            evaluate_regression(y_true, y_pred, "test")

    def test_raises_on_audit(self):
        y_true, y_pred = self._make_data()
        with pytest.raises(SealedSplitError):
            evaluate_regression(y_true, y_pred, "audit")
