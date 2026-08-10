"""S32 Model Performance tests."""
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.models.model_bundle import ModelBundle


class TestS32Performance:
    @pytest.fixture
    def _setup(self, db_session: Session):
        import json
        metrics = {
            "stage": "integration_baseline",
            "threshold": 0.35,
            "classifier": {
                "calibration_method": "sigmoid",
                "validation_pr_auc": 0.72,
                "validation_roc_auc": 0.81,
                "validation_brier": 0.18,
                "validation_f1": 0.64,
                "validation_precision": 0.61,
                "validation_recall": 0.67,
                "classification_validation_row_count": 245,
                "cv_pr_auc_mean": 0.68,
                "cv_pr_auc_std": 0.05,
                "cv_roc_auc_mean": 0.78,
                "cv_roc_auc_std": 0.04,
            },
            "regression": {
                "validation_mae": 12.3,
                "validation_median_ae": 8.1,
                "validation_rmse": 18.9,
                "validation_p90_ae": 35.2,
                "regression_validation_row_count": 210,
                "cv_mae_mean": 14.1,
                "cv_mae_std": 2.3,
                "cv_rmse_mean": 21.5,
                "cv_rmse_std": 3.1,
            },
        }
        metrics_json = json.dumps(metrics)
        bundle = ModelBundle(
            model_version="s32-test", model_type="bundle",
            artifact_path="s32.joblib", is_active=1,
            metrics_json=metrics_json,
        )
        db_session.add(bundle); db_session.commit()
        return bundle, metrics

    def test_bundle_metrics_accessible(self, db_session, _setup):
        bundle, metrics = _setup
        import json
        stored = json.loads(bundle.metrics_json)
        assert stored["classifier"]["validation_pr_auc"] == 0.72
        assert stored["regression"]["validation_mae"] == 12.3

    def test_no_test_metrics_in_bundle(self, db_session, _setup):
        bundle, metrics = _setup
        import json
        stored = json.loads(bundle.metrics_json)
        assert "test" not in stored.get("classifier", {})
        assert "test_accuracy" not in stored.get("classifier", {})
        assert "audit" not in stored.get("classifier", {})

    def test_classification_regression_separate(self, db_session, _setup):
        bundle, metrics = _setup
        import json
        stored = json.loads(bundle.metrics_json)
        assert "classifier" in stored
        assert "regression" in stored
        assert "validation_pr_auc" in stored["classifier"]
        assert "validation_mae" in stored["regression"]

    def test_threshold_in_metadata(self, db_session, _setup):
        bundle, metrics = _setup
        import json
        stored = json.loads(bundle.metrics_json)
        assert stored["threshold"] == 0.35

    def test_cv_metrics_present(self, db_session, _setup):
        bundle, metrics = _setup
        import json
        stored = json.loads(bundle.metrics_json)
        assert "cv_pr_auc_mean" in stored["classifier"]
        assert "cv_mae_mean" in stored["regression"]
        assert stored["classifier"]["cv_pr_auc_std"] == 0.05
