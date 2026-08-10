import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api import predictions
from app.core.errors import AppError
from app.core.runtime import runtime_state
from app.models.model_bundle import ModelBundle
from app.models.prediction import PredictionRun
from app.models.process import Process
from app.schemas.prediction import BatchPredictionRequest
from app.services.prediction_service import _to_risk_level
from app.services.process_service import get_prediction_history, get_process_list
from app.web.routes import _model_metrics_metadata
from scripts.enrich_active_bundle_metrics import enrich_metrics


def add_bundle(session, version: str) -> ModelBundle:
    bundle = ModelBundle(
        model_version=version,
        model_type="bundle",
        artifact_path=f"{version}.joblib",
        is_active=0,
    )
    session.add(bundle)
    session.flush()
    return bundle


class TestRiskPresentation:
    @pytest.mark.parametrize(
        ("score", "level"),
        [(0, "low"), (39, "low"), (40, "medium"), (69, "medium"), (70, "high"), (100, "high"), (None, None)],
    )
    def test_risk_level_boundaries(self, score, level):
        assert _to_risk_level(score) == level


class TestPredictionHistory:
    def test_history_returns_all_contexts_in_reverse_chronological_order(self, db_session):
        bundle = add_bundle(db_session, "history-bundle")
        process = Process(external_id="HISTORY-001", created_at=datetime(2024, 6, 1, tzinfo=UTC))
        db_session.add(process)
        db_session.flush()
        opening = PredictionRun(
            process_id=process.id,
            model_bundle_id=bundle.id,
            model_version=bundle.model_version,
            status="success",
            prediction_context="opening",
            prediction_type="normal",
            delay_probability=0.2,
            risk_score=20,
            risk_level="low",
            predicted_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        simulation = PredictionRun(
            process_id=process.id,
            model_bundle_id=bundle.id,
            model_version=bundle.model_version,
            status="success",
            prediction_context="simulation",
            prediction_type="simulation",
            delay_probability=0.8,
            risk_score=80,
            risk_level="high",
            predicted_at=datetime(2024, 6, 2, tzinfo=UTC),
        )
        db_session.add_all([opening, simulation])
        db_session.flush()

        history = get_prediction_history(db_session, process.id)

        assert [item["prediction_context"] for item in history] == ["simulation", "opening"]
        assert history[0]["risk_level"] == "high"
        assert history[1]["risk_score"] == 20

    def test_regression_only_prediction_is_listed(self, db_session):
        bundle = add_bundle(db_session, "regression-only-bundle")
        process = Process(external_id="REGRESSION-ONLY", created_at=datetime(2024, 6, 1, tzinfo=UTC))
        db_session.add(process)
        db_session.flush()
        db_session.add(PredictionRun(
            process_id=process.id,
            model_bundle_id=bundle.id,
            model_version=bundle.model_version,
            status="success",
            prediction_context="opening",
            prediction_type="normal",
            predicted_hours=12.5,
            predicted_at=datetime(2024, 6, 1, tzinfo=UTC),
        ))
        db_session.flush()

        result = get_process_list(db_session, bundle_id=bundle.id)

        assert result["processes"][0]["has_prediction"] is True
        assert result["processes"][0]["delay_probability"] is None
        assert result["processes"][0]["predicted_hours"] == 12.5


class TestBatchPrediction:
    def test_batch_request_rejects_duplicate_or_excessive_ids(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(process_ids=[1, 1])
        with pytest.raises(ValidationError):
            BatchPredictionRequest(process_ids=list(range(1, 52)))

    @pytest.mark.parametrize("process_ids", [[True], ["1"], [1.0], [0], [-1]])
    def test_batch_request_requires_strict_positive_integer_ids(self, process_ids):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(process_ids=process_ids)

    def test_batch_preserves_success_and_failure_per_process(self, db_session, monkeypatch):
        def fake_prediction(_session, process_id, _bundle):
            if process_id == 2:
                raise AppError(error_code="SNAPSHOT_NOT_FOUND", status_code=404)
            prediction = SimpleNamespace(
                id=process_id,
                process_id=process_id,
                snapshot_id=None,
                model_bundle_id=1,
                prediction_context="opening",
                status="success",
                delay_probability=0.8,
                risk_score=80,
                risk_level="high",
                predicted_is_delayed=1,
                predicted_hours=10.0,
                model_version="batch-test",
                predicted_at=datetime.now(UTC),
            )
            return SimpleNamespace(
                prediction_run=prediction,
                reused=False,
                classification_available=True,
                integration_threshold=0.35,
                regression_available=True,
            )

        monkeypatch.setattr(predictions, "predict_single", fake_prediction)
        original_bundle = runtime_state.bundle
        runtime_state.bundle = object()
        try:
            result = predictions.create_batch_predictions(
                BatchPredictionRequest(process_ids=[1, 2]),
                db_session,
            )
        finally:
            runtime_state.bundle = original_bundle

        assert result["succeeded"] == 1
        assert result["failed"] == 1
        assert result["results"][0]["prediction"]["risk_level"] == "high"
        assert result["results"][1]["error_code"] == "SNAPSHOT_NOT_FOUND"


class TestModelMetricsAndDocumentation:
    def test_registry_metrics_extend_artifact_metadata(self):
        bundle = SimpleNamespace(
            metadata={"classifier": {"validation_pr_auc": 0.8}},
            bundle_record=SimpleNamespace(metrics_json=json.dumps({
                "classifier": {"validation_confusion_matrix": [[10, 2], [3, 8]]}
            })),
        )

        metadata = _model_metrics_metadata(bundle)

        assert metadata["classifier"]["validation_pr_auc"] == 0.8
        assert metadata["classifier"]["validation_confusion_matrix"] == [[10, 2], [3, 8]]

    def test_enrichment_writes_brier_score(self):
        metrics = enrich_metrics(
            {},
            {
                "roc_auc": 0.8,
                "pr_auc": 0.7,
                "brier": 0.12,
                "f1": 0.6,
                "precision": 0.5,
                "recall": 0.75,
                "row_count": 10,
                "confusion_matrix": [[4, 1], [2, 3]],
            },
            {
                "mae": 1.0,
                "median_ae": 0.8,
                "rmse": 1.2,
                "p90_abs_error": 2.0,
                "row_count": 8,
            },
        )

        assert metrics["classifier"]["validation_brier"] == 0.12

    def test_model_card_and_ui_components_exist(self):
        card = Path("docs/MODEL_KARTI.md").read_text(encoding="utf-8")
        detail = Path("app/templates/process_detail.html").read_text(encoding="utf-8")
        listing = Path("app/templates/process_list.html").read_text(encoding="utf-8")
        performance = Path("app/templates/model_performance.html").read_text(encoding="utf-8")

        assert "risk_level" in card
        assert "prediction-history-heading" in detail
        assert "if (!hasSla) show(noSla);" in Path("app/static/js/process_detail.js").read_text(encoding="utf-8")
        assert "batch-predict" in listing
        assert "selectedProcessIds: new Set()" in Path("app/static/js/process_list.js").read_text(encoding="utf-8")
        assert "state.selectedProcessIds.size >= 50" in Path("app/static/js/process_list.js").read_text(encoding="utf-8")
        assert "perf-confusion" in performance
