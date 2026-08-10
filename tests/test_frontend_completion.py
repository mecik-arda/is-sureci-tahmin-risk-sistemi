import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.main import app
from app.core.errors import AppError
from app.models.import_run import DataQualityIssue, ImportRun
from app.models.process import Process
from app.schemas.interaction import FeedbackRequest, SimulationRequest
from app.services.dashboard_service import get_dashboard_data
from app.services.data_quality_service import get_data_quality_summary
from app.services.model_monitoring_service import get_model_monitoring_data
from app.services.process_service import get_process_detail
from app.web import routes


class TestInteractionRequests:
    def test_simulation_request_accepts_valid_body(self):
        request = SimulationRequest(
            base_prediction_id=1,
            overrides={"source": "citizens_connect_app", "sla_duration_hours": 24},
        )

        assert request.base_prediction_id == 1
        assert request.overrides.source == "citizens_connect_app"

    @pytest.mark.parametrize(
        "payload",
        [
            {"base_prediction_id": 0, "overrides": {"source": "citizens_connect_app"}},
            {"base_prediction_id": 1, "overrides": {}},
            {"base_prediction_id": 1, "overrides": {"open_month": 13}},
            {"base_prediction_id": 1, "overrides": {"sla_duration_hours": -1}},
            {"base_prediction_id": 1, "overrides": {"unexpected": "value"}},
            {"base_prediction_id": 1, "overrides": {"source": "not valid"}},
        ],
    )
    def test_simulation_request_rejects_invalid_body(self, payload):
        with pytest.raises(ValidationError):
            SimulationRequest(**payload)

    def test_feedback_comment_has_size_limit(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback_type="usefulness", comment="x" * 5001)

    def test_usefulness_feedback_requires_comment(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback_type="usefulness")


class TestFrontendDataServices:
    def test_process_detail_includes_deadline(self, db_session):
        process = Process(
            external_id="FRONTEND-DEADLINE",
            created_at=datetime(2024, 6, 1, tzinfo=UTC),
            deadline=datetime(2024, 6, 2, tzinfo=UTC),
        )
        db_session.add(process)
        db_session.flush()

        detail = get_process_detail(db_session, process.id)

        assert detail is not None
        assert detail["current_fields"]["deadline"] == "2024-06-02T00:00:00+00:00"

    def test_dashboard_includes_daily_and_process_type_data(self, db_session):
        db_session.add_all([
            Process(external_id="TYPE-A-1", process_type="A", created_at=datetime(2024, 6, 1, tzinfo=UTC)),
            Process(external_id="TYPE-A-2", process_type="A", created_at=datetime(2024, 6, 2, tzinfo=UTC)),
            Process(external_id="TYPE-B-1", process_type="B", created_at=datetime(2024, 6, 3, tzinfo=UTC)),
        ])
        db_session.flush()

        result = get_dashboard_data(db_session)

        assert result["daily_volume"] == []
        assert result["process_type_distribution"] == [
            {"label": "A", "count": 2},
            {"label": "B", "count": 1},
        ]

    def test_data_quality_summary_uses_audit_records(self, db_session):
        run = ImportRun(
            file_name="source.csv",
            status="completed_with_issues",
            total_rows=10,
            imported_rows=8,
            quarantined_rows=1,
            error_rows=1,
            warning_count=2,
            completed_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        db_session.add(run)
        db_session.flush()
        db_session.add(DataQualityIssue(
            import_run_id=run.id,
            issue_code="INVALID_DATE",
            severity="error",
        ))
        db_session.flush()

        result = get_data_quality_summary(db_session)

        assert result["total_import_runs"] == 1
        assert result["issue_count"] == 1
        assert result["issues_by_code"] == [{"code": "INVALID_DATE", "count": 1}]
        assert result["recent_runs"][0]["file_name"] == "source.csv"

    def test_model_monitoring_uses_existing_runtime_and_cache_state(self):
        bundle = SimpleNamespace(
            bundle_record=SimpleNamespace(
                model_version="monitor-v1",
                model_type="bundle",
                trained_at=datetime(2024, 6, 1, tzinfo=UTC),
            ),
            metadata={
                "stage": "candidate",
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
            },
            artifact_hash="a" * 64,
            threshold=0.35,
        )
        cache = SimpleNamespace(
            built_at="2024-06-01T00:00:00+00:00",
            build_count=1,
            cache_hit_count=4,
            cached_rows=lambda: 20,
            cached_memory_bytes=lambda: 1024,
        )

        result = get_model_monitoring_data(bundle, cache)

        assert result["available"] is True
        assert result["model_version"] == "monitor-v1"
        assert result["analysis_cache"]["build_count"] == 1


class TestFrontendCompletionContract:
    def test_runtime_registers_frontend_completion_api_routes(self):
        paths = {route.path for route in app.routes}
        assert "/api/simulation-options" in paths
        assert "/api/model-monitoring" in paths
        assert "/api/data-quality" in paths

    def test_routes_use_json_request_models(self):
        simulation = inspect.get_annotations(routes.create_simulation, eval_str=True)
        feedback = inspect.get_annotations(routes.create_feedback, eval_str=True)
        assert simulation["request"] is SimulationRequest
        assert feedback["request"] is FeedbackRequest

    def test_simulation_categories_are_checked_against_canonical_options(self):
        valid = SimulationRequest(
            base_prediction_id=1,
            overrides={"source": "citizens_connect_app"},
        )
        assert valid.base_prediction_id == 1
        with pytest.raises(ValidationError):
            SimulationRequest(
                base_prediction_id=1,
                overrides={"source": "not_allowed"},
            )

    def test_templates_and_scripts_include_completed_screens(self):
        paths = {route.path for route in app.routes}
        assert "/model-monitoring" in paths
        assert "/data-quality" in paths
        assert Path("app/static/js/model_monitoring.js").is_file()
        assert Path("app/static/js/data_quality.js").is_file()
