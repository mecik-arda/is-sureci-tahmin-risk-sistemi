import inspect
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.api import predictions
from app.core.runtime import runtime_state
from app.models.model_bundle import ModelBundle
from app.models.prediction import PredictionRun
from app.models.process import Process, ProcessSnapshot
from app.services.dashboard_service import get_dashboard_data
from app.services.process_service import get_process_list


class Classifier:
    def predict_proba(self, _frame):
        return np.array([[0.2, 0.8]])


class Regressor:
    def predict(self, _frame):
        return np.array([12.0])


def create_bundle(session: Session, version: str) -> ModelBundle:
    bundle = ModelBundle(
        model_version=version,
        model_type="bundle",
        artifact_path=f"{version}.joblib",
        is_active=0,
    )
    session.add(bundle)
    session.flush()
    return bundle


def create_process(session: Session, external_id: str, completed_at=None, deadline=None, process_type="type_a") -> Process:
    process = Process(
        external_id=external_id,
        process_type=process_type,
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        completed_at=completed_at,
        deadline=deadline,
    )
    session.add(process)
    session.flush()
    return process


class TestDashboardAggregates:
    def test_dashboard_preserves_prediction_and_actual_scopes(self, db_session):
        now = datetime.now(UTC)
        bundle = create_bundle(db_session, "dashboard-current")
        other_bundle = create_bundle(db_session, "dashboard-other")
        on_time = create_process(
            db_session,
            "DASH-ONTIME",
            completed_at=datetime(2024, 6, 2, tzinfo=UTC),
            deadline=datetime(2024, 6, 3, tzinfo=UTC),
            process_type="type_a",
        )
        delayed = create_process(
            db_session,
            "DASH-DELAYED",
            completed_at=datetime(2024, 6, 4, tzinfo=UTC),
            deadline=datetime(2024, 6, 3, tzinfo=UTC),
            process_type="type_b",
        )
        open_process = create_process(db_session, "DASH-OPEN", process_type="type_a")
        db_session.add_all([
            PredictionRun(
                process_id=on_time.id,
                model_bundle_id=bundle.id,
                model_version=bundle.model_version,
                status="success",
                prediction_context="opening",
                prediction_type="normal",
                delay_probability=0.8,
                predicted_is_delayed=1,
                predicted_hours=12,
                predicted_at=now,
            ),
            PredictionRun(
                process_id=delayed.id,
                model_bundle_id=bundle.id,
                model_version=bundle.model_version,
                status="success",
                prediction_context="simulation",
                prediction_type="simulation",
                delay_probability=0.9,
                predicted_is_delayed=1,
                predicted_hours=10,
                predicted_at=now,
            ),
            PredictionRun(
                process_id=open_process.id,
                model_bundle_id=bundle.id,
                model_version=bundle.model_version,
                status="success",
                prediction_context="opening",
                prediction_type="normal",
                delay_probability=0.7,
                predicted_is_delayed=1,
                predicted_hours=8,
                predicted_at=now - timedelta(days=31),
            ),
            PredictionRun(
                process_id=open_process.id,
                model_bundle_id=other_bundle.id,
                model_version=other_bundle.model_version,
                status="success",
                prediction_context="opening",
                prediction_type="normal",
                delay_probability=0.6,
                predicted_is_delayed=1,
                predicted_hours=6,
                predicted_at=now,
            ),
        ])
        db_session.flush()

        result = get_dashboard_data(db_session, bundle.id)

        assert result["prediction_kpis"] == {
            "total_predictions": 1,
            "high_risk_count": 1,
            "avg_delay_probability": 0.8,
            "avg_predicted_hours": 12.0,
        }
        assert result["actual_kpis"] == {
            "total_completed": 2,
            "on_time": 1,
            "actually_delayed": 1,
            "actual_delay_rate": 0.5,
        }
        assert result["daily_volume"] == [{"date": now.date().isoformat(), "count": 1}]
        assert result["process_type_distribution"] == [
            {"label": "type_a", "count": 2},
            {"label": "type_b", "count": 1},
        ]

    def test_dashboard_index_is_present(self, db_session):
        indexes = db_session.execute(text("PRAGMA index_list('prediction_runs')")).all()
        assert "ix_prediction_runs_dashboard_recent" in {index[1] for index in indexes}

    def test_process_list_uses_bounded_query_count(self, db_session):
        bundle = create_bundle(db_session, "process-list-query-count")
        create_process(db_session, "LIST-ONE")
        create_process(db_session, "LIST-TWO")
        create_process(db_session, "LIST-THREE")
        statements = []

        def record_statement(_connection, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(db_session.get_bind(), "before_cursor_execute", record_statement)
        try:
            result = get_process_list(db_session, bundle.id, page=1, per_page=2)
        finally:
            event.remove(db_session.get_bind(), "before_cursor_execute", record_statement)

        assert result["total"] == 3
        assert len(result["processes"]) == 2
        assert len(statements) == 2


class TestPredictionAction:
    def test_prediction_endpoint_commits_created_prediction(self, db_session):
        bundle_record = create_bundle(db_session, "prediction-action")
        process = create_process(
            db_session,
            "PREDICTION-ACTION",
            deadline=datetime(2024, 6, 2, tzinfo=UTC),
        )
        snapshot = ProcessSnapshot(
            process_id=process.id,
            snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 1, tzinfo=UTC),
            input_json=json.dumps({
                "source": "citizens_connect_app",
                "subject": "animal_control",
                "reason": "animal_issues",
                "type": "animal_control",
                "neighborhood": "allston_brighton",
                "created_at": "2024-06-01T00:00:00",
                "deadline": "2024-06-02T00:00:00",
            }),
            input_fingerprint="prediction-action-fingerprint",
        )
        db_session.add(snapshot)
        db_session.flush()
        loaded_bundle = SimpleNamespace(
            bundle_id=bundle_record.id,
            bundle_record=bundle_record,
            classifier=Classifier(),
            regressor=Regressor(),
            calibration_model=None,
            threshold=0.5,
        )
        original_bundle = runtime_state.bundle
        runtime_state.bundle = loaded_bundle
        try:
            response = predictions.create_prediction(process.id, db_session)
        finally:
            runtime_state.bundle = original_bundle

        other_session = Session(bind=db_session.get_bind())
        try:
            persisted = other_session.execute(
                text("SELECT COUNT(*) FROM prediction_runs WHERE process_id = :process_id"),
                {"process_id": process.id},
            ).scalar_one()
        finally:
            other_session.close()

        assert response.process_id == process.id
        assert response.reused is False
        assert persisted == 1

    def test_detail_script_uses_existing_single_prediction_endpoint(self):
        source = inspect.getsourcefile(get_dashboard_data)
        assert source is not None
        script = open("app/static/js/process_detail.js", encoding="utf-8").read()
        assert "'/api/processes/' + encodeURIComponent(processId) + '/predictions'" in script
        assert "btn-predict" in script

    def test_dashboard_keeps_empty_sections_visible(self):
        script = open("app/static/js/dashboard.js", encoding="utf-8").read()
        assert "predictionSection.style.display" not in script
        assert "actualSection.style.display" not in script
