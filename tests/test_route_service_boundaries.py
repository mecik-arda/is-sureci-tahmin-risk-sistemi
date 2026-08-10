import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from app.core.errors import ProcessNotFoundError, SnapshotNotFoundError
from app.main import app
from app.models.model_bundle import ModelBundle
from app.models.prediction import PredictionRun
from app.models.process import Process, ProcessSnapshot
from app.services.analysis_dataset import AnalysisDatasetService
from app.services.simulation_service import run_simulation
from app.web import routes


SIMULATION_PATH = "/api/processes/{process_id}/simulations"
MODEL_PERFORMANCE_PATH = "/api/model-performance"


def registered_paths() -> set[str]:
    return {route.path for route in app.routes}


def opening_input() -> dict[str, str]:
    return {
        "source": "phone",
        "subject": "test",
        "reason": "test",
        "type": "Graffiti Removal",
        "neighborhood": "Back Bay",
        "created_at": "2024-06-15T12:00:00",
        "deadline": "2024-07-01T00:00:00",
    }


def create_process_with_snapshot(db_session, external_id: str = "BOUNDARY-001"):
    process = Process(
        external_id=external_id,
        created_at=datetime(2024, 6, 15, tzinfo=UTC),
        deadline=datetime(2024, 7, 1, tzinfo=UTC),
    )
    db_session.add(process)
    db_session.flush()
    snapshot = ProcessSnapshot(
        process_id=process.id,
        snapshot_type="opening",
        snapshot_at=datetime(2024, 6, 15, tzinfo=UTC),
        input_json=json.dumps(opening_input()),
        input_fingerprint=f"fp-{external_id}",
    )
    db_session.add(snapshot)
    db_session.flush()
    return process, snapshot


def create_bundle(db_session, model_version: str = "boundary-bundle"):
    bundle = ModelBundle(
        model_version=model_version,
        model_type="bundle",
        artifact_path=f"{model_version}.joblib",
        is_active=0,
    )
    db_session.add(bundle)
    db_session.flush()
    return bundle


class RecordingClassifier:
    def __init__(self):
        self.frame = None

    def predict_proba(self, frame):
        self.frame = frame.copy()
        return np.array([[0.2, 0.8]])


class RecordingRegressor:
    def __init__(self):
        self.frame = None

    def predict(self, frame):
        self.frame = frame.copy()
        return np.array([24.0])


def simulation_bundle():
    return SimpleNamespace(
        classifier=RecordingClassifier(),
        regressor=RecordingRegressor(),
        calibration_model=None,
        threshold=0.5,
    )


class TestRouteRegistration:
    def test_simulation_single_prefix_registered(self):
        assert SIMULATION_PATH in registered_paths()

    def test_simulation_double_prefix_not_registered(self):
        assert "/api/api/processes/{process_id}/simulations" not in registered_paths()

    def test_model_performance_single_prefix_registered(self):
        assert MODEL_PERFORMANCE_PATH in registered_paths()

    def test_model_performance_double_prefix_not_registered(self):
        assert "/api/api/model-performance" not in registered_paths()

    def test_no_double_api_prefix_registered(self):
        assert not [path for path in registered_paths() if "/api/api/" in path]

    def test_frontend_simulation_path_matches_backend(self):
        source = Path("app/static/js/process_detail.js").read_text(encoding="utf-8")
        assert "'/api/processes/' + encodeURIComponent(processId) + '/simulations" in source
        assert SIMULATION_PATH in registered_paths()

    def test_frontend_model_performance_path_matches_backend(self):
        source = Path("app/static/js/model_performance.js").read_text(encoding="utf-8")
        assert "fetch('/api/model-performance')" in source
        assert MODEL_PERFORMANCE_PATH in registered_paths()


class TestRouteServiceBoundary:
    def test_similarity_route_has_no_ml_preparation(self):
        source = inspect.getsource(routes.similar_processes)
        assert "derive_features" not in source
        assert "DataFrame" not in source
        assert "ProcessSnapshot" not in source

    def test_simulation_route_has_no_ml_preparation(self):
        source = inspect.getsource(routes.create_simulation)
        assert "PredictionRun" not in source
        assert "ProcessSnapshot" not in source
        assert "derive_features" not in source


class TestSimilarityServiceBoundary:
    def test_process_level_similarity_prepares_opening_features(self, db_session, monkeypatch):
        process, _ = create_process_with_snapshot(db_session)
        service = AnalysisDatasetService()
        captured = {}

        def get_similar_processes(session, process_id, bundle, query_features):
            captured["session"] = session
            captured["process_id"] = process_id
            captured["bundle"] = bundle
            captured["query_features"] = query_features
            return {"neighbors": [], "available": True}

        monkeypatch.setattr(service, "get_similar_processes", get_similar_processes)
        bundle = object()
        result = service.get_similar_processes_for_process(db_session, process.id, bundle)

        assert result == {"neighbors": [], "available": True}
        assert captured["session"] is db_session
        assert captured["process_id"] == process.id
        assert captured["bundle"] is bundle
        assert captured["query_features"].iloc[0]["source"] == "phone"
        assert captured["query_features"].iloc[0]["open_month"] == 6

    def test_missing_process_preserves_error(self, db_session):
        service = AnalysisDatasetService()
        with pytest.raises(ProcessNotFoundError):
            service.get_similar_processes_for_process(db_session, 999999, object())

    def test_missing_snapshot_preserves_unavailable_result(self, db_session):
        process = Process(
            external_id="NO-SNAPSHOT",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        db_session.add(process)
        db_session.flush()

        result = AnalysisDatasetService().get_similar_processes_for_process(
            db_session,
            process.id,
            object(),
        )

        assert result == {"neighbors": [], "available": False}

    def test_repeated_similarity_reuses_cached_dataset(self, db_session, monkeypatch):
        process, _ = create_process_with_snapshot(db_session, "CACHE-QUERY")
        neighbor = Process(
            external_id="CACHE-NEIGHBOR",
            process_type="neighbor-type",
            created_at=datetime(2024, 5, 1, tzinfo=UTC),
        )
        db_session.add(neighbor)
        db_session.flush()

        split = SimpleNamespace(
            X=pd.DataFrame([opening_input()]),
            external_ids=[process.external_id],
            y=np.array([0]),
        )
        dataset = SimpleNamespace(train=split, validation=split)
        dataset_result = SimpleNamespace(classification=dataset, regression=dataset)
        build_calls = []

        def build_dataset(session, schema):
            build_calls.append((session, schema))
            return dataset_result

        class SimilarityService:
            def __init__(self, **kwargs):
                pass

            def find_similar(self, query_features, pipeline):
                self_neighbor = SimpleNamespace(
                    external_id=process.external_id,
                    is_delayed=False,
                    total_duration_hours=10.0,
                )
                other_neighbor = SimpleNamespace(
                    external_id=neighbor.external_id,
                    is_delayed=True,
                    total_duration_hours=20.0,
                )
                return [[self_neighbor, other_neighbor]]

        monkeypatch.setattr("app.services.analysis_dataset.build_dataset", build_dataset)
        monkeypatch.setattr(
            "ml.similarity.similarity_service.SimilarityService",
            SimilarityService,
        )
        service = AnalysisDatasetService()
        bundle = SimpleNamespace(classifier=object())

        first = service.get_similar_processes_for_process(db_session, process.id, bundle)
        second = service.get_similar_processes_for_process(db_session, process.id, bundle)

        assert first == second
        assert first["neighbors"][0]["process_type"] == "neighbor-type"
        assert len(build_calls) == 1
        assert service.build_count == 1
        assert service.cache_hit_count == 1


class TestSimulationServiceBoundary:
    def test_full_workflow_applies_overrides_and_persists_only_simulation(self, db_session):
        process, snapshot = create_process_with_snapshot(db_session, "SIM-WORKFLOW")
        model_bundle = create_bundle(db_session)
        base = PredictionRun(
            process_id=process.id,
            snapshot_id=snapshot.id,
            model_bundle_id=model_bundle.id,
            model_version=model_bundle.model_version,
            prediction_context="opening",
            status="success",
            predicted_at=datetime.now(UTC),
            prediction_type="normal",
        )
        db_session.add(base)
        db_session.flush()
        original_input = snapshot.input_json
        bundle = simulation_bundle()

        simulation = run_simulation(
            db_session,
            process.id,
            base.id,
            {"source": "mobile_app"},
            bundle,
        )

        assert simulation.prediction_context == "simulation"
        assert simulation.risk_score == 80
        assert simulation.risk_level == "high"
        assert json.loads(simulation.simulation_overrides_json) == {"source": "mobile_app"}
        assert bundle.classifier.frame.iloc[0]["source"] == "mobile_app"
        assert bundle.regressor.frame.iloc[0]["source"] == "mobile_app"
        assert snapshot.input_json == original_input
        simulations = db_session.execute(
            select(PredictionRun).where(PredictionRun.prediction_context == "simulation")
        ).scalars().all()
        assert simulations == [simulation]

    def test_missing_base_prediction_preserves_error(self, db_session):
        process, _ = create_process_with_snapshot(db_session, "SIM-NO-PREDICTION")
        with pytest.raises(ProcessNotFoundError):
            run_simulation(db_session, process.id, 999999, {}, simulation_bundle())

    def test_missing_snapshot_preserves_error(self, db_session):
        process = Process(
            external_id="SIM-NO-SNAPSHOT",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
        )
        db_session.add(process)
        db_session.flush()
        model_bundle = create_bundle(db_session, "no-snapshot-bundle")
        base = PredictionRun(
            process_id=process.id,
            snapshot_id=None,
            model_bundle_id=model_bundle.id,
            model_version=model_bundle.model_version,
            prediction_context="opening",
            status="success",
            predicted_at=datetime.now(UTC),
            prediction_type="normal",
        )
        db_session.add(base)
        db_session.flush()

        with pytest.raises(SnapshotNotFoundError):
            run_simulation(db_session, process.id, base.id, {}, simulation_bundle())
