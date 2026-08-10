"""S30 Simulation tests — what-if scenarios."""
import json
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRun
from app.models.process import Process, ProcessSnapshot
from app.models.model_bundle import ModelBundle


class TestS30Simulation:
    @pytest.fixture
    def _setup(self, db_session: Session):
        bundle = ModelBundle(model_version="s30-test", model_type="bundle", artifact_path="s30.joblib", is_active=0)
        db_session.add(bundle); db_session.flush()
        process = Process(external_id="S30-001", created_at=datetime(2024, 6, 15, tzinfo=UTC), deadline=datetime(2024, 7, 1, tzinfo=UTC))
        db_session.add(process); db_session.flush()
        default_input = json.dumps({"source": "phone", "subject": "test", "reason": "test", "type": "Graffiti Removal", "neighborhood": "Back Bay", "created_at": "2024-06-15T12:00:00", "deadline": "2024-07-01T00:00:00"})
        snap = ProcessSnapshot(process_id=process.id, snapshot_type="opening", snapshot_at=datetime(2024, 6, 15, tzinfo=UTC), input_json=default_input, input_fingerprint="fp-s30")
        db_session.add(snap); db_session.commit()
        return bundle, process, snap

    def test_simulation_prediction_context_is_simulation(self, db_session, _setup):
        bundle, process, snap = _setup
        base = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(base); db_session.flush()
        sim = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="simulation", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(sim); db_session.flush()
        assert sim.prediction_context == "simulation"

    def test_simulation_persists_overrides(self, db_session, _setup):
        bundle, process, snap = _setup
        overrides_json = json.dumps({"source": "mobile_app"})
        base = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(base); db_session.flush()
        sim = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="simulation", status="success", simulation_overrides_json=overrides_json, predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(sim); db_session.flush()
        assert sim.simulation_overrides_json == overrides_json
        parsed = json.loads(sim.simulation_overrides_json)
        assert parsed["source"] == "mobile_app"

    def test_simulation_uses_same_bundle(self, db_session, _setup):
        bundle, process, snap = _setup
        base = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(base); db_session.flush()
        sim = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="simulation", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(sim); db_session.flush()
        assert sim.model_bundle_id == base.model_bundle_id

    def test_snapshot_immutable_after_simulation(self, db_session, _setup):
        _, process, snap = _setup
        orig_input = snap.input_json
        base = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=1, model_version="v1", prediction_context="opening", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(base); db_session.flush()
        sim = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=1, model_version="v1", prediction_context="simulation", status="success", simulation_overrides_json=json.dumps({"source": "altered"}), predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(sim); db_session.flush()
        db_session.refresh(snap)
        assert snap.input_json == orig_input

    def test_predicted_is_delayed_persisted(self, db_session, _setup):
        bundle, process, snap = _setup
        pred = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", delay_probability=0.8, predicted_is_delayed=1, predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(pred); db_session.flush()
        db_session.refresh(pred)
        assert pred.predicted_is_delayed == 1

    def test_predicted_is_delayed_null_for_sla_missing(self, db_session, _setup):
        bundle, process, snap = _setup
        pred = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", delay_probability=None, predicted_is_delayed=None, predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(pred); db_session.flush()
        assert pred.predicted_is_delayed is None
        assert pred.delay_probability is None

    def test_simulation_excluded_from_opening_predictions(self, db_session, _setup):
        bundle, process, snap = _setup
        opening = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        simulation = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="simulation", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add_all([opening, simulation]); db_session.flush()
        from sqlalchemy import select
        opening_query = db_session.execute(select(PredictionRun).where(PredictionRun.prediction_context == "opening")).scalars().all()
        sim_query = db_session.execute(select(PredictionRun).where(PredictionRun.prediction_context == "simulation")).scalars().all()
        assert len(opening_query) == 1
        assert len(sim_query) == 1
