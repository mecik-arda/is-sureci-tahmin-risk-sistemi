"""S31 Feedback tests — accuracy/usefulness separation."""
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.prediction import PredictionFeedback, PredictionRun
from app.models.process import Process, ProcessSnapshot
from app.models.model_bundle import ModelBundle
from app.services.feedback_service import compute_actual_outcome, submit_feedback


class TestS31Feedback:
    @pytest.fixture
    def _setup(self, db_session: Session):
        bundle = ModelBundle(model_version="s31-test", model_type="bundle", artifact_path="s31.joblib", is_active=0)
        db_session.add(bundle); db_session.flush()
        process = Process(external_id="S31-001", created_at=datetime(2024, 6, 15, tzinfo=UTC), deadline=datetime(2024, 7, 1, tzinfo=UTC), completed_at=datetime(2024, 7, 5, tzinfo=UTC), closure_reason="resolved")
        db_session.add(process); db_session.flush()
        snap = ProcessSnapshot(process_id=process.id, snapshot_type="opening", snapshot_at=datetime(2024, 6, 15, tzinfo=UTC), input_json="{}", input_fingerprint="fp-s31")
        db_session.add(snap); db_session.flush()
        pred = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="opening", status="success", delay_probability=0.7, predicted_is_delayed=1, predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(pred); db_session.commit()
        return bundle, process, snap, pred

    def test_accuracy_backend_derived_delayed(self, db_session, _setup):
        bundle, process, snap, pred = _setup
        outcome = compute_actual_outcome(db_session, pred)
        assert outcome == 1  # completed_at > deadline → delayed

    def test_accuracy_backend_derived_not_delayed(self, db_session, _setup):
        bundle, process, snap, pred = _setup
        process.deadline = datetime(2024, 8, 1)
        db_session.flush()
        outcome = compute_actual_outcome(db_session, pred)
        assert outcome == 0

    def test_sla_missing_accuracy_unavailable(self, db_session, _setup):
        bundle, process, snap, pred = _setup
        process.deadline = None
        db_session.flush()
        outcome = compute_actual_outcome(db_session, pred)
        assert outcome is None

    def test_unresolved_outcome_unavailable(self, db_session, _setup):
        bundle, process, snap, pred = _setup
        process.completed_at = None
        db_session.flush()
        outcome = compute_actual_outcome(db_session, pred)
        assert outcome is None

    def test_simulation_feedback_rejected(self, db_session, _setup):
        bundle, process, snap, pred = _setup
        sim = PredictionRun(process_id=process.id, snapshot_id=snap.id, model_bundle_id=bundle.id, model_version=bundle.model_version, prediction_context="simulation", status="success", predicted_at=datetime.now(UTC), prediction_type="normal")
        db_session.add(sim); db_session.flush()
        from app.core.errors import AppError
        with pytest.raises(AppError, match="Simulasyon"):
            submit_feedback(db_session, sim.id, "accuracy", "test")

    def test_usefulness_comment_accepted(self, db_session, _setup):
        _, _, _, pred = _setup
        fb = submit_feedback(db_session, pred.id, "usefulness", "\u00c7ok faydal\u0131")
        assert fb.feedback_type == "usefulness"
        assert fb.comment == "\u00c7ok faydal\u0131"
        assert fb.actual_outcome is None

    def test_duplicate_feedback_upsert(self, db_session, _setup):
        _, _, _, pred = _setup
        fb1 = submit_feedback(db_session, pred.id, "usefulness", "ilk yorum")
        fb1_id = fb1.id
        fb2 = submit_feedback(db_session, pred.id, "usefulness", "g\u00fcncellenmi\u015f yorum")
        assert fb2.id == fb1_id
        assert fb2.comment == "g\u00fcncellenmi\u015f yorum"

        from sqlalchemy import select, func
        count = db_session.execute(
            select(func.count()).where(
                PredictionFeedback.prediction_id == pred.id,
                PredictionFeedback.feedback_type == "usefulness",
            )
        ).scalar()
        assert count == 1

    def test_feedback_prediction_immutable(self, db_session, _setup):
        _, _, _, pred = _setup
        orig_prob = pred.delay_probability
        orig_delayed = pred.predicted_is_delayed
        submit_feedback(db_session, pred.id, "usefulness", "test")
        db_session.refresh(pred)
        assert pred.delay_probability == orig_prob
        assert pred.predicted_is_delayed == orig_delayed
