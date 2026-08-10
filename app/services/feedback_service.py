"""S31 Feedback service — accuracy vs usefulness separation."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.prediction import PredictionFeedback, PredictionRun


def compute_actual_outcome(session: Session, prediction: PredictionRun) -> int | None:
    from app.models.process import Process

    if prediction.process_id is None:
        return None

    process = session.execute(
        select(Process).where(Process.id == prediction.process_id)
    ).scalars().first()
    if process is None or process.deadline is None:
        return None
    if process.completed_at is None:
        return None

    return 1 if process.completed_at > process.deadline else 0


def submit_feedback(
    session: Session,
    prediction_id: int,
    feedback_type: str,
    comment: str | None = None,
) -> PredictionFeedback:
    prediction = session.get(PredictionRun, prediction_id)

    if prediction is not None and prediction.prediction_context == "simulation":
        from app.core.errors import AppError
        raise AppError(
            message="Simulasyon tahminlerine geri bildirim verilemez.",
            error_code="FEEDBACK_NOT_ALLOWED",
            status_code=422,
        )

    existing = session.execute(
        select(PredictionFeedback).where(
            PredictionFeedback.prediction_id == prediction_id,
            PredictionFeedback.feedback_type == feedback_type,
        )
    ).scalars().first()

    actual_outcome = None
    if feedback_type == "accuracy" and prediction is not None:
        actual_outcome = compute_actual_outcome(session, prediction)

    if existing is not None:
        existing.comment = comment
        existing.actual_outcome = actual_outcome
        existing.created_at = datetime.now(UTC)
        session.flush()
        return existing

    fb = PredictionFeedback(
        prediction_id=prediction_id,
        feedback_type=feedback_type,
        comment=comment,
        actual_outcome=actual_outcome,
        created_at=datetime.now(UTC),
    )
    session.add(fb)
    session.flush()
    return fb
