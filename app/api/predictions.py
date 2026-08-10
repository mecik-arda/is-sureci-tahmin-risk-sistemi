"""Prediction API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import (
    AppError,
    ModelUnavailableError,
    SnapshotNotFoundError,
)
from app.core.runtime import runtime_state
from app.schemas.prediction import BatchPredictionRequest, PredictionResponse
from app.services.prediction_service import predict_single

router = APIRouter(tags=["predictions"])


def _to_response(result) -> PredictionResponse:
    prediction = result.prediction_run
    pp = getattr(result, "delay_prob_std", None)
    ps = getattr(result, "predicted_hours_std", None)
    delay_lower = None
    delay_upper = None
    hours_lower = None
    hours_upper = None
    if pp is not None and prediction.delay_probability is not None:
        delay_lower = max(0.0, prediction.delay_probability - 2 * pp)
        delay_upper = min(1.0, prediction.delay_probability + 2 * pp)
    if ps is not None and prediction.predicted_hours is not None:
        hours_lower = max(0.0, prediction.predicted_hours - 2 * ps)
        hours_upper = max(0.0, prediction.predicted_hours + 2 * ps)
    return PredictionResponse(
        prediction_run_id=prediction.id,
        process_id=prediction.process_id,
        snapshot_id=prediction.snapshot_id,
        model_bundle_id=prediction.model_bundle_id,
        prediction_context=prediction.prediction_context,
        status=prediction.status,
        reused=result.reused,
        classification_available=result.classification_available,
        delay_probability=prediction.delay_probability,
        delay_probability_lower=delay_lower,
        delay_probability_upper=delay_upper,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        predicted_is_delayed=(prediction.predicted_is_delayed == 1) if prediction.predicted_is_delayed is not None else None,
        integration_threshold=result.integration_threshold,
        regression_available=result.regression_available,
        predicted_duration_hours=prediction.predicted_hours,
        predicted_duration_hours_lower=hours_lower,
        predicted_duration_hours_upper=hours_upper,
        model_stage="integration_baseline",
        created_at=prediction.predicted_at,
    )


@router.post("/processes/{process_id}/predictions", response_model=PredictionResponse)
def create_prediction(process_id: int, db: Session = Depends(get_db)):
    if runtime_state.bundle is None:
        raise ModelUnavailableError()

    try:
        result = predict_single(db, process_id, runtime_state.bundle)
        db.commit()
    except SnapshotNotFoundError:
        db.rollback()
        raise
    except AppError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise AppError(
            message="Tahmin sirasinda beklenmeyen hata.",
        )

    return _to_response(result)


@router.post("/predictions/batch")
def create_batch_predictions(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    if runtime_state.bundle is None:
        raise ModelUnavailableError()

    results = []
    for process_id in request.process_ids:
        try:
            result = predict_single(db, process_id, runtime_state.bundle)
            db.commit()
            results.append({
                "process_id": process_id,
                "ok": True,
                "prediction": _to_response(result).model_dump(mode="json"),
            })
        except AppError as error:
            db.rollback()
            results.append({
                "process_id": process_id,
                "ok": False,
                "error_code": error.error_code,
            })
        except Exception:
            db.rollback()
            results.append({
                "process_id": process_id,
                "ok": False,
                "error_code": "PREDICTION_FAILED",
            })

    return {
        "results": results,
        "succeeded": sum(1 for result in results if result["ok"]),
        "failed": sum(1 for result in results if not result["ok"]),
    }
