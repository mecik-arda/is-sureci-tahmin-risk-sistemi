"""Process list and detail service — S22/S24."""

from __future__ import annotations

import json
import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRun
from app.models.process import Process, ProcessSnapshot


def get_process_list(
    session: Session,
    bundle_id: int | None = None,
    page: int = 1,
    per_page: int = 20,
    status_filter: str | None = None,
    risk_filter: str | None = None,
) -> dict[str, Any]:
    filters = []

    if status_filter:
        if status_filter == "open":
            filters.append(Process.completed_at.is_(None))
        elif status_filter == "closed":
            filters.append(Process.completed_at.is_not(None))

    if bundle_id is None:
        total = int(session.scalar(select(func.count(Process.id)).where(*filters)) or 0)
        processes = session.execute(
            select(Process)
            .where(*filters)
            .order_by(Process.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        ).scalars().all()
        rows = [(process, None) for process in processes]
    else:
        latest_prediction_id = (
            select(PredictionRun.id)
            .where(
                PredictionRun.process_id == Process.id,
                PredictionRun.model_bundle_id == bundle_id,
                PredictionRun.prediction_context == "opening",
                PredictionRun.status == "success",
            )
            .order_by(PredictionRun.predicted_at.desc())
            .limit(1)
            .correlate(Process)
            .scalar_subquery()
        )
        query = select(Process, PredictionRun).outerjoin(
            PredictionRun,
            PredictionRun.id == latest_prediction_id,
        )
        count_query = select(func.count(Process.id)).select_from(Process).outerjoin(
            PredictionRun,
            PredictionRun.id == latest_prediction_id,
        )
        if risk_filter == "high_risk":
            filters.append(PredictionRun.predicted_is_delayed == 1)
        elif risk_filter == "low_risk":
            filters.extend([
                PredictionRun.predicted_is_delayed.is_not(None),
                PredictionRun.predicted_is_delayed != 1,
            ])
        total = int(session.scalar(count_query.where(*filters)) or 0)
        rows = session.execute(
            query
            .where(*filters)
            .order_by(Process.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        ).all()

    processes = []
    for process, prediction in rows:
        has_prediction = prediction is not None and (
            prediction.delay_probability is not None or prediction.predicted_hours is not None
        )
        processes.append({
            "id": process.id,
            "external_id": process.external_id,
            "process_type": process.process_type,
            "current_status": process.current_status,
            "created_at": process.created_at.isoformat() if process.created_at else None,
            "completed_at": process.completed_at.isoformat() if process.completed_at else None,
            "deadline": process.deadline.isoformat() if process.deadline else None,
            "delay_probability": float(prediction.delay_probability) if prediction and prediction.delay_probability is not None else None,
            "risk_score": prediction.risk_score if prediction else None,
            "risk_level": prediction.risk_level if prediction else None,
            "predicted_is_delayed": (prediction.predicted_is_delayed == 1) if has_prediction and prediction.predicted_is_delayed is not None else None,
            "predicted_hours": float(prediction.predicted_hours) if prediction and prediction.predicted_hours is not None else None,
            "has_prediction": has_prediction,
        })

    return {"processes": processes, "total": total, "page": page, "per_page": per_page}


def get_process_detail(session: Session, process_id: int) -> dict[str, Any] | None:
    """S22: opening_fields + current_fields + latest prediction."""
    process = session.execute(
        select(Process).where(Process.id == process_id)
    ).scalars().first()
    if process is None:
        return None

    snapshot = session.execute(
        select(ProcessSnapshot).where(
            ProcessSnapshot.process_id == process_id,
            ProcessSnapshot.snapshot_type == "opening",
        )
    ).scalars().first()

    opening_fields: dict[str, Any] = {}
    input_fingerprint = None
    snapshot_at = None
    if snapshot:
        input_data = json.loads(snapshot.input_json)
        from ml.features.feature_derivation import derive_features
        features = derive_features(input_data)
        opening_fields = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}
        input_fingerprint = snapshot.input_fingerprint
        snapshot_at = snapshot.snapshot_at.isoformat() if snapshot.snapshot_at else None

    current_fields = {
        "current_status": process.current_status,
        "completed_at": process.completed_at.isoformat() if process.completed_at else None,
        "deadline": process.deadline.isoformat() if process.deadline else None,
        "closure_reason": process.closure_reason,
        "total_duration_hours": None,
    }
    if process.completed_at is not None and process.created_at is not None:
        delta = (process.completed_at - process.created_at).total_seconds() / 3600
        current_fields["total_duration_hours"] = round(max(0, delta), 2)

    return {
        "id": process.id,
        "external_id": process.external_id,
        "process_type": process.process_type,
        "opening_fields": opening_fields,
        "current_fields": current_fields,
        "input_fingerprint": input_fingerprint,
        "snapshot_at": snapshot_at,
        "has_sla": process.deadline is not None,
    }


def get_process_prediction(session: Session, process_id: int, bundle_id: int | None = None) -> dict[str, Any] | None:
    """S25: latest successful prediction for this process."""
    query = select(PredictionRun).where(
        PredictionRun.process_id == process_id,
        PredictionRun.status == "success",
        PredictionRun.prediction_context == "opening",
    )
    if bundle_id is not None:
        query = query.where(PredictionRun.model_bundle_id == bundle_id)
    query = query.order_by(PredictionRun.predicted_at.desc()).limit(1)

    pred = session.execute(query).scalars().first()
    if pred is None:
        return None

    return {
        "prediction_id": pred.id,
        "delay_probability": float(pred.delay_probability) if pred.delay_probability is not None else None,
        "risk_score": pred.risk_score,
        "risk_level": pred.risk_level,
        "predicted_is_delayed": (pred.predicted_is_delayed == 1) if pred.predicted_is_delayed is not None else None,
        "predicted_hours": float(pred.predicted_hours) if pred.predicted_hours is not None else None,
        "predicted_at": pred.predicted_at.isoformat() if pred.predicted_at else None,
        "model_version": pred.model_version,
        "model_bundle_id": pred.model_bundle_id,
    }


def get_prediction_history(
    session: Session,
    process_id: int,
    limit: int = 50,
) -> list[dict[str, Any]]:
    predictions = session.execute(
        select(PredictionRun)
        .where(PredictionRun.process_id == process_id)
        .order_by(PredictionRun.predicted_at.desc())
        .limit(limit)
    ).scalars().all()
    return [
        {
            "prediction_id": prediction.id,
            "prediction_context": prediction.prediction_context,
            "status": prediction.status,
            "delay_probability": float(prediction.delay_probability) if prediction.delay_probability is not None else None,
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level,
            "predicted_is_delayed": prediction.predicted_is_delayed == 1 if prediction.predicted_is_delayed is not None else None,
            "predicted_hours": float(prediction.predicted_hours) if prediction.predicted_hours is not None else None,
            "model_version": prediction.model_version,
            "predicted_at": prediction.predicted_at.isoformat() if prediction.predicted_at else None,
        }
        for prediction in predictions
    ]
