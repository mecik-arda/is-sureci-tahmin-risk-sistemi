"""Dashboard data service — S23 prediction/actual separation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRun
from app.models.process import Process


def get_dashboard_data(session: Session, bundle_id: int | None = None) -> dict[str, Any]:
    """Returns prediction_kpis and actual_kpis separately (S23)."""
    cutoff = datetime.now(UTC) - timedelta(days=30)

    prediction_filters = [
        PredictionRun.predicted_at >= cutoff,
        PredictionRun.status == "success",
        PredictionRun.prediction_context == "opening",
    ]
    if bundle_id is not None:
        prediction_filters.append(PredictionRun.model_bundle_id == bundle_id)

    prediction_values = session.execute(
        select(
            func.count(PredictionRun.id),
            func.coalesce(
                func.sum(case((PredictionRun.predicted_is_delayed == 1, 1), else_=0)),
                0,
            ),
            func.avg(PredictionRun.delay_probability),
            func.avg(PredictionRun.predicted_hours),
        ).where(*prediction_filters)
    ).one()

    pred_kpis = {
        "total_predictions": int(prediction_values[0]),
        "high_risk_count": int(prediction_values[1]),
        "avg_delay_probability": float(prediction_values[2]) if prediction_values[2] is not None else 0,
        "avg_predicted_hours": float(prediction_values[3]) if prediction_values[3] is not None else 0,
    }

    completed = Process.completed_at.is_not(None)
    has_sla = Process.deadline.is_not(None)
    actual_values = session.execute(
        select(
            func.coalesce(func.sum(case((completed, 1), else_=0)), 0),
            func.coalesce(func.sum(case((and_(completed, has_sla, Process.completed_at <= Process.deadline), 1), else_=0)), 0),
            func.coalesce(func.sum(case((and_(completed, has_sla, Process.completed_at > Process.deadline), 1), else_=0)), 0),
        )
    ).one()
    total_completed = int(actual_values[0])
    on_time = int(actual_values[1])
    actually_delayed = int(actual_values[2])
    classified_completed = on_time + actually_delayed

    actual_kpis = {
        "total_completed": total_completed,
        "on_time": on_time,
        "actually_delayed": actually_delayed,
        "actual_delay_rate": actually_delayed / max(1, classified_completed),
    }

    day = func.date(PredictionRun.predicted_at).label("day")
    daily_counts = session.execute(
        select(day, func.count(PredictionRun.id))
        .where(*prediction_filters)
        .group_by(day)
        .order_by(day)
    ).all()

    process_type = func.coalesce(Process.process_type, "Tanımlanmamış").label("process_type")
    process_type_counts = session.execute(
        select(process_type, func.count(Process.id))
        .group_by(process_type)
        .order_by(func.count(Process.id).desc(), process_type)
        .limit(10)
    ).all()

    return {
        "prediction_kpis": pred_kpis,
        "actual_kpis": actual_kpis,
        "daily_volume": [
            {"date": date, "count": int(count)}
            for date, count in daily_counts
        ],
        "process_type_distribution": [
            {"label": label, "count": int(count)}
            for label, count in process_type_counts
        ],
    }
