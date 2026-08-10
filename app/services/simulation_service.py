"""S30 Simulation service — what-if scenario on opening-v1 features."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ProcessNotFoundError, SnapshotNotFoundError
from app.models.prediction import PredictionRun
from app.models.process import ProcessSnapshot
from app.services.prediction_service import _to_risk_level, _to_risk_score
from ml.features.feature_derivation import derive_features


def run_simulation(
    session: Session,
    process_id: int,
    base_prediction_id: int,
    overrides: dict[str, int | str | float | None],
    bundle,
) -> PredictionRun:
    prediction = session.execute(
        select(PredictionRun).where(
            PredictionRun.id == base_prediction_id,
            PredictionRun.process_id == process_id,
            PredictionRun.prediction_context == "opening",
            PredictionRun.status == "success",
        )
    ).scalars().first()
    if prediction is None:
        raise ProcessNotFoundError(
            message="Simulasyon icin once bu surec icin bir tahmin uretilmelidir."
        )

    snapshot = session.execute(
        select(ProcessSnapshot).where(
            ProcessSnapshot.process_id == process_id,
            ProcessSnapshot.snapshot_type == "opening",
        )
    ).scalars().first()
    if snapshot is None:
        raise SnapshotNotFoundError()

    features = derive_features(json.loads(snapshot.input_json))
    effective = dict(features)
    effective.update({k: v for k, v in overrides.items() if v is not None})

    import pandas as pd
    df = pd.DataFrame([effective])

    delay_prob = None
    is_delayed = None
    has_sla = effective.get("sla_duration_hours") is not None and not (
        isinstance(effective.get("sla_duration_hours"), float)
        and np.isnan(effective.get("sla_duration_hours", 0))
    )

    if has_sla:
        raw = float(bundle.classifier.predict_proba(df)[0, 1])
        if bundle.calibration_model is not None:
            delay_prob = float(bundle.calibration_model.predict_proba(np.array([raw]))[0])
        else:
            delay_prob = raw
        is_delayed = delay_prob >= bundle.threshold

    raw_pred = float(bundle.regressor.predict(df)[0])
    if not np.isfinite(raw_pred):
        predicted_hours = None
    else:
        predicted_hours = max(0.0, raw_pred)

    risk_score = _to_risk_score(delay_prob)

    sim = PredictionRun(
        process_id=prediction.process_id,
        snapshot_id=prediction.snapshot_id,
        model_bundle_id=prediction.model_bundle_id,
        model_version=prediction.model_version,
        prediction_context="simulation",
        status="success",
        delay_probability=delay_prob,
        risk_score=risk_score,
        risk_level=_to_risk_level(risk_score),
        predicted_is_delayed=(1 if is_delayed else 0) if is_delayed is not None and has_sla else None,
        predicted_hours=predicted_hours,
        simulation_overrides_json=json.dumps(overrides, ensure_ascii=False) if overrides else None,
        input_fingerprint=None,
        predicted_at=datetime.now(UTC),
    )
    session.add(sim)
    session.flush()
    return sim
