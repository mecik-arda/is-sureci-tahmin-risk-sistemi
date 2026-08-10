"""Prediction servisi.

S16: Feature'lar yalnız immutable opening snapshot'tan türetilir.
S18: Başarılı prediction reuse (idempotency).
S19: Partial UNIQUE index ile concurrent duplicate engelleme.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import (
    AppError,
    ModelUnavailableError,
    SnapshotNotFoundError,
)
from app.models.prediction import PredictionRun
from app.models.process import ProcessSnapshot
from app.services.model_loader import LoadedBundle
from ml.datasets.target_builder import OBSERVATION_CUTOFF, compute_is_delayed
from ml.features.feature_derivation import derive_features
from ml.features.schema_loader import load_feature_schema


def predict_single(
    session: Session,
    process_id: int,
    loaded_bundle: LoadedBundle,
) -> PredictionResult:
    snapshot = _get_opening_snapshot(session, process_id)
    if snapshot is None:
        raise SnapshotNotFoundError(
            message=f"Process {process_id} icin opening snapshot bulunamadi."
        )

    input_json = json.loads(snapshot.input_json)
    input_fp = snapshot.input_fingerprint
    integration_threshold = loaded_bundle.threshold

    reuse = _find_existing_success(
        session, snapshot.id, input_fp, loaded_bundle.bundle_id
    )
    if reuse is not None:
            return _to_result(reuse, input_json, reused=True, threshold=integration_threshold)

    features = derive_features(input_json)
    feature_df = _features_to_dataframe(features)

    clf_available = _has_sla(input_json)
    delay_prob = None
    is_delayed = None
    delay_prob_std = None

    if clf_available:
        raw_proba = float(loaded_bundle.classifier.predict_proba(feature_df)[0, 1])
        if loaded_bundle.calibration_model is not None:
            delay_prob = float(loaded_bundle.calibration_model.predict_proba(
                np.array([raw_proba])
            )[0])
        else:
            delay_prob = raw_proba
        is_delayed = delay_prob >= integration_threshold
        clf_mean, clf_std = _compute_forest_std(loaded_bundle.classifier, feature_df)
        if clf_std is not None:
            delay_prob_std = clf_std

    reg_available = True
    predicted_hours = None
    predicted_hours_std = None

    raw_pred = float(loaded_bundle.regressor.predict(feature_df)[0])
    if not np.isfinite(raw_pred):
        raise AppError(
            message="Regresyon modeli gecersiz sonuc uretti.",
            error_code="INVALID_REGRESSION_PREDICTION",
            status_code=500,
        )
    predicted_hours = max(0.0, raw_pred)
    reg_mean, reg_std = _compute_forest_std(loaded_bundle.regressor, feature_df)
    if reg_std is not None:
        predicted_hours_std = reg_std

    status = "success"
    predicted_at = datetime.now(UTC)

    risk_score = _to_risk_score(delay_prob)
    prediction = PredictionRun(
        process_id=process_id,
        snapshot_id=snapshot.id,
        model_bundle_id=loaded_bundle.bundle_id,
        model_version=loaded_bundle.bundle_record.model_version,
        prediction_type="normal",
        prediction_context="opening",
        status=status,
        delay_probability=delay_prob,
        risk_score=risk_score,
        risk_level=_to_risk_level(risk_score),
        predicted_is_delayed=(1 if is_delayed else 0) if is_delayed is not None and clf_available else None,
        predicted_hours=predicted_hours,
        input_fingerprint=input_fp,
        predicted_at=predicted_at,
    )
    session.add(prediction)

    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        reuse = _find_existing_success(
            session, snapshot.id, input_fp, loaded_bundle.bundle_id
        )
        if reuse is not None:
            return _to_result(reuse, input_json, reused=True, threshold=integration_threshold)
        raise

    return _to_result(
        prediction, input_json, reused=False, threshold=integration_threshold,
        delay_prob_std=delay_prob_std, predicted_hours_std=predicted_hours_std,
    )


def _get_opening_snapshot(session: Session, process_id: int) -> ProcessSnapshot | None:
    return session.execute(
        select(ProcessSnapshot).where(
            ProcessSnapshot.process_id == process_id,
            ProcessSnapshot.snapshot_type == "opening",
        )
    ).scalar_one_or_none()


def _find_existing_success(
    session: Session,
    snapshot_id: int,
    input_fingerprint: str | None,
    bundle_id: int,
) -> PredictionRun | None:
    return session.execute(
        select(PredictionRun).where(
            PredictionRun.snapshot_id == snapshot_id,
            PredictionRun.input_fingerprint == input_fingerprint,
            PredictionRun.model_bundle_id == bundle_id,
            PredictionRun.prediction_context == "opening",
            PredictionRun.status == "success",
        )
    ).scalars().first()


def _features_to_dataframe(features: dict[str, Any]) -> Any:
    import pandas as pd
    return pd.DataFrame([features])


def _has_sla(input_json: dict[str, Any]) -> bool:
    deadline = input_json.get("deadline")
    if deadline is None or str(deadline).strip() == "":
        return False
    return True


def _to_risk_score(delay_prob: float | None) -> int | None:
    if delay_prob is None:
        return None
    return min(100, max(0, round(delay_prob * 100)))


def _to_risk_level(risk_score: int | None) -> str | None:
    if risk_score is None:
        return None
    if risk_score <= 39:
        return "low"
    if risk_score <= 69:
        return "medium"
    return "high"


class PredictionResult:
    def __init__(
        self,
        prediction_run: PredictionRun,
        reused: bool,
        classification_available: bool,
        integration_threshold: float | None,
        regression_available: bool,
        delay_prob_std: float | None = None,
        predicted_hours_std: float | None = None,
    ):
        self.prediction_run = prediction_run
        self.reused = reused
        self.classification_available = classification_available
        self.integration_threshold = integration_threshold
        self.regression_available = regression_available
        self.delay_prob_std = delay_prob_std
        self.predicted_hours_std = predicted_hours_std


def _compute_forest_std(model, X) -> tuple[float | None, float | None]:
    try:
        estimator = model
        if hasattr(model, "named_steps"):
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, "estimators_"):
                estimator = last_step
        if not hasattr(estimator, "estimators_"):
            return None, None
        trees = estimator.estimators_
        if len(trees) < 2:
            return None, None
        try:
            tree_preds = np.array([float(t.predict_proba(X)[0, 1]) for t in trees])
        except (AttributeError, IndexError):
            tree_preds = np.array([float(t.predict(X)[0]) for t in trees])
        return float(np.mean(tree_preds)), float(np.std(tree_preds, ddof=1))
    except Exception:
        return None, None


def _to_result(prediction: PredictionRun, input_json: dict, reused: bool, threshold: float | None = None,
               delay_prob_std: float | None = None, predicted_hours_std: float | None = None) -> PredictionResult:
    clf_available = _has_sla(input_json) and prediction.delay_probability is not None
    return PredictionResult(
        prediction_run=prediction,
        reused=reused,
        classification_available=clf_available,
        integration_threshold=threshold,
        regression_available=prediction.predicted_hours is not None,
        delay_prob_std=delay_prob_std,
        predicted_hours_std=predicted_hours_std,
    )
