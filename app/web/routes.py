"""Faz 6 API endpoints for frontend data consumption.

S22-S29 compliant: prediction/actual separation, single bundle scope,
no fake values, proper empty states.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.runtime import runtime_state
from app.schemas.interaction import FeedbackRequest, SimulationRequest
from app.services.analysis_dataset import analysis_dataset_service
from app.services.dashboard_service import get_dashboard_data
from app.services.process_service import (
    get_process_detail,
    get_prediction_history,
    get_process_list,
    get_process_prediction,
)

router = APIRouter(tags=["faz6"])

_dashboard_cache: TTLCache = TTLCache(maxsize=1, ttl=60)
_perf_cache: TTLCache = TTLCache(maxsize=1, ttl=60)


def _cache_bundle_version() -> str:
    b = runtime_state.bundle
    if b is None:
        return "none"
    bid = str(b.bundle_id)
    ver = b.bundle_record.model_version or "0"
    t = str(b.threshold)
    try:
        mhash = str(hash(b.bundle_record.metrics_json or ""))
    except Exception:
        mhash = "0"
    return bid + ":" + ver + ":" + t + ":" + mhash


@lru_cache(maxsize=1)
def _simulation_options() -> dict[str, list[str]]:
    mapping_path = Path(__file__).resolve().parent.parent.parent / "ml" / "mappings" / "canonical_map_v1.json"
    columns = json.loads(mapping_path.read_text(encoding="utf-8"))["columns"]
    return {
        column: sorted(set(values.values()) | {"missing", "unknown"})
        for column, values in columns.items()
        if column in {"source", "subject", "reason", "type", "neighborhood"}
    }


def _model_metrics_metadata(bundle) -> dict:
    metadata = dict(bundle.metadata or {})
    try:
        stored_metrics = json.loads(bundle.bundle_record.metrics_json or "{}")
    except json.JSONDecodeError:
        return metadata
    for section in ("classifier", "regression"):
        if section in stored_metrics:
            metadata[section] = {
                **metadata.get(section, {}),
                **stored_metrics[section],
            }
    return metadata


def _get_bundle_id() -> int | None:
    """Returns active bundle_id or None if degraded."""
    if runtime_state.bundle is not None:
        return runtime_state.bundle.bundle_id
    return None


def _get_threshold() -> float:
    """Returns bundle threshold (S25: backend owns threshold)."""
    if runtime_state.bundle is not None:
        return runtime_state.bundle.threshold
    return 0.5


def _get_banner_data() -> dict:
    """S28: banner data for every page."""
    from app.core.config import get_settings
    settings = get_settings()
    bundle_stage = "integration_baseline"
    model_version = None
    threshold = 0.5
    calibration_method = None
    if runtime_state.bundle is not None:
        metadata = runtime_state.bundle.metadata
        bundle_stage = metadata.get("stage", "integration_baseline")
        model_version = runtime_state.bundle.bundle_record.model_version
        threshold = runtime_state.bundle.threshold
        clf_meta = metadata.get("classifier", {})
        calibration_method = clf_meta.get("calibration_method")
    return {
        "app_mode": settings.app_mode.value,
        "model_available": runtime_state.model_available,
        "bundle_stage": bundle_stage,
        "model_version": model_version,
        "threshold": threshold,
        "calibration_method": calibration_method,
    }


@router.get("/banner")
def banner_data():
    """S28: Global banner data."""
    return _get_banner_data()


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    """S23: Dashboard with prediction/actual separation."""
    cache_key = "dash:" + _cache_bundle_version()
    if cache_key in _dashboard_cache:
        return _dashboard_cache[cache_key]
    bundle_id = _get_bundle_id()
    data = get_dashboard_data(db, bundle_id)
    data["banner"] = _get_banner_data()
    _dashboard_cache[cache_key] = data
    return data


@router.get("/processes")
def process_list(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    risk: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """S24: Process list with single bundle scope."""
    bundle_id = _get_bundle_id()
    data = get_process_list(db, bundle_id, page, per_page, status, risk)
    data["banner"] = _get_banner_data()
    return data


@router.get("/processes/{process_id}")
def process_detail(process_id: int, db: Session = Depends(get_db)):
    """S22: Process detail with opening/current dual card."""
    from app.core.errors import ProcessNotFoundError
    detail = get_process_detail(db, process_id)
    if detail is None:
        raise ProcessNotFoundError()
    bundle_id = _get_bundle_id()
    detail["prediction"] = get_process_prediction(db, process_id, bundle_id)
    detail["banner"] = _get_banner_data()
    return detail


@router.get("/processes/{process_id}/prediction-history")
def prediction_history(
    process_id: int,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    from app.core.errors import ProcessNotFoundError

    if get_process_detail(db, process_id) is None:
        raise ProcessNotFoundError()
    return {"predictions": get_prediction_history(db, process_id, limit)}


@router.get("/processes/{process_id}/similar")
def similar_processes(process_id: int, db: Session = Depends(get_db)):
    """S27: Similar processes via KNN on opening-v1 features."""
    if runtime_state.bundle is None:
        return {"neighbors": [], "available": False}

    return analysis_dataset_service.get_similar_processes_for_process(
        db, process_id, runtime_state.bundle,
    )


@router.get("/processes/{process_id}/xai")
def process_xai(process_id: int, db: Session = Depends(get_db)):
    """S26: Global feature importance + per-instance SHAP for the active model."""
    return analysis_dataset_service.get_xai(db, runtime_state.bundle, process_id=process_id)


@router.get("/model-performance")
def model_performance(db: Session = Depends(get_db)):
    """S32: Model performance — Validation metrics only."""
    cache_key = "perf:" + _cache_bundle_version()
    if cache_key in _perf_cache:
        return _perf_cache[cache_key]
    if runtime_state.bundle is None:
        return {"available": False}

    bundle = runtime_state.bundle
    metadata = _model_metrics_metadata(bundle)
    clf_meta = metadata.get("classifier", {})
    reg_meta = metadata.get("regression", {})

    result = {
        "available": True,
        "bundle": {
            "model_version": bundle.bundle_record.model_version,
            "model_type": bundle.bundle_record.model_type,
            "stage": metadata.get("stage"),
            "threshold": bundle.threshold,
            "calibration_method": clf_meta.get("calibration_method"),
        },
        "classification": {
            "validation_pr_auc": clf_meta.get("validation_pr_auc"),
            "validation_roc_auc": clf_meta.get("validation_roc_auc"),
            "validation_brier": clf_meta.get("validation_brier"),
            "validation_f1": clf_meta.get("validation_f1"),
            "validation_precision": clf_meta.get("validation_precision"),
            "validation_recall": clf_meta.get("validation_recall"),
            "classification_validation_row_count": clf_meta.get("classification_validation_row_count"),
            "validation_confusion_matrix": clf_meta.get("validation_confusion_matrix"),
            "cv_pr_auc_mean": clf_meta.get("cv_pr_auc_mean"),
            "cv_pr_auc_std": clf_meta.get("cv_pr_auc_std"),
            "cv_roc_auc_mean": clf_meta.get("cv_roc_auc_mean"),
            "cv_roc_auc_std": clf_meta.get("cv_roc_auc_std"),
        },
        "regression": {
            "validation_mae": reg_meta.get("validation_mae"),
            "validation_median_ae": reg_meta.get("validation_median_ae"),
            "validation_rmse": reg_meta.get("validation_rmse"),
            "validation_p90_ae": reg_meta.get("validation_p90_ae"),
            "regression_validation_row_count": reg_meta.get("regression_validation_row_count"),
            "cv_mae_mean": reg_meta.get("cv_mae_mean"),
            "cv_mae_std": reg_meta.get("cv_mae_std"),
            "cv_rmse_mean": reg_meta.get("cv_rmse_mean"),
            "cv_rmse_std": reg_meta.get("cv_rmse_std"),
        },
        "banner": _get_banner_data(),
    }
    _perf_cache[cache_key] = result
    return result


@router.post("/processes/{process_id}/simulations")
def create_simulation(
    process_id: int,
    request: SimulationRequest,
    db: Session = Depends(get_db),
):
    """S30: Run what-if simulation on opening-v1 features."""
    if runtime_state.bundle is None:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError()

    from app.services.simulation_service import run_simulation
    try:
        sim = run_simulation(
            db,
            process_id,
            request.base_prediction_id,
            request.overrides.model_dump(exclude_none=True),
            runtime_state.bundle,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "simulation_id": sim.id,
        "delay_probability": float(sim.delay_probability) if sim.delay_probability is not None else None,
        "predicted_is_delayed": sim.predicted_is_delayed == 1 if sim.predicted_is_delayed is not None else None,
        "predicted_hours": float(sim.predicted_hours) if sim.predicted_hours is not None else None,
        "threshold": runtime_state.bundle.threshold,
    }


@router.post("/predictions/{prediction_id}/feedback")
def create_feedback(
    prediction_id: int,
    request: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """S31: Submit accuracy or usefulness feedback."""
    from app.services.feedback_service import submit_feedback

    try:
        fb = submit_feedback(db, prediction_id, request.feedback_type, request.comment)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "feedback_id": fb.id,
        "feedback_type": fb.feedback_type,
        "actual_outcome": fb.actual_outcome,
        "comment": fb.comment,
    }


@router.get("/simulation-options")
def simulation_options():
    return _simulation_options()


@router.get("/model-monitoring")
def model_monitoring():
    from app.services.model_monitoring_service import get_model_monitoring_data

    return get_model_monitoring_data(runtime_state.bundle, analysis_dataset_service)


@router.get("/data-quality")
def data_quality(db: Session = Depends(get_db)):
    from app.services.data_quality_service import get_data_quality_summary

    return get_data_quality_summary(db)


@router.get("/label-catalog")
def label_catalog(request: Request):
    """Serve label_catalog_v1.json from app.state (lifespan cache)."""
    catalog = getattr(request.app.state, "label_catalog", None)
    if catalog:
        return JSONResponse(content=catalog)
    return JSONResponse(content={}, status_code=404)
