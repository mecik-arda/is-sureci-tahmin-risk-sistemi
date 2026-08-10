from __future__ import annotations

from typing import Any

from app.services.analysis_dataset import AnalysisDatasetService


def get_model_monitoring_data(bundle: Any, analysis_service: AnalysisDatasetService) -> dict[str, Any]:
    if bundle is None:
        return {"available": False}

    metadata = bundle.metadata or {}
    record = bundle.bundle_record
    return {
        "available": True,
        "model_version": record.model_version,
        "model_type": record.model_type,
        "trained_at": record.trained_at.isoformat() if record.trained_at else None,
        "artifact_hash": bundle.artifact_hash,
        "stage": metadata.get("stage"),
        "feature_schema_version": metadata.get("feature_schema_version"),
        "canonical_mapping_version": metadata.get("canonical_mapping_version"),
        "threshold": bundle.threshold,
        "analysis_cache": {
            "built_at": analysis_service.built_at,
            "build_count": analysis_service.build_count,
            "cache_hit_count": analysis_service.cache_hit_count,
            "cached_rows": analysis_service.cached_rows(),
            "cached_memory_bytes": analysis_service.cached_memory_bytes(),
        },
    }
