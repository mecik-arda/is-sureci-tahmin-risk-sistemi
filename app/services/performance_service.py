"""S32 Model Performance service — Validation metrics only."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.model_bundle import ModelBundle
from sqlalchemy import select


def get_model_performance(session: Session, bundle_id: int) -> dict[str, Any] | None:
    bundle = session.execute(
        select(ModelBundle).where(ModelBundle.id == bundle_id)
    ).scalars().first()
    if bundle is None:
        return None

    return {
        "bundle_id": bundle.id,
        "model_version": bundle.model_version,
        "model_type": bundle.model_type,
        "artifact_hash": (bundle.artifact_hash[:16] + "..." if bundle.artifact_hash else None),
        "trained_at": bundle.trained_at.isoformat() if bundle.trained_at else None,
        "is_active": bool(bundle.is_active),
    }
