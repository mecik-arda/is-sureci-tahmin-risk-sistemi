"""ML model versiyonları (model registry).

`model_bundles`: Eğitilen modellerin artifact yolu, metrikleri ve
özellik listesi. Aynı anda yalnızca bir model `is_active=1` olabilir.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ModelBundle(Base):
    """Eğitilmiş bir model paketinin (bundle) kayıt sürümü."""

    __tablename__ = "model_bundles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    artifact_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_list_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    predictions: Mapped[list[Any]] = relationship("PredictionRun", back_populates="model_bundle")

    __table_args__ = (
        CheckConstraint("is_active IN (0, 1)", name="ck_model_bundles_is_active"),
        CheckConstraint(
            "model_type IN ('classifier', 'regressor', 'bundle')",
            name="ck_model_bundles_model_type",
        ),
    )
