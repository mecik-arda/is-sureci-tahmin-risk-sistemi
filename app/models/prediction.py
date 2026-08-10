"""Tahmin logları ve geri bildirim.

`prediction_runs`: Her tahmin/simülasyon işleminin denetim kaydı.
`prediction_feedback`: Kullanıcı geri bildirimleri (doğruluk ve kullanışlılık).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class PredictionRun(Base):
    """Tek bir tahmin veya simülasyon işleminin kaydı."""

    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    process_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processes.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("process_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    model_bundle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_bundles.id"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    prediction_type: Mapped[str] = mapped_column(String, nullable=False, default="normal")
    prediction_context: Mapped[str] = mapped_column(String, nullable=False, default="opening")
    status: Mapped[str] = mapped_column(String, nullable=False)
    delay_probability: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_hours: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    predicted_is_delayed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    simulation_overrides_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    process: Mapped[Any] = relationship("Process", back_populates="predictions")
    model_bundle: Mapped[Any] = relationship("ModelBundle", back_populates="predictions")
    feedback: Mapped[list[PredictionFeedback]] = relationship(
        "PredictionFeedback",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "prediction_type IN ('normal', 'simulation')",
            name="ck_prediction_runs_prediction_type",
        ),
        CheckConstraint(
            "status IN ('success', 'failed')",
            name="ck_prediction_runs_status",
        ),
        CheckConstraint(
            "risk_level IS NULL OR risk_level IN ('low', 'medium', 'high')",
            name="ck_prediction_runs_risk_level",
        ),
        CheckConstraint(
            "risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)",
            name="ck_prediction_runs_risk_score",
        ),
        Index("ix_prediction_runs_process_id", "process_id"),
    )


class PredictionFeedback(Base):
    """Bir tahmine verilmiş kullanıcı geri bildirimi."""

    __tablename__ = "prediction_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("prediction_runs.id", ondelete="CASCADE"), nullable=False
    )
    feedback_type: Mapped[str] = mapped_column(String, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    prediction: Mapped[PredictionRun] = relationship("PredictionRun", back_populates="feedback")

    __table_args__ = (
        CheckConstraint(
            "feedback_type IN ('accuracy', 'usefulness')",
            name="ck_prediction_feedback_feedback_type",
        ),
        CheckConstraint(
            "actual_outcome IS NULL OR actual_outcome IN (0, 1)",
            name="ck_prediction_feedback_actual_outcome",
        ),
        Index("ix_prediction_feedback_prediction_id", "prediction_id"),
        UniqueConstraint(
            "prediction_id", "feedback_type",
            name="uq_prediction_feedback_prediction_id_feedback_type",
        ),
    )
