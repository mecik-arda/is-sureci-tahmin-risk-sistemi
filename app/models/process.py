"""Süreç kayıtları ve snapshot tabloları.

`processes`: Sürecin görece sabit bilgileri ve ham kaynak payload.
`process_snapshots`: Modele verilen girdinin denetlenebilir kopyası.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Process(Base):
    """Süreç kaydı.

    external_id Boston 311 verisindeki `case_enquiry_id` değerini taşır.
    source_payload_json, kaynak dosyadaki bilinmeyen kolonları korunaklı
    şekilde saklar; ancak iş kuralları bu alandan kontrolsüz veri çekmez.
    """

    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    process_type: Mapped[str | None] = mapped_column(String, nullable=True)
    current_status: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    deadline: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closure_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    source_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_row_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    last_import_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("import_runs.id", ondelete="SET NULL"), nullable=True
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    snapshots: Mapped[list[ProcessSnapshot]] = relationship(
        "ProcessSnapshot",
        back_populates="process",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list[PredictionRun]] = relationship(
        "PredictionRun",
        back_populates="process",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_processes_external_id", "external_id"),
    )


class ProcessSnapshot(Base):
    """Modele verilen girdinin tam kopyası (denetlenebilirlik).

    V1'de snapshot_type daima 'opening' ve snapshot_at = created_at'tir.
    """

    __tablename__ = "process_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    process_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processes.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_type: Mapped[str] = mapped_column(String, nullable=False, default="opening")
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String, nullable=True)
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    source_import_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("import_runs.id", ondelete="SET NULL"), nullable=True
    )

    process: Mapped[Process] = relationship("Process", back_populates="snapshots")

    __table_args__ = (
        CheckConstraint(
            "snapshot_type IN ('opening')",
            name="ck_process_snapshots_snapshot_type",
        ),
        Index("ix_process_snapshots_process_id", "process_id"),
    )
