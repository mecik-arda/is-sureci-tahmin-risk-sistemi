"""İçe aktarma denetim günlüğü ve veri kalite sorunları.

`import_runs`: Her CSV/Excel yükleme işleminin kaydı.
`data_quality_issues`: İçe aktarma sırasında tespit edilen hatalı satırlar.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ImportRun(Base):
    """Dosya içe aktarma işlemi denetim kaydı."""

    __tablename__ = "import_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    canonical_mapping_version: Mapped[str | None] = mapped_column(String, nullable=True)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    imported_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_duplicate_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quarantined_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    quality_issues: Mapped[list[DataQualityIssue]] = relationship(
        "DataQualityIssue",
        back_populates="import_run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'completed_with_issues', "
            "'failed', 'duplicate_file')",
            name="ck_import_runs_status",
        ),
        CheckConstraint("total_rows >= 0", name="ck_import_runs_total_rows"),
        CheckConstraint("imported_rows >= 0", name="ck_import_runs_imported_rows"),
        CheckConstraint("inserted_rows >= 0", name="ck_import_runs_inserted_rows"),
        CheckConstraint("updated_rows >= 0", name="ck_import_runs_updated_rows"),
        CheckConstraint("skipped_duplicate_rows >= 0", name="ck_import_runs_skipped_duplicate_rows"),
        CheckConstraint("quarantined_rows >= 0", name="ck_import_runs_quarantined_rows"),
        CheckConstraint("error_rows >= 0", name="ck_import_runs_error_rows"),
        CheckConstraint("warning_count >= 0", name="ck_import_runs_warning_count"),
    )


class DataQualityIssue(Base):
    """İçe aktarma sırasında tespit edilen tek bir veri kalite sorunu."""

    __tablename__ = "data_quality_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    import_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("import_runs.id", ondelete="CASCADE"), nullable=False
    )
    row_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_name: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_type: Mapped[str | None] = mapped_column(String, nullable=True)
    issue_code: Mapped[str | None] = mapped_column(String, nullable=True)
    severity: Mapped[str] = mapped_column(String, nullable=False, default="warning")
    issue_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_value: Mapped[str | None] = mapped_column(Text, nullable=True)

    import_run: Mapped[ImportRun] = relationship("ImportRun", back_populates="quality_issues")

    __table_args__ = (
        CheckConstraint(
            "severity IN ('warning', 'error')",
            name="ck_data_quality_issues_severity",
        ),
        CheckConstraint(
            "issue_code IS NULL OR issue_code IN ('DUPLICATE_FILE', "
            "'UNKNOWN_CANONICAL_MAPPING', 'REQUIRED_FIELD_MISSING', 'INVALID_DATE', "
            "'INVALID_FILE_SCHEMA', 'OPENING_DATA_CONFLICT', "
            "'CANONICAL_MAPPING_MISMATCH', 'UNSUPPORTED_FILE_FORMAT', "
            "'MISSING_REQUIRED_COLUMN')",
            name="ck_data_quality_issues_issue_code",
        ),
        Index("ix_data_quality_issues_import_run_id", "import_run_id"),
    )
