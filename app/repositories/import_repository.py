"""Import run veri erişim katmanı."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.import_run import DataQualityIssue, ImportRun


def create_import_run(
    session: Session,
    file_name: str,
    file_hash: str,
    canonical_mapping_version: str,
) -> ImportRun:
    """Yeni bir import run kaydı oluşturur."""
    run = ImportRun(
        file_name=file_name,
        file_hash=file_hash,
        status="running",
        canonical_mapping_version=canonical_mapping_version,
        started_at=datetime.now(UTC),
    )
    session.add(run)
    session.flush()
    return run


def find_successful_by_hash(session: Session, file_hash: str) -> ImportRun | None:
    """Aynı SHA256 ile daha önce başarılı işlenmiş bir import run arar.

    Başarılı durumlar: completed, completed_with_issues
    duplicate_file ve failed durumları engel oluşturmaz.
    """
    stmt = select(ImportRun).where(
        ImportRun.file_hash == file_hash,
        ImportRun.status.in_(["completed", "completed_with_issues"]),
    )
    return session.execute(stmt).scalar_one_or_none()


def update_run_finalized(
    session: Session,
    run: ImportRun,
    status: str,
    total_rows: int,
    inserted_rows: int,
    updated_rows: int,
    skipped_duplicate_rows: int,
    quarantined_rows: int,
    error_rows: int,
    warning_count: int,
) -> None:
    """Import run kaydını nihai durumla günceller."""
    run.status = status
    run.total_rows = total_rows
    run.imported_rows = inserted_rows + updated_rows
    run.inserted_rows = inserted_rows
    run.updated_rows = updated_rows
    run.skipped_duplicate_rows = skipped_duplicate_rows
    run.quarantined_rows = quarantined_rows
    run.error_rows = error_rows
    run.warning_count = warning_count
    run.completed_at = datetime.now(UTC)


def record_quality_issue(
    session: Session,
    import_run_id: int,
    row_number: int | None,
    issue_code: str,
    severity: str,
    field_name: str | None,
    message: str,
) -> None:
    """Veri kalite sorunu kaydı oluşturur. Hassas ham değer yazılmaz."""
    issue = DataQualityIssue(
        import_run_id=import_run_id,
        row_number=row_number,
        column_name=field_name,
        issue_code=issue_code,
        issue_type=None,
        severity=severity,
        issue_message=message,
        raw_value=None,
    )
    session.add(issue)
