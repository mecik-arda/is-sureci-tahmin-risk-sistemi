"""Import orkestrasyon servisi.

Dosya okuma → doğrulama → kanonik mapping → fingerprint → kontrollü upsert
akışını yönetir. Transaction sınırları bu servis tarafından belirlenir.

Kurallar:
    - Per-row commit yapılmaz.
    - Conflict satırları karantinaya alınır, diğerleri işlenir.
    - Tüm durumlar import_runs ve data_quality_issues içinde denetlenebilir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.repositories import import_repository, process_repository
from app.services.canonical_mapper import CanonicalMapper
from app.services.fingerprint import (
    compute_file_fingerprint,
    compute_opening_fingerprint,
    compute_row_fingerprint,
)
from app.services.file_reader import FileReadError, UnsupportedFileFormatError, read_data_file
from app.services.validator import (
    CATEGORICAL_COLUMNS,
    ParsedRow,
    ValidationIssue,
    parse_date,
    validate_and_parse_row,
    validate_columns,
)

FEATURE_SCHEMA_VERSION = "opening-v1"

OPENING_FIELD_NAMES = [
    "external_id",
    "created_at",
    "deadline",
    "source",
    "subject",
    "reason",
    "type",
    "neighborhood",
]


@dataclass
class ImportCounts:
    """Import sayaçları."""

    total_rows: int = 0
    inserted_rows: int = 0
    updated_rows: int = 0
    skipped_duplicate_rows: int = 0
    quarantined_rows: int = 0
    error_rows: int = 0
    warning_count: int = 0


@dataclass
class ImportResult:
    """Import işleminin nihai sonucu."""

    status: str
    counts: ImportCounts
    file_hash: str
    import_run_id: int
    errors: list[str] = field(default_factory=list)


def _build_opening_fields(parsed: ParsedRow) -> dict:
    """Opening fingerprint için alanları hazırlar."""
    fields = {
        "external_id": parsed.external_id,
        "created_at": parsed.created_at.isoformat(),
        "deadline": parsed.deadline.isoformat() if parsed.deadline else None,
    }
    for col in ["source", "subject", "reason", "type", "neighborhood"]:
        fields[col] = parsed.canonical_values.get(col, "missing")
    return fields


def _build_outcome_fields(parsed: ParsedRow) -> dict:
    """Outcome fingerprint için alanları hazırlar."""
    return {
        "completed_at": parsed.completed_at.isoformat() if parsed.completed_at else None,
        "current_status": parsed.current_status,
        "closure_reason": parsed.closure_reason,
    }


def _has_error(issues: list[ValidationIssue]) -> bool:
    """Hata seviyesinde issue var mı?"""
    return any(i.severity == "error" for i in issues)


def run_import(
    session: Session,
    file_path: str | Path,
    mapper: CanonicalMapper,
) -> ImportResult:
    """Tam import akışını çalıştırır.

    Args:
        session: Açık SQLAlchemy session.
        file_path: İçe aktarılacak dosya yolu.
        mapper: Yüklü canonical mapper.

    Returns:
        ImportResult: Import işleminin özet sonucu.
    """
    path = Path(file_path)
    counts = ImportCounts()

    file_hash = compute_file_fingerprint(path)

    existing_run = import_repository.find_successful_by_hash(session, file_hash)
    if existing_run is not None:
        run = import_repository.create_import_run(
            session, path.name, file_hash, mapper.version
        )
        import_repository.update_run_finalized(
            session, run,
            status="duplicate_file",
            total_rows=0, inserted_rows=0, updated_rows=0,
            skipped_duplicate_rows=0, quarantined_rows=0,
            error_rows=0, warning_count=0,
        )
        session.flush()
        return ImportResult(
            status="duplicate_file",
            counts=counts,
            file_hash=file_hash,
            import_run_id=run.id,
        )

    try:
        read_result = read_data_file(path)
    except UnsupportedFileFormatError as exc:
        run = import_repository.create_import_run(
            session, path.name, file_hash, mapper.version
        )
        import_repository.record_quality_issue(
            session, run.id, None, "UNSUPPORTED_FILE_FORMAT", "error",
            None, str(exc),
        )
        import_repository.update_run_finalized(
            session, run,
            status="failed",
            total_rows=0, inserted_rows=0, updated_rows=0,
            skipped_duplicate_rows=0, quarantined_rows=1,
            error_rows=1, warning_count=0,
        )
        session.flush()
        return ImportResult(
            status="failed", counts=counts, file_hash=file_hash,
            import_run_id=run.id, errors=[str(exc)],
        )
    except FileReadError as exc:
        run = import_repository.create_import_run(
            session, path.name, file_hash, mapper.version
        )
        import_repository.record_quality_issue(
            session, run.id, None, "INVALID_FILE_SCHEMA", "error",
            None, str(exc),
        )
        import_repository.update_run_finalized(
            session, run,
            status="failed",
            total_rows=0, inserted_rows=0, updated_rows=0,
            skipped_duplicate_rows=0, quarantined_rows=1,
            error_rows=1, warning_count=0,
        )
        session.flush()
        return ImportResult(
            status="failed", counts=counts, file_hash=file_hash,
            import_run_id=run.id, errors=[str(exc)],
        )

    column_issues = validate_columns(read_result.columns)
    if column_issues:
        run = import_repository.create_import_run(
            session, path.name, file_hash, mapper.version
        )
        for issue in column_issues:
            import_repository.record_quality_issue(
                session, run.id, None, issue.issue_code, issue.severity,
                issue.field_name, issue.message,
            )
        import_repository.update_run_finalized(
            session, run,
            status="failed",
            total_rows=0, inserted_rows=0, updated_rows=0,
            skipped_duplicate_rows=0, quarantined_rows=len(column_issues),
            error_rows=len(column_issues), warning_count=0,
        )
        session.flush()
        return ImportResult(
            status="failed", counts=counts, file_hash=file_hash,
            import_run_id=run.id,
            errors=[i.message for i in column_issues],
        )

    run = import_repository.create_import_run(
        session, path.name, file_hash, mapper.version
    )

    counts.total_rows = read_result.row_count

    for idx, raw_row in enumerate(read_result.rows, start=1):
        parsed, issues = validate_and_parse_row(raw_row, idx, mapper)

        for issue in issues:
            if issue.severity == "warning":
                import_repository.record_quality_issue(
                    session, run.id, idx, issue.issue_code, issue.severity,
                    issue.field_name, issue.message,
                )
                counts.warning_count += 1

        if parsed is None or _has_error(issues):
            for issue in issues:
                if issue.severity == "error":
                    import_repository.record_quality_issue(
                        session, run.id, idx, issue.issue_code, issue.severity,
                        issue.field_name, issue.message,
                    )
            counts.quarantined_rows += 1
            counts.error_rows += 1
            continue

        opening_fields = _build_opening_fields(parsed)
        outcome_fields = _build_outcome_fields(parsed)
        opening_fp = compute_opening_fingerprint(opening_fields)
        row_fp = compute_row_fingerprint(opening_fields, outcome_fields)

        existing = process_repository.find_by_external_id(session, parsed.external_id)

        if existing is None:
            process_repository.insert_process(
                session,
                external_id=parsed.external_id,
                process_type_canonical=parsed.canonical_values.get("type", "unknown"),
                created_at=parsed.created_at,
                deadline=parsed.deadline,
                completed_at=parsed.completed_at,
                current_status=parsed.current_status,
                closure_reason=parsed.closure_reason,
                raw_payload=parsed.raw_payload,
                canonical_values=parsed.canonical_values,
                row_fingerprint=row_fp,
                import_run_id=run.id,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
            counts.inserted_rows += 1

        else:
            existing_run = _get_import_run_for_process(session, existing.last_import_id)
            if (
                existing_run is not None
                and existing_run.canonical_mapping_version != mapper.version
            ):
                import_repository.record_quality_issue(
                    session, run.id, idx,
                    "CANONICAL_MAPPING_MISMATCH", "error",
                    None,
                    f"Satir {idx}: Mevcut kayit farkli mapping surumu kullaniyor.",
                )
                counts.quarantined_rows += 1
                counts.error_rows += 1
                continue

            if existing.current_row_fingerprint == row_fp:
                counts.skipped_duplicate_rows += 1
                continue

            existing_snapshot = _get_opening_snapshot(session, existing.id)

            if existing_snapshot is not None and existing_snapshot.input_fingerprint == opening_fp:
                process_repository.update_outcome_fields(
                    session,
                    existing,
                    completed_at=parsed.completed_at,
                    current_status=parsed.current_status,
                    closure_reason=parsed.closure_reason,
                    row_fingerprint=row_fp,
                    import_run_id=run.id,
                )
                counts.updated_rows += 1
            else:
                import_repository.record_quality_issue(
                    session, run.id, idx,
                    "OPENING_DATA_CONFLICT", "error",
                    None,
                    f"Satir {idx}: Açılış verileri mevcut kayitle celisiyor.",
                )
                counts.quarantined_rows += 1
                counts.error_rows += 1

    if counts.error_rows > 0 and counts.inserted_rows + counts.updated_rows > 0:
        final_status = "completed_with_issues"
    elif counts.error_rows > 0:
        final_status = "failed"
    else:
        final_status = "completed"

    import_repository.update_run_finalized(
        session, run,
        status=final_status,
        total_rows=counts.total_rows,
        inserted_rows=counts.inserted_rows,
        updated_rows=counts.updated_rows,
        skipped_duplicate_rows=counts.skipped_duplicate_rows,
        quarantined_rows=counts.quarantined_rows,
        error_rows=counts.error_rows,
        warning_count=counts.warning_count,
    )

    session.flush()

    return ImportResult(
        status=final_status,
        counts=counts,
        file_hash=file_hash,
        import_run_id=run.id,
    )


def _get_opening_snapshot(session: Session, process_id: int):
    """Süreç için opening snapshot'ını getirir."""
    from app.models.process import ProcessSnapshot
    from sqlalchemy import select

    stmt = select(ProcessSnapshot).where(
        ProcessSnapshot.process_id == process_id,
        ProcessSnapshot.snapshot_type == "opening",
    )
    return session.execute(stmt).scalar_one_or_none()


def _get_import_run_for_process(session: Session, import_run_id: int | None):
    """Süreç ile ilişkili import run'u getirir."""
    if import_run_id is None:
        return None
    from app.models.import_run import ImportRun
    from sqlalchemy import select

    stmt = select(ImportRun).where(ImportRun.id == import_run_id)
    return session.execute(stmt).scalar_one_or_none()
