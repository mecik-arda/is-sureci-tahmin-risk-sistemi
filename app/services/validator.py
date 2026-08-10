"""Dosya ve satır doğrulama servisi.

Yapısal hatalar (eksik ID, parse edilemeyen tarih) error seviyesindedir
ve satırın karantinaya alınmasını gerektirir.

Bilinmeyen kategorik değer warning seviyesindedir ve satır reddedilmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.services.canonical_mapper import CanonicalMapper, MappingResult


COLUMN_MAP = {
    "case_enquiry_id": "external_id",
    "open_dt": "created_at",
    "closed_dt": "completed_at",
    "sla_target_dt": "deadline",
    "case_status": "current_status",
    "closure_reason": "closure_reason",
}

CATEGORICAL_COLUMNS = ["source", "subject", "reason", "type", "neighborhood", "department"]

REQUIRED_COLUMNS = {"case_enquiry_id", "open_dt"}


@dataclass
class ValidationIssue:
    """Tek bir doğrulama sorunu."""

    issue_code: str
    severity: str
    field_name: str | None
    message: str


@dataclass
class ParsedRow:
    """Doğrulanan ve parse edilen satır."""

    external_id: str
    created_at: datetime
    deadline: datetime | None
    completed_at: datetime | None
    current_status: str | None
    closure_reason: str | None
    raw_payload: dict[str, object]
    canonical_values: dict[str, str] = field(default_factory=dict)


def validate_columns(columns: list[str]) -> list[ValidationIssue]:
    """Dosya kolonlarını zorunlu kolonlara karşı doğrular."""
    issues: list[ValidationIssue] = []
    col_set = {c.strip() for c in columns}

    for required in REQUIRED_COLUMNS:
        if required not in col_set:
            issues.append(
                ValidationIssue(
                    issue_code="MISSING_REQUIRED_COLUMN",
                    severity="error",
                    field_name=required,
                    message=f"Zorunlu kolon eksik: {required}",
                )
            )

    return issues


def parse_date(value: object) -> tuple[datetime | None, bool]:
    """Tarih değerini parse eder.

    Returns:
        (datetime | None, success: bool)
    """
    if value is None or str(value).strip() == "":
        return None, True

    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt), True
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text), True
    except ValueError:
        return None, False


def validate_and_parse_row(
    row: dict[str, object],
    row_number: int,
    mapper: CanonicalMapper,
) -> tuple[ParsedRow | None, list[ValidationIssue]]:
    """Tek bir ham satırı doğrular ve parse eder.

    Returns:
        (ParsedRow | None, list[ValidationIssue])
        ParsedRow None ise satır karantinaya alınır (error seviyesi issue var).
    """
    issues: list[ValidationIssue] = []

    raw_external_id = row.get("case_enquiry_id")
    external_id = str(raw_external_id).strip() if raw_external_id is not None else ""
    if external_id == "":
        issues.append(
            ValidationIssue(
                issue_code="REQUIRED_FIELD_MISSING",
                severity="error",
                field_name="case_enquiry_id",
                message=f"Satir {row_number}: Zorunlu alan case_enquiry_id bos veya eksik.",
            )
        )
        return None, issues

    raw_open_dt = row.get("open_dt")
    created_at, open_ok = parse_date(raw_open_dt)
    if not open_ok:
        issues.append(
            ValidationIssue(
                issue_code="INVALID_DATE",
                severity="error",
                field_name="open_dt",
                message=f"Satir {row_number}: open_dt parse edilemedi.",
            )
        )
        return None, issues

    if created_at is None:
        issues.append(
            ValidationIssue(
                issue_code="REQUIRED_FIELD_MISSING",
                severity="error",
                field_name="open_dt",
                message=f"Satir {row_number}: Zorunlu alan open_dt bos.",
            )
        )
        return None, issues

    deadline, deadline_ok = parse_date(row.get("sla_target_dt"))
    if not deadline_ok:
        issues.append(
            ValidationIssue(
                issue_code="INVALID_DATE",
                severity="error",
                field_name="sla_target_dt",
                message=f"Satir {row_number}: sla_target_dt parse edilemedi.",
            )
        )
        return None, issues

    completed_at, completed_ok = parse_date(row.get("closed_dt"))
    if not completed_ok:
        issues.append(
            ValidationIssue(
                issue_code="INVALID_DATE",
                severity="error",
                field_name="closed_dt",
                message=f"Satir {row_number}: closed_dt parse edilemedi.",
            )
        )
        return None, issues

    current_status = row.get("case_status")
    current_status_str = str(current_status).strip() if current_status is not None else None
    if current_status_str == "":
        current_status_str = None

    closure_reason = row.get("closure_reason")
    closure_reason_str = str(closure_reason).strip() if closure_reason is not None else None
    if closure_reason_str == "":
        closure_reason_str = None

    canonical_values: dict[str, str] = {}
    for col in CATEGORICAL_COLUMNS:
        raw_val = row.get(col)
        result: MappingResult = mapper.map(col, str(raw_val) if raw_val is not None else None)
        canonical_values[col] = result.canonical_code
        if not result.is_known and not result.is_missing:
            issues.append(
                ValidationIssue(
                    issue_code="UNKNOWN_CANONICAL_MAPPING",
                    severity="warning",
                    field_name=col,
                    message=f"Satir {row_number}: Kolon '{col}' bilinmeyen deger iceriyor.",
                )
            )

    parsed = ParsedRow(
        external_id=external_id,
        created_at=created_at,
        deadline=deadline,
        completed_at=completed_at,
        current_status=current_status_str,
        closure_reason=closure_reason_str,
        raw_payload=dict(row),
        canonical_values=canonical_values,
    )

    return parsed, issues
