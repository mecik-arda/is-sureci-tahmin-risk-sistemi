"""Süreç (process) veri erişim katmanı.

Kontrollü upsert kurallarını uygular:
    - Yalnız whitelist edilmiş outcome alanları güncellenebilir.
    - Opening snapshot immutable'dır.
    - source_payload_json değiştirilmez.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.process import Process, ProcessSnapshot

OPENING_FEATURE_COLUMNS = ["source", "subject", "reason", "type", "neighborhood"]


def find_by_external_id(session: Session, external_id: str) -> Process | None:
    """external_id'ye göre süreç kaydını bulur."""
    stmt = select(Process).where(Process.external_id == external_id)
    return session.execute(stmt).scalar_one_or_none()


def _build_snapshot_input(
    external_id: str,
    created_at: datetime,
    deadline: datetime | None,
    canonical_values: dict[str, str],
) -> str:
    """Snapshot input_json içeriğini kanonik kodlarla oluşturur."""
    payload: dict[str, Any] = {
        "external_id": external_id,
        "created_at": created_at.isoformat(),
        "deadline": deadline.isoformat() if deadline else None,
    }
    for col in OPENING_FEATURE_COLUMNS:
        payload[col] = canonical_values.get(col, "missing")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def insert_process(
    session: Session,
    external_id: str,
    process_type_canonical: str,
    created_at: datetime,
    deadline: datetime | None,
    completed_at: datetime | None,
    current_status: str | None,
    closure_reason: str | None,
    raw_payload: dict[str, object],
    canonical_values: dict[str, str],
    row_fingerprint: str,
    import_run_id: int,
    feature_schema_version: str,
) -> Process:
    """Yeni süreç kaydı ve opening snapshot oluşturur."""
    process = Process(
        external_id=external_id,
        process_type=process_type_canonical,
        current_status=current_status,
        created_at=created_at,
        deadline=deadline,
        completed_at=completed_at,
        source_payload_json=json.dumps(raw_payload, ensure_ascii=False, default=str),
        current_row_fingerprint=row_fingerprint,
        last_import_id=import_run_id,
        imported_at=datetime.now(UTC),
    )
    session.add(process)
    session.flush()

    input_json = _build_snapshot_input(external_id, created_at, deadline, canonical_values)
    from app.services.fingerprint import compute_opening_fingerprint

    opening_fields = {
        "external_id": external_id,
        "created_at": created_at.isoformat(),
        "deadline": deadline.isoformat() if deadline else None,
    }
    for col in OPENING_FEATURE_COLUMNS:
        opening_fields[col] = canonical_values.get(col, "missing")

    opening_fp = compute_opening_fingerprint(opening_fields)

    snapshot = ProcessSnapshot(
        process_id=process.id,
        snapshot_type="opening",
        snapshot_at=created_at,
        feature_schema_version=feature_schema_version,
        input_json=input_json,
        input_fingerprint=opening_fp,
        source_import_id=import_run_id,
    )
    session.add(snapshot)
    return process


def update_outcome_fields(
    session: Session,
    process: Process,
    completed_at: datetime | None,
    current_status: str | None,
    closure_reason: str | None,
    row_fingerprint: str,
    import_run_id: int,
) -> None:
    """Yalnız whitelist edilmiş outcome alanlarını günceller.

    Opening snapshot ve source_payload_json değiştirilmez.
    """
    process.completed_at = completed_at
    process.current_status = current_status
    process.closure_reason = closure_reason
    process.current_row_fingerprint = row_fingerprint
    process.last_import_id = import_run_id
    process.updated_at = datetime.now(UTC)
