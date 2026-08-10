"""Import service upsert ve idempotency testleri."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.import_run import DataQualityIssue, ImportRun
from app.models.process import Process, ProcessSnapshot
from app.services.canonical_mapper import CanonicalMapper
from app.services.import_service import run_import


def _make_row(
    case_enquiry_id: str = "10100-001",
    open_dt: str = "2024-01-15 10:00:00",
    closed_dt: str = "",
    sla_target_dt: str = "2024-01-20 17:00:00",
    case_status: str = "Open",
    source: str = "Citizens Connect App",
    subject: str = "Public Works Department",
    reason: str = "Highway Maintenance",
    type: str = "Request for Pothole Repair",
    neighborhood: str = "Roxbury",
    closure_reason: str = "",
) -> dict:
    return {
        "case_enquiry_id": case_enquiry_id,
        "open_dt": open_dt,
        "closed_dt": closed_dt,
        "sla_target_dt": sla_target_dt,
        "case_status": case_status,
        "source": source,
        "subject": subject,
        "reason": reason,
        "type": type,
        "neighborhood": neighborhood,
        "closure_reason": closure_reason,
    }


def test_import_inserts_new_process(db_session: Session, make_csv, canonical_mapper: CanonicalMapper):
    """Geçerli CSV yeni süreç kaydı oluşturuyor."""
    csv_path = make_csv([_make_row()])
    result = run_import(db_session, csv_path, canonical_mapper)

    assert result.status == "completed"
    assert result.counts.inserted_rows == 1
    assert result.counts.total_rows == 1

    processes = db_session.execute(select(Process)).scalars().all()
    assert len(processes) == 1
    assert processes[0].external_id == "10100-001"
    assert processes[0].process_type == "request_for_pothole_repair"


def test_import_creates_opening_snapshot(db_session: Session, make_csv, canonical_mapper):
    """Import opening snapshot oluşturuyor."""
    csv_path = make_csv([_make_row()])
    run_import(db_session, csv_path, canonical_mapper)

    snapshots = db_session.execute(select(ProcessSnapshot)).scalars().all()
    assert len(snapshots) == 1
    assert snapshots[0].snapshot_type == "opening"
    assert snapshots[0].input_fingerprint is not None


def test_duplicate_file_skipped(db_session: Session, make_csv, canonical_mapper):
    """Aynı dosyanın iki kez yüklenmesi kayıt sayısını değiştirmiyor."""
    rows = [_make_row(), _make_row(case_enquiry_id="10100-002")]
    csv_path = make_csv(rows)

    result1 = run_import(db_session, csv_path, canonical_mapper)
    assert result1.counts.inserted_rows == 2

    result2 = run_import(db_session, csv_path, canonical_mapper)
    assert result2.status == "duplicate_file"
    assert result2.counts.total_rows == 0

    processes = db_session.execute(select(Process)).scalars().all()
    assert len(processes) == 2


def test_same_row_skipped(db_session: Session, make_csv, canonical_mapper):
    """Aynı satırın tekrar yüklenmesi no-op oluyor (farklı dosya)."""
    row = _make_row()
    csv1 = make_csv([row], "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    row2 = _make_row(case_enquiry_id="10100-OTHER")
    csv2 = make_csv([row, row2], "file2.csv")
    result2 = run_import(db_session, csv2, canonical_mapper)

    assert result2.counts.skipped_duplicate_rows == 1
    assert result2.counts.inserted_rows == 1

    processes = db_session.execute(select(Process)).scalars().all()
    assert len(processes) == 2


def test_outcome_update_open_to_closed(db_session: Session, make_csv, canonical_mapper):
    """Açık süreç sonradan Closed geldiğinde yalnız outcome alanları güncelleniyor."""
    csv1 = make_csv([_make_row(case_status="Open", closed_dt="")], "open.csv")
    result1 = run_import(db_session, csv1, canonical_mapper)
    assert result1.counts.inserted_rows == 1

    process = db_session.execute(select(Process)).scalars().first()
    assert process.current_status == "Open"
    assert process.completed_at is None
    original_payload = process.source_payload_json

    csv2 = make_csv(
        [_make_row(case_status="Closed", closed_dt="2024-01-18 14:00:00")],
        "closed.csv",
    )
    result2 = run_import(db_session, csv2, canonical_mapper)
    assert result2.counts.updated_rows == 1

    db_session.refresh(process)
    assert process.current_status == "Closed"
    assert process.completed_at is not None
    assert process.source_payload_json == original_payload


def test_opening_snapshot_immutable(db_session: Session, make_csv, canonical_mapper):
    """Opening snapshot güncellemeyle değişmiyor."""
    csv1 = make_csv([_make_row()], "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    snapshot = db_session.execute(select(ProcessSnapshot)).scalars().first()
    original_input_json = snapshot.input_json
    original_fp = snapshot.input_fingerprint

    csv2 = make_csv(
        [_make_row(case_status="Closed", closed_dt="2024-01-18 14:00:00")],
        "file2.csv",
    )
    run_import(db_session, csv2, canonical_mapper)

    db_session.refresh(snapshot)
    assert snapshot.input_json == original_input_json
    assert snapshot.input_fingerprint == original_fp


def test_source_payload_immutable(db_session: Session, make_csv, canonical_mapper):
    """İlk source_payload_json güncellemeyle değişmiyor."""
    csv1 = make_csv([_make_row()], "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    process = db_session.execute(select(Process)).scalars().first()
    original_payload = process.source_payload_json

    csv2 = make_csv(
        [_make_row(case_status="Closed", closed_dt="2024-01-18 14:00:00")],
        "file2.csv",
    )
    run_import(db_session, csv2, canonical_mapper)

    db_session.refresh(process)
    assert process.source_payload_json == original_payload


def test_opening_data_conflict_quarantined(db_session: Session, make_csv, canonical_mapper):
    """Opening alanı değiştiğinde OPENING_DATA_CONFLICT oluşuyor."""
    csv1 = make_csv([_make_row(type="Request for Pothole Repair")], "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    csv2 = make_csv([_make_row(type="Graffiti Removal")], "file2.csv")
    result2 = run_import(db_session, csv2, canonical_mapper)

    assert result2.counts.quarantined_rows == 1

    issues = db_session.execute(
        select(DataQualityIssue).where(DataQualityIssue.issue_code == "OPENING_DATA_CONFLICT")
    ).scalars().all()
    assert len(issues) == 1


def test_conflict_does_not_lose_valid_rows(db_session: Session, make_csv, canonical_mapper):
    """Conflict bulunan bir satır diğer geçerli satırları kaybettirmiyor."""
    rows = [
        _make_row(case_enquiry_id="CONFLICT-001", type="Request for Pothole Repair"),
        _make_row(case_enquiry_id="VALID-001", type="Request for Pothole Repair"),
    ]
    csv1 = make_csv(rows, "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    rows2 = [
        _make_row(case_enquiry_id="CONFLICT-001", type="Graffiti Removal"),
        _make_row(case_enquiry_id="VALID-002", type="Request for Pothole Repair"),
    ]
    csv2 = make_csv(rows2, "file2.csv")
    result2 = run_import(db_session, csv2, canonical_mapper)

    assert result2.counts.quarantined_rows == 1
    assert result2.counts.inserted_rows == 1
    assert result2.status == "completed_with_issues"


def test_unknown_category_accepted_with_warning(db_session: Session, make_csv, canonical_mapper):
    """Bilinmeyen kategori warning üretip satırı kabul ediyor."""
    row = _make_row(source="Some New Unknown Source")
    csv_path = make_csv([row])
    result = run_import(db_session, csv_path, canonical_mapper)

    assert result.counts.inserted_rows == 1
    assert result.counts.warning_count == 1
    assert result.counts.quarantined_rows == 0

    process = db_session.execute(select(Process)).scalars().first()
    import json
    payload = json.loads(process.source_payload_json)
    assert payload["source"] == "Some New Unknown Source"


def test_missing_required_field_quarantined(db_session: Session, make_csv, canonical_mapper):
    """Eksik zorunlu alan satırı karantinaya alıyor."""
    row = _make_row()
    row["case_enquiry_id"] = ""
    csv_path = make_csv([row])
    result = run_import(db_session, csv_path, canonical_mapper)

    assert result.counts.quarantined_rows == 1
    assert result.counts.error_rows == 1


def test_missing_required_column_fails(db_session: Session, make_csv, canonical_mapper):
    """Zorunlu kolon eksik dosya failed oluyor."""
    row = {"open_dt": "2024-01-01 10:00:00", "type": "Pothole"}
    csv_path = make_csv([row])
    result = run_import(db_session, csv_path, canonical_mapper)

    assert result.status == "failed"


def test_canonical_mapping_mismatch(db_session: Session, make_csv, canonical_mapper, tmp_path):
    """Mapping versiyonu degistiginde satir karantinaya aliniyor."""
    row = _make_row(case_enquiry_id="MISMATCH-001")
    csv1 = make_csv([row], "file1.csv")
    result1 = run_import(db_session, csv1, canonical_mapper)
    assert result1.counts.inserted_rows == 1

    process = db_session.execute(select(Process)).scalars().first()
    original_payload = process.source_payload_json
    original_process_type = process.process_type
    snapshot = db_session.execute(select(ProcessSnapshot)).scalars().first()
    original_input_json = snapshot.input_json
    original_input_fp = snapshot.input_fingerprint

    import json
    v2_map = json.loads(Path("ml/mappings/canonical_map_v1.json").read_text(encoding="utf-8"))
    v2_map["version"] = "2.0.0"
    v2_path = tmp_path / "canonical_map_v2.json"
    v2_path.write_text(json.dumps(v2_map, ensure_ascii=False), encoding="utf-8")
    mapper_v2 = CanonicalMapper(v2_path)

    rows2 = [
        _make_row(case_enquiry_id="MISMATCH-001"),
        _make_row(case_enquiry_id="EXTRA-001"),
    ]
    csv2 = make_csv(rows2, "file2.csv")
    result2 = run_import(db_session, csv2, mapper_v2)

    assert result2.counts.quarantined_rows == 1
    assert result2.counts.error_rows == 1
    assert result2.counts.inserted_rows == 1

    issues = db_session.execute(
        select(DataQualityIssue).where(DataQualityIssue.issue_code == "CANONICAL_MAPPING_MISMATCH")
    ).scalars().all()
    assert len(issues) == 1
    assert issues[0].severity == "error"

    db_session.refresh(process)
    assert process.source_payload_json == original_payload
    assert process.process_type == original_process_type

    db_session.refresh(snapshot)
    assert snapshot.input_json == original_input_json
    assert snapshot.input_fingerprint == original_input_fp


def test_canonical_mapping_mismatch_does_not_lose_valid_rows(
    db_session: Session, make_csv, canonical_mapper, tmp_path
):
    """Mapping mismatch satiri diger gecerli satirlari kaybettirmiyor."""
    rows = [
        _make_row(case_enquiry_id="MISMATCH-002", type="Request for Pothole Repair"),
        _make_row(case_enquiry_id="VALID-003", type="Request for Pothole Repair"),
    ]
    csv1 = make_csv(rows, "file1.csv")
    run_import(db_session, csv1, canonical_mapper)

    import json
    v2_map = json.loads(Path("ml/mappings/canonical_map_v1.json").read_text(encoding="utf-8"))
    v2_map["version"] = "2.0.0"
    v2_path = tmp_path / "canonical_map_v2b.json"
    v2_path.write_text(json.dumps(v2_map, ensure_ascii=False), encoding="utf-8")
    mapper_v2 = CanonicalMapper(v2_path)

    rows2 = [
        _make_row(case_enquiry_id="MISMATCH-002", type="Request for Pothole Repair"),
        _make_row(case_enquiry_id="VALID-004", type="Request for Pothole Repair"),
    ]
    csv2 = make_csv(rows2, "file2.csv")
    result2 = run_import(db_session, csv2, mapper_v2)

    assert result2.counts.quarantined_rows == 1
    assert result2.counts.inserted_rows == 1
    assert result2.status == "completed_with_issues"


def test_import_run_counters_accurate(db_session: Session, make_csv, canonical_mapper):
    """import_runs sayaçları gerçek sonuçlarla uyuşuyor."""
    rows = [
        _make_row(case_enquiry_id="C-001"),
        _make_row(case_enquiry_id="C-002"),
        _make_row(case_enquiry_id="C-003", source="Unknown Source X"),
    ]
    csv_path = make_csv(rows)
    result = run_import(db_session, csv_path, canonical_mapper)

    run = db_session.execute(select(ImportRun)).scalars().first()
    assert run.total_rows == 3
    assert run.inserted_rows == 3
    assert run.warning_count == 1
    assert run.canonical_mapping_version == canonical_mapper.version
