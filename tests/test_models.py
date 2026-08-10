"""ORM model testleri."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session


def test_process_creation(db_session: Session):
    """Process ORM modeli oluşturulup kaydedilebiliyor."""
    from app.models import Process

    process = Process(
        external_id="10100-20240101-0001",
        process_type="Sidewalk Repair",
        current_status="Open",
        created_at=datetime(2024, 1, 1, 10, 0, 0),
    )
    db_session.add(process)
    db_session.commit()
    db_session.refresh(process)

    assert process.id is not None
    assert process.external_id == "10100-20240101-0001"


def test_process_unique_external_id(db_session: Session):
    """external_id UNIQUE constraint'i çalışıyor."""
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from app.models import Process

    process1 = Process(
        external_id="DUPLICATE-001",
        created_at=datetime(2024, 1, 1),
    )
    db_session.add(process1)
    db_session.commit()

    process2 = Process(
        external_id="DUPLICATE-001",
        created_at=datetime(2024, 1, 2),
    )
    db_session.add(process2)
    with __import__("pytest").raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_model_bundle_creation(db_session: Session):
    """ModelBundle ORM modeli oluşturulabiliyor."""
    from app.models import ModelBundle

    bundle = ModelBundle(
        model_version="clf-v0.1.0-demo",
        model_type="classifier",
        artifact_path="artifacts/demo/clf_v0.1.0.joblib",
        is_active=0,
    )
    db_session.add(bundle)
    db_session.commit()
    db_session.refresh(bundle)

    assert bundle.id is not None
    assert bundle.model_version == "clf-v0.1.0-demo"


def test_prediction_run_creation(db_session: Session):
    """PredictionRun ORM modeli ilişkilerle birlikte oluşturulabiliyor."""
    from app.models import ModelBundle, PredictionRun, Process

    process = Process(
        external_id="PRED-TEST-001",
        created_at=datetime(2024, 6, 15, 9, 0, 0),
    )
    db_session.add(process)

    bundle = ModelBundle(
        model_version="faz4-integration-v1",
        model_type="bundle",
        artifact_path="artifacts/demo/faz4-v1.joblib",
        artifact_hash="abc123",
        is_active=1,
        trained_at=datetime.now(UTC),
    )
    db_session.add(bundle)
    db_session.commit()
    db_session.refresh(process)
    db_session.refresh(bundle)

    prediction = PredictionRun(
        process_id=process.id,
        model_bundle_id=bundle.id,
        model_version="faz4-integration-v1",
        prediction_type="normal",
        prediction_context="opening",
        status="success",
        delay_probability=0.76,
        risk_score=78,
        risk_level="high",
        predicted_at=datetime.now(UTC),
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assert prediction.id is not None
    assert prediction.risk_level == "high"
    assert prediction.process_id == process.id
    assert prediction.model_bundle_id == bundle.id
    assert prediction.status == "success"


def test_import_run_and_quality_issue(db_session: Session):
    """ImportRun ve DataQualityIssue ilişkisi çalışıyor."""
    from app.models import DataQualityIssue, ImportRun

    run = ImportRun(
        file_name="test.csv",
        status="completed",
        total_rows=100,
        imported_rows=95,
        error_rows=5,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    issue = DataQualityIssue(
        import_run_id=run.id,
        row_number=42,
        column_name="open_dt",
        issue_type="missing_required",
        issue_message="Zorunlu alan bos",
    )
    db_session.add(issue)
    db_session.commit()
    db_session.refresh(issue)

    assert issue.id is not None
    assert issue.import_run_id == run.id


def test_all_tables_exist(db_session: Session):
    """Tüm 7 tablo veritabanında mevcut."""
    from sqlalchemy import text

    result = db_session.execute(
        text(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version' "
            "ORDER BY name"
        )
    ).fetchall()
    table_names = {r[0] for r in result}
    expected = {
        "processes",
        "process_snapshots",
        "import_runs",
        "data_quality_issues",
        "model_bundles",
        "prediction_runs",
        "prediction_feedback",
    }
    assert expected.issubset(table_names), f"Eksik tablolar: {expected - table_names}"
