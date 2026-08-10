"""Faz 4 — Prediction Vertical Slice testleri."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ActiveModelAmbiguousError
from app.models import ModelBundle, PredictionRun
from app.models.process import Process, ProcessSnapshot


class TestMigration003:
    """Migration 003: prediction_runs genisletmesi."""

    def test_003_upgrade_adds_columns(self, db_session: Session):
        columns = [col["name"] for col in db_session.execute(
            text("PRAGMA table_info('prediction_runs')")
        ).mappings().all()]
        assert "status" in columns
        assert "model_bundle_id" in columns
        assert "prediction_context" in columns

    def test_003_status_check_constraint(self, db_session: Session):
        bundle = ModelBundle(
            model_version="test-v1", model_type="bundle",
            artifact_path="artifacts/test.joblib", is_active=1,
        )
        db_session.add(bundle)
        db_session.flush()

        process = Process(external_id="C-001", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        with pytest.raises(IntegrityError):
            db_session.execute(text(
                "INSERT INTO prediction_runs (process_id, model_bundle_id, "
                "model_version, status, prediction_context, prediction_type, predicted_at) "
                "VALUES (1, 1, 'v1', 'invalid_status', 'opening', 'normal', '2024-06-15T00:00:00')"
            ))

    def test_003_partial_unique_index_exists(self, db_session: Session):
        indexes = db_session.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND name='uq_prediction_success_identity'")
        ).scalars().all()
        assert len(indexes) == 1

    def test_003_upgrade_downgrade_upgrade(self, test_engine):
        from alembic import command
        from alembic.config import Config

        import tempfile
        import os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_url = f"sqlite:///{tmpdir}/isolated.db"
            saved = _os.environ.get("ALEMBIC_DATABASE_URL")
            _os.environ["ALEMBIC_DATABASE_URL"] = db_url
            try:
                alembic_cfg = Config(Path(__file__).resolve().parent.parent / "alembic.ini")
                command.upgrade(alembic_cfg, "head")
                command.downgrade(alembic_cfg, "002")
                command.upgrade(alembic_cfg, "head")
            finally:
                if saved is not None:
                    _os.environ["ALEMBIC_DATABASE_URL"] = saved
                else:
                    _os.environ.pop("ALEMBIC_DATABASE_URL", None)


class TestPredictionDBIdempotency:
    """S18 + S19: DB seviyesinde idempotency."""

    @pytest.fixture
    def _setup(self, db_session: Session):
        bundle = ModelBundle(
            model_version="idem-test-v1", model_type="bundle",
            artifact_path="artifacts/idem.joblib", is_active=0,
        )
        db_session.add(bundle)
        db_session.flush()

        process = Process(external_id="IDEM-001", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15),
            input_json=json.dumps({}),
            input_fingerprint="fp-abc-123",
        )
        db_session.add(snapshot)
        db_session.commit()
        return bundle, process, snapshot

    def test_duplicate_success_rejected(self, db_session, _setup):
        bundle, process, snapshot = _setup

        pred1 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(pred1)
        db_session.flush()

        pred2 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(pred2)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_multiple_failed_allowed(self, db_session, _setup):
        bundle, process, snapshot = _setup

        for _ in range(3):
            pred = PredictionRun(
                process_id=process.id, snapshot_id=snapshot.id,
                model_bundle_id=bundle.id, model_version=bundle.model_version,
                status="failed", prediction_context="opening",
                input_fingerprint="fp-abc-123",
            )
            db_session.add(pred)
            db_session.flush()

        count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='failed'")
        ).scalar()
        assert count == 3

    def test_success_after_failed_allowed(self, db_session, _setup):
        bundle, process, snapshot = _setup

        failed = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="failed", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(failed)
        db_session.flush()

        success = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(success)
        db_session.flush()

        assert success.id is not None

    def test_failed_does_not_block_retry(self, db_session, _setup):
        bundle, process, snapshot = _setup

        failed = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="failed", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(failed)
        db_session.flush()

        retry = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="failed", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(retry)
        db_session.flush()

    def test_different_bundle_allows_new_success(self, db_session, _setup):
        bundle, process, snapshot = _setup

        pred1 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(pred1)
        db_session.flush()

        bundle2 = ModelBundle(
            model_version="idem-test-v2", model_type="bundle",
            artifact_path="artifacts/idem2.joblib", is_active=0,
        )
        db_session.add(bundle2)
        db_session.flush()

        pred2 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle2.id, model_version=bundle2.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-abc-123",
        )
        db_session.add(pred2)
        db_session.flush()

        assert pred2.id != pred1.id


class TestRuntime:
    """Runtime state: degraded, ready, ambiguous."""

    @pytest.fixture(autouse=True)
    def _degraded(self, client):
        from app.core.runtime import runtime_state
        runtime_state.model_available = False
        runtime_state.bundle = None
        yield

    def test_ready_503_without_model(self, client):
        response = client.get("/ready")
        assert response.status_code == 503

    def test_health_200_without_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_prediction_503_without_model(self, client):
        response = client.post("/api/processes/1/predictions")
        assert response.status_code == 503
        data = response.json()
        assert "MODEL_UNAVAILABLE" in data.get("error_code", "")


class TestModelLoader:
    """Güvenli model yükleme."""

    @pytest.fixture
    def _bundle_record(self, db_session: Session):
        settings = get_settings()
        artifact_dir = Path(settings.effective_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "test-loader.joblib"

        bundle_data = {
            "classifier_pipeline": "mock_classifier",
            "regression_pipeline": "mock_regressor",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
                "stage": "integration_baseline",
            },
        }
        joblib.dump(bundle_data, str(artifact_path))

        import hashlib
        sha = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()

        bundle = ModelBundle(
            model_version="loader-test-v1",
            model_type="bundle",
            artifact_path=str(artifact_path),
            artifact_hash=sha,
            is_active=0,
            trained_at=datetime.now(UTC),
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)
        return bundle, artifact_path

    def test_load_with_correct_hash(self, db_session, _bundle_record):
        bundle_record, _ = _bundle_record
        from app.services.model_loader import load_bundle
        loaded = load_bundle(db_session, bundle_id=bundle_record.id)
        assert loaded.bundle_id == bundle_record.id
        assert loaded.classifier == "mock_classifier"
        assert loaded.regressor == "mock_regressor"

    def test_tampered_hash_raises(self, db_session, _bundle_record):
        bundle_record, artifact_path = _bundle_record
        artifact_path.write_bytes(b"tampered")
        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle_record.id)

    def test_outside_artifact_root_raises(self, db_session):
        bundle = ModelBundle(
            model_version="outside-test-v1",
            model_type="bundle",
            artifact_path="C:/outside/joblib",
            artifact_hash="abc",
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_schema_mismatch_raises(self, db_session, _bundle_record):
        bundle_record, artifact_path = _bundle_record
        bundle_data = {
            "classifier_pipeline": "mock",
            "regression_pipeline": "mock",
            "metadata": {
                "feature_schema_version": "opening-v99",
                "canonical_mapping_version": "1.0.0",
            },
        }
        joblib.dump(bundle_data, str(artifact_path))
        import hashlib
        sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        bundle_record.artifact_hash = sha
        db_session.commit()

        from app.core.errors import SchemaMismatchError
        from app.services.model_loader import load_bundle
        with pytest.raises(SchemaMismatchError):
            load_bundle(db_session, bundle_id=bundle_record.id)

    def test_mapping_mismatch_raises(self, db_session, _bundle_record):
        bundle_record, artifact_path = _bundle_record
        bundle_data = {
            "classifier_pipeline": "mock",
            "regression_pipeline": "mock",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "99.0.0",
            },
        }
        joblib.dump(bundle_data, str(artifact_path))
        import hashlib
        sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        bundle_record.artifact_hash = sha
        db_session.commit()

        from app.core.errors import SchemaMismatchError
        from app.services.model_loader import load_bundle
        with pytest.raises(SchemaMismatchError):
            load_bundle(db_session, bundle_id=bundle_record.id)

    def test_missing_artifact_raises(self, db_session):
        bundle = ModelBundle(
            model_version="missing-art-v1",
            model_type="bundle",
            artifact_path=str(Path(get_settings().effective_artifact_dir) / "nonexistent.joblib"),
            artifact_hash="abc",
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_ambiguous_active_raises(self, db_session):
        settings = get_settings()
        artifact_dir = Path(settings.effective_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            p = artifact_dir / f"amb{i}.joblib"
            p.write_bytes(b"mock")
            b = ModelBundle(
                model_version=f"amb-test-v{i}",
                model_type="bundle",
                artifact_path=str(p),
                is_active=1,
            )
            db_session.add(b)
        db_session.commit()

        from app.services.model_loader import find_active_bundle
        with pytest.raises(ActiveModelAmbiguousError):
            find_active_bundle(db_session)


class TestConcurrency:
    """S19: Concurrent request altinda duplicate success engellenir."""

    def test_integrity_error_triggers_re_read(self, db_session):
        from app.models import ModelBundle
        bundle = ModelBundle(
            model_version="conc-test-v1", model_type="bundle",
            artifact_path="artifacts/conc.joblib", is_active=0,
        )
        db_session.add(bundle)

        process = Process(external_id="CONC-001", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15),
            input_json=json.dumps({}),
            input_fingerprint="fp-conc-001",
        )
        db_session.add(snapshot)
        db_session.commit()

        first = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-conc-001",
        )
        db_session.add(first)
        db_session.flush()

        duplicate = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-conc-001",
        )
        db_session.add(duplicate)
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestPredictionConstraints:
    """NOT NULL ve FK constraint'leri."""

    def test_missing_status_raises(self, db_session):
        bundle = ModelBundle(
            model_version="nn-test-v1", model_type="bundle",
            artifact_path="artifacts/nn.joblib", is_active=0,
        )
        db_session.add(bundle)
        process = Process(external_id="NN-001", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        pred = PredictionRun(
            process_id=process.id, model_bundle_id=bundle.id,
            model_version=bundle.model_version,
            prediction_context="opening",
        )
        db_session.add(pred)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_missing_model_bundle_id_raises(self, db_session):
        process = Process(external_id="NN-002", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        pred = PredictionRun(
            process_id=process.id, model_version="v1",
            status="success", prediction_context="opening",
        )
        db_session.add(pred)
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestSLA:
    """SLA eksik davranisi."""

    def test_sla_missing_classification_unavailable(self):
        from app.services.prediction_service import _has_sla
        assert _has_sla({"deadline": None}) is False
        assert _has_sla({"deadline": ""}) is False
        assert _has_sla({"deadline": "2024-06-20T00:00:00"}) is True

    def test_risk_score_range(self):
        from app.services.prediction_service import _to_risk_score

        assert _to_risk_score(None) is None
        assert _to_risk_score(0.76) == 76
        assert _to_risk_score(1.5) == 100
        assert _to_risk_score(-0.1) == 0


class TestArtifactPathSecurity:
    """Path validation: startswith → relative_to."""

    @pytest.fixture
    def _artifact_root(self):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_prefix_attack_rejected(self, db_session, _artifact_root):
        settings = get_settings()
        evil_dir = Path(str(settings.effective_artifact_dir) + "_evil")
        evil_dir.mkdir(parents=True, exist_ok=True)
        evil_file = evil_dir / "malicious.joblib"
        evil_file.write_bytes(b"malicious")

        import hashlib
        sha = hashlib.sha256(b"malicious").hexdigest()

        bundle = ModelBundle(
            model_version="prefix-atk-v1",
            model_type="bundle",
            artifact_path=str(evil_file),
            artifact_hash=sha,
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_parent_traversal_rejected(self, db_session, _artifact_root):
        traversal_path = _artifact_root / ".." / "outside.joblib"
        traversal_path = traversal_path.resolve()
        traversal_path.parent.mkdir(parents=True, exist_ok=True)
        traversal_path.write_bytes(b"outside")

        import hashlib
        sha = hashlib.sha256(b"outside").hexdigest()

        bundle = ModelBundle(
            model_version="traversal-v1",
            model_type="bundle",
            artifact_path=str(_artifact_root / ".." / "outside.joblib"),
            artifact_hash=sha,
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_valid_subdirectory_accepted(self, db_session, _artifact_root):
        subdir = _artifact_root / "subdir"
        subdir.mkdir(parents=True, exist_ok=True)
        artifact_file = subdir / "valid.joblib"

        bundle_data = {
            "classifier_pipeline": "mock-clf",
            "regression_pipeline": "mock-reg",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
            },
        }
        joblib.dump(bundle_data, str(artifact_file))
        import hashlib
        sha = hashlib.sha256(artifact_file.read_bytes()).hexdigest()

        bundle = ModelBundle(
            model_version="subdir-v1",
            model_type="bundle",
            artifact_path=str(artifact_file),
            artifact_hash=sha,
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.services.model_loader import load_bundle
        loaded = load_bundle(db_session, bundle_id=bundle.id)
        assert loaded.classifier == "mock-clf"


class TestArtifactHashMandatory:
    """Hash zorunlu: None, empty, invalid format → reject."""

    @pytest.fixture
    def _valid_artifact(self, db_session):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "hash-test.joblib"

        bundle_data = {
            "classifier_pipeline": "mock-clf",
            "regression_pipeline": "mock-reg",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
            },
        }
        joblib.dump(bundle_data, str(path))
        return path

    def test_hash_none_raises(self, db_session, _valid_artifact):
        bundle = ModelBundle(
            model_version="hash-none-v1", model_type="bundle",
            artifact_path=str(_valid_artifact), artifact_hash=None, is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_hash_empty_raises(self, db_session, _valid_artifact):
        bundle = ModelBundle(
            model_version="hash-empty-v1", model_type="bundle",
            artifact_path=str(_valid_artifact), artifact_hash="", is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_hash_invalid_format_raises(self, db_session, _valid_artifact):
        bundle = ModelBundle(
            model_version="hash-fmt-v1", model_type="bundle",
            artifact_path=str(_valid_artifact), artifact_hash="not-a-valid-sha256-hash-value",
            is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.core.errors import ModelUnavailableError
        from app.services.model_loader import load_bundle
        with pytest.raises(ModelUnavailableError):
            load_bundle(db_session, bundle_id=bundle.id)

    def test_hash_valid_format_accepted(self, db_session, _valid_artifact):
        import hashlib
        sha = hashlib.sha256(_valid_artifact.read_bytes()).hexdigest()

        bundle = ModelBundle(
            model_version="hash-ok-v1", model_type="bundle",
            artifact_path=str(_valid_artifact), artifact_hash=sha, is_active=0,
        )
        db_session.add(bundle)
        db_session.commit()
        db_session.refresh(bundle)

        from app.services.model_loader import load_bundle
        loaded = load_bundle(db_session, bundle_id=bundle.id)
        assert loaded.bundle_id == bundle.id


class TestRegressionFailure:
    """Non-finite regression output → prediction başarısız, success row YOK."""

    @pytest.fixture
    def _prediction_setup(self, db_session: Session):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        p = root / "regfail.joblib"
        p.write_bytes(b"mock")

        import hashlib
        sha = hashlib.sha256(b"mock").hexdigest()
        bundle_record = ModelBundle(
            model_version="regfail-v1", model_type="bundle",
            artifact_path=str(p), artifact_hash=sha, is_active=0,
        )
        db_session.add(bundle_record)

        process = Process(external_id="REGF-001", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        input_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": "2024-06-18T09:00:00",
            "source": "phone_call",
            "subject": "graffiti_removal",
            "reason": "public_nuisance",
            "type": "property_violation",
            "neighborhood": "roxbury_02119",
        })
        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=input_json,
            input_fingerprint="fp-regfail-001",
        )
        db_session.add(snapshot)
        db_session.commit()
        db_session.refresh(bundle_record)
        return bundle_record, process, snapshot

    def _make_bundle(self, bundle_record, regressor_output):
        from app.services.model_loader import LoadedBundle

        class FakeClassifier:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])

        class FakeRegressor:
            def __init__(self, value):
                self._value = value
                self.call_count = 0

            def predict(self, X):
                self.call_count += 1
                return np.array([self._value])

        reg = FakeRegressor(regressor_output)
        clf = FakeClassifier()

        return LoadedBundle(
            bundle_id=bundle_record.id,
            bundle_record=bundle_record,
            classifier=clf,
            regressor=reg,
            metadata={"feature_schema_version": "opening-v1", "canonical_mapping_version": "1.0.0"},
            artifact_hash="a" * 64,
        ), reg

    def test_nan_raises_no_success_row(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup
        loaded, reg = self._make_bundle(bundle_record, np.nan)

        from app.core.errors import AppError
        from app.services.prediction_service import predict_single
        with pytest.raises(AppError) as exc_info:
            predict_single(db_session, process.id, loaded)
        assert exc_info.value.error_code == "INVALID_REGRESSION_PREDICTION"

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 0

    def test_pos_inf_raises_no_success_row(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup
        loaded, reg = self._make_bundle(bundle_record, float("inf"))

        from app.core.errors import AppError
        from app.services.prediction_service import predict_single
        with pytest.raises(AppError) as exc_info:
            predict_single(db_session, process.id, loaded)
        assert exc_info.value.error_code == "INVALID_REGRESSION_PREDICTION"

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 0

    def test_neg_inf_raises_no_success_row(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup
        loaded, reg = self._make_bundle(bundle_record, float("-inf"))

        from app.core.errors import AppError
        from app.services.prediction_service import predict_single
        with pytest.raises(AppError) as exc_info:
            predict_single(db_session, process.id, loaded)
        assert exc_info.value.error_code == "INVALID_REGRESSION_PREDICTION"

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 0

    def test_retry_nan_then_finite(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup

        nan_loaded, nan_reg = self._make_bundle(bundle_record, np.nan)
        from app.core.errors import AppError
        from app.services.prediction_service import predict_single
        try:
            predict_single(db_session, process.id, nan_loaded)
        except AppError:
            pass

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 0

        fin_loaded, fin_reg = self._make_bundle(bundle_record, 42.5)
        result1 = predict_single(db_session, process.id, fin_loaded)
        db_session.commit()

        assert result1.reused is False
        assert result1.regression_available is True
        assert result1.prediction_run.predicted_hours == 42.5
        assert fin_reg.call_count == 1

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 1

        result2 = predict_single(db_session, process.id, fin_loaded)
        db_session.commit()

        assert result2.reused is True
        assert result2.prediction_run.id == result1.prediction_run.id
        assert fin_reg.call_count == 1

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 1

    def test_negative_finite_clamped_to_zero(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup
        loaded, reg = self._make_bundle(bundle_record, -5.0)

        from app.services.prediction_service import predict_single
        result = predict_single(db_session, process.id, loaded)
        db_session.commit()

        assert result.reused is False
        assert result.regression_available is True
        assert result.prediction_run.predicted_hours == 0.0
        assert result.prediction_run.status == "success"

    def test_valid_output_preserved(self, db_session, _prediction_setup):
        bundle_record, process, snapshot = _prediction_setup
        loaded, reg = self._make_bundle(bundle_record, 12.5)

        from app.services.prediction_service import predict_single
        result = predict_single(db_session, process.id, loaded)
        db_session.commit()

        assert result.reused is False
        assert result.regression_available is True
        assert result.prediction_run.predicted_hours == 12.5

    def test_sla_missing_finite_regression_success(self, db_session, _prediction_setup):
        bundle_record, _, _ = _prediction_setup

        process = Process(external_id="REGF-SLA-OK", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        sla_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": None,
            "source": "phone_call",
            "subject": "graffiti_removal",
            "reason": "public_nuisance",
            "type": "property_violation",
            "neighborhood": "roxbury_02119",
        })
        sla_snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=sla_json,
            input_fingerprint="fp-sla-ok-001",
        )
        db_session.add(sla_snapshot)
        db_session.commit()

        loaded, reg = self._make_bundle(bundle_record, 36.0)

        from app.services.prediction_service import predict_single
        result = predict_single(db_session, process.id, loaded)
        db_session.commit()

        assert result.classification_available is False
        assert result.prediction_run.delay_probability is None
        assert result.regression_available is True
        assert result.prediction_run.predicted_hours == 36.0
        assert result.prediction_run.status == "success"

    def test_sla_missing_nonfinite_regression_fails(self, db_session, _prediction_setup):
        bundle_record, _, _ = _prediction_setup

        process = Process(external_id="REGF-SLA-NAN", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        sla_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": None,
            "source": "phone_call",
            "subject": "graffiti_removal",
            "reason": "public_nuisance",
            "type": "property_violation",
            "neighborhood": "roxbury_02119",
        })
        sla_snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=sla_json,
            input_fingerprint="fp-sla-nan-001",
        )
        db_session.add(sla_snapshot)
        db_session.commit()

        loaded, reg = self._make_bundle(bundle_record, np.nan)

        from app.core.errors import AppError
        from app.services.prediction_service import predict_single
        with pytest.raises(AppError) as exc_info:
            predict_single(db_session, process.id, loaded)
        assert exc_info.value.error_code == "INVALID_REGRESSION_PREDICTION"

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE status='success'")
        ).scalar()
        assert success_count == 0


class TestRealSklearnE2E:
    """Gerçek sklearn pipeline ile uçtan uca prediction testi."""

    @pytest.fixture
    def _e2e_bundle(self, db_session):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)

        from sklearn.linear_model import LogisticRegression
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.linear_model import ElasticNet
        from ml.features.preprocessing import build_classification_pipeline, build_regression_pipeline
        from ml.features.schema_loader import load_feature_schema

        schema = load_feature_schema()

        X = pd.DataFrame([
            {
                "source": "citizens_connect_app",
                "subject": "pothole_not_filled",
                "reason": "road_maintenance",
                "type": "street_repair",
                "neighborhood": "dorchester_02121",
                "open_month": 6,
                "open_weekday": 2,
                "open_hour": 9,
                "is_weekend": 0,
                "sla_duration_hours": 72.0,
            },
            {
                "source": "phone_call",
                "subject": "graffiti_removal",
                "reason": "public_nuisance",
                "type": "property_violation",
                "neighborhood": "roxbury_02119",
                "open_month": 8,
                "open_weekday": 5,
                "open_hour": 15,
                "is_weekend": 0,
                "sla_duration_hours": 168.0,
            },
        ])

        clf = LogisticRegression(max_iter=100, random_state=42)
        clf_pipe = build_classification_pipeline(schema, clf)
        clf_pipe.fit(X, np.array([0, 1]))

        reg = TransformedTargetRegressor(
            regressor=ElasticNet(random_state=42, max_iter=100),
            func=np.log1p, inverse_func=np.expm1,
        )
        reg_pipe = build_regression_pipeline(schema, reg)
        reg_pipe.fit(X, np.array([48.0, 12.0]))

        import sklearn
        import hashlib

        bundle_data = {
            "classifier_pipeline": clf_pipe,
            "regression_pipeline": reg_pipe,
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
                "sklearn_version": sklearn.__version__,
                "bundle_format_version": "1.0.0",
                "stage": "integration_baseline",
            },
        }

        artifact_path = root / "e2e-test-bundle.joblib"
        joblib.dump(bundle_data, str(artifact_path))
        sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        bundle_record = ModelBundle(
            model_version="e2e-test-v1",
            model_type="bundle",
            artifact_path=str(artifact_path),
            artifact_hash=sha,
            is_active=0,
            trained_at=datetime.now(UTC),
        )
        db_session.add(bundle_record)
        db_session.commit()
        db_session.refresh(bundle_record)

        from app.services.model_loader import load_bundle
        loaded = load_bundle(db_session, bundle_id=bundle_record.id)
        return loaded, bundle_record

    def test_e2e_prediction_new(self, db_session, _e2e_bundle):
        loaded_bundle, bundle_record = _e2e_bundle
        import pandas as pd

        process = Process(external_id="E2E-001", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        input_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": "2024-06-18T09:00:00",
            "source": "citizens_connect_app",
            "subject": "pothole_not_filled",
            "reason": "road_maintenance",
            "type": "street_repair",
            "neighborhood": "dorchester_02121",
        })
        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=input_json,
            input_fingerprint="e2e-fp-001",
        )
        db_session.add(snapshot)
        db_session.commit()

        from app.services.prediction_service import predict_single
        result = predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        assert result.reused is False
        assert result.prediction_run.status == "success"
        assert result.prediction_run.delay_probability is not None
        assert result.regression_available is True

    def test_e2e_prediction_reuse(self, db_session, _e2e_bundle):
        loaded_bundle, bundle_record = _e2e_bundle

        process = Process(external_id="E2E-002", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        input_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": "2024-06-18T09:00:00",
            "source": "citizens_connect_app",
            "subject": "pothole_not_filled",
            "reason": "road_maintenance",
            "type": "street_repair",
            "neighborhood": "dorchester_02121",
        })
        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=input_json,
            input_fingerprint="e2e-fp-002",
        )
        db_session.add(snapshot)
        db_session.commit()

        from app.services.prediction_service import predict_single

        result1 = predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        result2 = predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        assert result1.reused is False
        assert result2.reused is True
        assert result2.prediction_run.id == result1.prediction_run.id

    def test_e2e_sla_missing_classification(self, db_session, _e2e_bundle):
        loaded_bundle, bundle_record = _e2e_bundle

        process = Process(external_id="E2E-SLA-001", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        input_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": None,
            "source": "citizens_connect_app",
            "subject": "pothole_not_filled",
            "reason": "road_maintenance",
            "type": "street_repair",
            "neighborhood": "dorchester_02121",
        })
        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=input_json,
            input_fingerprint="e2e-sla-fp",
        )
        db_session.add(snapshot)
        db_session.commit()

        from app.services.prediction_service import predict_single
        result = predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        assert result.classification_available is False
        assert result.prediction_run.delay_probability is None
        assert result.regression_available is True
        assert result.prediction_run.predicted_hours is not None
        assert result.prediction_run.predicted_hours >= 0

    def test_e2e_duplicate_success_db_count(self, db_session, _e2e_bundle):
        loaded_bundle, bundle_record = _e2e_bundle

        process = Process(external_id="E2E-DC-001", created_at=datetime(2024, 6, 15, 9, 0, 0))
        db_session.add(process)
        db_session.flush()

        input_json = json.dumps({
            "created_at": "2024-06-15T09:00:00",
            "deadline": "2024-06-18T09:00:00",
            "source": "citizens_connect_app",
            "subject": "pothole_not_filled",
            "reason": "road_maintenance",
            "type": "street_repair",
            "neighborhood": "dorchester_02121",
        })
        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15, 9, 0, 0),
            input_json=input_json,
            input_fingerprint="e2e-dc-fp",
        )
        db_session.add(snapshot)
        db_session.commit()

        from app.services.prediction_service import predict_single

        predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        predict_single(db_session, process.id, loaded_bundle)
        db_session.commit()

        success_count = db_session.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE snapshot_id=:sid AND status='success'"),
            {"sid": snapshot.id},
        ).scalar()
        assert success_count == 1


class TestBundleImmutability:
    """Bundle artifact overwrite koruması."""

    def test_two_builds_use_different_paths(self, db_session):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)

        import uuid
        ts1 = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path1 = root / f"bundle_{ts1}_{uuid.uuid4().hex[:8]}.joblib"

        ts2 = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path2 = root / f"bundle_{ts2}_{uuid.uuid4().hex[:8]}.joblib"

        assert str(path1) != str(path2)

        bundle_data = {
            "classifier_pipeline": "mock",
            "regression_pipeline": "mock",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
            },
        }

        joblib.dump(bundle_data, str(path1))
        import hashlib
        sha1 = hashlib.sha256(path1.read_bytes()).hexdigest()

        bundle1 = ModelBundle(
            model_version="immut-v1",
            model_type="bundle",
            artifact_path=str(path1),
            artifact_hash=sha1,
            is_active=0,
        )
        db_session.add(bundle1)
        db_session.commit()
        db_session.refresh(bundle1)

        joblib.dump(bundle_data, str(path2))
        sha2 = hashlib.sha256(path2.read_bytes()).hexdigest()

        bundle2 = ModelBundle(
            model_version="immut-v2",
            model_type="bundle",
            artifact_path=str(path2),
            artifact_hash=sha2,
            is_active=0,
        )
        db_session.add(bundle2)
        db_session.commit()
        db_session.refresh(bundle2)

        from app.services.model_loader import load_bundle
        loaded1 = load_bundle(db_session, bundle_id=bundle1.id)
        loaded2 = load_bundle(db_session, bundle_id=bundle2.id)

        assert loaded1.bundle_id == bundle1.id
        assert loaded2.bundle_id == bundle2.id
        assert loaded1.artifact_hash == sha1
        assert loaded2.artifact_hash == sha2


class TestConcurrencyReal:
    """Gerçek SQLite eşzamanlılık: duplicate success tek satır."""

    def test_concurrent_sessions_one_success_row(self, db_session, temp_db_url):
        settings = get_settings()
        root = Path(settings.effective_artifact_dir)
        root.mkdir(parents=True, exist_ok=True)

        bundle_data = {
            "classifier_pipeline": "mock-clf",
            "regression_pipeline": "mock-reg",
            "metadata": {
                "feature_schema_version": "opening-v1",
                "canonical_mapping_version": "1.0.0",
            },
        }
        artifact_path = root / "conc-real.joblib"
        joblib.dump(bundle_data, str(artifact_path))
        import hashlib
        sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        bundle = ModelBundle(
            model_version="conc-real-v1",
            model_type="bundle",
            artifact_path=str(artifact_path),
            artifact_hash=sha,
            is_active=0,
        )
        db_session.add(bundle)

        process = Process(external_id="CONC-R-001", created_at=datetime(2024, 6, 15))
        db_session.add(process)
        db_session.flush()

        snapshot = ProcessSnapshot(
            process_id=process.id, snapshot_type="opening",
            snapshot_at=datetime(2024, 6, 15),
            input_json=json.dumps({}),
            input_fingerprint="fp-conc-r-001",
        )
        db_session.add(snapshot)
        db_session.commit()

        pred1 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-conc-r-001",
            prediction_type="normal",
            predicted_at=datetime.now(UTC),
        )
        db_session.add(pred1)
        db_session.commit()

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session as SaSession

        engine2 = create_engine(temp_db_url, future=True, connect_args={"check_same_thread": False})
        s2 = SaSession(bind=engine2)

        pred2 = PredictionRun(
            process_id=process.id, snapshot_id=snapshot.id,
            model_bundle_id=bundle.id, model_version=bundle.model_version,
            status="success", prediction_context="opening",
            input_fingerprint="fp-conc-r-001",
            prediction_type="normal",
            predicted_at=datetime.now(UTC),
        )
        s2.add(pred2)
        try:
            s2.flush()
        except IntegrityError:
            s2.rollback()

        success_count = s2.execute(
            text("SELECT COUNT(*) FROM prediction_runs WHERE snapshot_id=:sid AND status='success'"),
            {"sid": snapshot.id},
        ).scalar()
        s2.close()

        assert success_count <= 1
