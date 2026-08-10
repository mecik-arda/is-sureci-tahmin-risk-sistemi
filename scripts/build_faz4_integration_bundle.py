"""Faz 4 integration bundle oluşturma scripti.

Kullanim:
    python scripts/build_faz4_integration_bundle.py

Yalniz TRAIN split üzerinde fit eder.
Test/Audit'e dokunulmaz (sealed guard).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.model_bundle import ModelBundle
from ml.datasets.dataset_builder import build_dataset
from ml.features.preprocessing import build_classification_pipeline, build_regression_pipeline
from ml.features.schema_loader import load_feature_schema

BUNDLE_VERSION = "faz4-integration-v1"


def main() -> None:
    settings = get_settings()
    schema = load_feature_schema()

    session = SessionLocal()
    try:
        result = build_dataset(session, schema)
    finally:
        session.close()

    clf_dataset = result.classification
    reg_dataset = result.regression

    X_train_clf = clf_dataset.train.X
    y_train_clf = clf_dataset.train.y
    X_train_reg = reg_dataset.train.X
    y_train_reg = reg_dataset.train.y

    print(f"Train (classification): {X_train_clf.shape[0]:,} satir")
    print(f"Train (regression):     {X_train_reg.shape[0]:,} satir")
    print(f"Validation:             {clf_dataset.validation.X.shape[0]:,} satir")
    print(f"Test [MÜHÜRLÜ]:         {clf_dataset.test_info.row_count:,} kayit")
    print(f"Audit [MÜHÜRLÜ]:        {clf_dataset.audit_info.row_count:,} kayit")
    print()

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    clf_pipeline = build_classification_pipeline(schema, classifier)
    clf_pipeline.fit(X_train_clf, y_train_clf)
    print("Classifier fit tamamlandi.")

    regressor = TransformedTargetRegressor(
        regressor=ElasticNet(random_state=42, max_iter=5000),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    reg_pipeline = build_regression_pipeline(schema, regressor)
    reg_pipeline.fit(X_train_reg, y_train_reg)
    print("Regressor fit tamamlandi.")

    import sklearn
    metadata = {
        "feature_schema_version": schema.feature_schema_version,
        "canonical_mapping_version": schema.canonical_mapping_version,
        "observation_cutoff": "2025-01-13T00:00:00",
        "classifier_model_type": "LogisticRegression",
        "regressor_model_type": "TransformedTargetRegressor(log1p, ElasticNet)",
        "sklearn_version": sklearn.__version__,
        "bundle_format_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "training_scope": "train_only",
        "stage": "integration_baseline",
        "classifier_fingerprint": clf_dataset.fingerprint,
        "regression_fingerprint": reg_dataset.fingerprint,
        "train_classification_rows": X_train_clf.shape[0],
        "train_regression_rows": X_train_reg.shape[0],
    }

    bundle = {
        "classifier_pipeline": clf_pipeline,
        "regression_pipeline": reg_pipeline,
        "metadata": metadata,
    }

    artifact_dir = Path(settings.effective_artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    artifact_path = artifact_dir / f"{BUNDLE_VERSION}_{ts}_{unique_id}.joblib"

    joblib.dump(bundle, str(artifact_path))
    print(f"\nArtifact kaydedildi: {artifact_path}")

    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    print(f"SHA256: {artifact_hash}")

    session = SessionLocal()
    try:
        session.execute(
            __import__("sqlalchemy").text("UPDATE model_bundles SET is_active = 0 WHERE is_active = 1")
        )
        session.flush()

        bundle_record = ModelBundle(
            model_version=BUNDLE_VERSION,
            model_type="bundle",
            artifact_path=str(artifact_path),
            artifact_hash=artifact_hash,
            metrics_json=json.dumps({}, ensure_ascii=False),
            feature_list_json=json.dumps(schema.all_features, ensure_ascii=False),
            trained_at=datetime.now(UTC),
            is_active=1,
        )
        session.add(bundle_record)
        session.commit()
        session.refresh(bundle_record)

        print(f"\nModel bundle DB'ye kaydedildi:")
        print(f"  ID: {bundle_record.id}")
        print(f"  Version: {bundle_record.model_version}")
        print(f"  Stage: integration_baseline")
        print(f"  is_active: {bundle_record.is_active}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
