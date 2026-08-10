"""Faz 5 production candidate bundle olusturma.

Calibration: OOF Train fit. Validation evaluation only.
Test/Audit sealed.

Kullanim:
    python scripts/build_faz5_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.model_bundle import ModelBundle
from ml.calibration.calibrator import CalibrationModel, fit_calibration
from ml.datasets.dataset_builder import build_dataset
from ml.features.preprocessing import (
    build_classification_pipeline,
    build_regression_pipeline,
)
from ml.features.schema_loader import load_feature_schema
from ml.training.classifier_trainer import (
    _build_classifier,
    find_best_threshold,
    train_classifier_cv,
)
from ml.training.regressor_trainer import (
    _build_regressor,
    train_regressor_cv,
)

BUNDLE_VERSION = "faz5-production-candidate-v1"
N_SEARCH_CANDIDATES = 5


def main() -> None:
    settings = get_settings()
    schema = load_feature_schema()

    session = SessionLocal()
    try:
        result = build_dataset(session, schema)
        cls_fingerprint = result.classification.fingerprint
        reg_fingerprint = result.regression.fingerprint
    finally:
        session.close()

    clf_dataset = result.classification
    reg_dataset = result.regression

    X_train_clf = clf_dataset.train.X
    y_train_clf = clf_dataset.train.y
    X_val_clf = clf_dataset.validation.X
    y_val_clf = clf_dataset.validation.y
    X_train_reg = reg_dataset.train.X
    y_train_reg = reg_dataset.train.y
    X_val_reg = reg_dataset.validation.X
    y_val_reg = reg_dataset.validation.y

    print(f"Train (classification): {X_train_clf.shape[0]:,} satir")
    print(f"Validation:             {clf_dataset.validation.X.shape[0]:,} satir")
    print(f"Test [MUHURLU]:         {clf_dataset.test_info.row_count:,} kayit")
    print(f"Audit [MUHURLU]:        {clf_dataset.audit_info.row_count:,} kayit")
    print()

    print("=== Classification Winner: RandomForest ===")
    clf_cv = train_classifier_cv(
        X_train_clf, y_train_clf, schema, "RandomForest",
        n_splits=5, search=True, n_search_candidates=N_SEARCH_CANDIDATES,
    )
    print(f"  CV PR-AUC: {clf_cv.pr_auc_mean:.4f} +/- {clf_cv.pr_auc_std:.4f}")
    print(f"  Best params: {clf_cv.best_params}")

    clf_base = _build_classifier("RandomForest", **(clf_cv.best_params or {}))
    clf_pipe = build_classification_pipeline(schema, clf_base)
    clf_pipe.fit(X_train_clf, y_train_clf)

    cal_model = fit_calibration(clf_cv.oof_proba, clf_cv.oof_true, "sigmoid")
    print(f"  Calibration: {cal_model.method}")

    val_proba = cal_model.predict_proba(clf_pipe.predict_proba(X_val_clf)[:, 1])
    best_th = find_best_threshold(y_val_clf, val_proba)
    print(f"  Threshold: {best_th['threshold']:.2f} (F1={best_th['f1']:.4f})")

    from sklearn.metrics import average_precision_score, brier_score_loss
    from ml.evaluation.metrics import evaluate_classification, evaluate_regression
    val_pr_auc = float(average_precision_score(y_val_clf, val_proba))
    val_brier = float(brier_score_loss(y_val_clf, val_proba))
    val_cls_metrics = evaluate_classification(
        y_val_clf,
        (val_proba >= best_th["threshold"]).astype(int),
        val_proba,
        "validation",
    )
    print(f"  Validation PR-AUC: {val_pr_auc:.4f}")
    print(f"  Validation Brier: {val_brier:.4f}")

    print()
    print("=== Regression Winner: ElasticNet_log1p ===")
    reg_cv = train_regressor_cv(
        X_train_reg, y_train_reg, schema, "ElasticNet_log1p",
        n_splits=5, search=True, n_search_candidates=N_SEARCH_CANDIDATES,
    )
    print(f"  CV MAE: {reg_cv.mae_mean:.2f}h +/- {reg_cv.mae_std:.2f}h")
    print(f"  Best params: {reg_cv.best_params}")

    reg_base = _build_regressor("ElasticNet_log1p", **(reg_cv.best_params or {}))
    reg_pipe = build_regression_pipeline(schema, reg_base)
    reg_pipe.fit(X_train_reg, y_train_reg)

    val_pred_reg = np.maximum(0.0, reg_pipe.predict(X_val_reg))
    from sklearn.metrics import mean_absolute_error
    val_mae = float(mean_absolute_error(y_val_reg, val_pred_reg))
    val_reg_metrics = evaluate_regression(y_val_reg, val_pred_reg, "validation")
    print(f"  Validation MAE: {val_mae:.2f}h")

    import sklearn
    import sys as _sys
    metadata: dict[str, Any] = {
        "bundle_format_version": "1.0.0",
        "phase": "faz5",
        "stage": "production_candidate",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_schema_version": schema.feature_schema_version,
        "canonical_mapping_version": schema.canonical_mapping_version,
        "observation_cutoff": "2025-01-13T00:00:00",
        "training_scope": "train_only",
        "classifier": {
            "model_type": "RandomForestClassifier",
            "best_params": clf_cv.best_params,
            "cv_pr_auc_mean": clf_cv.pr_auc_mean,
            "cv_pr_auc_std": clf_cv.pr_auc_std,
            "cv_roc_auc_mean": clf_cv.roc_auc_mean,
            "cv_roc_auc_std": clf_cv.roc_auc_std,
            "calibration_method": cal_model.method,
            "threshold": best_th["threshold"],
            "threshold_f1": best_th["f1"],
            "validation_pr_auc": val_pr_auc,
            "validation_brier": val_brier,
            "validation_roc_auc": val_cls_metrics["roc_auc"],
            "validation_f1": val_cls_metrics["f1"],
            "validation_precision": val_cls_metrics["precision"],
            "validation_recall": val_cls_metrics["recall"],
            "classification_validation_row_count": val_cls_metrics["row_count"],
            "validation_confusion_matrix": val_cls_metrics["confusion_matrix"],
        },
        "regression": {
            "model_type": "TransformedTargetRegressor(log1p, ElasticNet)",
            "best_params": reg_cv.best_params,
            "cv_mae_mean": reg_cv.mae_mean,
            "cv_mae_std": reg_cv.mae_std,
            "cv_rmse_mean": reg_cv.rmse_mean,
            "cv_rmse_std": reg_cv.rmse_std,
            "validation_mae": val_mae,
            "validation_median_ae": val_reg_metrics["median_ae"],
            "validation_rmse": val_reg_metrics["rmse"],
            "validation_p90_ae": val_reg_metrics["p90_abs_error"],
            "regression_validation_row_count": val_reg_metrics["row_count"],
        },
        "classification_fingerprint": cls_fingerprint,
        "regression_fingerprint": reg_fingerprint,
        "train_classification_rows": X_train_clf.shape[0],
        "train_regression_rows": X_train_reg.shape[0],
        "validation_classification_rows": X_val_clf.shape[0],
        "validation_regression_rows": X_val_reg.shape[0],
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "python_version": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        "cv_splits": 5,
        "hp_search_candidates": N_SEARCH_CANDIDATES,
    }

    bundle = {
        "classifier_pipeline": clf_pipe,
        "calibration_model": cal_model,
        "threshold": best_th["threshold"],
        "regression_pipeline": reg_pipe,
        "metadata": metadata,
    }

    artifact_dir = settings.effective_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    artifact_path = artifact_dir / f"{BUNDLE_VERSION}_{ts}_{unique_id}.joblib"

    joblib.dump(bundle, str(artifact_path))
    print(f"\nArtifact saved: {artifact_path}")

    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    print(f"SHA256: {artifact_hash}")

    session = SessionLocal()
    try:
        bundle_record = ModelBundle(
            model_version=BUNDLE_VERSION,
            model_type="bundle",
            artifact_path=str(artifact_path),
            artifact_hash=artifact_hash,
            metrics_json=json.dumps(
                {k: v for k, v in metadata.items()
                 if k in ("classifier", "regression")},
                ensure_ascii=False, default=str,
            ),
            feature_list_json=json.dumps(schema.all_features, ensure_ascii=False),
            trained_at=datetime.now(UTC),
            is_active=0,
        )
        session.add(bundle_record)
        session.commit()
        session.refresh(bundle_record)

        print(f"\nModelBundle DB kaydi:")
        print(f"  ID: {bundle_record.id}")
        print(f"  Version: {bundle_record.model_version}")
        print(f"  Stage: production_candidate")
        print(f"  is_active: {bundle_record.is_active} (candidate - NOT active)")
    finally:
        session.close()

    print("\nTest/Audit degerlendirmesi YAPILMADI (muhurlu).")
    print("Faz 5 production_candidate bundle olusturuldu.")


if __name__ == "__main__":
    main()
