"""Faz 5 — Model Gelistirme ve Karsilastirma.

S20: Train icinde TimeSeriesSplit CV, Validation final karar, Test/Audit muhurlu.
S21: Classification: PR-AUC, Regression: MAE ana secim metrigi.

Kullanim:
    python scripts/run_faz5_training.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from app.core.database import SessionLocal
from ml.calibration.calibrator import CALIBRATION_METHODS
from ml.datasets.dataset_builder import build_dataset
from ml.evaluation.comparison import (
    format_calibration_comparison,
    format_classification_table,
    format_cv_summary,
    format_regression_table,
    get_winner_classifier,
    get_winner_regressor,
)
from ml.features.schema_loader import load_feature_schema
from ml.training.classifier_trainer import (
    CLASSIFIER_NAMES,
    ClassifierResult,
    calibrate_and_evaluate,
    find_best_threshold,
    train_classifier_cv,
)
from ml.training.regressor_trainer import (
    REGRESSOR_NAMES,
    RegressorResult,
    fit_and_evaluate_regressor,
    train_regressor_cv,
)


def main() -> None:
    print("=" * 70)
    print("FAZ 5 — MODEL GELISTIRME")
    print("=" * 70)

    schema = load_feature_schema()
    print(f"\nFeature schema: {schema.feature_schema_version}")
    print(f"Canonical mapping: {schema.canonical_mapping_version}")
    print(f"Features: {len(schema.all_features)} ({', '.join(schema.all_features)})")

    session = SessionLocal()
    try:
        result = build_dataset(session, schema)
        dataset_fingerprint_cls = result.classification.fingerprint
        dataset_fingerprint_reg = result.regression.fingerprint
    finally:
        session.close()

    clf_dataset = result.classification
    reg_dataset = result.regression

    print(f"\nDataset:")
    print(f"  Classification: {clf_dataset.train.row_count:,} train | {clf_dataset.validation.row_count:,} val | {clf_dataset.test_info.row_count:,} test [M] | {clf_dataset.audit_info.row_count:,} audit [M]")
    print(f"  Regression:     {reg_dataset.train.row_count:,} train | {reg_dataset.validation.row_count:,} val | {reg_dataset.test_info.row_count:,} test [M] | {reg_dataset.audit_info.row_count:,} audit [M]")

    X_train_clf = clf_dataset.train.X
    y_train_clf = clf_dataset.train.y
    X_val_clf = clf_dataset.validation.X
    y_val_clf = clf_dataset.validation.y

    X_train_reg = reg_dataset.train.X
    y_train_reg = reg_dataset.train.y
    X_val_reg = reg_dataset.validation.X
    y_val_reg = reg_dataset.validation.y

    delayed_pct = 100 * float(np.mean(y_train_clf))
    print(f"\n  Train delayed rate: {delayed_pct:.1f}%")
    print(f"  Train duration median: {float(np.median(y_train_reg)):.1f}h")

    print("\n" + "=" * 70)
    print("CLASSIFICATION — TimeSeriesSplit CV + HP Search (Train-only)")
    print("=" * 70)

    clf_results: list[ClassifierResult] = []

    for model_name in CLASSIFIER_NAMES:
        print(f"\n--- {model_name} ---")

        cv = train_classifier_cv(
            X_train_clf, y_train_clf, schema, model_name,
            search=True, n_search_candidates=5,
        )
        print(f"  CV PR-AUC:  {cv.pr_auc_mean:.4f} +/- {cv.pr_auc_std:.4f}")
        print(f"  CV ROC-AUC: {cv.roc_auc_mean:.4f} +/- {cv.roc_auc_std:.4f}")
        if cv.best_params:
            print(f"  Best params: {cv.best_params}")

        clf_result = calibrate_and_evaluate(
            X_train_clf, y_train_clf, X_val_clf, y_val_clf,
            schema, model_name, cv,
            calibration_methods=CALIBRATION_METHODS,
        )

        print(f"\n  {format_calibration_comparison(clf_result.calibration_results, clf_result.selected_calibration)}")

        val_proba = clf_result.calibrated_model.predict_proba(
            clf_result.base_pipeline.predict_proba(X_val_clf)[:, 1],
        )
        best_th = find_best_threshold(y_val_clf, val_proba)
        clf_result.threshold = best_th["threshold"]
        clf_result.threshold_details = best_th
        print(f"  Best threshold: {best_th['threshold']:.2f} (F1={best_th['f1']:.4f}, P={best_th['precision']:.4f}, R={best_th['recall']:.4f})")

        clf_results.append(clf_result)

    print("\n" + "=" * 70)
    print("REGRESSION — TimeSeriesSplit CV + HP Search (Train-only)")
    print("=" * 70)

    reg_results: list[RegressorResult] = []

    for model_name in REGRESSOR_NAMES:
        print(f"\n--- {model_name} ---")

        cv = train_regressor_cv(
            X_train_reg, y_train_reg, schema, model_name,
            search=True, n_search_candidates=5,
        )
        print(f"  CV MAE:  {cv.mae_mean:.2f}h +/- {cv.mae_std:.2f}h")
        print(f"  CV RMSE: {cv.rmse_mean:.2f}h +/- {cv.rmse_std:.2f}h")
        if cv.best_params:
            print(f"  Best params: {cv.best_params}")

        reg_result = fit_and_evaluate_regressor(
            X_train_reg, y_train_reg, X_val_reg, y_val_reg,
            schema, model_name,
            best_params=cv.best_params,
        )
        reg_result.cv_result = cv
        reg_results.append(reg_result)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n" + format_cv_summary(clf_results))
    print(f"\n--- Validation (Eylul) ---")
    print(format_classification_table(clf_results))

    print(f"\n{'-' * 70}")
    print("\nREGRESSION RESULTS")
    print(format_regression_table(reg_results))

    winner_clf = get_winner_classifier(clf_results)
    winner_reg = get_winner_regressor(reg_results)

    print(f"\n{'-' * 70}")
    if winner_clf:
        print(f"\nClassification winner: {winner_clf.model_name}")
        print(f"  Calibration: {winner_clf.selected_calibration}")
        print(f"  PR-AUC: {winner_clf.validation_metrics['pr_auc']:.4f}")
        print(f"  Threshold: {winner_clf.threshold:.3f}")
        print(f"  Brier: {winner_clf.brier_score:.4f}")
        print(f"  Best HP: {winner_clf.cv_result.best_params}")

    if winner_reg:
        print(f"\nRegression winner: {winner_reg.model_name}")
        print(f"  MAE: {winner_reg.validation_metrics['mae']:.2f}h")
        print(f"  RMSE: {winner_reg.validation_metrics['rmse']:.2f}h")
        print(f"  Best HP: {winner_reg.cv_result.best_params}")

    metrics: dict[str, Any] = {
        "phase": "faz5",
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": schema.feature_schema_version,
        "mapping_version": schema.canonical_mapping_version,
        "train_period": "2024-01..2024-08",
        "validation_period": "2024-09",
        "dataset_fingerprint_classification": dataset_fingerprint_cls,
        "dataset_fingerprint_regression": dataset_fingerprint_reg,
        "classification": {},
        "regression": {},
    }

    for r in clf_results:
        metrics["classification"][r.model_name] = {
            "cv_pr_auc_mean": r.cv_result.pr_auc_mean,
            "cv_pr_auc_std": r.cv_result.pr_auc_std,
            "cv_roc_auc_mean": r.cv_result.roc_auc_mean,
            "cv_roc_auc_std": r.cv_result.roc_auc_std,
            "best_params": r.cv_result.best_params,
            "selected_calibration": r.selected_calibration,
            "calibration_comparison": {
                m: {"brier": e.brier, "pr_auc": e.pr_auc, "roc_auc": e.roc_auc}
                for m, e in r.calibration_results.items()
            },
            "threshold": r.threshold,
            "threshold_details": r.threshold_details,
            "brier": r.brier_score,
            "validation": r.validation_metrics,
        }

    for r in reg_results:
        metrics["regression"][r.model_name] = {
            "cv_mae_mean": r.cv_result.mae_mean,
            "cv_mae_std": r.cv_result.mae_std,
            "cv_rmse_mean": r.cv_result.rmse_mean,
            "cv_rmse_std": r.cv_result.rmse_std,
            "best_params": r.cv_result.best_params,
            "validation": r.validation_metrics,
        }

    if winner_clf:
        metrics["classification"]["_winner"] = winner_clf.model_name
    if winner_reg:
        metrics["regression"]["_winner"] = winner_reg.model_name

    output_dir = project_root / "ml" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "faz5_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"\nMetrics saved: {output_path}")

    print("\nTest (Ekim-Kasim): MUHURLU")
    print("Audit (Aralik): MUHURLU")
    print("\nFaz 5 training tamamlandi.")


if __name__ == "__main__":
    main()
