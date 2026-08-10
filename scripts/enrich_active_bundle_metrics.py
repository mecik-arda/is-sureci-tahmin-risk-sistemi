from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import brier_score_loss

from app.core.database import SessionLocal
from app.services.model_loader import find_active_bundle, load_bundle
from ml.datasets.dataset_builder import build_dataset
from ml.evaluation.metrics import evaluate_classification, evaluate_regression
from ml.features.schema_loader import load_feature_schema


def enrich_metrics(
    stored: dict,
    cls_metrics: dict,
    reg_metrics: dict,
) -> dict:
    classifier = stored.setdefault("classifier", {})
    classifier.update({
        "validation_roc_auc": cls_metrics["roc_auc"],
        "validation_pr_auc": cls_metrics["pr_auc"],
        "validation_brier": cls_metrics["brier"],
        "validation_f1": cls_metrics["f1"],
        "validation_precision": cls_metrics["precision"],
        "validation_recall": cls_metrics["recall"],
        "classification_validation_row_count": cls_metrics["row_count"],
        "validation_confusion_matrix": cls_metrics["confusion_matrix"],
    })
    regression = stored.setdefault("regression", {})
    regression.update({
        "validation_mae": reg_metrics["mae"],
        "validation_median_ae": reg_metrics["median_ae"],
        "validation_rmse": reg_metrics["rmse"],
        "validation_p90_ae": reg_metrics["p90_abs_error"],
        "regression_validation_row_count": reg_metrics["row_count"],
    })
    return stored


def main() -> None:
    session = SessionLocal()
    try:
        record = find_active_bundle(session)
        if record is None:
            raise RuntimeError("Aktif model bundle bulunamadı.")
        bundle = load_bundle(session, record.id)
        dataset = build_dataset(session, load_feature_schema())
        metadata = bundle.metadata
        if dataset.classification.fingerprint != metadata.get("classification_fingerprint"):
            raise RuntimeError("Classification dataset fingerprint uyuşmuyor.")
        if dataset.regression.fingerprint != metadata.get("regression_fingerprint"):
            raise RuntimeError("Regression dataset fingerprint uyuşmuyor.")

        cls_validation = dataset.classification.validation
        raw_proba = bundle.classifier.predict_proba(cls_validation.X)[:, 1]
        proba = bundle.calibration_model.predict_proba(raw_proba) if bundle.calibration_model is not None else raw_proba
        cls_metrics = evaluate_classification(
            cls_validation.y,
            (proba >= bundle.threshold).astype(int),
            proba,
            "validation",
        )
        cls_metrics["brier"] = float(brier_score_loss(cls_validation.y, proba))

        reg_validation = dataset.regression.validation
        reg_pred = np.maximum(0.0, bundle.regressor.predict(reg_validation.X))
        reg_metrics = evaluate_regression(reg_validation.y, reg_pred, "validation")

        stored = json.loads(record.metrics_json or "{}")
        record.metrics_json = json.dumps(enrich_metrics(stored, cls_metrics, reg_metrics), ensure_ascii=False)
        session.commit()
        print(json.dumps({
            "model_version": record.model_version,
            "classification_validation_rows": cls_metrics["row_count"],
            "regression_validation_rows": reg_metrics["row_count"],
        }))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
