"""Faz 3 baseline çalistirma scripti.

Kullanim:
    $env:APP_MODE="local"
    python scripts/run_faz3_baseline.py

Önce veri import edilmis olmali:
    python scripts/import_process_data.py --file tmpm461rr5o.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from app.core.config import get_settings
from app.core.database import SessionLocal
from ml.datasets.dataset_builder import build_dataset
from ml.features.schema_loader import load_feature_schema
from ml.training.baseline_classifier import train_classification_baselines
from ml.training.baseline_regressor import train_regression_baselines


def _format_confusion_matrix(cm: list[list[int]]) -> str:
    return f"TN={cm[0][0]}  FP={cm[0][1]}  |  FN={cm[1][0]}  TP={cm[1][1]}"


def main() -> None:
    settings = get_settings()
    print(f"APP_MODE: {settings.app_mode.value}")
    print(f"Database: {settings.effective_database_url}")
    print()

    schema = load_feature_schema()
    print(f"Feature Schema: {schema.feature_schema_version}")
    print(f"Canonical Mapping: {schema.canonical_mapping_version}")
    print(f"Features: {schema.all_features}")
    print()

    session = SessionLocal()
    try:
        result = build_dataset(session, schema)
    finally:
        session.close()

    print("=" * 70)
    print("DATASET BUILD SONUÇLARI")
    print("=" * 70)
    print(f"Toplam snapshot:           {result.total_snapshots:>10,}")
    print(f"Schema hatali satir:       {len(result.schema_errors):>10,}")
    print(f"Split disi (tarih disi):   {result.unassigned_split:>10,}")
    print()

    cls_meta = result.classification.metadata
    reg_meta = result.regression.metadata
    print("SINIFLANDIRMA KOHORTU")
    print(f"  Train:       {cls_meta.split_counts['train']:>10,}")
    print(f"  Validation:  {cls_meta.split_counts['validation']:>10,}")
    print(f"  Test [SEALED]: {cls_meta.split_counts['test']:>10,}")
    print(f"  Audit [SEALED]: {cls_meta.split_counts['audit']:>10,}")
    print(f"  Excluded:    {result.classification_excluded:>10,}")
    print(f"  Fingerprint: {result.classification.fingerprint[:16]}...")
    print()

    print("REGRESYON KOORTU")
    print(f"  Train:       {reg_meta.split_counts['train']:>10,}")
    print(f"  Validation:  {reg_meta.split_counts['validation']:>10,}")
    print(f"  Test [SEALED]: {reg_meta.split_counts['test']:>10,}")
    print(f"  Audit [SEALED]: {reg_meta.split_counts['audit']:>10,}")
    print(f"  Excluded:    {result.regression_excluded:>10,}")
    print(f"  Fingerprint: {result.regression.fingerprint[:16]}...")
    print()

    if result.schema_errors:
        print(f"UYARI: {len(result.schema_errors)} satirda schema hatasi:")
        for err in result.schema_errors[:5]:
            print(f"  {err}")
        if len(result.schema_errors) > 5:
            print(f"  ... ve {len(result.schema_errors) - 5} satir daha")
        print()

    if result.classification.train.row_count == 0:
        print("HATA: Siniflandirma train kümesi bos. Veri import edildi mi?")
        return

    print("=" * 70)
    print("SINIFLANDIRMA BASELINE (VALIDATION)")
    print("=" * 70)
    cls_results = train_classification_baselines(result.classification, schema)
    for name, metrics in cls_results.items():
        print(f"\n  [{name}]")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Confusion: {_format_confusion_matrix(metrics['confusion_matrix'])}")

    print()
    print("=" * 70)
    print("REGRESYON BASELINE (VALIDATION)")
    print("=" * 70)
    reg_results = train_regression_baselines(result.regression, schema)
    for name, metrics in reg_results.items():
        print(f"\n  [{name}]")
        print(f"    MAE:        {metrics['mae']:.2f} saat")
        print(f"    Median AE:  {metrics['median_ae']:.2f} saat")
        print(f"    RMSE:       {metrics['rmse']:.2f} saat")
        print(f"    p90 AE:     {metrics['p90_abs_error']:.2f} saat")

    print()
    print("=" * 70)
    print("SEALED HOLDOUT DURUMU")
    print("=" * 70)
    print(f"  Test:  {result.classification.test_info.row_count:,} kayit [MÜHÜRLÜ]")
    print(f"  Audit: {result.classification.audit_info.row_count:,} kayit [MÜHÜRLÜ]")
    print(f"  Test/Audit üzerinde metrik hesaplanMADI.")
    print()


if __name__ == "__main__":
    main()
