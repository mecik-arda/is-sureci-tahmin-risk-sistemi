"""CatBoost model egitimi ve karsilastirmasi.

RandomForest/ElasticNet baseline'a alternatif olarak CatBoost classifier +
regressor denemesi. Kategorik feature'larin one-hot encoding olmadan
dogrudan desteklenmesi nedeniyle avantajlidir.

Kullanim:
    python scripts/train_catboost.py [--mode {local,demo}]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("APP_MODE", "local")

from app.core.database import SessionLocal, engine
from app.core.config import get_settings
from ml.features.schema_loader import load_feature_schema

settings = get_settings()
MODE = settings.app_mode.value
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / MODE


def build_datasets():
    """Train/Validation/Test split from process_snapshots (kronolojik)."""
    from sqlalchemy import and_, func, select
    from app.models.process import Process, ProcessSnapshot
    from app.models.prediction import PredictionRun

    db = SessionLocal()
    try:
        schema = load_feature_schema()
        categorical = schema.get("categorical", [])
        numeric = schema.get("numeric", [])

        rows = (
            db.query(ProcessSnapshot, Process)
            .join(Process, Process.id == ProcessSnapshot.process_id)
            .filter(ProcessSnapshot.snapshot_type == "opening")
            .order_by(Process.created_at)
            .all()
        )

        X_all = []
        y_clf = []
        y_reg = []
        dates = []
        snapshots_all = []

        from ml.features.feature_derivation import derive_features
        from ml.datasets.target_builder import compute_is_delayed, compute_total_duration_hours

        for snap, proc in rows:
            input_data = json.loads(snap.input_json)
            features = derive_features(input_data)
            row = []
            for f in categorical + numeric:
                val = features.get(f)
                row.append(val)
            X_all.append(row)
            snapshots_all.append(snap)

            is_delayed = compute_is_delayed(proc.completed_at, proc.deadline)
            duration = compute_total_duration_hours(proc.created_at, proc.completed_at)
            y_clf.append(is_delayed)
            y_reg.append(duration)
            dates.append(proc.created_at)

        X = np.array(X_all, dtype=object)
        y_clf_arr = np.array(y_clf, dtype=float)
        y_reg_arr = np.array(y_reg, dtype=float)
        dates_arr = np.array(dates)

        cutoff_train = datetime(2024, 9, 1, tzinfo=UTC)
        cutoff_val = datetime(2024, 10, 1, tzinfo=UTC)

        train_mask = dates_arr < cutoff_train
        val_mask = (dates_arr >= cutoff_train) & (dates_arr < cutoff_val)
        test_mask = dates_arr >= cutoff_val

        clf_mask_train = train_mask & (y_clf_arr >= 0)
        clf_mask_val = val_mask & (y_clf_arr >= 0)
        clf_mask_test = test_mask & (y_clf_arr >= 0)

        reg_mask_train = train_mask & (y_reg_arr >= 0) & np.isfinite(y_reg_arr)
        reg_mask_val = val_mask & (y_reg_arr >= 0) & np.isfinite(y_reg_arr)
        reg_mask_test = test_mask & (y_reg_arr >= 0) & np.isfinite(y_reg_arr)

        result = {
            "X_train_clf": X[clf_mask_train],
            "X_val_clf": X[clf_mask_val],
            "X_test_clf": X[clf_mask_test],
            "y_train_clf": y_clf_arr[clf_mask_train],
            "y_val_clf": y_clf_arr[clf_mask_val],
            "y_test_clf": y_clf_arr[clf_mask_test],
            "X_train_reg": X[reg_mask_train],
            "X_val_reg": X[reg_mask_val],
            "X_test_reg": X[reg_mask_test],
            "y_train_reg": y_reg_arr[reg_mask_train],
            "y_val_reg": y_reg_arr[reg_mask_val],
            "y_test_reg": y_reg_arr[reg_mask_test],
            "feature_names": categorical + numeric,
            "cat_features": list(range(len(categorical))),
        }
        print(f"Classification: Train={result['y_train_clf'].shape[0]} "
              f"Val={result['y_val_clf'].shape[0]} "
              f"Test={result['y_test_clf'].shape[0]}")
        print(f"Regression:     Train={result['y_train_reg'].shape[0]} "
              f"Val={result['y_val_reg'].shape[0]} "
              f"Test={result['y_test_reg'].shape[0]}")
        return result
    finally:
        db.close()


def train_catboost(datasets: dict) -> dict:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, f1_score,
        mean_absolute_error, precision_score, recall_score, roc_auc_score,
    )

    results = {}

    X_train = datasets["X_train_clf"]
    X_val = datasets["X_val_clf"]
    y_train = datasets["y_train_clf"]
    y_val = datasets["y_val_clf"]
    cat_features = datasets["cat_features"]

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        print("CatBoost kurulu degil. pip install catboost")
        return {}

    try:
        clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            cat_features=cat_features,
            random_seed=42,
            verbose=100,
            task_type="CPU",
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.35).astype(int)

        clf_results = {
            "pr_auc": float(average_precision_score(y_val, y_prob)),
            "roc_auc": float(roc_auc_score(y_val, y_prob)),
            "brier": float(brier_score_loss(y_val, y_prob)),
            "f1": float(f1_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        }
        results["classifier"] = clf_results
        print(f"\nCatBoost Classifier (Validation):")
        for k, v in clf_results.items():
            print(f"  {k}: {v:.4f}")

    X_train_reg = datasets["X_train_reg"]
    X_val_reg = datasets["X_val_reg"]
    y_train_reg = datasets["y_train_reg"]
    y_val_reg = datasets["y_val_reg"]

    reg = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            cat_features=cat_features,
            random_seed=42,
            verbose=100,
            task_type="CPU",
        )
        y_train_reg_log = np.log1p(np.clip(y_train_reg, 0, None))
        y_val_reg_log = np.log1p(np.clip(y_val_reg, 0, None))
        reg.fit(X_train_reg, y_train_reg_log, eval_set=(X_val_reg, y_val_reg_log), early_stopping_rounds=30)
        y_pred_raw = reg.predict(X_val_reg)
        y_pred = np.expm1(np.clip(y_pred_raw, 0, None))
        y_true = np.clip(y_val_reg, 0, None)

        reg_results = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "median_ae": float(np.median(np.abs(y_true - y_pred))),
        }
        results["regression"] = reg_results
        print(f"\nCatBoost Regressor (Validation, log1p):")
        for k, v in reg_results.items():
            print(f"  {k}: {v:.4f} hours")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "demo"], default=None)
    args = parser.parse_args()
    if args.mode:
        os.environ["APP_MODE"] = args.mode

    print(f"CatBoost egitimi basliyor... (APP_MODE={os.environ.get('APP_MODE', 'local')})")
    datasets = build_datasets()
    if datasets is None:
        return

    results = train_catboost(datasets)

    out_path = ARTIFACTS_DIR / f"catboost_results_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSonuclar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
