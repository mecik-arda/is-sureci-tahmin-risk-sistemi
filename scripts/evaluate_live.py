"""Canlı model performans değerlendirmesi — yeni kapanan süreçler üzerinde.

Aktif model bundle'ı yeni kapanan süreçler üzerinde değerlendirir ve
model drift göstergesi olarak metrikleri hesaplar.

Kullanım:
    python scripts/evaluate_live.py [--mode {local,demo}]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("APP_MODE", "local")

import argparse
import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.model_bundle import ModelBundle
from app.models.process import Process, ProcessSnapshot
from app.services.model_loader import find_active_bundle, load_bundle
from app.services.prediction_service import predict_single

settings = get_settings()
MODE = settings.app_mode.value
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / MODE / "evaluations"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    result = {
        "classification": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
            "brier": float(brier_score_loss(y_true, y_prob)),
            "sample_count": len(y_true),
        },
    }

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    result["confusion_matrix"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    return result


def evaluate(threshold: float = 0.35) -> dict | None:
    db = SessionLocal()
    try:
        active = find_active_bundle(db)
        if active is None:
            print("Aktif model bulunamadı.")
            return None

        loaded = load_bundle(db, bundle_id=active.id)
        print(f"Aktif model: {active.model_version} ({active.model_type}), threshold={threshold}")

        closed_processes = list(db.execute(
            select(Process).where(
                Process.completed_at.is_not(None),
                Process.deadline.is_not(None),
            )
        ).scalars().all())

        if not closed_processes:
            print("Değerlendirilecek kapanmış süreç (SLA'lı) bulunamadı.")
            return None

        print(f"Kapanmış SLA'lı süreç sayısı: {len(closed_processes)}")

        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        skipped = 0
        prediction_failures = 0

        for proc in closed_processes:
            snap_q = db.execute(
                select(ProcessSnapshot).where(
                    ProcessSnapshot.process_id == proc.id,
                    ProcessSnapshot.snapshot_type == "opening",
                )
            ).scalars().first()

            if snap_q is None:
                skipped += 1
                continue

            is_delayed = 1 if proc.completed_at > proc.deadline else 0

            try:
                result = predict_single(db, proc.id, loaded)
                db.commit()
                delay_prob = float(result.prediction_run.delay_probability or 0.0)
                y_prob_list.append(delay_prob)
                y_pred_list.append(1 if delay_prob >= threshold else 0)
                y_true_list.append(is_delayed)
            except Exception:
                db.rollback()
                prediction_failures += 1

        if not y_true_list:
            return None

        y_true = np.array(y_true_list, dtype=int)
        y_pred = np.array(y_pred_list, dtype=int)
        y_prob = np.array(y_prob_list, dtype=float)

        clf_results = compute_metrics(y_true, y_pred, y_prob)

        report = {
            "model_version": active.model_version,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "threshold": threshold,
            "total_evaluated": len(y_true),
            "skipped_no_snapshot": skipped,
            "prediction_failures": prediction_failures,
            "classification": clf_results["classification"],
            "confusion_matrix": clf_results["confusion_matrix"],
        }

        prev_reports = sorted(ARTIFACTS_DIR.glob("evaluation_*.json"))
        if prev_reports:
            prev = json.loads(prev_reports[-1].read_text(encoding="utf-8"))
            prev_clf = prev.get("classification", {})
            curr_clf = report["classification"]
            for metric in ("f1", "brier"):
                pv = prev_clf.get(metric)
                cv = curr_clf.get(metric)
                if pv is not None and cv is not None:
                    diff = cv - pv
                    direction = "YUKSELDI" if diff > 0 else ("DUSTU" if diff < 0 else "DEGISMEDI")
                    print(f"  {metric}: {pv:.4f} -> {cv:.4f} ({diff:+.4f}, {direction})")

        print(f"\nSiniflandirma (n={len(y_true)}):")
        print(f"  Accuracy:  {report['classification']['accuracy']:.4f}")
        print(f"  F1:        {report['classification']['f1']:.4f}")
        roc_val = report['classification']['roc_auc']
        print(f"  ROC AUC:   {roc_val:.4f}" if roc_val is not None else "  ROC AUC:   N/A (tek sinif)")
        print(f"  Brier:     {report['classification']['brier']:.4f}")
        print(f"  Confusion: TP={report['confusion_matrix']['tp']} FP={report['confusion_matrix']['fp']} TN={report['confusion_matrix']['tn']} FN={report['confusion_matrix']['fn']}")

        out_path = ARTIFACTS_DIR / f"evaluation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRapor kaydedildi: {out_path}")

        return report
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "demo"], default=None)
    args = parser.parse_args()
    if args.mode:
        os.environ["APP_MODE"] = args.mode

    global ARTIFACTS_DIR
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / os.environ.get("APP_MODE", MODE) / "evaluations"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Canli model degerlendirmesi basliyor... (APP_MODE={os.environ.get('APP_MODE', 'local')})")
    evaluate()


if __name__ == "__main__":
    main()
