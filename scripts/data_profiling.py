"""Veri profilleme — processes ve process_snapshots tablolarından kalite ölçümleri.

Kullanım:
    python scripts/data_profiling.py

Çıktı:
    - Console log (eksik değer, aykırı değer, mükerrer özeti)
    - artifacts/{mode}/data_quality/profile_report.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("APP_MODE", "local")

import numpy as np

from app.core.config import get_settings
from app.core.database import SessionLocal, engine
from app.models.import_run import ImportRun
from app.models.process import Process, ProcessSnapshot
from sqlalchemy import select

settings = get_settings()
MODE = settings.app_mode.value
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / MODE / "data_quality"


def profile() -> dict:
    db = SessionLocal()
    try:
        processes = list(db.execute(select(Process)).scalars().all())
        total = len(processes)
        if total == 0:
            print("Hiç süreç kaydı bulunamadı.")
            return {"error": "empty"}

        numeric_cols = ["total_duration_hours"]
        categorical_cols = ["process_type", "current_status", "closure_reason"]
        date_cols = ["created_at", "completed_at", "deadline"]

        report: dict = {
            "mode": MODE,
            "profiled_at": datetime.now(UTC).isoformat(),
            "total_processes": total,
            "columns": {},
        }

        missing_ext_id = sum(1 for p in processes if not p.external_id)
        if missing_ext_id:
            report["columns"]["external_id"] = {"missing": missing_ext_id, "missing_pct": round(missing_ext_id / total * 100, 2)}

        for col in categorical_cols:
            vals = [getattr(p, col) for p in processes]
            null_count = sum(1 for v in vals if v is None or (isinstance(v, str) and v.strip() == ""))
            report["columns"][col] = {
                "missing": null_count,
                "missing_pct": round(null_count / total * 100, 2),
                "unique": len(set(v for v in vals if v is not None and (not isinstance(v, str) or v.strip() != ""))),
                "top_5": _top_n([v for v in vals if v is not None and (isinstance(v, str) and v.strip() != "")], 5),
            }

        durations = []
        for p in processes:
            if p.created_at and p.completed_at:
                dur = (p.completed_at - p.created_at).total_seconds() / 3600
                if dur >= 0:
                    durations.append(dur)
        if durations:
            arr = np.array(durations)
            q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(np.sum((arr < lower) | (arr > upper)))
            report["columns"]["total_duration_hours"] = {
                "count": len(arr),
                "missing": total - len(arr),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outliers_iqr": outliers,
            }

        dup_external = {}
        for p in processes:
            eid = p.external_id
            if eid:
                dup_external[eid] = dup_external.get(eid, 0) + 1
        duplicate_ids = {k: v for k, v in dup_external.items() if v > 1}
        report["duplicate_external_ids"] = len(duplicate_ids)
        if duplicate_ids:
            report["duplicate_external_ids_detail"] = dict(sorted(duplicate_ids.items(), key=lambda x: -x[1])[:10])

        snapshots = list(db.execute(select(ProcessSnapshot)).scalars().all())
        report["total_snapshots"] = len(snapshots)
        snap_types = {}
        for s in snapshots:
            snap_types[s.snapshot_type] = snap_types.get(s.snapshot_type, 0) + 1
        report["snapshot_types"] = snap_types

        return report
    finally:
        db.close()


def _top_n(values: list[str], n: int = 5) -> list[dict]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return [{"value": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])[:n]]


def main() -> None:
    print(f"Veri profilleme başlıyor... (APP_MODE={MODE})")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = profile()
    out_path = ARTIFACTS_DIR / "profile_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Profil raporu kaydedildi: {out_path}")

    if "total_processes" in report:
        total = report["total_processes"]
        print(f"\nToplam süreç: {total}")
        dup = report.get("duplicate_external_ids", 0)
        if dup:
            print(f"Mükerrer external_id: {dup} grup")

        for col, info in report.get("columns", {}).items():
            miss = info.get("missing", 0)
            miss_pct = info.get("missing_pct", 0)
            print(f"  {col}: {miss} eksik (%{miss_pct:.2f})")
            if "outliers_iqr" in info:
                print(f"    -> {info['outliers_iqr']} aykırı değer (IQR yöntemi)")

    print("\nProfilleme tamamlandı.")


if __name__ == "__main__":
    main()
