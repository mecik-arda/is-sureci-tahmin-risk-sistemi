"""Veri temizleme scripti.

Mevcut processes tablosundaki:
  - Negatif sureli (completed_at < created_at) kayitlari raporlar
  - Aykiri sure degerlerini IQR/IQR yontemiyle tespit eder
  - Winsorization sinirlarini hesaplar (veriye mudahale etmez)

Konfigurasyon: ml/config/cleaning_config.json (opsiyonel)

Not: Winsorization islemi model egitimi sirasinda feature pipeline'da uygulanir.
Bu script yalniz raporlama amaci tasir.

Kullanim:
    python scripts/clean_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.core.database import SessionLocal


def _load_cleaning_config() -> dict:
    config_path = project_root / "ml" / "config" / "cleaning_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _compute_bounds(values: list[float], iqr_mult: float = 1.5) -> tuple[float, float, float, float]:
    s = sorted(values)
    n = len(s)
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0
    q1 = s[int(n * 0.25)]
    q3 = s[int(n * 0.75)]
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    return lower, upper, q1, q3


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    config = _load_cleaning_config()

    try:
        import sqlalchemy as sa
        from app.models.process import Process

        rules = config.get("rules", {})
        iqr_mult = 1.5
        cap_config = rules.get("cap_outliers", {})
        if cap_config.get("enabled"):
            iqr_mult = float(cap_config.get("iqr_multiplier", 1.5))
        show_negatives = rules.get("remove_negative_durations", {}).get("enabled", True)

        rows = db.execute(
            sa.select(
                Process.id,
                Process.external_id,
                Process.completed_at,
                Process.created_at,
            ).where(
                Process.completed_at.isnot(None),
                Process.created_at.isnot(None),
            )
        ).fetchall()

        if not rows:
            print("Tamamlanmis surec kaydi bulunamadi.")
            return

        durations = []
        negative: list[dict] = []

        for r in rows:
            created = r.created_at
            completed = r.completed_at
            dur_h = (completed - created).total_seconds() / 3600
            if dur_h < 0:
                negative.append({
                    "id": r.id,
                    "external_id": r.external_id,
                    "created_at": str(created),
                    "completed_at": str(completed),
                    "duration_hours": round(dur_h, 2),
                })
            else:
                durations.append(dur_h)

        lower, upper, q1, q3 = _compute_bounds(durations, iqr_mult)

        outliers_above = [d for d in durations if d > upper]
        outliers_below = [d for d in durations if d < lower] if lower > 0 else []

        total = len(rows)

        print("=" * 60)
        print("         VERI TEMIZLEME RAPORU")
        print("=" * 60)
        print(f"  Toplam tamamlanmis kayit:    {total:,}")
        print(f"  Gecerli sureli (>=0):        {len(durations):,}")
        print(f"  Negatif sureli:              {len(negative)}")
        print(f"  Q1: {q1:.1f}h  |  Q3: {q3:.1f}h  |  IQR: {q3 - q1:.1f}h")
        print(f"  Winsorization alt sinir:     {lower:.1f}h")
        print(f"  Winsorization ust sinir:     {upper:.1f}h")
        print(f"  Alt sinirda aykiri:          {len(outliers_below)}")
        print(f"  Ust sinirda aykiri:          {len(outliers_above)} "
              f"({round(len(outliers_above)/len(durations)*100, 1) if durations else 0}%)")
        print()

        if negative:
            print("--- Negatif sureli kayitlar (ilk 10) ---")
            for n in negative[:10]:
                print(f"  ID={n['id']}  ext={n['external_id']}  "
                      f"{n['created_at'][:16]} -> {n['completed_at'][:16]}  "
                      f"sure={n['duration_hours']}h")
            print(f"  Toplam {len(negative)} negatif sureli. "
                  f"Bu kayitlar model egitiminde veri kalite sorunu olarak isaretlenir.")
            print()

        if outliers_above:
            print(f"--- Ust aykiri sureler (ilk 5 / {len(outliers_above)}) ---")
            for v in sorted(outliers_above, reverse=True)[:5]:
                print(f"  {v:.1f}h")
            print(f"  Winsorization ust sinir: {upper:.1f}h. "
                  f"Bu degerin uzerindeki sureler egitimde cap'lenir.")
            print()

        out_dir = settings.effective_artifact_dir / "data_quality"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cleaning_report.json"
        report_data = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {"iqr_multiplier": iqr_mult},
            "total_completed": total,
            "valid_durations": len(durations),
            "negative_duration_count": len(negative),
            "negative_duration_sample": negative[:10] if show_negatives else [],
            "iqr_bounds": {"q1": round(q1, 2), "q3": round(q3, 2),
                           "lower": round(lower, 2), "upper": round(upper, 2)},
            "outliers_below_lower": len(outliers_below),
            "outliers_above_upper": len(outliers_above),
            "outliers_above_pct": round(len(outliers_above)/len(durations)*100, 1) if durations else 0,
        }
        out_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"  Rapor kaydedildi: {out_path}")

    except Exception as exc:
        print(f"  HATA: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
