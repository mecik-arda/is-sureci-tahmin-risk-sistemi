"""Demo verisi olusturma scripti.

Kullanim:
    python scripts/generate_demo_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.process import Process, ProcessSnapshot

CAT_SOURCE = ["citizens_connect_app", "phone_call", "web_form", "internal_staff", "mayors_office", "city_council"]
CAT_SUBJECT = ["pothole_not_filled", "graffiti_removal", "street_light_outage", "trash_missed_pickup",
               "road_maintenance", "sidewalk_repair", "tree_hazard", "rodent_sighting", "noise_complaint", "parking_violation"]
CAT_REASON = ["road_maintenance", "public_nuisance", "health_sanitation", "traffic_safety", "property_violation",
              "environmental", "utility_issue", "public_safety", "administrative", "infrastructure"]
CAT_TYPE = ["street_repair", "property_violation", "sanitation", "traffic_signal", "public_safety",
            "environmental", "noise_control", "park_maintenance", "building_inspection", "water_drainage"]
CAT_NEIGHBORHOOD = ["dorchester_02121", "roxbury_02119", "south_boston_02127", "jamaica_plain_02130",
                    "east_boston_02128", "brighton_02135", "mattapan_02126", "hyde_park_02136",
                    "roslindale_02131", "west_roxbury_02132", "charlestown_02129", "allston_02134"]

NP_RANDOM = np.random.RandomState(42)
TOTAL_RECORDS = 8000


def main() -> None:
    settings = get_settings()
    if not settings.is_demo:
        print("Bu script yalnizca demo modunda calisir. APP_MODE=demo olarak ayarlayin.")
        return

    db = SessionLocal()
    existing = db.execute(text("SELECT COUNT(*) FROM processes")).scalar()
    if existing > 0:
        print(f"Veritabaninda zaten {existing} kayit var. Temizlenip yeniden olusturulacak.")
        db.execute(text("DELETE FROM processes"))
        db.commit()

    processes = []
    snapshots = []

    rec = 0
    day = datetime(2024, 1, 1, 0, 0, 0)

    while rec < TOTAL_RECORDS and day < datetime(2024, 12, 31, 23, 59, 0):
        n_today = NP_RANDOM.poisson(20) + 1
        for _ in range(min(n_today, TOTAL_RECORDS - rec)):
            hour = NP_RANDOM.randint(0, 23)
            minute = NP_RANDOM.randint(0, 59)
            created_at = day.replace(hour=hour, minute=minute)

            has_sla = NP_RANDOM.rand() > 0.1
            sla_hours = int(NP_RANDOM.choice([24, 48, 72, 120, 168, 336, 720]))
            deadline = created_at + timedelta(hours=sla_hours) if has_sla else None

            will_be_closed = NP_RANDOM.rand() > 0.15
            if will_be_closed:
                duration_hours = NP_RANDOM.lognormal(mean=3.5, sigma=1.2)
                completed_at = created_at + timedelta(hours=float(duration_hours))
            else:
                completed_at = None

            is_delayed_val = 0
            if completed_at is not None and deadline is not None and completed_at > deadline:
                is_delayed_val = 1

            closure_reason = None
            current_status = "Open"
            if completed_at is not None:
                current_status = "Closed"
                closure_reason = NP_RANDOM.choice(["Resolved", "No Violation Found", "Duplicate", "Administratively Closed"])

            external_id = f"DEMO-{rec:06d}"
            source = NP_RANDOM.choice(CAT_SOURCE)
            subject = NP_RANDOM.choice(CAT_SUBJECT)
            reason = NP_RANDOM.choice(CAT_REASON)
            ptype = NP_RANDOM.choice(CAT_TYPE)
            neighborhood = NP_RANDOM.choice(CAT_NEIGHBORHOOD)

            source_payload = json.dumps({
                "source": source.replace("_", " ").title(),
                "subject": subject.replace("_", " ").title(),
                "reason": reason.replace("_", " ").title(),
                "type": ptype.replace("_", " ").title(),
                "neighborhood": neighborhood.replace("_", " ").title(),
            })

            process = Process(
                external_id=external_id,
                process_type=ptype,
                current_status=current_status,
                created_at=created_at,
                deadline=deadline,
                completed_at=completed_at,
                closure_reason=closure_reason,
                source_payload_json=source_payload,
                imported_at=created_at,
            )
            db.add(process)
            db.flush()

            input_json = json.dumps({
                "created_at": created_at.isoformat(),
                "deadline": deadline.isoformat() if deadline else None,
                "source": source,
                "subject": subject,
                "reason": reason,
                "type": ptype,
                "neighborhood": neighborhood,
            })
            input_fp = f"demo-fp-{external_id}"

            snapshot = ProcessSnapshot(
                process_id=process.id,
                snapshot_type="opening",
                snapshot_at=created_at,
                feature_schema_version="opening-v1",
                input_json=input_json,
                input_fingerprint=input_fp,
            )
            db.add(snapshot)
            db.flush()

            rec += 1
            if rec % 500 == 0:
                db.commit()
                print(f"  {rec}/{TOTAL_RECORDS} kayit olusturuldu...")

        day += timedelta(days=1)

    db.commit()
    db.close()

    print(f"\nToplam {rec} kayit olusturuldu (process + snapshot).")
    print("Demo veri seti hazir.")


if __name__ == "__main__":
    main()
