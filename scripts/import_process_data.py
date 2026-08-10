"""CSV/XLSX dosyası içe aktaran CLI betiği.

Kullanım:
    python scripts/import_process_data.py --file <dosya_yolu>

Örnek:
    python scripts/import_process_data.py --file tmpm461rr5o.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.canonical_mapper import CanonicalMapper
from app.services.import_service import run_import


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV/XLSX dosyasini iceri aktarir."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Içe aktarilacak dosya yolu (CSV veya XLSX).",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"HATA: Dosya bulunamadi: {file_path}")
        sys.exit(1)

    settings = get_settings()
    mapping_path = project_root / "ml" / "mappings" / "canonical_map_v1.json"

    if not mapping_path.exists():
        print(f"HATA: Mapping katalogu bulunamadi: {mapping_path}")
        sys.exit(1)

    mapper = CanonicalMapper(mapping_path)

    print(f"Dosya: {file_path.name}")
    print(f"Ortam: {settings.app_mode.value}")
    print(f"Mapping surumu: {mapper.version}")
    print("Isleniyor...")

    session = SessionLocal()
    try:
        result = run_import(session, file_path, mapper)
        session.commit()
    except Exception as exc:
        session.rollback()
        print(f"HATA: Import basarisiz: {type(exc).__name__}")
        sys.exit(1)
    finally:
        session.close()

    c = result.counts
    print()
    print("=== IMPORT SONUCU ===")
    print(f"Durum:           {result.status}")
    print(f"Toplam satir:    {c.total_rows}")
    print(f"Yeni kayit:      {c.inserted_rows}")
    print(f"Guncellenen:     {c.updated_rows}")
    print(f"Tekrar (skip):   {c.skipped_duplicate_rows}")
    print(f"Karantina:       {c.quarantined_rows}")
    print(f"Hata satiri:     {c.error_rows}")
    print(f"Uyari sayisi:    {c.warning_count}")
    print(f"Import run ID:   {result.import_run_id}")
    print(f"Dosya SHA256:    {result.file_hash[:16]}...")

    if result.errors:
        print("\nHatalar:")
        for err in result.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
