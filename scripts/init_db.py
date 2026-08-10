"""Veritabanını sıfırdan oluşturan kurulum betiği.

Kullanım:
    python scripts/init_db.py

Mevcut veritabanını siler ve migration'ları sıfırdan uygular.
UYARI: Bu betik tüm verileri siler; yalnızca ilk kurulumda kullanın.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from alembic import command
from alembic.config import Config

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    db_url = settings.effective_database_url

    # SQLite dosya yolunu çıkar
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", ""))
    else:
        db_path = None

    print(f"Ortam modu: {settings.app_mode.value}")
    print(f"Veritabani: {db_url}")
    print(f"Artifact dizini: {settings.effective_artifact_dir}")

    # Mevcut veritabanını sil
    if db_path and db_path.exists():
        print(f"Mevcut veritabani siliniyor: {db_path}")
        db_path.unlink()

    # Migration'ları uygula
    cfg = Config(str(project_root / "alembic.ini"))
    command.upgrade(cfg, "head")

    print("Veritabani basariyla olusturuldu.")


if __name__ == "__main__":
    main()
