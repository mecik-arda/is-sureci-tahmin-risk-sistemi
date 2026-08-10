"""Pytest ortak fixture'ları.

Testler harici ağ bağlantısı olmadan, geçici (temp) veritabanında çalışır.
Gerçek demo.db veya process_risk.db dosyaları asla kullanılmaz.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

os.environ["APP_MODE"] = "demo"
os.environ.setdefault("LOG_LEVEL", "WARNING")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASETEMP = _PROJECT_ROOT / ".pytest_tmp"


def pytest_configure(config):
    """Pytest basetemp'i proje altina al (Windows ACL PermissionError onlemi)."""
    _BASETEMP.mkdir(parents=True, exist_ok=True)
    if not config.option.basetemp:
        config.option.basetemp = str(_BASETEMP)


@pytest.fixture(scope="session")
def temp_db_path(tmp_path_factory) -> Path:
    """Session boyunca geçici veritabanı dosya yolu."""
    base = tmp_path_factory.mktemp("session_db", numbered=False)
    return base / "test.db"


@pytest.fixture(scope="session")
def temp_db_url(temp_db_path: Path) -> str:
    """Session boyunca geçici veritabanı URL'si."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture(scope="session")
def test_engine(temp_db_url: str) -> Engine:
    """Testler için engine oluşturur ve PRAGMA'ları ayarlar."""
    engine = create_engine(
        temp_db_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(temp_db_url: str) -> Generator[None, None, None]:
    """Migration'ları test veritabanına uygular."""
    from alembic import command
    from alembic.config import Config

    os.environ["ALEMBIC_DATABASE_URL"] = temp_db_url

    alembic_cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))

    command.upgrade(alembic_cfg, "head")
    yield
    os.environ.pop("ALEMBIC_DATABASE_URL", None)


@pytest.fixture(scope="session", autouse=True)
def reset_analysis_cache() -> Generator[None, None, None]:
    """Her test session'ında analysis dataset cache'ini sıfırla."""
    from app.services.analysis_dataset import analysis_dataset_service
    analysis_dataset_service.reset()
    yield
    analysis_dataset_service.reset()


@pytest.fixture(scope="function")
def db_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Her test için temiz bir veritabanı session'ı sağlar.

    Her test öncesi tüm tablolar temizlenir (TRUNCATE).
    """
    from app.models.base import Base

    session = Session(bind=test_engine)
    session.execute(text("PRAGMA foreign_keys=OFF"))
    for table in reversed(Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.execute(text("PRAGMA foreign_keys=ON"))
    session.commit()

    yield session

    session.close()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient; get_db dependency'sini test session'ına yönlendirir."""
    from app.dependencies import get_db
    from app.main import app

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def canonical_mapper() -> "CanonicalMapper":
    """Testler için yüklü canonical mapper sağlar."""
    from app.services.canonical_mapper import CanonicalMapper

    mapping_path = Path(__file__).resolve().parent.parent / "ml" / "mappings" / "canonical_map_v1.json"
    return CanonicalMapper(mapping_path)


@pytest.fixture
def make_csv(tmp_path: Path):
    """Test CSV dosyası oluşturur. make_csv(rows: list[dict]) -> Path."""
    import csv

    def _make(rows: list[dict], filename: str = "test.csv") -> Path:
        path = tmp_path / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return _make
