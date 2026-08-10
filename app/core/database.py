"""Senkron SQLAlchemy 2.0 veritabanı altyapısı.

- Engine her uygulama ömrü boyunca tekildir.
- Her HTTP isteği için ayrı bir Session oluşturulur (dependency injection).
- SQLite bağlantısında foreign_keys ve busy_timeout PRAGMA'ları uygulanır.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _build_engine() -> Engine:
    """Konfigürasyondan SQLAlchemy engine oluşturur."""
    settings = get_settings()
    url = settings.effective_database_url
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine = create_engine(
        url,
        echo=False,
        future=True,
        connect_args=connect_args,
    )

    busy_timeout = settings.sqlite_busy_timeout_ms

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        """Her yeni SQLite bağlantısında PRAGMA'ları uygular."""
        if engine.dialect.name != "sqlite":
            return
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={busy_timeout}")
        cursor.close()

    return engine


# Tekil engine ve session fabrikası
engine: Engine = _build_engine()
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: her istek için ayrı bir Session sağlar.

    İstek bittiğinde (ister başarılı ister hatalı) session kapatılır.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection(db: Session) -> bool:
    """Veritabanı bağlantısının aktif olduğunu doğrular."""
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
