"""Veritabanı altyapı testleri."""

from __future__ import annotations

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine


def test_foreign_keys_pragma_enabled(test_engine: Engine):
    """SQLite bağlantısında foreign_keys PRAGMA'sı ON."""
    with test_engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys")).fetchone()
        assert result[0] == 1


def test_busy_timeout_pragma_set(test_engine: Engine):
    """SQLite bağlantısında busy_timeout ayarlı."""
    with test_engine.connect() as conn:
        result = conn.execute(text("PRAGMA busy_timeout")).fetchone()
        assert result[0] == 5000


def test_foreign_key_enforcement(test_engine: Engine):
    """Foreign key constraint gerçekten uygulanıyor mu?"""
    with test_engine.connect() as conn:
        from sqlalchemy.exc import IntegrityError

        with __import__("pytest").raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO process_snapshots "
                    "(process_id, snapshot_type, snapshot_at, input_json) "
                    "VALUES (999999, 'opening', '2024-01-01', '{}')"
                )
            )
            conn.commit()


def test_session_isolation(db_session):
    """Session düzgün çalışıyor."""
    result = db_session.execute(text("SELECT 1")).fetchone()
    assert result[0] == 1
