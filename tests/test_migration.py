"""Migration testleri."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


@pytest.fixture(autouse=True)
def _restore_alembic_url():
    """Her migration testinden sonra ALEMBIC_DATABASE_URL'i geri yükler."""
    saved = os.environ.get("ALEMBIC_DATABASE_URL")
    yield
    if saved is not None:
        os.environ["ALEMBIC_DATABASE_URL"] = saved
    else:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    os.environ["ALEMBIC_DATABASE_URL"] = db_url
    return cfg


def _get_table_names(db_url: str) -> set[str]:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        names = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
                )
            ).fetchall()
        }
    engine.dispose()
    return names


_EXPECTED_TABLES = {
    "processes",
    "process_snapshots",
    "import_runs",
    "data_quality_issues",
    "model_bundles",
    "prediction_runs",
    "prediction_feedback",
}


def test_migration_creates_all_tables(tmp_path):
    """Sıfırdan migration 7 tabloyu oluşturuyor."""
    db_url = f"sqlite:///{tmp_path / 'migration_test.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")

    assert _EXPECTED_TABLES == _get_table_names(db_url)


def test_migration_downgrade_removes_tables(tmp_path):
    """Downgrade tabloları kaldırıyor."""
    db_url = f"sqlite:///{tmp_path / 'downgrade_test.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    assert _get_table_names(db_url) == set()


def test_migration_upgrade_downgrade_upgrade(tmp_path):
    """Upgrade -> downgrade -> upgrade döngüsü sorunsuz çalışıyor."""
    db_url = f"sqlite:///{tmp_path / 'cycle_test.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    names = _get_table_names(db_url)
    assert "processes" in names
    assert "prediction_feedback" in names


def test_check_constraints_exist(tmp_path):
    """CHECK constraint'leri migration'da oluşturuluyor."""
    db_url = f"sqlite:///{tmp_path / 'constraint_test.db'}"
    cfg = _alembic_config(db_url)

    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    # is_active=5 CHECK constraint'ine takılmalı (SQLite execute anında kontrol eder)
    with pytest.raises(Exception):
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO model_bundles "
                    "(model_version, model_type, artifact_path, is_active) "
                    "VALUES ('test', 'classifier', 'test', 5)"
                )
            )
            conn.commit()
    engine.dispose()
