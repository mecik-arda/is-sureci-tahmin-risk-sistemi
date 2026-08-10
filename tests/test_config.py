"""Konfigürasyon testleri."""

from __future__ import annotations

import importlib

import pytest


def test_config_demo_mode(monkeypatch):
    """APP_MODE=demo doğru şekilde yükleniyor."""
    monkeypatch.setenv("APP_MODE", "demo")
    from app.core.config import AppMode, Settings

    settings = Settings()
    assert settings.app_mode is AppMode.DEMO
    assert settings.is_demo is True
    assert "demo.db" in settings.effective_database_url
    assert "demo" in str(settings.effective_artifact_dir)


def test_config_local_mode(monkeypatch):
    """APP_MODE=local doğru şekilde yükleniyor."""
    monkeypatch.setenv("APP_MODE", "local")
    from app.core.config import AppMode, Settings

    settings = Settings()
    assert settings.app_mode is AppMode.LOCAL
    assert settings.is_demo is False
    assert "process_risk.db" in settings.effective_database_url
    assert "production" in str(settings.effective_artifact_dir)


def test_config_invalid_mode(monkeypatch):
    """Tanımsız APP_MODE ValueError fırlatıyor."""
    monkeypatch.setenv("APP_MODE", "production")
    from app.core.config import Settings

    with pytest.raises(ValueError, match="APP_MODE"):
        Settings()


def test_config_defaults(monkeypatch):
    """Varsayılan değerler doğru."""
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.delenv("APP_HOST", raising=False)
    monkeypatch.delenv("APP_PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    from app.core.config import Settings

    settings = Settings()
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 8000
    assert settings.log_level == "INFO"


def test_demo_and_local_db_paths_differ(monkeypatch):
    """Demo ve local veritabanları farklı dosyalara işaret ediyor."""
    from app.core.config import Settings

    monkeypatch.setenv("APP_MODE", "demo")
    demo = Settings()
    demo_url = demo.effective_database_url

    monkeypatch.setenv("APP_MODE", "local")
    local = Settings()
    local_url = local.effective_database_url

    assert demo_url != local_url


def test_demo_and_local_artifact_dirs_differ(monkeypatch):
    """Demo ve local artifact dizinleri farklı."""
    from app.core.config import Settings

    monkeypatch.setenv("APP_MODE", "demo")
    demo = Settings()

    monkeypatch.setenv("APP_MODE", "local")
    local = Settings()

    assert demo.effective_artifact_dir != local.effective_artifact_dir
