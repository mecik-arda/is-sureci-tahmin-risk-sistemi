"""Merkezi ve tip güvenli konfigürasyon.

İki çalışma modu vardır:
    - APP_MODE=demo  : Sentetik veri ve denemeler.
    - APP_MODE=local : Gerçek kurum verisi ve üretim modelleri.

Demo ve local ortamları farklı SQLite veritabanı ve artifact dizinleri
kullanır; böylece sentetik ve gerçek veriler hiçbir şekilde karışmaz.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    """Uygulama çalışma modu."""

    DEMO = "demo"
    LOCAL = "local"


# Proje kök dizini (bu dosyanın iki üst dizini)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Uygulama genelinde kullanılan tüm konfigürasyon değerleri."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Çalışma modu ---
    app_mode: AppMode = AppMode.DEMO

    # --- Sunucu ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # --- Loglama ---
    log_level: str = "INFO"

    # --- Veritabanı (moda göre dinamik) ---
    database_url: str | None = None

    # --- Model artifact dizini ---
    artifact_dir: str | None = None

    # --- SQLite ayarları ---
    sqlite_busy_timeout_ms: int = 5000

    @field_validator("app_mode", mode="before")
    @classmethod
    def _validate_app_mode(cls, value: object) -> AppMode:
        if isinstance(value, AppMode):
            return value
        text = str(value).strip().lower()
        try:
            return AppMode(text)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in AppMode)
            raise ValueError(
                f"APP_MODE yalnızca şu değerlerden birini alabilir: {allowed}"
            ) from exc

    @property
    def effective_database_url(self) -> str:
        """Mode göre çözümlenen veritabanı URL'si."""
        if self.database_url:
            return self.database_url
        if self.app_mode is AppMode.DEMO:
            return f"sqlite:///{PROJECT_ROOT / 'data' / 'demo.db'}"
        return f"sqlite:///{PROJECT_ROOT / 'data' / 'process_risk.db'}"

    @property
    def effective_artifact_dir(self) -> Path:
        """Mode göre çözümlenen artifact dizini."""
        if self.artifact_dir:
            return Path(self.artifact_dir)
        if self.app_mode is AppMode.DEMO:
            return PROJECT_ROOT / "artifacts" / "demo"
        return PROJECT_ROOT / "artifacts" / "production"

    @property
    def is_demo(self) -> bool:
        return self.app_mode is AppMode.DEMO


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Konfigürasyon tekil (singleton) örneğini döndürür."""
    settings = Settings()
    # Artifact dizinlerinin varlığını garanti et
    settings.effective_artifact_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    return settings
