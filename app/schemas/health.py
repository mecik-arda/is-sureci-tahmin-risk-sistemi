"""Sağlık ve hazırlık kontrolü yanıt şemaları."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness yanıtı: uygulama çalışıyor mu?"""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness yanıtı: tahmin servisi hazır mı?"""

    status: str
    mode: str
    model_available: bool
    database_ok: bool
