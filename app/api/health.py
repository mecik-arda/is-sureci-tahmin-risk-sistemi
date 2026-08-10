"""Liveness ve readiness endpoint'leri.

/health : Uygulama canlı mı? Modelden bağımsız, daima 200.
/ready  : Tahmin servisi hazır mı? Model yoksa 503 MODEL_UNAVAILABLE.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ModelUnavailableError
from app.core.runtime import runtime_state
from app.dependencies import get_db
from app.schemas.health import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: uygulama yanıt veriyor mu?"""
    return HealthResponse(status="ok")


@router.get("/ready")
def ready(request: Request, db: Session = Depends(get_db)) -> JSONResponse:
    """Readiness: tahmin servisi ve bağımlılıklar hazır mı?

    Model yoksa HTTP 503 ve MODEL_UNAVAILABLE kodu döner.
    Veritabanı erişilemiyorsa HTTP 503 döner.
    """
    settings = get_settings()

    # Veritabanı kontrolü
    database_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_ok = False

    model_available = runtime_state.model_available
    response_status = "ready" if (model_available and database_ok) else "degraded"

    body = ReadyResponse(
        status=response_status,
        mode=settings.app_mode.value,
        model_available=model_available,
        database_ok=database_ok,
    )

    if not model_available:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": ModelUnavailableError.error_code,
                "message": ModelUnavailableError.message,
                "details": {
                    "status": response_status,
                    "mode": settings.app_mode.value,
                    "model_available": model_available,
                    "database_ok": database_ok,
                },
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    http_status = 200 if database_ok else 503
    return JSONResponse(status_code=http_status, content=body.model_dump())
