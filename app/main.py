"""FastAPI uygulama giriş noktası.

Lifespan:
    Başlangıçta aktif model aranır. Bulunursa doğrulanıp belleğe yüklenir
    ve durum 'ready' olur. Bulunamazsa uygulama 'degraded' modda çalışmaya
    devam eder; model gerektiren endpoint'ler MODEL_UNAVAILABLE döndürür.

Middleware:
    Her isteğe benzersiz bir request_id atanır.
    Rate limiting (slowapi) uygulanır.

Hata yönetimi:
    AppError alt sınıfları standart formatta yanıtlanır.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.limiter import limiter
from app.core.runtime import runtime_state


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Uygulama yaşam döngüsü: başlangıçta aktif model bundle yükleme."""
    from pathlib import Path
    from app.core.database import SessionLocal

    import json as _json
    _label_catalog_path = Path(__file__).parent.parent / "ml" / "config" / "label_catalog_v1.json"
    if _label_catalog_path.exists():
        application.state.label_catalog = _json.loads(_label_catalog_path.read_text(encoding="utf-8"))
    else:
        application.state.label_catalog = {}

    try:
        db = SessionLocal()
        try:
            from app.services.model_loader import find_active_bundle, load_bundle
            active = find_active_bundle(db)
            if active is not None:
                loaded = load_bundle(db, bundle_id=active.id)
                runtime_state.mark_ready(active.model_version, loaded)
            else:
                runtime_state.mark_degraded(
                    "Aktif model bundle bulunamadi (is_active=1 kayit yok)."
                )
        finally:
            db.close()
    except Exception as exc:
        runtime_state.mark_degraded(f"Model yukleme hatasi: {exc}")

    runtime_state.mark_startup_done()
    yield


def create_app() -> FastAPI:
    """FastAPI uygulamasını oluşturur ve yapılandırır."""
    settings = get_settings()

    app = FastAPI(
        title="AI Destekli İş Süreci Tahmin ve Risk Sistemi",
        description=(
            "Geçmiş süreç kayıtlarından gecikme riski, tahmini tamamlanma "
            "süresi ve açıklanabilir karar desteği sağlayan yerel uygulama."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Her isteğe benzersiz bir request_id ekler."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(api_router)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Uygulama hatalarını standart formatta yanıtlar."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    from pathlib import Path

    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    static_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates_env = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        """Dashboard HTML page."""
        return templates_env.TemplateResponse(
            request, "dashboard.html", {"page": "dashboard"}
        )

    @app.get("/processes", response_class=HTMLResponse)
    async def process_list_page(request: Request):
        """Process list HTML page."""
        return templates_env.TemplateResponse(
            request, "process_list.html", {"page": "processes"}
        )

    @app.get("/processes/{process_id}", response_class=HTMLResponse)
    async def process_detail_page(request: Request, process_id: int):
        """Process detail HTML page."""
        return templates_env.TemplateResponse(
            request,
            "process_detail.html",
            {"page": "process_detail", "process_id": process_id},
        )

    @app.get("/data-import", response_class=HTMLResponse)
    async def data_import_page(request: Request):
        return templates_env.TemplateResponse(
            request, "data_import.html", {"page": "data_import"}
        )

    @app.get("/model-performance", response_class=HTMLResponse)
    async def model_performance_page(request: Request):
        return templates_env.TemplateResponse(
            request, "model_performance.html", {"page": "model_performance"}
        )

    @app.get("/model-monitoring", response_class=HTMLResponse)
    async def model_monitoring_page(request: Request):
        return templates_env.TemplateResponse(
            request, "model_monitoring.html", {"page": "model_monitoring"}
        )

    @app.get("/data-quality", response_class=HTMLResponse)
    async def data_quality_page(request: Request):
        return templates_env.TemplateResponse(
            request, "data_quality.html", {"page": "data_quality"}
        )

    return app


app = create_app()
