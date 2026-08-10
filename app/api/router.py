"""Tüm API router'larını birleştiren kök router.

/api/v1 :  Sürümlenmiş API yolları (birincil)
/api     :  Geriye uyumlu yollar (kaldırılacak)
/health, /ready :  Kök seviyede (sürümsüz)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.imports import router as imports_router
from app.api.predictions import router as predictions_router
from app.web.routes import router as faz6_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(imports_router)
v1_router.include_router(predictions_router)
v1_router.include_router(faz6_router)

legacy_router = APIRouter(prefix="/api")
legacy_router.include_router(imports_router)
legacy_router.include_router(predictions_router)
legacy_router.include_router(faz6_router)

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(v1_router)
api_router.include_router(legacy_router)
