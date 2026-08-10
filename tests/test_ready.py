"""/ready endpoint testleri."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _degraded_runtime(client: TestClient):
    """TestClient lifespan model yuklemis olabilir; testler icin sifirla."""
    from app.core.runtime import runtime_state
    runtime_state.model_available = False
    runtime_state.bundle = None
    yield


def test_ready_returns_503_without_model(client: TestClient):
    """Model yokken /ready 503 döner."""
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_returns_model_unavailable_error_code(client: TestClient):
    """Model yokken /ready MODEL_UNAVAILABLE kodu taşır."""
    response = client.get("/ready")
    data = response.json()
    assert data.get("error_code") == "MODEL_UNAVAILABLE"


def test_ready_returns_degraded_status(client: TestClient):
    """Model yokken status degraded'dir."""
    response = client.get("/ready")
    data = response.json()
    details = data.get("details", data)
    assert details.get("status", data.get("status")) == "degraded"


def test_ready_reports_model_available_false(client: TestClient):
    """Model yokken model_available False'dir."""
    response = client.get("/ready")
    data = response.json()
    details = data.get("details", data)
    assert details.get("model_available") is False


def test_ready_reports_database_ok(client: TestClient):
    """Veritabanı erişilebilir durumdadır."""
    response = client.get("/ready")
    data = response.json()
    details = data.get("details", data)
    assert details.get("database_ok") is True


def test_ready_reports_mode(client: TestClient):
    """Yanıt uygulama modunu içerir."""
    response = client.get("/ready")
    data = response.json()
    details = data.get("details", data)
    assert "mode" in details
    assert details["mode"] in ("demo", "local")


def test_ready_has_request_id(client: TestClient):
    """/ready yanıtı request_id taşır."""
    response = client.get("/ready")
    data = response.json()
    assert "request_id" in data
    assert data["request_id"] != ""
