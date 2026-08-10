"""/health endpoint testleri."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient):
    """Liveness: /health her zaman 200 döner."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client: TestClient):
    """/health yanıtı status=ok içerir."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


def test_health_has_request_id_header(client: TestClient):
    """/health yanıtı X-Request-ID header'ı taşır."""
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_health_works_without_model(client: TestClient):
    """Model yokken bile /health 200 döner (degraded modda canlı)."""
    response = client.get("/health")
    assert response.status_code == 200
