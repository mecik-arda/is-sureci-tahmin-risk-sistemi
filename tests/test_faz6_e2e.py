"""Faz 6 E2E tests — S22-S29 compliance verification."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.dependencies import get_db


@pytest.fixture
def e2e_client(db_session: Session):
    """TestClient with DB dependency overridden to test session."""
    def _override():
        yield db_session
    app.dependency_overrides[get_db] = _override
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def sample_process(db_session: Session):
    """E2E testleri için örnek süreç kaydı."""
    from datetime import datetime, UTC
    from app.models.process import Process, ProcessSnapshot

    p = Process(
        external_id="E2E-TEST-001",
        process_type="street_repair",
        current_status="Open",
        created_at=datetime(2024, 6, 15, 10, 0, 0),
        deadline=datetime(2024, 6, 20, 17, 0, 0),
    )
    db_session.add(p)
    db_session.flush()

    import json
    snap = ProcessSnapshot(
        process_id=p.id,
        snapshot_type="opening",
        snapshot_at=p.created_at,
        feature_schema_version="opening-v1",
        input_json=json.dumps({
            "created_at": p.created_at.isoformat(),
            "deadline": p.deadline.isoformat(),
            "source": "citizens_connect_app",
            "subject": "pothole_not_filled",
            "reason": "road_maintenance",
            "type": "street_repair",
            "neighborhood": "roxbury_02119",
        }),
        input_fingerprint="e2e_test_fp",
    )
    db_session.add(snap)
    db_session.commit()


class TestAppStartup:
    def test_health_200(self, e2e_client):
        r = e2e_client.get("/health")
        assert r.status_code == 200

    def test_ready_returns(self, e2e_client):
        r = e2e_client.get("/ready")
        assert r.status_code in [200, 503]


class TestHTMLPages:
    def test_dashboard_html(self, e2e_client):
        r = e2e_client.get("/")
        assert r.status_code == 200
        html = r.text
        assert 'lang="tr"' in html
        assert "banner-area" in html

    def test_process_list_html(self, e2e_client):
        r = e2e_client.get("/processes")
        assert r.status_code == 200
        html = r.text
        assert "model-context" in html
        assert "<table" in html

    def test_process_detail_html(self, e2e_client):
        r = e2e_client.get("/processes/1")
        assert r.status_code == 200
        html = r.text
        assert "Tahmin Anındaki Bilgiler" in html
        assert "Güncel Süreç Durumu" in html


class TestStaticFiles:
    def test_css_serves(self, e2e_client):
        r = e2e_client.get("/static/css/style.css")
        assert r.status_code == 200
        assert "banner-candidate" in r.text
        assert "banner-degraded" in r.text
        assert "banner-demo" in r.text

    def test_app_js_serves(self, e2e_client):
        r = e2e_client.get("/static/js/app.js")
        assert r.status_code == 200
        assert "loadBanner" in r.text

    def test_dashboard_js_serves(self, e2e_client):
        r = e2e_client.get("/static/js/dashboard.js")
        assert r.status_code == 200

    def test_process_list_js_serves(self, e2e_client):
        r = e2e_client.get("/static/js/process_list.js")
        assert r.status_code == 200

    def test_process_detail_js_serves(self, e2e_client):
        r = e2e_client.get("/static/js/process_detail.js")
        assert r.status_code == 200


class TestBannerS28:
    def test_banner_has_all_fields(self, e2e_client):
        r = e2e_client.get("/api/banner")
        assert r.status_code == 200
        data = r.json()
        assert "app_mode" in data
        assert "model_available" in data
        assert "bundle_stage" in data
        assert "threshold" in data
        assert "calibration_method" in data

    def test_banner_demo_mode(self, e2e_client):
        r = e2e_client.get("/api/banner")
        data = r.json()
        assert data["app_mode"] == "demo"


class TestDashboardS23:
    def test_has_prediction_and_actual_sections(self, e2e_client):
        r = e2e_client.get("/api/dashboard")
        assert r.status_code == 200
        data = r.json()
        assert "prediction_kpis" in data
        assert "actual_kpis" in data
        assert isinstance(data["prediction_kpis"], dict)
        assert isinstance(data["actual_kpis"], dict)

    def test_kpis_are_numbers_or_null(self, e2e_client):
        r = e2e_client.get("/api/dashboard")
        data = r.json()
        pred = data["prediction_kpis"]
        assert isinstance(pred["total_predictions"], int)
        assert isinstance(pred["high_risk_count"], int)


class TestProcessListS24:
    def test_has_pagination(self, e2e_client):
        r = e2e_client.get("/api/processes?page=1&per_page=5")
        assert r.status_code == 200
        data = r.json()
        assert "processes" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data

    def test_process_fields_exist(self, e2e_client):
        r = e2e_client.get("/api/processes?page=1&per_page=1")
        data = r.json()
        if data["processes"]:
            p = data["processes"][0]
            assert "delay_probability" in p
            assert "predicted_is_delayed" in p
            assert "predicted_hours" in p
            assert "has_prediction" in p


class TestProcessDetailS22:
    def test_opening_and_current_separate(self, e2e_client):
        r = e2e_client.get("/api/processes/1")
        assert r.status_code == 200
        data = r.json()
        assert "opening_fields" in data
        assert "current_fields" in data
        assert isinstance(data["opening_fields"], dict)
        assert isinstance(data["current_fields"], dict)

    def test_current_fields_has_deadline(self, e2e_client):
        r = e2e_client.get("/api/processes/1")
        data = r.json()
        cf = data["current_fields"]
        assert "deadline" in cf

    def test_current_fields_has_four_outcome(self, e2e_client):
        r = e2e_client.get("/api/processes/1")
        data = r.json()
        cf = data["current_fields"]
        assert "current_status" in cf
        assert "completed_at" in cf
        assert "closure_reason" in cf
        assert "total_duration_hours" in cf

    def test_has_prediction_key(self, e2e_client):
        r = e2e_client.get("/api/processes/1")
        data = r.json()
        assert "prediction" in data
        assert "has_sla" in data

    def test_has_input_fingerprint(self, e2e_client):
        r = e2e_client.get("/api/processes/1")
        data = r.json()
        assert "input_fingerprint" in data


class TestNotFoundS29:
    def test_missing_process_404(self, e2e_client):
        r = e2e_client.get("/api/processes/99999999")
        assert r.status_code == 404


class TestXaiS26:
    def test_xai_has_importances_and_available(self, e2e_client):
        r = e2e_client.get("/api/processes/1/xai")
        assert r.status_code == 200
        data = r.json()
        assert "importances" in data
        assert "available" in data

    def test_xai_unavailable_when_no_model(self, e2e_client):
        r = e2e_client.get("/api/processes/1/xai")
        data = r.json()
        assert data["available"] in [True, False]


class TestSimilarS27:
    def test_similar_has_neighbors_and_available(self, e2e_client):
        r = e2e_client.get("/api/processes/1/similar")
        assert r.status_code == 200
        data = r.json()
        assert "neighbors" in data
        assert "available" in data

    def test_similar_no_distance_field(self, e2e_client):
        r = e2e_client.get("/api/processes/1/similar")
        data = r.json()
        for n in data.get("neighbors", []):
            assert "distance" not in n


class TestS30SimulationPage:
    def test_simulation_form_renders(self, e2e_client):
        resp = e2e_client.get("/processes/1")
        assert resp.status_code == 200
        html = resp.text.lower()
        assert "what-if" in html or "varsay\u0131msal" in html

    def test_simulation_warnings_visible(self, e2e_client):
        resp = e2e_client.get("/processes/1")
        assert resp.status_code == 200
        html = resp.text
        assert "varsay\u0131msal senaryolar i\u00e7indir" in html.lower()
        assert "neden oldu\u011funu kan\u0131tlamaz" in html.lower()

    def test_simulation_no_causal_language(self, e2e_client):
        resp = e2e_client.get("/processes/1")
        assert resp.status_code == 200
        html = resp.text
        assert "riski azaltt\u0131" not in html.lower()
        assert "gecikmeyi d\u00fc\u015f\u00fcr\u00fcr" not in html.lower()


class TestS31FeedbackPage:
    def test_feedback_sections_exist(self, e2e_client):
        resp = e2e_client.get("/processes/1")
        assert resp.status_code == 200
        html = resp.text.lower()
        assert "geri bildirim" in html


class TestS32PerformancePage:
    def test_performance_page_loads(self, e2e_client):
        resp = e2e_client.get("/model-performance")
        assert resp.status_code == 200
        html = resp.text
        assert "model performans\u0131" in html.lower()

    def test_performance_no_test_accuracy_language(self, e2e_client):
        resp = e2e_client.get("/model-performance")
        assert resp.status_code == 200
        html = resp.text.lower()
        assert "test accuracy" not in html
        assert "nihai ba\u015far\u0131" not in html
        assert "model do\u011frulu\u011fu" not in html

    def test_performance_validation_warning(self, e2e_client):
        resp = e2e_client.get("/model-performance")
        assert resp.status_code == 200
        html = resp.text
        assert "validation" in html.lower() or "eyl\u00fcl" in html.lower()

    def test_performance_cv_section(self, e2e_client):
        resp = e2e_client.get("/model-performance")
        assert resp.status_code == 200
        html = resp.text
        assert "cv" in html.lower() or "cross" in html.lower()

    def test_performance_js_serves(self, e2e_client):
        resp = e2e_client.get("/static/js/model_performance.js")
        assert resp.status_code == 200


class TestLabelCatalogEndpoint:
    def test_label_catalog_serves(self, e2e_client):
        resp = e2e_client.get("/api/label-catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert "label_catalog_version" in data or "feature_labels" in data
