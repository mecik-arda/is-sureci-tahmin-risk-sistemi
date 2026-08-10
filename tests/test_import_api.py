from io import BytesIO
from pathlib import Path

from openpyxl import Workbook
from sqlalchemy import select

from app.models.process import Process
from app.services.import_service import ImportCounts, ImportResult


def csv_content(external_id: str = "API-IMPORT-001") -> bytes:
    return (
        "case_enquiry_id,open_dt,closed_dt,sla_target_dt,case_status,source,subject,reason,type,neighborhood,closure_reason\n"
        f"{external_id},2024-01-15 10:00:00,,2024-01-20 17:00:00,Open,Citizens Connect App,Public Works Department,Highway Maintenance,Request for Pothole Repair,Roxbury,\n"
    ).encode("utf-8")


def xlsx_content(external_id: str = "API-IMPORT-XLSX") -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append([
        "case_enquiry_id", "open_dt", "closed_dt", "sla_target_dt", "case_status",
        "source", "subject", "reason", "type", "neighborhood", "closure_reason",
    ])
    worksheet.append([
        external_id, "2024-01-15 10:00:00", "", "2024-01-20 17:00:00", "Open",
        "Citizens Connect App", "Public Works Department", "Highway Maintenance",
        "Request for Pothole Repair", "Roxbury", "",
    ])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


class TestImportApi:
    def test_upload_delegates_to_pipeline_and_removes_temporary_file(self, client, monkeypatch):
        captured = {}

        def fake_run_import(_session, file_path, mapper):
            captured["path"] = file_path
            captured["mapper_version"] = mapper.version
            captured["content"] = file_path.read_bytes()
            return ImportResult(
                status="completed",
                counts=ImportCounts(total_rows=1),
                file_hash="a" * 64,
                import_run_id=1,
            )

        monkeypatch.setattr("app.api.imports.run_import", fake_run_import)
        monkeypatch.setattr("app.api.imports.CHUNK_SIZE", 7)

        response = client.post(
            "/api/imports",
            files={"file": ("../../sensitive.csv", csv_content(), "text/csv")},
        )

        assert response.status_code == 200
        assert captured["path"].suffix == ".csv"
        assert captured["path"].name == "upload.csv"
        assert captured["mapper_version"] == "1.0.0"
        assert captured["content"] == csv_content()
        assert not captured["path"].exists()

    def test_csv_upload_runs_existing_import_pipeline(self, client, db_session, monkeypatch):
        reset_calls = []
        monkeypatch.setattr(
            "app.api.imports.analysis_dataset_service.reset",
            lambda: reset_calls.append(True),
        )

        response = client.post(
            "/api/imports",
            files={"file": ("upload.csv", csv_content(), "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["counts"]["inserted_rows"] == 1
        assert data["import_run_id"] > 0
        assert reset_calls == [True]
        process = db_session.execute(select(Process)).scalar_one()
        assert process.external_id == "API-IMPORT-001"

    def test_xlsx_upload_runs_existing_import_pipeline(self, client, db_session):
        response = client.post(
            "/api/imports",
            files={
                "file": (
                    "upload.xlsx",
                    xlsx_content(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        process = db_session.execute(select(Process)).scalar_one()
        assert process.external_id == "API-IMPORT-XLSX"

    def test_duplicate_upload_preserves_existing_data_without_cache_reset(self, client, db_session, monkeypatch):
        reset_calls = []
        monkeypatch.setattr(
            "app.api.imports.analysis_dataset_service.reset",
            lambda: reset_calls.append(True),
        )
        upload = {"file": ("upload.csv", csv_content(), "text/csv")}
        first = client.post("/api/imports", files=upload)
        second = client.post("/api/imports", files=upload)

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == "duplicate_file"
        assert reset_calls == [True]
        assert len(db_session.execute(select(Process)).scalars().all()) == 1

    def test_unsupported_file_is_rejected_before_import(self, client):
        response = client.post(
            "/api/imports",
            files={"file": ("upload.txt", b"not a supported dataset", "text/plain")},
        )

        assert response.status_code == 422
        assert response.json()["error_code"] == "UNSUPPORTED_FILE_FORMAT"

    def test_unsupported_file_never_invokes_import_pipeline(self, client, monkeypatch):
        invoked = []
        monkeypatch.setattr(
            "app.api.imports.run_import",
            lambda *_args: invoked.append(True),
        )

        response = client.post(
            "/api/imports",
            files={"file": ("upload.txt", b"unsupported", "text/plain")},
        )

        assert response.status_code == 422
        assert invoked == []

    def test_upload_size_limit_is_enforced(self, client, monkeypatch):
        monkeypatch.setattr("app.api.imports.MAX_UPLOAD_BYTES", 1)
        invoked = []
        monkeypatch.setattr(
            "app.api.imports.run_import",
            lambda *_args: invoked.append(True),
        )

        response = client.post(
            "/api/imports",
            files={"file": ("upload.csv", b"ab", "text/csv")},
        )

        assert response.status_code == 413
        assert response.json()["error_code"] == "UPLOAD_TOO_LARGE"
        assert invoked == []

    def test_pipeline_failure_rolls_back_and_uses_safe_error(self, client, db_session, monkeypatch):
        def fail_import(*_args):
            raise RuntimeError("sensitive internal failure")

        monkeypatch.setattr("app.api.imports.run_import", fail_import)

        response = client.post(
            "/api/imports",
            files={"file": ("upload.csv", csv_content(), "text/csv")},
        )

        assert response.status_code == 500
        assert response.json()["error_code"] == "IMPORT_FAILED"
        assert "sensitive internal failure" not in response.text
        assert not db_session.in_transaction()

    def test_data_import_page_and_script_are_available(self, client):
        page = client.get("/data-import")
        script = client.get("/static/js/data_import.js")

        assert page.status_code == 200
        assert "Dosyayı Aktar" in page.text
        assert script.status_code == 200
        assert "FormData" in script.text

    def test_import_script_blocks_duplicate_submission(self):
        script = Path("app/static/js/data_import.js").read_text(encoding="utf-8")

        assert "button.disabled = true" in script
        assert "button.setAttribute('aria-busy', 'true')" in script
        assert "button.disabled = false" in script
