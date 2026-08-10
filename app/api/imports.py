from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import AppError
from app.services.analysis_dataset import analysis_dataset_service
from app.services.canonical_mapper import CanonicalMapper
from app.services.file_reader import SUPPORTED_EXTENSIONS
from app.services.import_service import run_import


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "mappings" / "canonical_map_v1.json"

router = APIRouter(tags=["imports"])


@router.post("/imports")
def import_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        raise AppError(
            message="Yalnizca CSV veya XLSX dosyalari aktarilabilir.",
            error_code="UNSUPPORTED_FILE_FORMAT",
            status_code=422,
        )

    try:
        with TemporaryDirectory(prefix="process-import-") as directory:
            upload_path = Path(directory) / f"upload{suffix}"
            uploaded_bytes = 0
            with upload_path.open("wb") as destination:
                while chunk := file.file.read(CHUNK_SIZE):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > MAX_UPLOAD_BYTES:
                        raise AppError(
                            message="Dosya izin verilen en fazla 512 MB boyutunu asiyor.",
                            error_code="UPLOAD_TOO_LARGE",
                            status_code=413,
                        )
                    destination.write(chunk)

            mapper = CanonicalMapper(MAPPING_PATH)
            result = run_import(db, upload_path, mapper)
            db.commit()
    except AppError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise AppError(
            message="Dosya aktarimi tamamlanamadi.",
            error_code="IMPORT_FAILED",
            status_code=500,
        )
    finally:
        file.file.close()

    if result.counts.inserted_rows or result.counts.updated_rows:
        analysis_dataset_service.reset()

    return {
        "status": result.status,
        "import_run_id": result.import_run_id,
        "counts": {
            "total_rows": result.counts.total_rows,
            "inserted_rows": result.counts.inserted_rows,
            "updated_rows": result.counts.updated_rows,
            "skipped_duplicate_rows": result.counts.skipped_duplicate_rows,
            "quarantined_rows": result.counts.quarantined_rows,
            "error_rows": result.counts.error_rows,
            "warning_count": result.counts.warning_count,
        },
    }
