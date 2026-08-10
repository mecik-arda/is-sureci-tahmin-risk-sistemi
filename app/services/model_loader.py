"""Güvenli model bundle yükleme.

Artifact yalnız izin verilen artifacts root altındaki yollardan yüklenir.
Hash, schema, mapping version, bundle format kontrolleri uygulanır.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.model_bundle import ModelBundle

EXPECTED_SCHEMA_VERSION = "opening-v1"
EXPECTED_CANONICAL_MAPPING_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class LoadedBundle:
    bundle_id: int
    bundle_record: ModelBundle
    classifier: Any
    regressor: Any
    metadata: dict[str, Any]
    artifact_hash: str
    calibration_model: Any | None = None
    threshold: float = 0.5


def find_active_bundle(session: Session) -> ModelBundle | None:
    active = session.execute(
        select(ModelBundle).where(ModelBundle.is_active == 1)
    ).scalars().all()
    if len(active) == 0:
        return None
    if len(active) > 1:
        from app.core.errors import ActiveModelAmbiguousError
        raise ActiveModelAmbiguousError(
            details={"active_count": len(active)},
        )
    return active[0]


def _validate_artifact_path(artifact_path: Path, artifact_root: Path) -> None:
    try:
        artifact_path.relative_to(artifact_root)
    except ValueError:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model artifact izin verilen dizin disinda.",
            details={"path": str(artifact_path), "root": str(artifact_root)},
        )


def load_bundle(session: Session, bundle_id: int | None = None) -> LoadedBundle:
    settings = get_settings()
    artifact_root = Path(settings.effective_artifact_dir).resolve()

    if bundle_id is not None:
        bundle_record = session.execute(
            select(ModelBundle).where(ModelBundle.id == bundle_id)
        ).scalar_one_or_none()
    else:
        bundle_record = find_active_bundle(session)

    if bundle_record is None:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError()

    artifact_path = Path(bundle_record.artifact_path).resolve()

    _validate_artifact_path(artifact_path, artifact_root)

    if not artifact_path.is_file():
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model artifact dosyasi bulunamadi.",
            details={"path": str(artifact_path)},
        )

    if bundle_record.artifact_hash is None or not _SHA256_RE.match(bundle_record.artifact_hash):
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model artifact SHA256 hash eksik veya gecersiz format.",
            details={
                "hash": bundle_record.artifact_hash,
            },
        )

    actual_hash = _compute_sha256(artifact_path)

    if actual_hash != bundle_record.artifact_hash:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model artifact SHA256 uyusmazligi. Dosya degistirilmis olabilir.",
            details={
                "expected": bundle_record.artifact_hash,
                "actual": actual_hash,
            },
        )

    bundle = joblib.load(str(artifact_path))

    if not isinstance(bundle, dict):
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model bundle gecersiz format.",
        )

    metadata = bundle.get("metadata", {})
    if not metadata:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Model bundle metadata eksik.",
        )

    schema = metadata.get("feature_schema_version", "")
    if schema != EXPECTED_SCHEMA_VERSION:
        from app.core.errors import SchemaMismatchError
        raise SchemaMismatchError(
            message=f"Feature schema uyusmazligi. Beklenen: {EXPECTED_SCHEMA_VERSION}, alinan: {schema}",
            details={"expected": EXPECTED_SCHEMA_VERSION, "actual": schema},
        )

    mapping = metadata.get("canonical_mapping_version", "")
    if mapping != EXPECTED_CANONICAL_MAPPING_VERSION:
        from app.core.errors import SchemaMismatchError
        raise SchemaMismatchError(
            message=f"Kanonik mapping versiyonu uyusmazligi. Beklenen: {EXPECTED_CANONICAL_MAPPING_VERSION}, alinan: {mapping}",
            details={"expected": EXPECTED_CANONICAL_MAPPING_VERSION, "actual": mapping},
        )

    clf = bundle.get("classifier_pipeline")
    reg = bundle.get("regression_pipeline")
    if clf is None or reg is None:
        from app.core.errors import ModelUnavailableError
        raise ModelUnavailableError(
            message="Bundle icinde classifier veya regressor pipeline bulunamadi.",
        )

    calibration = bundle.get("calibration_model", None)
    threshold = float(bundle.get("threshold", 0.5))

    return LoadedBundle(
        bundle_id=bundle_record.id,
        bundle_record=bundle_record,
        classifier=clf,
        regressor=reg,
        metadata=metadata,
        artifact_hash=actual_hash,
        calibration_model=calibration,
        threshold=threshold,
    )


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
