"""Analysis dataset service — cached dataset for XAI and similarity.

Provides thread-safe lazy caching of the full ~282K-row dataset.
Builds once on first access; all subsequent requests reuse the same
in-memory pandas DataFrames. No invalidation — runtime dataset is
immutable (Process/ProcessSnapshot rows never change via API).

Concurrency: threading.Lock prevents thundering herd on cold cache.
Cache stores pandas DataFrames, not ORM objects (DB session independent).
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ProcessNotFoundError
from app.models.process import Process, ProcessSnapshot
from ml.datasets.dataset_builder import DatasetBuildResult, build_dataset
from ml.features.feature_derivation import derive_features
from ml.features.schema_loader import FeatureSchema, load_feature_schema


class AnalysisDatasetService:
    """Cached analysis dataset with thread-safe lazy initialization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: DatasetBuildResult | None = None
        self._build_count: int = 0
        self._cache_hit_count: int = 0
        self._built_at: str | None = None

    def get_dataset(
        self, session: Session, schema: FeatureSchema | None = None
    ) -> DatasetBuildResult:
        """Return cached dataset, building once if cold."""
        if self._result is not None:
            self._cache_hit_count += 1
            return self._result

        with self._lock:
            if self._result is not None:
                self._cache_hit_count += 1
                return self._result

            _schema = schema if schema is not None else load_feature_schema()
            self._result = build_dataset(session, _schema)
            self._build_count += 1

            from datetime import UTC, datetime
            self._built_at = datetime.now(UTC).isoformat()

        return self._result

    def get_similar_processes(
        self,
        session: Session,
        process_id: int,
        bundle: Any,
        query_features: pd.DataFrame,
    ) -> dict[str, Any]:
        """S27: Similar processes via KNN on opening-v1 features.

        Uses cached dataset; only build on first access.
        """
        if bundle is None:
            return {"neighbors": [], "available": False}

        from ml.similarity.similarity_service import SimilarityService

        result = self.get_dataset(session)
        cls_dataset = result.classification
        reg_dataset = result.regression

        svc = SimilarityService(
            pipeline=bundle.classifier,
            X_reference=cls_dataset.train.X,
            external_ids=cls_dataset.train.external_ids,
            y_cls=cls_dataset.train.y,
            y_reg=reg_dataset.train.y,
            n_neighbors=6,
        )

        results = svc.find_similar(query_features, bundle.classifier)

        neighbors = []
        for group in results:
            for n in group[1:]:
                neighbors.append({
                    "external_id": n.external_id,
                    "is_delayed": n.is_delayed,
                    "total_duration_hours": n.total_duration_hours,
                })

        external_ids = [n["external_id"] for n in neighbors]
        type_map: dict[str, str] = {}
        if external_ids:
            procs = session.execute(
                select(Process.external_id, Process.process_type).where(
                    Process.external_id.in_(external_ids)
                )
            ).all()
            type_map = {p.external_id: p.process_type for p in procs}

        enriched = []
        for n in neighbors[:5]:
            n["process_type"] = type_map.get(n["external_id"])
            enriched.append(n)

        return {"neighbors": enriched, "available": True}

    def get_similar_processes_for_process(
        self,
        session: Session,
        process_id: int,
        bundle: Any,
    ) -> dict[str, Any]:
        process = session.get(Process, process_id)
        if process is None:
            raise ProcessNotFoundError()

        snapshot = session.execute(
            select(ProcessSnapshot).where(
                ProcessSnapshot.process_id == process_id,
                ProcessSnapshot.snapshot_type == "opening",
            )
        ).scalars().first()
        if snapshot is None:
            return {"neighbors": [], "available": False}

        features = derive_features(json.loads(snapshot.input_json))
        query_features = pd.DataFrame([features])
        return self.get_similar_processes(
            session,
            process_id,
            bundle,
            query_features,
        )

    def get_xai(
        self, session: Session, bundle: Any, process_id: int | None = None
    ) -> dict[str, Any]:
        """S26: Global feature importance + optional per-instance SHAP.

        Uses cached dataset; only build on first access.
        """
        if bundle is None:
            return {"importances": [], "shap_values": [], "available": False}

        from ml.xai.explainer import compute_feature_importance, compute_shap_values

        result = self.get_dataset(session)
        cls_dataset = result.classification
        importances = compute_feature_importance(
            bundle.classifier,
            cls_dataset.validation.X,
            cls_dataset.validation.y,
        )

        shap_values: list[dict[str, Any]] = []
        if process_id is not None:
            shap_values = self._compute_instance_shap(
                session, bundle, cls_dataset.validation.X, process_id
            )

        return {
            "importances": [
                {"feature": fi.feature, "importance": fi.importance, "label_tr": fi.label_tr}
                for fi in importances
            ],
            "shap_values": shap_values,
            "available": True,
        }

    def _compute_instance_shap(
        self,
        session: Session,
        bundle: Any,
        X_background: pd.DataFrame,
        process_id: int,
    ) -> list[dict[str, Any]]:
        """Compute per-instance SHAP values for a specific process."""
        import json

        try:
            from ml.xai.explainer import compute_shap_values
            from app.models.process import Process, ProcessSnapshot
            from ml.features.feature_derivation import derive_features
            from sqlalchemy import select as sa_select
            import pandas as pd

            process = session.get(Process, process_id)
            if process is None:
                return []

            snapshot = session.execute(
                sa_select(ProcessSnapshot).where(
                    ProcessSnapshot.process_id == process_id,
                    ProcessSnapshot.snapshot_type == "opening",
                ).limit(1)
            ).scalars().first()

            if snapshot is None:
                return []

            try:
                snapshot_data = json.loads(snapshot.input_json)
            except (json.JSONDecodeError, TypeError):
                return []

            features_dict = derive_features(snapshot_data)
            if not features_dict:
                return []

            X_instance = pd.DataFrame([features_dict])

            contributions = compute_shap_values(
                bundle.classifier,
                X_instance,
                X_background=X_background,
            )

            return [
                {
                    "feature": c.feature,
                    "shap_value": c.shap_value,
                    "label_tr": c.label_tr,
                    "direction": "risk_artiran" if c.shap_value > 0 else "risk_azaltan",
                }
                for c in contributions
            ]
        except Exception:
            return []

    @property
    def build_count(self) -> int:
        return self._build_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count

    @property
    def built_at(self) -> str | None:
        return self._built_at

    def reset(self) -> None:
        """Clear cache. Used by tests to ensure isolation."""
        with self._lock:
            self._result = None
            self._build_count = 0
            self._cache_hit_count = 0
            self._built_at = None

    def cached_memory_bytes(self) -> int:
        if self._result is None:
            return 0
        total = 0
        for dataset in (self._result.classification, self._result.regression):
            for split in (dataset.train, dataset.validation):
                total += split.X.memory_usage(deep=True).sum()
        return int(total)

    def cached_rows(self) -> int:
        if self._result is None:
            return 0
        return (
            self._result.classification.train.row_count
            + self._result.classification.validation.row_count
            + self._result.regression.train.row_count
            + self._result.regression.validation.row_count
        )


# Application-scoped singleton — owned by lifespan
analysis_dataset_service = AnalysisDatasetService()
