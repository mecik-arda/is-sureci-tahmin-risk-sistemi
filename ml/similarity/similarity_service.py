"""SimilarityService — validation-only nearest neighbors.

Faz 5 scope: benzer gecmis isler servisi (KNN).

Yalniz opening-v1 feature'lari kullanilir.
Outcome kolonlari (is_delayed, total_duration_hours) similarity mesafesine
girmez.

S20 uyumlu: yalniz Train + Validation verisi kullanilir.
Test/Audit similarity index'e girmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


@dataclass
class SimilarProcess:
    external_id: str
    distance: float
    features: dict[str, Any]
    is_delayed: bool | None
    total_duration_hours: float | None


class SimilarityService:
    def __init__(
        self,
        pipeline: Pipeline,
        X_reference: pd.DataFrame,
        external_ids: list[str],
        y_cls: np.ndarray | None = None,
        y_reg: np.ndarray | None = None,
        n_neighbors: int = 5,
    ):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            X_ref_transformed = preprocessor.transform(X_reference)
        else:
            X_ref_transformed = X_reference.values

        self._external_ids = external_ids
        self._y_cls = y_cls
        self._y_reg = y_reg
        self._X_ref = X_reference

        self._nn = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, len(X_ref_transformed)),
            metric="euclidean",
            n_jobs=-1,
        )
        self._nn.fit(X_ref_transformed)

    def find_similar(
        self,
        X_query: pd.DataFrame,
        pipeline: Pipeline,
    ) -> list[list[SimilarProcess]]:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            X_transformed = preprocessor.transform(X_query)
        else:
            X_transformed = X_query.values

        distances, indices = self._nn.kneighbors(X_transformed)

        results: list[list[SimilarProcess]] = []
        for i in range(len(X_query)):
            neighbors: list[SimilarProcess] = []
            for j, idx in enumerate(indices[i]):
                ext_id = self._external_ids[idx]
                dist = float(distances[i][j])
                features = self._X_ref.iloc[idx].to_dict()
                is_delayed = bool(self._y_cls[idx]) if self._y_cls is not None else None
                duration = float(self._y_reg[idx]) if self._y_reg is not None else None
                neighbors.append(SimilarProcess(
                    external_id=ext_id,
                    distance=dist,
                    features=features,
                    is_delayed=is_delayed,
                    total_duration_hours=duration,
                ))
            results.append(neighbors)
        return results

    def find_similar_for_ids(
        self,
        external_ids: list[str],
        X: pd.DataFrame,
        pipeline: Pipeline,
    ) -> list[list[SimilarProcess]]:
        query_mask = X.index.isin(external_ids)
        X_query = X[query_mask]
        return self.find_similar(X_query, pipeline)
