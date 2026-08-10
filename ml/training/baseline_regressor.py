"""Faz 3 regresyon baseline: DummyRegressor + ElasticNet (raw + log1p).

Raw total_duration_hours ile log1p TransformedTargetRegressor
ayni validation datasetinde kiyaslanir.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet

from ml.datasets.dataset_builder import RegressionDataset
from ml.evaluation.metrics import evaluate_regression
from ml.features.preprocessing import build_regression_pipeline
from ml.features.schema_loader import FeatureSchema


def train_regression_baselines(
    dataset: RegressionDataset,
    schema: FeatureSchema,
) -> dict[str, dict]:
    """Train + validation üzerinde Dummy, ElasticNet raw ve log1p baseline'larini çalistirir.

    Tüm metrikler yalniz validation üzerinde, gerçek saat ölçeginde hesaplanir.
    Test/Audit'e dokunulmaz (sealed_guard tarafindan korunur).
    Negatif tahminler inference boundary'de max(0, prediction) ile sifirlanir.
    """
    pipelines = {
        "dummy_median": build_regression_pipeline(
            schema,
            DummyRegressor(strategy="median"),
        ),
        "elasticnet_raw": build_regression_pipeline(
            schema,
            ElasticNet(random_state=42, max_iter=5000),
        ),
        "elasticnet_log1p": build_regression_pipeline(
            schema,
            TransformedTargetRegressor(
                regressor=ElasticNet(random_state=42, max_iter=5000),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        ),
    }

    results: dict[str, dict] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(dataset.train.X, dataset.train.y)

        y_pred = pipeline.predict(dataset.validation.X)
        y_pred = np.maximum(0, y_pred)

        results[name] = evaluate_regression(
            dataset.validation.y, y_pred, "validation"
        )

    return results
