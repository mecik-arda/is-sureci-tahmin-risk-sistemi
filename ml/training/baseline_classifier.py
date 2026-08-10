"""Faz 3 siniflandirma baseline: DummyClassifier + LogisticRegression.

SMOTE, oversampling, threshold tuning, calibration YOK.
Dogal class distribution korunur.
0.5 default threshold yalniz referans amaçlidir.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ml.datasets.dataset_builder import ClassificationDataset
from ml.evaluation.metrics import evaluate_classification
from ml.features.preprocessing import build_classification_pipeline
from ml.features.schema_loader import FeatureSchema


def train_classification_baselines(
    dataset: ClassificationDataset,
    schema: FeatureSchema,
) -> dict[str, dict]:
    """Train + validation üzerinde Dummy ve Logistic baseline'larini çalistirir.

    Tüm metrikler yalniz validation üzerinde hesaplanir.
    Test/Audit'e dokunulmaz (sealed_guard tarafindan korunur).
    """
    pipelines = {
        "dummy": build_classification_pipeline(
            schema,
            DummyClassifier(strategy="prior", random_state=42),
        ),
        "logistic": build_classification_pipeline(
            schema,
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    }

    results: dict[str, dict] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(dataset.train.X, dataset.train.y)

        y_pred = pipeline.predict(dataset.validation.X)
        y_proba = pipeline.predict_proba(dataset.validation.X)[:, 1]

        results[name] = evaluate_classification(
            dataset.validation.y, y_pred, y_proba, "validation"
        )

    return results
