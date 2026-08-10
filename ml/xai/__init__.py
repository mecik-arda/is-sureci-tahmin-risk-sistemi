"""XAI feature importance for Faz 5 winner models.

Uses sklearn built-in methods:
  - Tree models: feature_importances_ + permutation importance
  - Linear models: coefficients

No external XAI library dependency (shap not installed).
Causal claim uretilmez.
Turkce sunum etiketleri canonical isimlerden ayri tutulur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline


FEATURE_LABELS_TR: dict[str, str] = {
    "source": "Bildirim Kaynagi",
    "subject": "Konu",
    "reason": "Neden",
    "type": "Surec Tipi",
    "neighborhood": "Mahalle",
    "open_month": "Acilis Ayi",
    "open_weekday": "Acilis Gunu (Haftanin)",
    "open_hour": "Acilis Saati",
    "is_weekend": "Haftasonu Acilis",
    "sla_duration_hours": "SLA Suresi (saat)",
}


@dataclass
class FeatureImportance:
    feature: str
    importance: float
    label_tr: str


def compute_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[FeatureImportance]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return _direct_importance(pipeline, X, y, n_repeats, random_state)

    X_transformed = preprocessor.transform(X)
    feature_names_out = _get_transformed_feature_names(preprocessor, X)
    model = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")

    if model is None:
        return []

    if hasattr(model, "feature_importances_"):
        fi = _tree_importance(model, X_transformed, y, n_repeats, random_state)
    elif isinstance(model, (LogisticRegression, ElasticNet)):
        fi = _linear_importance(model, X_transformed, y, n_repeats, random_state)
    else:
        fi = _permutation_only(pipeline, X, y, n_repeats, random_state)

    if feature_names_out and len(feature_names_out) == len(fi):
        scores = fi
    else:
        scores = _permutation_only(pipeline, X, y, n_repeats, random_state)
        feature_names_out = X.columns.tolist()

    results: list[FeatureImportance] = []
    for name, imp in zip(feature_names_out, scores):
        results.append(FeatureImportance(
            feature=name,
            importance=float(imp),
            label_tr=FEATURE_LABELS_TR.get(name, name),
        ))
    results.sort(key=lambda r: abs(r.importance), reverse=True)
    return results


def _get_transformed_feature_names(preprocessor: Any, X: pd.DataFrame) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return X.columns.tolist()


def _tree_importance(
    model: Any, X: np.ndarray, y: np.ndarray,
    n_repeats: int, random_state: int,
) -> list[float]:
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )
    return result.importances_mean.tolist()


def _linear_importance(
    model: Any, X: np.ndarray, y: np.ndarray,
    n_repeats: int, random_state: int,
) -> list[float]:
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return np.abs(coef).tolist()
    return _permutation_only(model, X, y, n_repeats, random_state)


def _permutation_only(
    estimator: Any, X: Any, y: np.ndarray,
    n_repeats: int, random_state: int,
) -> list[float]:
    result = permutation_importance(
        estimator, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return result.importances_mean.tolist()


def _direct_importance(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray,
    n_repeats: int, random_state: int,
) -> list[FeatureImportance]:
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return [
        FeatureImportance(feature=col, importance=float(imp), label_tr=FEATURE_LABELS_TR.get(col, col))
        for col, imp in zip(X.columns, result.importances_mean)
    ]


def format_importance_table(importances: list[FeatureImportance], top_n: int = 10) -> str:
    lines = []
    lines.append(f"{'Ozellik':35s} {'Onem':>10s}  {'Yorum':45s}")
    lines.append("-" * 95)
    for fi in importances[:top_n]:
        direction = "Gecikme riskini ARTIRIR" if fi.importance > 0 else "Gecikme riskini AZALTIR"
        lines.append(
            f"{fi.feature:<35s} {fi.importance:>10.4f}  {direction:<45s}"
        )
    return "\n".join(lines)
