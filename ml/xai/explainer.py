"""XAI — global ve per-instance feature importance.

V1 (Faz 6): global permutation importance.
V2 (Faz 7): per-instance SHAP degerleri eklendi.
  - TreeExplainer: RandomForest icin optimize.
  - SHAP yuklu degilse global permutation'a fallback.
  - Per-instance directional contribution: "Bu ozellik riski ARTIRDI/AZALTTI"
  - Turkce sunum etiketleri canonical isimlerden ayri tutulur.
  - Causal claim uretilmez.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline

_SHAP_AVAILABLE = False
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    pass

_LABEL_CATALOG_PATH = Path(__file__).resolve().parent.parent / "config" / "label_catalog_v1.json"


@lru_cache(maxsize=1)
def _load_label_catalog() -> dict[str, str]:
    with open(_LABEL_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("feature_labels", {}))


FEATURE_LABELS_TR: dict[str, str] = _load_label_catalog()


@dataclass
class FeatureImportance:
    feature: str
    importance: float
    label_tr: str


@dataclass
class ShapContribution:
    feature: str
    shap_value: float
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
        fi_scores = _tree_importance(
            model, X_transformed, y, n_repeats, random_state,
        )
    elif isinstance(model, (LogisticRegression, ElasticNet)):
        fi_scores = _linear_importance(
            model, X_transformed, y, n_repeats, random_state,
        )
    else:
        fi_scores = _permutation_only(
            pipeline, X, y, n_repeats, random_state,
        )

    if not feature_names_out or len(feature_names_out) != len(fi_scores):
        fi_scores = _permutation_only(pipeline, X, y, n_repeats, random_state)
        feature_names_out = X.columns.tolist()

    results: list[FeatureImportance] = []
    for name, imp in zip(feature_names_out, fi_scores):
        results.append(FeatureImportance(
            feature=name,
            importance=float(imp),
            label_tr=FEATURE_LABELS_TR.get(name, name),
        ))
    results.sort(key=lambda r: abs(r.importance), reverse=True)
    return results


def compute_shap_values(
    pipeline: Pipeline,
    X_instance: pd.DataFrame,
    X_background: pd.DataFrame | None = None,
) -> list[ShapContribution]:
    if not _SHAP_AVAILABLE:
        return []

    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        model_step = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")

        if preprocessor is not None and model_step is not None:
            X_inst_t = preprocessor.transform(X_instance)
            feature_names_out = _get_transformed_feature_names(preprocessor, X_instance)

            if hasattr(model_step, "predict_proba"):
                try:
                    explainer = shap.TreeExplainer(model_step)
                    shap_vals = explainer.shap_values(X_inst_t)

                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]

                    if shap_vals.ndim == 1:
                        shap_vals = shap_vals.reshape(1, -1)
                except Exception:
                    return []

                contributions: list[ShapContribution] = []
                for i, name in enumerate(feature_names_out):
                    if i < shap_vals.shape[1]:
                        contributions.append(ShapContribution(
                            feature=name,
                            shap_value=float(shap_vals[0, i]),
                            label_tr=FEATURE_LABELS_TR.get(name, name),
                        ))
                contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
                return contributions
    except Exception:
        pass

    return []


def _get_transformed_feature_names(
    preprocessor: Any,
    X: pd.DataFrame,
) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return X.columns.tolist()


def _tree_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> list[float]:
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )
    return list(result.importances_mean)


def _linear_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> list[float]:
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return list(np.abs(coef))
    return _permutation_only(model, X, y, n_repeats, random_state)


def _permutation_only(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> list[float]:
    result = permutation_importance(
        estimator, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return list(result.importances_mean)


def _direct_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> list[FeatureImportance]:
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return [
        FeatureImportance(
            feature=col,
            importance=float(imp),
            label_tr=FEATURE_LABELS_TR.get(col, col),
        )
        for col, imp in zip(X.columns, result.importances_mean)
    ]


def format_importance_table(
    importances: list[FeatureImportance],
    top_n: int = 10,
) -> str:
    lines = []
    lines.append(f"{'Ozellik':35s} {'Onem (PI)':>10s}  Yorum")
    lines.append("-" * 80)
    for fi in importances[:top_n]:
        lines.append(
            f"{fi.feature:<35s} {fi.importance:>10.4f}  "
            f"Bu ozellik modelin global tahmin performansina katki saglar"
        )
    return "\n".join(lines)

