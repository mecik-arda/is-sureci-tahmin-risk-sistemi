"""Faz 5 regression model training.

S20: TimeSeriesSplit CV yalniz Train üzerinde.
S20: Hyperparameter search yalniz Train.
S21: MAE ana secim metrigi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ml.evaluation.metrics import evaluate_regression
from ml.features.preprocessing import build_regression_pipeline
from ml.features.schema_loader import FeatureSchema

DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_HP_SEARCH_CANDIDATES = 15


@dataclass
class RegressorCVResult:
    model_name: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    oof_preds: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    best_params: dict[str, Any] | None = None


@dataclass
class RegressorResult:
    model_name: str
    cv_result: RegressorCVResult
    fitted_pipeline: Any = None
    validation_metrics: dict[str, Any] | None = None


def _build_regressor(model_name: str, **kwargs: Any) -> BaseEstimator:
    base_kwargs = {"random_state": DEFAULT_RANDOM_STATE}
    base_kwargs.update(kwargs)

    if model_name == "ElasticNet_log1p":
        allowed = {"alpha", "l1_ratio", "max_iter", "random_state"}
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("max_iter", 5000)
        return TransformedTargetRegressor(
            regressor=ElasticNet(**filtered),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    if model_name == "HistGradientBoostingRegressor":
        allowed = {
            "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
            "min_samples_leaf", "l2_regularization", "random_state",
        }
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("early_stopping", False)
        return HistGradientBoostingRegressor(**filtered)
    if model_name == "RandomForestRegressor":
        allowed = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "random_state",
        }
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("n_jobs", -1)
        return RandomForestRegressor(**filtered)
    raise ValueError(f"Bilinmeyen model: {model_name}")


def _get_regressor_search_space(model_name: str) -> dict[str, list[Any]]:
    if model_name == "ElasticNet_log1p":
        return {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
            "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "max_iter": [5000],
        }
    if model_name == "HistGradientBoostingRegressor":
        return {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_iter": [100, 200, 300],
            "max_leaf_nodes": [15, 31, 63, None],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
        }
    if model_name == "RandomForestRegressor":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        }
    raise ValueError(f"Bilinmeyen model: {model_name}")


def _random_param_search_regressor(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    n_candidates: int = DEFAULT_HP_SEARCH_CANDIDATES,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, Any], float]:
    search_space = _get_regressor_search_space(model_name)
    rng = np.random.RandomState(random_state)

    param_names = list(search_space.keys())
    candidates: list[dict[str, Any]] = []
    for _ in range(n_candidates):
        candidate: dict[str, Any] = {}
        for name in param_names:
            values = search_space[name]
            if values is not None and len(values) > 0:
                candidate[name] = values[rng.randint(len(values))]
        candidates.append(candidate)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_np = X.values

    best_params: dict[str, Any] = {}
    best_score = float("inf")

    for candidate in candidates:
        maes: list[float] = []
        try:
            for train_idx, oof_idx in tscv.split(X_np):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_oof_fold = X.iloc[oof_idx]
                y_oof_fold = y[oof_idx]

                reg = _build_regressor(model_name, **candidate)
                pipe = build_regression_pipeline(schema, reg)
                pipe.fit(X_train_fold, y_train_fold)
                oof_pred = np.maximum(0.0, pipe.predict(X_oof_fold))
                maes.append(mean_absolute_error(y_oof_fold, oof_pred))
        except Exception:
            continue

        if maes:
            mean_mae = float(np.mean(maes))
            if mean_mae < best_score:
                best_score = mean_mae
                best_params = candidate

    if not best_params:
        best_params = {}
    return best_params, best_score


def train_regressor_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    n_splits: int = DEFAULT_CV_SPLITS,
    search: bool = False,
    n_search_candidates: int = DEFAULT_HP_SEARCH_CANDIDATES,
) -> RegressorCVResult:
    if search:
        best_params, search_score = _random_param_search_regressor(
            X, y, schema, model_name, n_search_candidates, n_splits,
        )
    else:
        best_params = {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = []
    all_y_true = []
    maes = []
    rmses = []

    X_np = X.values
    split_indices = list(tscv.split(X_np))

    for _fold_idx, (train_idx, oof_idx) in enumerate(split_indices):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_oof_fold = X.iloc[oof_idx]
        y_oof_fold = y[oof_idx]

        reg = _build_regressor(model_name, **best_params)
        pipe = build_regression_pipeline(schema, reg)
        pipe.fit(X_train_fold, y_train_fold)

        oof_pred = np.maximum(0.0, pipe.predict(X_oof_fold))
        preds.extend(oof_pred.tolist())
        all_y_true.extend(y_oof_fold.tolist())

        maes.append(mean_absolute_error(y_oof_fold, oof_pred))
        rmses.append(np.sqrt(mean_squared_error(y_oof_fold, oof_pred)))

    pred_array = np.array(preds)
    true_array = np.array(all_y_true)

    return RegressorCVResult(
        model_name=model_name,
        mae_mean=float(np.mean(maes)),
        mae_std=float(np.std(maes)),
        rmse_mean=float(np.mean(rmses)),
        rmse_std=float(np.std(rmses)),
        oof_preds=pred_array,
        oof_true=true_array,
        best_params=best_params if best_params else None,
    )


def fit_and_evaluate_regressor(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    best_params: dict[str, Any] | None = None,
) -> RegressorResult:
    params = best_params or {}
    reg = _build_regressor(model_name, **params)
    pipe = build_regression_pipeline(schema, reg)
    pipe.fit(X_train, y_train)

    val_pred = np.maximum(0.0, pipe.predict(X_val))
    val_metrics = evaluate_regression(y_val, val_pred, "validation")

    return RegressorResult(
        model_name=model_name,
        cv_result=RegressorCVResult(
            model_name=model_name,
            mae_mean=0.0, mae_std=0.0,
            rmse_mean=0.0, rmse_std=0.0,
        ),
        fitted_pipeline=pipe,
        validation_metrics=val_metrics,
    )


REGRESSOR_NAMES = ["ElasticNet_log1p", "HistGradientBoostingRegressor", "RandomForestRegressor"]
