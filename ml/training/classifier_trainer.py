"""Faz 5 classification model training.

S20: TimeSeriesSplit CV yalniz Train üzerinde.
S20: Hyperparameter search + calibration OOF fit yalniz Train.
S20: Validation final candidate karsilastirmasi + threshold secimi.
S21: PR-AUC ana secim metrigi.
S21: Calibration Brier ile degerlendirilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ml.calibration.calibrator import (
    CALIBRATION_METHODS,
    CalibrationModel,
    fit_calibration,
)
from ml.evaluation.metrics import evaluate_classification
from ml.features.preprocessing import build_classification_pipeline
from ml.features.schema_loader import FeatureSchema

DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_HP_SEARCH_CANDIDATES = 15


@dataclass
class CVResult:
    model_name: str
    pr_auc_mean: float
    pr_auc_std: float
    roc_auc_mean: float
    roc_auc_std: float
    oof_proba: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    best_params: dict[str, Any] | None = None


@dataclass
class ClassifierResult:
    model_name: str
    cv_result: CVResult
    calibration_results: dict[str, CalibrationEval] = field(default_factory=dict)
    selected_calibration: str = "uncalibrated"
    calibrated_model: CalibrationModel | None = None
    base_pipeline: Any | None = None
    threshold: float = 0.5
    threshold_details: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    brier_score: float | None = None


@dataclass
class CalibrationEval:
    method: str
    brier: float
    pr_auc: float
    roc_auc: float


def _build_classifier(model_name: str, **kwargs: Any) -> BaseEstimator:
    base_kwargs = {"random_state": DEFAULT_RANDOM_STATE}
    base_kwargs.update(kwargs)

    if model_name == "LogisticRegression":
        allowed = {"max_iter", "C", "random_state"}
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("max_iter", 2000)
        return LogisticRegression(**filtered)
    if model_name == "HistGradientBoosting":
        allowed = {
            "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
            "min_samples_leaf", "l2_regularization", "random_state",
        }
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("early_stopping", False)
        return HistGradientBoostingClassifier(**filtered)
    if model_name == "RandomForest":
        allowed = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "class_weight",
            "random_state",
        }
        filtered = {k: v for k, v in base_kwargs.items() if k in allowed}
        filtered.setdefault("n_jobs", -1)
        return RandomForestClassifier(**filtered)
    raise ValueError(f"Bilinmeyen model: {model_name}")


def _get_search_space(model_name: str) -> dict[str, list[Any]]:
    if model_name == "LogisticRegression":
        return {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "max_iter": [2000],
        }
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_iter": [100, 200, 300],
            "max_leaf_nodes": [15, 31, 63, None],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced", "balanced_subsample", None],
        }
    raise ValueError(f"Bilinmeyen model: {model_name}")


def _random_param_search(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    n_candidates: int = DEFAULT_HP_SEARCH_CANDIDATES,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, Any], float]:
    search_space = _get_search_space(model_name)
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
    best_score = -1.0

    for candidate in candidates:
        pr_aucs: list[float] = []
        try:
            for train_idx, oof_idx in tscv.split(X_np):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_oof_fold = X.iloc[oof_idx]
                y_oof_fold = y[oof_idx]

                clf = _build_classifier(model_name, **candidate)
                pipe = build_classification_pipeline(schema, clf)
                pipe.fit(X_train_fold, y_train_fold)
                oof_proba = pipe.predict_proba(X_oof_fold)[:, 1]
                pr_aucs.append(average_precision_score(y_oof_fold, oof_proba))
        except Exception:
            continue

        if pr_aucs:
            mean_pr = float(np.mean(pr_aucs))
            if mean_pr > best_score:
                best_score = mean_pr
                best_params = candidate

    if not best_params:
        best_params = {}
    return best_params, best_score


def train_classifier_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    n_splits: int = DEFAULT_CV_SPLITS,
    search: bool = False,
    n_search_candidates: int = DEFAULT_HP_SEARCH_CANDIDATES,
) -> CVResult:
    if search:
        best_params, search_score = _random_param_search(
            X, y, schema, model_name, n_search_candidates, n_splits,
        )
    else:
        best_params = {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    probas = []
    all_y_true = []
    pr_aucs = []
    roc_aucs = []

    X_np = X.values
    split_indices = list(tscv.split(X_np))

    for _fold_idx, (train_idx, oof_idx) in enumerate(split_indices):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_oof_fold = X.iloc[oof_idx]
        y_oof_fold = y[oof_idx]

        clf = _build_classifier(model_name, **best_params)
        pipe = build_classification_pipeline(schema, clf)
        pipe.fit(X_train_fold, y_train_fold)

        oof_proba = pipe.predict_proba(X_oof_fold)[:, 1]
        probas.extend(oof_proba.tolist())
        all_y_true.extend(y_oof_fold.tolist())

        pr_aucs.append(average_precision_score(y_oof_fold, oof_proba))
        roc_aucs.append(roc_auc_score(y_oof_fold, oof_proba))

    proba_array = np.array(probas)
    true_array = np.array(all_y_true)

    return CVResult(
        model_name=model_name,
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        roc_auc_mean=float(np.mean(roc_aucs)),
        roc_auc_std=float(np.std(roc_aucs)),
        oof_proba=proba_array,
        oof_true=true_array,
        best_params=best_params if best_params else None,
    )


def calibrate_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
    cv_result: CVResult,
    calibration_methods: list[str] | None = None,
) -> ClassifierResult:
    if cv_result.oof_proba is None or cv_result.oof_true is None:
        raise ValueError("CV OOF predictions required for calibration.")

    methods = calibration_methods or CALIBRATION_METHODS

    clf_base = _build_classifier(model_name, **(cv_result.best_params or {}))
    base_pipe = build_classification_pipeline(schema, clf_base)
    base_pipe.fit(X_train, y_train)

    cal_evals: dict[str, CalibrationEval] = {}
    cal_models: dict[str, CalibrationModel] = {}

    for method in methods:
        cal_model = fit_calibration(cv_result.oof_proba, cv_result.oof_true, method)
        cal_models[method] = cal_model

        val_proba = cal_model.predict_proba(base_pipe.predict_proba(X_val)[:, 1])
        brier = float(brier_score_loss(y_val, val_proba))
        pr_auc = float(average_precision_score(y_val, val_proba))
        roc_auc = float(roc_auc_score(y_val, val_proba))

        cal_evals[method] = CalibrationEval(
            method=method, brier=brier, pr_auc=pr_auc, roc_auc=roc_auc,
        )

    best_brier = min(cal_evals.values(), key=lambda e: e.brier)
    selected_method = best_brier.method
    selected_cal = cal_models[selected_method]

    val_proba = selected_cal.predict_proba(base_pipe.predict_proba(X_val)[:, 1])
    val_pred = val_proba >= 0.5
    val_metrics = evaluate_classification(y_val, val_pred, val_proba, "validation")
    brier = float(brier_score_loss(y_val, val_proba))

    return ClassifierResult(
        model_name=model_name,
        cv_result=cv_result,
        calibration_results=cal_evals,
        selected_calibration=selected_method,
        calibrated_model=selected_cal,
        base_pipeline=base_pipe,
        validation_metrics=val_metrics,
        brier_score=brier,
    )


def find_best_threshold(y_val: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    thresholds = np.arange(0.1, 0.91, 0.05)
    best: dict[str, Any] = {"threshold": 0.5, "f1": 0.0}

    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y_val == 1)))
        fp = int(np.sum((pred == 1) & (y_val == 0)))
        fn = int(np.sum((pred == 0) & (y_val == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "tp": tp, "fp": fp, "fn": fn,
            }

    return best


CLASSIFIER_NAMES = ["LogisticRegression", "HistGradientBoosting", "RandomForest"]
