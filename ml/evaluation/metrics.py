"""Siniflandirma ve regresyon metrikleri.

Tüm metrik fonksiyonlari sealed_guard tarafindan korunur.
Test/Audit split'leri için çagrilirlarsa hata firlatirlar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.evaluation.sealed_guard import assert_evaluable


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    """Siniflandirma metriklerini hesaplar.

    Metrikler: ROC-AUC, PR-AUC, Precision, Recall, F1, Confusion Matrix.
    """
    assert_evaluable(split_name)

    return {
        "split": split_name,
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "row_count": int(len(y_true)),
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    """Regresyon metriklerini gerçek saat ölçeginde hesaplar.

    Metrikler: MAE, MedianAE, RMSE, p90 Absolute Error.
    """
    assert_evaluable(split_name)

    abs_errors = np.abs(y_true - y_pred)

    return {
        "split": split_name,
        "mae": float(np.mean(abs_errors)),
        "median_ae": float(np.median(abs_errors)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "p90_abs_error": float(np.percentile(abs_errors, 90)),
        "row_count": int(len(y_true)),
    }
