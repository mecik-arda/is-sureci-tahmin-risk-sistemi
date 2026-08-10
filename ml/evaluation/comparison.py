"""Faz 5 model karsilastirma tablosu.

S20/S21 uyumlu: winner secimi Validation üzerindedir.
"""

from __future__ import annotations

from typing import Any

from ml.training.classifier_trainer import ClassifierResult, CalibrationEval
from ml.training.regressor_trainer import RegressorResult


def format_classification_table(results: list[ClassifierResult]) -> str:
    lines = []
    header = (
        f"{'Model':30s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1':>8s} "
        f"{'Prec':>8s} {'Rec':>8s} {'Brier':>8s} {'Thresh':>8s}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    best_idx = -1
    best_pr = -1.0
    for i, r in enumerate(results):
        if r.validation_metrics and r.validation_metrics["pr_auc"] > best_pr:
            best_pr = r.validation_metrics["pr_auc"]
            best_idx = i

    for i, r in enumerate(results):
        prefix = "-> " if i == best_idx else "   "
        if r.validation_metrics:
            v = r.validation_metrics
            lines.append(
                f"{prefix}{r.model_name:27s} "
                f"{v['pr_auc']:8.4f} "
                f"{v['roc_auc']:8.4f} "
                f"{v['f1']:8.4f} "
                f"{v['precision']:8.4f} "
                f"{v['recall']:8.4f} "
                f"{r.brier_score or 0:8.4f} "
                f"{r.threshold:8.3f}"
            )
    return "\n".join(lines)


def format_cv_summary(results: list[ClassifierResult]) -> str:
    lines = ["CV (TimeSeriesSplit, 5-fold, Train-only):"]
    lines.append(f"{'Model':30s} {'PR-AUC':>16s} {'ROC-AUC':>16s}")
    lines.append("-" * 65)
    for r in results:
        cv = r.cv_result
        params_str = ""
        if cv.best_params:
            params_str = f"  best={cv.best_params}"
        lines.append(
            f"{r.model_name:30s} "
            f"{cv.pr_auc_mean:.4f} +/- {cv.pr_auc_std:.4f}  "
            f"{cv.roc_auc_mean:.4f} +/- {cv.roc_auc_std:.4f}"
            f"{params_str}"
        )
    return "\n".join(lines)


def format_calibration_comparison(
    cal_results: dict[str, CalibrationEval],
    selected: str,
) -> str:
    lines = ["Calibration Comparison (Validation):"]
    lines.append(f"{'Method':15s} {'Brier':>8s} {'PR-AUC':>8s} {'ROC-AUC':>8s}")
    lines.append("-" * 45)
    for method, e in cal_results.items():
        prefix = "-> " if method == selected else "   "
        lines.append(
            f"{prefix}{method:<12s} "
            f"{e.brier:8.4f} "
            f"{e.pr_auc:8.4f} "
            f"{e.roc_auc:8.4f}"
        )
    return "\n".join(lines)


def format_regression_table(results: list[RegressorResult]) -> str:
    lines = []
    header = (
        f"{'Model':30s} {'MAE':>10s} {'MedAE':>10s} {'RMSE':>10s} {'p90AE':>10s}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    best_idx = -1
    best_mae = float("inf")
    for i, r in enumerate(results):
        if r.validation_metrics and r.validation_metrics["mae"] < best_mae:
            best_mae = r.validation_metrics["mae"]
            best_idx = i

    for i, r in enumerate(results):
        prefix = "-> " if i == best_idx else "   "
        if r.validation_metrics:
            v = r.validation_metrics
            lines.append(
                f"{prefix}{r.model_name:27s} "
                f"{v['mae']:10.2f} "
                f"{v['median_ae']:10.2f} "
                f"{v['rmse']:10.2f} "
                f"{v['p90_abs_error']:10.2f}"
            )
    return "\n".join(lines)


def get_winner_classifier(results: list[ClassifierResult]) -> ClassifierResult | None:
    best = None
    best_pr = -1.0
    for r in results:
        if r.validation_metrics and r.validation_metrics["pr_auc"] > best_pr:
            best_pr = r.validation_metrics["pr_auc"]
            best = r
    return best


def get_winner_regressor(results: list[RegressorResult]) -> RegressorResult | None:
    best = None
    best_mae = float("inf")
    for r in results:
        if r.validation_metrics and r.validation_metrics["mae"] < best_mae:
            best_mae = r.validation_metrics["mae"]
            best = r
    return best
