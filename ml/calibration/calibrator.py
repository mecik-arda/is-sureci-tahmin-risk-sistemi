"""OOF-safe probability calibration.

S20: Calibration fit yalniz Train OOF predictions üzerinde.
Her calibration sample'i kendisini görmemis bir model tarafindan predict edilir.

Validation ve Test/Audit calibration fit'e giremez.

Supports:
  - sigmoid (Platt scaling via LogisticRegression on logit-transformed OOF proba)
  - isotonic (IsotonicRegression on OOF proba)
  - uncalibrated (pass-through)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPSILON = 1e-15


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    fitted_model: Any | None

    def predict_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        if self.method == "uncalibrated":
            return np.asarray(raw_proba, dtype=np.float64).copy()
        if self.method == "sigmoid":
            clipped = np.clip(raw_proba, EPSILON, 1 - EPSILON)
            logit = np.log(clipped / (1 - clipped))
            return self.fitted_model.predict_proba(logit.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.fitted_model.predict(np.asarray(raw_proba, dtype=np.float64))
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_calibration(
    oof_proba: np.ndarray,
    y_true: np.ndarray,
    method: str,
) -> CalibrationModel:
    if method == "uncalibrated":
        return CalibrationModel(method="uncalibrated", fitted_model=None)

    if method == "sigmoid":
        clipped = np.clip(oof_proba, EPSILON, 1 - EPSILON)
        logit = np.log(clipped / (1 - clipped))
        model = LogisticRegression(C=1.0, fit_intercept=True, solver="lbfgs")
        model.fit(logit.reshape(-1, 1), y_true)
        return CalibrationModel(method="sigmoid", fitted_model=model)

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(oof_proba, y_true)
        return CalibrationModel(method="isotonic", fitted_model=model)

    raise ValueError(f"Unknown calibration method: {method}")


CALIBRATION_METHODS = ["uncalibrated", "sigmoid", "isotonic"]
