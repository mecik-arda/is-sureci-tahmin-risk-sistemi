"""Faz 5 calibration tests.

S20: Calibration fit yalniz Train OOF predictions.
Her calibration sample'i kendisini görmemis bir model tarafindan predict edilir.
Validation/Training hiçbir calibrator.fit islemine girmemelidir.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.calibration.calibrator import (
    CALIBRATION_METHODS,
    CalibrationModel,
    fit_calibration,
)


class TestCalibrationModel:
    def test_sigmoid_produces_calibrated_probabilities(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 500).astype(float)
        raw = rng.beta(2, 5, 500)
        raw = np.clip(raw, 1e-10, 1 - 1e-10)

        cal = fit_calibration(raw, y_true, "sigmoid")
        calibrated = cal.predict_proba(raw[:50])
        assert cal.method == "sigmoid"
        assert cal.fitted_model is not None
        assert calibrated.shape == (50,)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_isotonic_produces_calibrated_probabilities(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 500).astype(float)
        raw = rng.beta(2, 5, 500)
        raw = np.clip(raw, 1e-10, 1 - 1e-10)

        cal = fit_calibration(raw, y_true, "isotonic")
        calibrated = cal.predict_proba(raw[:50])
        assert cal.method == "isotonic"
        assert cal.fitted_model is not None
        assert calibrated.shape == (50,)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_uncalibrated_passes_through_raw(self):
        raw = np.array([0.1, 0.5, 0.9])
        cal = fit_calibration(raw, np.array([0, 1, 1]), "uncalibrated")
        calibrated = cal.predict_proba(raw)
        assert cal.method == "uncalibrated"
        assert cal.fitted_model is None
        np.testing.assert_array_almost_equal(calibrated, raw)

    def test_sigmoid_improves_calibration_on_miscalibrated_data(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 1000).astype(float)
        raw_overconfident = np.where(y_true == 1, 0.3 + rng.beta(10, 2, 1000) * 0.5, 0.0)
        raw_overconfident = np.clip(raw_overconfident, 1e-10, 1 - 1e-10)

        cal = fit_calibration(raw_overconfident, y_true, "sigmoid")

        test_y = rng.randint(0, 2, 500).astype(float)
        test_raw = np.where(test_y == 1, 0.3 + rng.beta(10, 2, 500) * 0.5, 0.0)
        test_raw = np.clip(test_raw, 1e-10, 1 - 1e-10)
        calibrated = cal.predict_proba(test_raw)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_isotonic_is_monotonic(self):
        raw = np.array([0.1, 0.2, 0.9])
        y_true = np.array([0, 0, 1])
        cal = fit_calibration(raw, y_true, "isotonic")
        calibrated = cal.predict_proba(raw)
        assert calibrated[0] <= calibrated[1] <= calibrated[2]

    def test_sigmoid_preserves_ordering(self):
        raw = np.array([0.1, 0.5, 0.9])
        y_true = np.array([0, 1, 1])
        cal = fit_calibration(raw, y_true, "sigmoid")
        calibrated = cal.predict_proba(raw)
        assert calibrated[0] <= calibrated[1] <= calibrated[2]

    def test_all_methods_in_calibration_methods(self):
        assert "uncalibrated" in CALIBRATION_METHODS
        assert "sigmoid" in CALIBRATION_METHODS
        assert "isotonic" in CALIBRATION_METHODS

    def test_calibration_model_raises_on_unknown_method(self):
        cal = CalibrationModel(method="unknown", fitted_model=None)
        with pytest.raises(ValueError, match="Unknown"):
            cal.predict_proba(np.array([0.5]))

    def test_sigmoid_handles_extreme_probabilities(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        raw = np.array([1e-20, 1e-10, 0.9999999, 1 - 1e-20])
        cal = fit_calibration(raw, y_true, "sigmoid")
        calibrated = cal.predict_proba(raw)
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_isotonic_out_of_bounds_clipped(self):
        raw = np.array([0.1, 0.5, 0.9])
        y_true = np.array([0, 1, 1])
        cal = fit_calibration(raw, y_true, "isotonic")
        calibrated = cal.predict_proba(np.array([-0.5, 0.0, 1.0, 1.5]))
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
