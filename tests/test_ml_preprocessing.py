"""Preprocessing pipeline testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.preprocessing import build_preprocessor
from ml.features.schema_loader import load_feature_schema


def _make_X(n=4, **overrides):
    data = {
        "source": ["a", "b", "a", "b"][:n],
        "subject": ["x", "y", "x", "y"][:n],
        "reason": ["r1", "r2", "r1", "r2"][:n],
        "type": ["t1", "t2", "t1", "t2"][:n],
        "neighborhood": ["n1", "n2", "n1", "n2"][:n],
        "open_month": [1, 3, 5, 7][:n],
        "open_weekday": [0, 1, 2, 3][:n],
        "open_hour": [10, 12, 14, 16][:n],
        "is_weekend": [0, 0, 1, 1][:n],
        "sla_duration_hours": [24.0, 48.0, 72.0, 96.0][:n],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestPreprocessor:
    def test_has_categorical_and_numeric_transformers(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        names = [t[0] for t in preprocessor.transformers]
        assert "categorical" in names
        assert "numeric" in names

    def test_fit_transform_no_nan(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        X = _make_X(4)
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert not np.isnan(transformed).any()

    def test_transform_validation_after_train_fit(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        train_X = _make_X(4)
        preprocessor.fit(train_X)

        val_X = _make_X(2)
        transformed = preprocessor.transform(val_X)
        assert transformed.shape[0] == 2

    def test_handle_unknown_ignore_for_unseen_category(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        train_X = _make_X(2)
        preprocessor.fit(train_X)

        val_X = _make_X(1, source=["unseen"], subject=["unseen"],
                         reason=["unseen"], type=["unseen"], neighborhood=["unseen"])
        transformed = preprocessor.transform(val_X)
        assert transformed.shape[0] == 1

    def test_missing_categorical_imputed(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        X = _make_X(4)
        X.loc[0, "source"] = np.nan
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert not np.isnan(transformed).any()

    def test_missing_numeric_imputed(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        X = _make_X(4)
        X.loc[0, "sla_duration_hours"] = np.nan
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert not np.isnan(transformed).any()

    def test_remainder_dropped(self):
        schema = load_feature_schema()
        preprocessor = build_preprocessor(schema)
        X = _make_X(4)
        X["extra_leakage_column"] = 42
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[0] == 4
