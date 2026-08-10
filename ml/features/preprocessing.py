"""Scikit-learn preprocessing pipeline.

Tüm ögrenme yalnizca Pipeline.fit(X_train) sirasinda train verisi üzerinde olur.
Validation üzerinde hiçbir statistic/encoder/scaler ögrenilmez.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features.schema_loader import FeatureSchema


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """V1 feature schema için ColumnTransformer olusturur.

    Kategorik: constant imputation ('missing') + OneHotEncoder (handle_unknown='ignore')
    Sayisal: median imputation + StandardScaler

    handle_unknown='ignore' yalniz savunma katmanidir.
    """
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, list(schema.categorical_features)),
            ("numeric", numeric_transformer, list(schema.numeric_features)),
        ],
        remainder="drop",
    )


def build_classification_pipeline(
    schema: FeatureSchema,
    classifier: BaseEstimator,
) -> Pipeline:
    """Full classification pipeline: preprocessing + classifier."""
    return Pipeline([
        ("preprocessor", build_preprocessor(schema)),
        ("classifier", classifier),
    ])


def build_regression_pipeline(
    schema: FeatureSchema,
    regressor: BaseEstimator | TransformerMixin,
) -> Pipeline:
    """Full regression pipeline: preprocessing + regressor."""
    return Pipeline([
        ("preprocessor", build_preprocessor(schema)),
        ("regressor", regressor),
    ])
