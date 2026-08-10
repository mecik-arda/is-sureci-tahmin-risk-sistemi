"""Feature schema ve feature derivation testleri."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ml.features.feature_derivation import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    derive_features,
    validate_feature_names,
)
from ml.features.schema_loader import (
    FeatureSchema,
    SchemaValidationError,
    load_feature_schema,
    validate_snapshot_input,
)


class TestSchemaLoader:
    def test_schema_loads_correctly(self):
        schema = load_feature_schema()
        assert schema.feature_schema_version == "opening-v1"
        assert schema.canonical_mapping_version == "1.0.0"
        assert schema.snapshot_type == "opening"

    def test_schema_has_5_categorical_features(self):
        schema = load_feature_schema()
        assert len(schema.categorical_features) == 5
        assert schema.categorical_features == ["source", "subject", "reason", "type", "neighborhood"]

    def test_schema_has_5_numeric_features(self):
        schema = load_feature_schema()
        assert len(schema.numeric_features) == 5
        assert schema.numeric_features == [
            "open_month", "open_weekday", "open_hour", "is_weekend", "sla_duration_hours",
        ]

    def test_schema_has_10_total_features(self):
        schema = load_feature_schema()
        assert len(schema.all_features) == 10

    def test_schema_excludes_department_and_queue(self):
        schema = load_feature_schema()
        assert "department" in schema.excluded_features
        assert "queue" in schema.excluded_features

    def test_schema_excludes_outcome_fields(self):
        schema = load_feature_schema()
        assert "case_status" in schema.excluded_features
        assert "closed_dt" in schema.excluded_features
        assert "completed_at" in schema.excluded_features


class TestSchemaValidation:
    def test_valid_input_passes(self):
        schema = load_feature_schema()
        input_json = {
            "external_id": "123",
            "created_at": "2024-01-15T10:30:00",
            "deadline": "2024-02-15T10:30:00",
            "source": "citizens_connect_app",
            "subject": "public_works_department",
            "reason": "highway_maintenance",
            "type": "request_for_pothole_repair",
            "neighborhood": "roxbury",
        }
        validate_snapshot_input(input_json, schema)

    def test_missing_created_at_raises(self):
        schema = load_feature_schema()
        input_json = {
            "external_id": "123",
            "deadline": "2024-02-15T10:30:00",
            "source": "test",
            "subject": "test",
            "reason": "test",
            "type": "test",
            "neighborhood": "test",
        }
        with pytest.raises(SchemaValidationError, match="created_at"):
            validate_snapshot_input(input_json, schema)

    def test_missing_categorical_key_raises(self):
        schema = load_feature_schema()
        input_json = {
            "external_id": "123",
            "created_at": "2024-01-15T10:30:00",
            "deadline": None,
            "source": "test",
            "subject": "test",
            "reason": "test",
            "type": "test",
        }
        with pytest.raises(SchemaValidationError, match="neighborhood"):
            validate_snapshot_input(input_json, schema)

    def test_deadline_none_is_valid(self):
        schema = load_feature_schema()
        input_json = {
            "external_id": "123",
            "created_at": "2024-01-15T10:30:00",
            "deadline": None,
            "source": "missing",
            "subject": "missing",
            "reason": "missing",
            "type": "missing",
            "neighborhood": "missing",
        }
        validate_snapshot_input(input_json, schema)


class TestFeatureDerivation:
    def _make_input(self, **overrides):
        base = {
            "external_id": "123",
            "created_at": "2024-03-15T14:30:00",
            "deadline": "2024-04-15T14:30:00",
            "source": "citizens_connect_app",
            "subject": "public_works_department",
            "reason": "highway_maintenance",
            "type": "request_for_pothole_repair",
            "neighborhood": "roxbury",
        }
        base.update(overrides)
        return base

    def test_returns_10_features(self):
        features = derive_features(self._make_input())
        assert len(features) == 10

    def test_categorical_values_passed_through(self):
        features = derive_features(self._make_input())
        assert features["source"] == "citizens_connect_app"
        assert features["subject"] == "public_works_department"
        assert features["reason"] == "highway_maintenance"
        assert features["type"] == "request_for_pothole_repair"
        assert features["neighborhood"] == "roxbury"

    def test_open_month(self):
        features = derive_features(self._make_input(created_at="2024-03-15T14:30:00"))
        assert features["open_month"] == 3

    def test_open_weekday(self):
        features = derive_features(self._make_input(created_at="2024-01-15T14:30:00"))
        assert features["open_weekday"] == 0

    def test_open_hour(self):
        features = derive_features(self._make_input(created_at="2024-03-15T14:30:00"))
        assert features["open_hour"] == 14

    def test_is_weekend_false(self):
        features = derive_features(self._make_input(created_at="2024-01-17T10:00:00"))
        assert features["is_weekend"] == 0

    def test_is_weekend_true(self):
        features = derive_features(self._make_input(created_at="2024-01-20T10:00:00"))
        assert features["is_weekend"] == 1

    def test_sla_duration_hours(self):
        features = derive_features(self._make_input(
            created_at="2024-01-01T00:00:00",
            deadline="2024-01-02T00:00:00",
        ))
        assert features["sla_duration_hours"] == 24.0

    def test_sla_duration_none_deadline(self):
        features = derive_features(self._make_input(deadline=None))
        assert np.isnan(features["sla_duration_hours"])

    def test_sla_duration_negative_becomes_nan(self):
        features = derive_features(self._make_input(
            created_at="2024-01-02T00:00:00",
            deadline="2024-01-01T00:00:00",
        ))
        assert np.isnan(features["sla_duration_hours"])

    def test_missing_categorical_defaults_to_missing(self):
        input_json = self._make_input()
        del input_json["neighborhood"]
        features = derive_features(input_json)
        assert features["neighborhood"] == "missing"

    def test_missing_created_at_raises(self):
        input_json = self._make_input()
        del input_json["created_at"]
        with pytest.raises(ValueError, match="created_at"):
            derive_features(input_json)


class TestFeatureNameConsistency:
    def test_validate_feature_names_passes(self):
        schema = load_feature_schema()
        validate_feature_names(schema)

    def test_constant_lists_match_schema(self):
        schema = load_feature_schema()
        assert CATEGORICAL_FEATURES == schema.categorical_features
        assert NUMERIC_FEATURES == schema.numeric_features
