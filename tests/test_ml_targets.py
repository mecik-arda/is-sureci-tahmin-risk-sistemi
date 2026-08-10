"""Target builder testleri."""

from __future__ import annotations

from datetime import datetime

import pytest

from ml.datasets.target_builder import (
    OBSERVATION_CUTOFF,
    compute_is_delayed,
    compute_total_duration_hours,
)


class TestClassificationTarget:
    def test_closed_before_sla_returns_0(self):
        created = datetime(2024, 1, 1, 0, 0, 0)
        deadline = datetime(2024, 2, 1, 0, 0, 0)
        closed = datetime(2024, 1, 15, 0, 0, 0)
        assert compute_is_delayed(closed, deadline) == 0

    def test_closed_exactly_at_sla_returns_0(self):
        deadline = datetime(2024, 2, 1, 12, 0, 0)
        closed = datetime(2024, 2, 1, 12, 0, 0)
        assert compute_is_delayed(closed, deadline) == 0

    def test_closed_after_sla_returns_1(self):
        deadline = datetime(2024, 2, 1, 0, 0, 0)
        closed = datetime(2024, 2, 2, 0, 0, 0)
        assert compute_is_delayed(closed, deadline) == 1

    def test_open_and_sla_before_cutoff_returns_1(self):
        deadline = datetime(2024, 12, 1, 0, 0, 0)
        assert compute_is_delayed(None, deadline) == 1

    def test_open_and_sla_after_cutoff_returns_none(self):
        deadline = datetime(2025, 6, 1, 0, 0, 0)
        assert compute_is_delayed(None, deadline) is None

    def test_open_and_sla_at_cutoff_returns_none(self):
        assert compute_is_delayed(None, OBSERVATION_CUTOFF) is None

    def test_sla_missing_returns_none(self):
        closed = datetime(2024, 1, 15, 0, 0, 0)
        assert compute_is_delayed(closed, None) is None

    def test_sla_missing_and_open_returns_none(self):
        assert compute_is_delayed(None, None) is None

    def test_custom_observation_cutoff(self):
        deadline = datetime(2024, 6, 1, 0, 0, 0)
        assert compute_is_delayed(None, deadline, datetime(2024, 7, 1, 0, 0, 0)) == 1
        assert compute_is_delayed(None, deadline, datetime(2024, 5, 1, 0, 0, 0)) is None


class TestRegressionTarget:
    def test_correct_duration(self):
        created = datetime(2024, 1, 1, 0, 0, 0)
        completed = datetime(2024, 1, 2, 0, 0, 0)
        assert compute_total_duration_hours(created, completed) == 24.0

    def test_partial_hours(self):
        created = datetime(2024, 1, 1, 0, 0, 0)
        completed = datetime(2024, 1, 1, 6, 30, 0)
        assert compute_total_duration_hours(created, completed) == 6.5

    def test_open_case_returns_none(self):
        created = datetime(2024, 1, 1, 0, 0, 0)
        assert compute_total_duration_hours(created, None) is None

    def test_negative_duration_returns_none(self):
        created = datetime(2024, 1, 2, 0, 0, 0)
        completed = datetime(2024, 1, 1, 0, 0, 0)
        assert compute_total_duration_hours(created, completed) is None

    def test_zero_duration_valid(self):
        created = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 0, 0)
        assert compute_total_duration_hours(created, completed) == 0.0

    def test_sla_missing_but_closed_is_included(self):
        created = datetime(2024, 1, 1, 0, 0, 0)
        completed = datetime(2024, 1, 5, 0, 0, 0)
        result = compute_total_duration_hours(created, completed)
        assert result == 96.0
