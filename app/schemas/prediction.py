"""Prediction response semalari."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class PredictionResponse(BaseModel):
    prediction_run_id: int
    process_id: int
    snapshot_id: int | None = None
    model_bundle_id: int
    prediction_context: str = "opening"
    status: str
    reused: bool
    classification_available: bool
    delay_probability: float | None = None
    delay_probability_lower: float | None = None
    delay_probability_upper: float | None = None
    risk_score: int | None = None
    risk_level: str | None = None
    predicted_is_delayed: bool | None = None
    integration_threshold: float | None = None
    regression_available: bool
    predicted_duration_hours: float | None = None
    predicted_duration_hours_lower: float | None = None
    predicted_duration_hours_upper: float | None = None
    model_stage: str = "integration_baseline"
    created_at: datetime


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_ids: list[StrictInt] = Field(min_length=1, max_length=50)

    @field_validator("process_ids")
    @classmethod
    def validate_process_ids(cls, value: list[int]) -> list[int]:
        if any(process_id <= 0 for process_id in value):
            raise ValueError("Süreç kimlikleri pozitif olmalıdır.")
        if len(set(value)) != len(value):
            raise ValueError("Aynı süreç kimliği birden fazla kez gönderilemez.")
        return value
