from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _load_simulation_options() -> dict[str, list[str]]:
    import json
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _cached() -> dict[str, list[str]]:
        from pathlib import Path
        mapping_path = Path(__file__).resolve().parent.parent.parent / "ml" / "mappings" / "canonical_map_v1.json"
        columns = json.loads(mapping_path.read_text(encoding="utf-8")).get("columns", {})
        return {
            column: sorted(set(values.values()) | {"missing", "unknown"})
            for column, values in columns.items()
            if column in {"source", "subject", "reason", "type", "neighborhood"}
        }
    return _cached()


class SimulationOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    subject: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    reason: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    type: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    neighborhood: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    open_month: int | None = Field(default=None, ge=1, le=12)
    open_weekday: int | None = Field(default=None, ge=0, le=6)
    open_hour: int | None = Field(default=None, ge=0, le=23)
    is_weekend: int | None = Field(default=None, ge=0, le=1)
    sla_duration_hours: float | None = Field(default=None, ge=0, le=87600)

    @field_validator("source", "subject", "reason", "type", "neighborhood")
    @classmethod
    def validate_categorical_value(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def check_categorical_values(self) -> "SimulationOverrides":
        options = _load_simulation_options()
        categorical_fields = ("source", "subject", "reason", "type", "neighborhood")
        invalid = []
        for field_name in categorical_fields:
            value = getattr(self, field_name)
            if value is not None and value not in options.get(field_name, set()):
                invalid.append(field_name)
        if invalid:
            raise ValueError(
                "Senaryo kategorik alanlarında izin verilmeyen değer var: "
                + ", ".join(sorted(invalid))
            )
        return self


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_prediction_id: int = Field(gt=0)
    overrides: SimulationOverrides

    @field_validator("overrides")
    @classmethod
    def require_override(cls, value: SimulationOverrides) -> SimulationOverrides:
        if not value.model_dump(exclude_none=True):
            raise ValueError("En az bir senaryo alanı değiştirilmelidir.")
        return value


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_type: Literal["accuracy", "usefulness"]
    comment: str | None = Field(default=None, max_length=5000)

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @model_validator(mode="after")
    def require_usefulness_comment(self) -> "FeedbackRequest":
        if self.feedback_type == "usefulness" and self.comment is None:
            raise ValueError("Fayda geri bildirimi için yorum gereklidir.")
        return self
