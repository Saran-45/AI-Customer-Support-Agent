from __future__ import annotations

from pydantic import BaseModel, field_validator


class IntentPrediction(BaseModel):
    intent: str
    confidence: float
    reason: str

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {v}")
        return v
