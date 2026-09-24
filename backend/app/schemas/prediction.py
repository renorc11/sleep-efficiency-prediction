from datetime import datetime
from pydantic import BaseModel, Field, conlist


class LifestyleInput(BaseModel):
    caffeine_mg_evening: float = Field(ge=0, le=800)
    alcohol_units: float = Field(ge=0, le=10)
    dinner_hour: float = Field(ge=16, le=24)
    stress_level: float = Field(ge=1, le=10)
    mood_score: float = Field(ge=-2, le=2)
    physical_activity_min: float = Field(ge=0, le=300)
    naps_count: int = Field(ge=0, le=5)
    naps_total_min: float = Field(ge=0, le=240)
    previous_sleep_hours: float = Field(ge=3, le=12)
    work_hours: float = Field(ge=0, le=16)
    blue_light_minutes: float = Field(ge=0, le=600)
    bedtime_consistency: float = Field(ge=0, le=1)
    hydration_liters: float = Field(ge=0.5, le=6)


class PredictionRequest(BaseModel):
    lifestyle: LifestyleInput
    # 5-min activity bins, 288 values for a full day. 1=interaction, 0=no interaction
    screen_activity_sequence: conlist(int, min_length=288, max_length=288)
    # Optional rough bedtime hint from user in HH:MM (24h), used to stabilize inference.
    sleep_time_hint: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class PredictionResponse(BaseModel):
    predicted_sleep_efficiency: float
    sleep_quality_label: str
    inferred_sleep_time: str
    model_outputs: dict
    top_influencing_factors: list[dict]
    recommendations: list[str]


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    predicted_sleep_efficiency: float
    sleep_quality_label: str
    inferred_sleep_time: str
