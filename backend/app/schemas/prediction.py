from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, conlist, field_validator


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

    @field_validator("screen_activity_sequence")
    @classmethod
    def _sequence_must_be_binary(cls, value: list[int]) -> list[int]:
        # The models were trained on binary bins; other values are outside the
        # input domain and would be silently passed to the Hybrid CNN.
        invalid = sorted({item for item in value if item not in (0, 1)})
        if invalid:
            raise ValueError(
                "screen_activity_sequence must contain only 0 (no interaction) "
                f"or 1 (interaction); found {invalid[:5]}."
            )
        return value


class PersonalizedRecommendation(BaseModel):
    """One recommendation. Wording may come from BioMistral; every number,
    attribution, source, priority and confidence is filled in by code."""

    title: str
    priority: int
    wording_source: Literal["biomistral", "mixed", "template"]
    factor_key: str
    related_features: list[str]
    kind: Literal["improve", "maintain"]
    model_evidence: dict[str, Any]
    scientific_evidence: dict[str, Any] | None
    source_ids: list[str]
    reason: str
    action: str
    confidence: Literal["low", "moderate", "high"]
    safety_note: str


class PersonalizedRecommendationResult(BaseModel):
    summary: str
    prediction_context: dict[str, Any]
    model_explanation: dict[str, Any]
    recommendations: list[PersonalizedRecommendation]
    retrieved_sources: list[dict[str, Any]]
    uncertainty: list[str]
    generation_status: Literal["generated", "fallback"]
    fallback_used: bool
    fallback_reason: str | None = None
    validation_issues: list[str] = []
    warnings: list[str] = []


class PredictionResponse(BaseModel):

    predicted_sleep_efficiency: float

    sleep_quality_label: str

    inferred_sleep_time: str

    model_outputs: dict

    top_influencing_factors: list[dict]

    recommendations: list[str]

    generated_explanation: str

    explanation_status: str

    # Additive diagnostics. Optional so existing clients and fixtures keep
    # working; `model_outputs` is deliberately unchanged because the web and
    # Android clients iterate over every key in it.
    prediction_diagnostics: dict[str, Any] | None = None

    # Automatic SHAP -> RAG -> BioMistral recommendations (additive).
    personalized_recommendations: PersonalizedRecommendationResult | None = None


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    predicted_sleep_efficiency: float
    sleep_quality_label: str
    inferred_sleep_time: str


class LocalExplanationResponse(BaseModel):
    """Additive-model diagnostics; does not alter the prediction response contract."""

    inferred_sleep_time: str
    model_predictions: dict[str, float]
    model_explanations: dict[str, Any]
    prediction_diagnostics: dict[str, Any] | None = None
    warnings: list[str]
