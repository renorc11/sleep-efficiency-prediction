
from fastapi import APIRouter, Depends, HTTPException, Request

from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.db.session import get_db
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.inactivity_service import build_engineered_features
from app.services.model_service import get_model_bundle
from app.services.recommendation_service import generate_recommendations
from app.services.personalized_recommendation_service import (
    recommend_for_prediction,
)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictionResponse:

    lifestyle = payload.lifestyle.model_dump()

    

    features, inactivity = build_engineered_features(
        lifestyle,
        payload.screen_activity_sequence,
        sleep_time_hint=payload.sleep_time_hint,
    )


    bundle = get_model_bundle()

    try:
        pred = bundle.predict_all(
            features,
            payload.screen_activity_sequence,
        )
    except ValueError as exc:
        # Model-input validation failures are client errors, not server faults.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    top_factors = pred["top_factors"][:5]

    recommendations = generate_recommendations(
        features,
        top_factors,
    )

    # Save the original ML prediction before attempting LLM generation.
    record = PredictionRecord(
        input_payload=payload.model_dump(),
        inferred_sleep_time=inactivity.inferred_sleep_time,
        predicted_efficiency=pred["ensemble"],
        predicted_quality=pred["sleep_quality"],
        model_outputs={
            k: pred[k]
            for k in ["rf", "xgb", "hybrid", "ensemble"]
        },
        top_factors=top_factors,
        recommendations=recommendations,
    )

    db.add(record)
    db.commit()

    # The ML prediction is complete and persisted before any LLM work. The
    # automatic SHAP -> RAG -> BioMistral pipeline replaces the former
    # free-text explanation call (one LLM call instead of two) and also fills
    # the existing generated_explanation/explanation_status fields.
    explanation_result = recommend_for_prediction(
        bundle=bundle,
        lifestyle=lifestyle,
        features=features,
        prediction=pred,
        sequence=payload.screen_activity_sequence,
        inferred_sleep_time=inactivity.inferred_sleep_time,
        knowledge_service=getattr(request.app.state, "knowledge_service", None),
        legacy_top_factors=top_factors,
        legacy_recommendations=recommendations,
    )

    return PredictionResponse(
        predicted_sleep_efficiency=round(pred["ensemble"], 2),
        sleep_quality_label=pred["sleep_quality"],
        inferred_sleep_time=inactivity.inferred_sleep_time,
        model_outputs={
            k: round(pred[k], 2)
            for k in ["rf", "xgb", "hybrid", "ensemble"]
        },
        top_influencing_factors=top_factors,
        recommendations=recommendations,
        generated_explanation=explanation_result[
            "generated_explanation"
        ],
        explanation_status=explanation_result[
            "explanation_status"
        ],
        prediction_diagnostics=pred.get("diagnostics"),
        personalized_recommendations=explanation_result["personalized_recommendations"],
    )
