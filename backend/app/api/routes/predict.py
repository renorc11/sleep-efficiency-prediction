from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.db.session import get_db
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.inactivity_service import build_engineered_features
from app.services.model_service import get_model_bundle
from app.services.recommendation_service import generate_recommendations

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    lifestyle = payload.lifestyle.model_dump()
    features, inactivity = build_engineered_features(
        lifestyle,
        payload.screen_activity_sequence,
        sleep_time_hint=payload.sleep_time_hint,
    )

    bundle = get_model_bundle()
    pred = bundle.predict_all(features, payload.screen_activity_sequence)

    top_factors = pred["top_factors"][:5]
    recommendations = generate_recommendations(features, top_factors)

    record = PredictionRecord(
        input_payload=payload.model_dump(),
        inferred_sleep_time=inactivity.inferred_sleep_time,
        predicted_efficiency=pred["ensemble"],
        predicted_quality=pred["sleep_quality"],
        model_outputs={k: pred[k] for k in ["rf", "xgb", "hybrid", "ensemble"]},
        top_factors=top_factors,
        recommendations=recommendations,
    )
    db.add(record)
    db.commit()

    return PredictionResponse(
        predicted_sleep_efficiency=round(pred["ensemble"], 2),
        sleep_quality_label=pred["sleep_quality"],
        inferred_sleep_time=inactivity.inferred_sleep_time,
        model_outputs={k: round(pred[k], 2) for k in ["rf", "xgb", "hybrid", "ensemble"]},
        top_influencing_factors=top_factors,
        recommendations=recommendations,
    )
