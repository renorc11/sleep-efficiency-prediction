from fastapi import APIRouter, HTTPException

from app.schemas.prediction import LocalExplanationResponse, PredictionRequest
from app.services.inactivity_service import build_engineered_features
from app.services.model_service import get_model_bundle

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.get("/global")
def global_feature_importance() -> dict:
    bundle = get_model_bundle()
    explanation = bundle.global_tree_shap("xgb")
    # Keep the established key while exposing the method and provenance.
    return {
        "global_feature_importance": explanation["feature_importance"][:15],
        "explanation": explanation,
    }


@router.post("/local", response_model=LocalExplanationResponse)
def local_explanation(payload: PredictionRequest) -> LocalExplanationResponse:
    """Return independent explanations for the deployed RF, XGB, and Hybrid models."""
    features, inactivity = build_engineered_features(
        payload.lifestyle.model_dump(),
        payload.screen_activity_sequence,
        sleep_time_hint=payload.sleep_time_hint,
    )
    try:
        result = get_model_bundle().explain_all(features, payload.screen_activity_sequence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return LocalExplanationResponse(
        inferred_sleep_time=inactivity.inferred_sleep_time,
        **result,
    )


REGRESSION_METRICS = ("mae", "rmse", "r2")
THRESHOLD_METRICS = ("accuracy", "precision", "recall", "f1")
# ROC-AUC in metadata.json was computed from (prediction - 50) / 50, a linear
# rescaling of the regression output, not a classifier probability. It is
# withheld from the API; the artifact itself is left unchanged.
WITHHELD_METRICS = {
    "roc_auc": "Withheld: computed from a linear rescaling of the regression output, not from a "
               "classifier probability, so it is not a valid metric for this regression model.",
}


@router.get("/model-metrics")
def model_metrics() -> dict:
    bundle = get_model_bundle()
    summary = bundle.meta.get("metrics_summary", {})
    threshold = bundle.meta.get("quality_threshold")
    return {
        # Same flat per-model shape as before, without withheld metrics.
        "metrics_summary": {
            model: {k: v for k, v in values.items() if k not in WITHHELD_METRICS}
            for model, values in summary.items()
        },
        "metric_groups": {
            "regression": {
                "metrics": list(REGRESSION_METRICS),
                "description": "Primary metrics of the sleep-efficiency index regression.",
            },
            "threshold_derived": {
                "metrics": list(THRESHOLD_METRICS),
                "threshold": threshold,
                "description": f"Secondary. The constructed index and the regression output are both "
                               f"thresholded at {threshold}; these measure agreement on that band, "
                               f"not a separately trained classifier.",
            },
        },
        "withheld_metrics": WITHHELD_METRICS,
        "evaluation_scope": "Historical 75-row test split of constructed development data; examined during "
                            "development and not an untouched hold-out.",
    }
