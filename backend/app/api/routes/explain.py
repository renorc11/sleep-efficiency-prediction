from fastapi import APIRouter

from app.services.model_service import get_model_bundle

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.get("/global")
def global_feature_importance() -> dict:
    bundle = get_model_bundle()
    # infer global importances from XGBoost model internals
    importance = bundle.xgb.get_booster().get_score(importance_type="gain")
    ranked = sorted(
        (
            {"feature": bundle.features[int(k[1:])] if k.startswith("f") else k, "importance": float(v)}
            for k, v in importance.items()
        ),
        key=lambda it: it["importance"],
        reverse=True,
    )
    return {"global_feature_importance": ranked[:15]}


@router.get("/model-metrics")
def model_metrics() -> dict:
    bundle = get_model_bundle()
    return {"metrics_summary": bundle.meta.get("metrics_summary", {})}
