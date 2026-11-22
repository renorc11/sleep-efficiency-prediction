from fastapi import APIRouter, Request

from app.services import model_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    # "status" is unchanged; component availability is additive so clients can
    # tell when chat and generated explanations are degraded.
    knowledge = getattr(request.app.state, "knowledge_service", None)
    return {
        "status": "ok",
        "knowledge_service": "available" if knowledge is not None else "unavailable",
        "models": "loaded" if model_service._model_bundle is not None else "not_loaded",
    }
