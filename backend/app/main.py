import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, predict, history, explain, chat
from app.core.config import settings
from app.db.session import Base, engine
from app.services.knowledge_service import create_knowledge_service
from app.services.model_service import get_model_bundle

logger = logging.getLogger(__name__)


def initialize_app_state(application: FastAPI) -> None:
    """Startup work, kept synchronous and separate so it is easy to test."""
    # `engine` is read from module globals at call time so tests can patch it.
    Base.metadata.create_all(bind=engine)

    # Retrieval is needed for chat and generated explanations, not for the
    # ML prediction itself. By default the API starts in a degraded mode when
    # the knowledge base cannot load; KNOWLEDGE_SERVICE_REQUIRED=true restores
    # the previous fail-fast behaviour.
    application.state.knowledge_service = None
    application.state.knowledge_service_error = None
    try:
        application.state.knowledge_service = create_knowledge_service()
    except Exception as exc:
        if os.getenv("KNOWLEDGE_SERVICE_REQUIRED", "false").lower() == "true":
            raise
        application.state.knowledge_service_error = type(exc).__name__
        logger.exception(
            "Knowledge service unavailable; starting without retrieval. "
            "Predictions still work; chat returns 503 and explanations use the fallback."
        )

    # Warm the model artifacts and the lazily built RF explainer so the first
    # prediction does not pay the loading cost. A failure here is logged and
    # surfaced by /health; it never prevents startup.
    application.state.models_loaded = False
    if settings.model_warmup:
        started = time.perf_counter()
        try:
            bundle = get_model_bundle()
            bundle._get_rf_explainer()
            application.state.models_loaded = True
            logger.info("Model artifacts warmed in %.1f s.", time.perf_counter() - started)
        except Exception:
            logger.exception("Model warm-up failed; /predict will retry loading on first use.")


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Replaces the deprecated @app.on_event("startup") hook; behaviour is unchanged.
    initialize_app_state(application)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(predict.router, prefix=settings.api_prefix)
app.include_router(history.router, prefix=settings.api_prefix)
app.include_router(explain.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
