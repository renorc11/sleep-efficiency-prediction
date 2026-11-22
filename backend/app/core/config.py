import os
from pathlib import Path
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


def resolve_database_url(url: str) -> str:
    """Anchor relative SQLite paths to backend/ instead of the current directory.

    `sqlite:///./sleep_app.db` used to resolve against wherever uvicorn was
    started, so running from the project root and from backend/ silently used
    different databases (and history appeared to vanish). backend/ is the
    documented startup directory and holds the existing database, so it is
    the anchor. Absolute paths and non-SQLite URLs are unchanged.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url == "sqlite://" or url.startswith(prefix + "/"):
        return url
    relative = url[len(prefix):]
    if relative.startswith(":memory:"):
        return url
    return prefix + str((BACKEND_DIR / relative).resolve())


class Settings(BaseModel):
    app_name: str = "Sleep Efficiency Predictor"
    api_prefix: str = "/api/v1"
    database_url: str = resolve_database_url(os.getenv("DATABASE_URL", "sqlite:///./sleep_app.db"))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "ml" / "artifacts"))).resolve()
    # Explicit origins, comma-separated in CORS_ORIGINS (e.g. a deployed frontend).
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]
    # Local development: any port on localhost/127.0.0.1. Next.js moves to 3001,
    # 3002, ... when 3000 is busy, and the fixed :3000 list then rejected the
    # browser's CORS preflight with 400 ("Failed to fetch"). Set CORS_ORIGIN_REGEX
    # to an empty string to disable.
    cors_origin_regex: str | None = os.getenv(
        "CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    ) or None
    # Load models and the RF explainer at startup so the first /predict is
    # not slowed by artifact loading.
    model_warmup: bool = os.getenv("MODEL_WARMUP", "true").lower() == "true"


settings = Settings()
