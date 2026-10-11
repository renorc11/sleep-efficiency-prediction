import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Sleep Efficiency Predictor"
    api_prefix: str = "/api/v1"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./sleep_app.db")
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(Path(__file__).resolve().parents[3] / "ml" / "artifacts"))).resolve()
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


settings = Settings()
