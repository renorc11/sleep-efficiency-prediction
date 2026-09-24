from datetime import datetime
from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    input_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    inferred_sleep_time: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_efficiency: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_quality: Mapped[str] = mapped_column(String(16), nullable=False)
    model_outputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    top_factors: Mapped[list] = mapped_column(JSON, nullable=False)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False)
