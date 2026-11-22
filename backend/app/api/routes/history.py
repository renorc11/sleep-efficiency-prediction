from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.db.session import get_db
from app.schemas.prediction import PredictionHistoryItem

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[PredictionHistoryItem])
def history(
    # Bounded: previously any value (including negative or huge) was accepted.
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[PredictionHistoryItem]:
    stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
    rows = db.scalars(stmt).all()
    return [
        PredictionHistoryItem(
            id=row.id,
            created_at=row.created_at,
            predicted_sleep_efficiency=row.predicted_efficiency,
            sleep_quality_label=row.predicted_quality,
            inferred_sleep_time=row.inferred_sleep_time,
        )
        for row in rows
    ]
