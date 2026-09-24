"""init predictions table

Revision ID: 0001_init_predictions
Revises:
Create Date: 2026-02-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_init_predictions"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("inferred_sleep_time", sa.String(length=10), nullable=False),
        sa.Column("predicted_efficiency", sa.Float(), nullable=False),
        sa.Column("predicted_quality", sa.String(length=16), nullable=False),
        sa.Column("model_outputs", sa.JSON(), nullable=False),
        sa.Column("top_factors", sa.JSON(), nullable=False),
        sa.Column("recommendations", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_id"), "predictions", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_id"), table_name="predictions")
    op.drop_table("predictions")
