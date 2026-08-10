"""Faz 6 runtime schema.

Simulasyon overrides, persisted predicted_is_delayed, feedback uniqueness.

Revision ID: 004
Revises: 003
Create Date: 2026-08-07
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("prediction_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("simulation_overrides_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("predicted_is_delayed", sa.Integer(), nullable=True))

    with op.batch_alter_table("prediction_feedback", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_prediction_feedback_prediction_id_feedback_type",
            ["prediction_id", "feedback_type"],
        )


def downgrade() -> None:
    with op.batch_alter_table("prediction_feedback", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_prediction_feedback_prediction_id_feedback_type",
            type_="unique",
        )

    with op.batch_alter_table("prediction_runs", schema=None) as batch_op:
        batch_op.drop_column("predicted_is_delayed")
        batch_op.drop_column("simulation_overrides_json")
