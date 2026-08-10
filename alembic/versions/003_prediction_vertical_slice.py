"""Prediction vertical slice schema.

prediction_runs: status, model_bundle_id, prediction_context,
                partial UNIQUE index uq_prediction_success_identity.

Revision ID: 003
Revises: 002
Create Date: 2025-08-07
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("prediction_runs", sa.Column("status", sa.String(), nullable=True))
    op.add_column("prediction_runs", sa.Column("model_bundle_id", sa.Integer(), nullable=True))
    op.add_column("prediction_runs", sa.Column("prediction_context", sa.String(), nullable=True))

    op.execute("UPDATE prediction_runs SET status = 'failed' WHERE status IS NULL")

    with op.batch_alter_table("prediction_runs", schema=None) as batch_op:
        batch_op.alter_column("status", existing_type=sa.String(), nullable=False)
        batch_op.alter_column("model_bundle_id", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("prediction_context", existing_type=sa.String(), nullable=False,
                              existing_server_default=None, server_default=sa.text("'opening'"))
        batch_op.create_check_constraint(
            "ck_prediction_runs_status",
            "status IN ('success', 'failed')",
        )
        batch_op.create_foreign_key(
            "fk_prediction_runs_model_bundle_id",
            "model_bundles",
            ["model_bundle_id"],
            ["id"],
        )

    op.execute(
        "CREATE UNIQUE INDEX uq_prediction_success_identity "
        "ON prediction_runs (snapshot_id, input_fingerprint, model_bundle_id, prediction_context) "
        "WHERE status = 'success'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_prediction_success_identity")

    with op.batch_alter_table("prediction_runs", schema=None) as batch_op:
        batch_op.drop_constraint("fk_prediction_runs_model_bundle_id", type_="foreignkey")
        batch_op.drop_constraint("ck_prediction_runs_status", type_="check")

    op.drop_column("prediction_runs", "prediction_context")
    op.drop_column("prediction_runs", "model_bundle_id")
    op.drop_column("prediction_runs", "status")
