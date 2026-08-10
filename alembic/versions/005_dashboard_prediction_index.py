from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_prediction_runs_dashboard_recent "
        "ON prediction_runs (model_bundle_id, predicted_at) "
        "WHERE status = 'success' AND prediction_context = 'opening'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_prediction_runs_dashboard_recent")
