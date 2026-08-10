"""İlk veritabanı şeması.

processes, process_snapshots, import_runs, data_quality_issues,
model_bundles, prediction_runs, prediction_feedback tablolarını oluşturur.

Revision ID: 001
Revises:
Create Date: 2025-08-06
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- processes ---
    op.create_table(
        "processes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("process_type", sa.String(), nullable=True),
        sa.Column("current_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deadline", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("source_payload_json", sa.Text(), nullable=True),
        sa.Column("imported_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_processes_external_id", "processes", ["external_id"])
    op.create_index("ix_processes_created_at", "processes", ["created_at"])

    # --- import_runs ---
    op.create_table(
        "import_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False),
        sa.Column("imported_rows", sa.Integer(), nullable=False),
        sa.Column("error_rows", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_import_runs_status",
        ),
        sa.CheckConstraint("total_rows >= 0", name="ck_import_runs_total_rows"),
        sa.CheckConstraint("imported_rows >= 0", name="ck_import_runs_imported_rows"),
        sa.CheckConstraint("error_rows >= 0", name="ck_import_runs_error_rows"),
    )

    # --- data_quality_issues ---
    op.create_table(
        "data_quality_issues",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("import_run_id", sa.Integer(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=True),
        sa.Column("column_name", sa.String(), nullable=True),
        sa.Column("issue_type", sa.String(), nullable=False),
        sa.Column("issue_message", sa.Text(), nullable=True),
        sa.Column("raw_value", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["import_run_id"], ["import_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "issue_type IN ('missing_required', 'type_mismatch', "
            "'date_inconsistency', 'duplicate_id', 'invalid_value')",
            name="ck_data_quality_issues_issue_type",
        ),
    )
    op.create_index(
        "ix_data_quality_issues_import_run_id",
        "data_quality_issues",
        ["import_run_id"],
    )

    # --- process_snapshots ---
    op.create_table(
        "process_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("process_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_type", sa.String(), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(), nullable=False),
        sa.Column("feature_schema_version", sa.String(), nullable=True),
        sa.Column("input_json", sa.Text(), nullable=False),
        sa.Column("input_fingerprint", sa.String(), nullable=True),
        sa.Column("source_import_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["process_id"], ["processes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_import_id"], ["import_runs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "snapshot_type IN ('opening')",
            name="ck_process_snapshots_snapshot_type",
        ),
    )
    op.create_index(
        "ix_process_snapshots_process_id", "process_snapshots", ["process_id"]
    )
    op.create_index(
        "ix_process_snapshots_snapshot_at", "process_snapshots", ["snapshot_at"]
    )

    # --- model_bundles ---
    op.create_table(
        "model_bundles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("artifact_path", sa.String(), nullable=False),
        sa.Column("artifact_hash", sa.String(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("feature_list_json", sa.Text(), nullable=True),
        sa.Column("trained_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version"),
        sa.CheckConstraint("is_active IN (0, 1)", name="ck_model_bundles_is_active"),
        sa.CheckConstraint(
            "model_type IN ('classifier', 'regressor', 'bundle')",
            name="ck_model_bundles_model_type",
        ),
    )

    # --- prediction_runs ---
    op.create_table(
        "prediction_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("process_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("prediction_type", sa.String(), nullable=False),
        sa.Column("delay_probability", sa.Numeric(), nullable=True),
        sa.Column("risk_score", sa.Integer(), nullable=True),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("predicted_hours", sa.Numeric(), nullable=True),
        sa.Column("explanation_json", sa.Text(), nullable=True),
        sa.Column("input_fingerprint", sa.String(), nullable=True),
        sa.Column("predicted_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["process_id"], ["processes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["process_snapshots.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "prediction_type IN ('normal', 'simulation')",
            name="ck_prediction_runs_prediction_type",
        ),
        sa.CheckConstraint(
            "risk_level IS NULL OR risk_level IN ('low', 'medium', 'high')",
            name="ck_prediction_runs_risk_level",
        ),
        sa.CheckConstraint(
            "risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)",
            name="ck_prediction_runs_risk_score",
        ),
    )
    op.create_index(
        "ix_prediction_runs_process_id", "prediction_runs", ["process_id"]
    )

    # --- prediction_feedback ---
    op.create_table(
        "prediction_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("feedback_type", sa.String(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("actual_outcome", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["prediction_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "feedback_type IN ('accuracy', 'usefulness')",
            name="ck_prediction_feedback_feedback_type",
        ),
        sa.CheckConstraint(
            "actual_outcome IS NULL OR actual_outcome IN (0, 1)",
            name="ck_prediction_feedback_actual_outcome",
        ),
    )
    op.create_index(
        "ix_prediction_feedback_prediction_id",
        "prediction_feedback",
        ["prediction_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_feedback_prediction_id", table_name="prediction_feedback")
    op.drop_table("prediction_feedback")
    op.drop_index("ix_prediction_runs_process_id", table_name="prediction_runs")
    op.drop_table("prediction_runs")
    op.drop_table("model_bundles")
    op.drop_index("ix_process_snapshots_snapshot_at", table_name="process_snapshots")
    op.drop_index("ix_process_snapshots_process_id", table_name="process_snapshots")
    op.drop_table("process_snapshots")
    op.drop_index("ix_data_quality_issues_import_run_id", table_name="data_quality_issues")
    op.drop_table("data_quality_issues")
    op.drop_table("import_runs")
    op.drop_index("ix_processes_created_at", table_name="processes")
    op.drop_index("ix_processes_external_id", table_name="processes")
    op.drop_table("processes")
