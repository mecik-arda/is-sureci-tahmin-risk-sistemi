"""Import pipeline şema genişletmesi.

import_runs: sayaçlar, canonical_mapping_version, yeni durumlar.
data_quality_issues: issue_code, severity.
processes: current_row_fingerprint, last_import_id, updated_at.

Revision ID: 002
Revises: 001
Create Date: 2025-08-06
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("import_runs", sa.Column("canonical_mapping_version", sa.String(), nullable=True))
    op.add_column("import_runs", sa.Column("inserted_rows", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("import_runs", sa.Column("updated_rows", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("import_runs", sa.Column("skipped_duplicate_rows", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("import_runs", sa.Column("quarantined_rows", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("import_runs", sa.Column("warning_count", sa.Integer(), nullable=False, server_default="0"))

    with op.batch_alter_table("import_runs", schema=None) as batch_op:
        batch_op.alter_column("status", existing_type=sa.String(), nullable=False, existing_server_default=None)
        batch_op.drop_constraint("ck_import_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_import_runs_status",
            "status IN ('pending', 'running', 'completed', 'completed_with_issues', 'failed', 'duplicate_file')",
        )
        batch_op.create_check_constraint("ck_import_runs_inserted_rows", "inserted_rows >= 0")
        batch_op.create_check_constraint("ck_import_runs_updated_rows", "updated_rows >= 0")
        batch_op.create_check_constraint("ck_import_runs_skipped_duplicate_rows", "skipped_duplicate_rows >= 0")
        batch_op.create_check_constraint("ck_import_runs_quarantined_rows", "quarantined_rows >= 0")
        batch_op.create_check_constraint("ck_import_runs_warning_count", "warning_count >= 0")

    op.add_column("data_quality_issues", sa.Column("issue_code", sa.String(), nullable=True))
    op.add_column("data_quality_issues", sa.Column("severity", sa.String(), nullable=False, server_default="warning"))

    with op.batch_alter_table("data_quality_issues", schema=None) as batch_op:
        batch_op.alter_column("issue_type", existing_type=sa.String(), nullable=True)
        batch_op.drop_constraint("ck_data_quality_issues_issue_type", type_="check")
        batch_op.create_check_constraint(
            "ck_data_quality_issues_severity",
            "severity IN ('warning', 'error')",
        )
        batch_op.create_check_constraint(
            "ck_data_quality_issues_issue_code",
            "issue_code IN ('DUPLICATE_FILE', 'UNKNOWN_CANONICAL_MAPPING', "
            "'REQUIRED_FIELD_MISSING', 'INVALID_DATE', 'INVALID_FILE_SCHEMA', "
            "'OPENING_DATA_CONFLICT', 'CANONICAL_MAPPING_MISMATCH', "
            "'UNSUPPORTED_FILE_FORMAT', 'MISSING_REQUIRED_COLUMN')",
        )

    op.add_column("processes", sa.Column("current_row_fingerprint", sa.String(), nullable=True))
    op.add_column("processes", sa.Column("last_import_id", sa.Integer(), nullable=True))
    op.add_column("processes", sa.Column("updated_at", sa.DateTime(), nullable=True))
    op.add_column("processes", sa.Column("closure_reason", sa.String(), nullable=True))

    with op.batch_alter_table("processes", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_processes_last_import_id",
            "import_runs",
            ["last_import_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("processes", schema=None) as batch_op:
        batch_op.drop_constraint("fk_processes_last_import_id", type_="foreignkey")

    op.drop_column("processes", "closure_reason")
    op.drop_column("processes", "updated_at")
    op.drop_column("processes", "last_import_id")
    op.drop_column("processes", "current_row_fingerprint")

    with op.batch_alter_table("data_quality_issues", schema=None) as batch_op:
        batch_op.drop_constraint("ck_data_quality_issues_issue_code", type_="check")
        batch_op.drop_constraint("ck_data_quality_issues_severity", type_="check")
        batch_op.alter_column("issue_type", existing_type=sa.String(), nullable=False)

    op.drop_column("data_quality_issues", "severity")
    op.drop_column("data_quality_issues", "issue_code")

    with op.batch_alter_table("import_runs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_import_runs_warning_count", type_="check")
        batch_op.drop_constraint("ck_import_runs_quarantined_rows", type_="check")
        batch_op.drop_constraint("ck_import_runs_skipped_duplicate_rows", type_="check")
        batch_op.drop_constraint("ck_import_runs_updated_rows", type_="check")
        batch_op.drop_constraint("ck_import_runs_inserted_rows", type_="check")
        batch_op.drop_constraint("ck_import_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_import_runs_status",
            "status IN ('pending', 'running', 'completed', 'failed')",
        )

    op.drop_column("import_runs", "warning_count")
    op.drop_column("import_runs", "quarantined_rows")
    op.drop_column("import_runs", "skipped_duplicate_rows")
    op.drop_column("import_runs", "updated_rows")
    op.drop_column("import_runs", "inserted_rows")
    op.drop_column("import_runs", "canonical_mapping_version")
