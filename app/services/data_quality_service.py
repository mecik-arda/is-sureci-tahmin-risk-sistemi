from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.import_run import DataQualityIssue, ImportRun


def get_data_quality_summary(session: Session) -> dict[str, Any]:
    runs = session.execute(
        select(ImportRun).order_by(ImportRun.completed_at.desc(), ImportRun.id.desc())
    ).scalars().all()
    issues = session.execute(select(DataQualityIssue)).scalars().all()
    issue_codes = Counter(issue.issue_code or "UNKNOWN" for issue in issues)
    severities = Counter(issue.severity for issue in issues)

    return {
        "total_import_runs": len(runs),
        "total_rows": sum(run.total_rows for run in runs),
        "imported_rows": sum(run.imported_rows for run in runs),
        "quarantined_rows": sum(run.quarantined_rows for run in runs),
        "error_rows": sum(run.error_rows for run in runs),
        "warning_count": sum(run.warning_count for run in runs),
        "issue_count": len(issues),
        "issues_by_severity": dict(sorted(severities.items())),
        "issues_by_code": [
            {"code": code, "count": count}
            for code, count in issue_codes.most_common()
        ],
        "recent_runs": [
            {
                "file_name": run.file_name,
                "status": run.status,
                "total_rows": run.total_rows,
                "imported_rows": run.imported_rows,
                "quarantined_rows": run.quarantined_rows,
                "error_rows": run.error_rows,
                "warning_count": run.warning_count,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            }
            for run in runs[:10]
        ],
    }
