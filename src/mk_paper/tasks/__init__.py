"""Tasks del crew."""

from mk_paper.tasks.analysis_tasks import create_analysis_task
from mk_paper.tasks.auditor_tasks import create_audit_task
from mk_paper.tasks.literature_tasks import create_literature_review_task
from mk_paper.tasks.writer_tasks import create_writer_task

__all__ = [
    "create_literature_review_task",
    "create_analysis_task",
    "create_writer_task",
    "create_audit_task",
]
