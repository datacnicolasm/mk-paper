"""Capa de persistencia local de artefactos del proyecto."""

from mk_paper.persistence.analysis_store import AnalysisArtifacts, save_analysis_report
from mk_paper.persistence.audit_store import AuditArtifacts, save_audit_verdict
from mk_paper.persistence.literature_store import (
    LiteratureArtifacts,
    ReviewArtifacts,
    save_literature_results,
    save_literature_review,
)
from mk_paper.persistence.paper_store import PaperArtifacts, save_paper_draft
from mk_paper.persistence.run_store import PipelineRunContext, create_pipeline_run

__all__ = [
    "LiteratureArtifacts",
    "ReviewArtifacts",
    "save_literature_results",
    "save_literature_review",
    "AnalysisArtifacts",
    "save_analysis_report",
    "PaperArtifacts",
    "save_paper_draft",
    "AuditArtifacts",
    "save_audit_verdict",
    "PipelineRunContext",
    "create_pipeline_run",
]
