"""Modelos Pydantic del proyecto."""

from mk_paper.models.audit_brief import (
    AuditBrief,
    AuditFeedback,
    AuditFinding,
    AuditVerdict,
    DimensionScores,
)
from mk_paper.models.pipeline import PipelineConfig, PipelineResult, PipelineStepResult
from mk_paper.models.research_brief import (
    ClassifiedPaper,
    LiteratureReviewOutput,
    ResearchBrief,
    SearchMatrix,
)
from mk_paper.models.writing_brief import (
    CitationCatalog,
    CitationEntry,
    CitationValidation,
    PaperDraft,
    WritingBrief,
)

__all__ = [
    "ResearchBrief",
    "SearchMatrix",
    "ClassifiedPaper",
    "LiteratureReviewOutput",
    "WritingBrief",
    "CitationEntry",
    "CitationCatalog",
    "CitationValidation",
    "PaperDraft",
    "AuditBrief",
    "AuditFinding",
    "DimensionScores",
    "AuditFeedback",
    "AuditVerdict",
    "PipelineConfig",
    "PipelineStepResult",
    "PipelineResult",
]
