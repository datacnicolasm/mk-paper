"""Modelos del Agente Auditor de Calidad (lit-review & writer)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AuditBrief(BaseModel):
    """Entrada del Quality Auditor: paths al draft + evidencia bibliográfica."""

    title: str = "Q1-Q2 quality audit"
    paper_draft_path: str | None = None
    writing_brief_path: str | None = None
    literature_review_path: str | None = None
    language: Literal["es", "en"] = "es"
    quality_threshold: float = Field(default=8.5, ge=0.0, le=10.0)
    max_revision_rounds: int = Field(default=2, ge=0, le=5)
    include_latex: bool = True
    use_llm: bool = True


class AuditFinding(BaseModel):
    """Hallazgo individual de auditoría."""

    category: Literal[
        "structure_bullets",
        "tone",
        "citations",
        "coherence",
        "orphan_claims",
        "research_question",
        "sections",
        "other",
    ]
    severity: Literal["critical", "major", "minor"] = "major"
    section: str = ""
    message: str
    suggested_fix: str = ""


class DimensionScores(BaseModel):
    """Puntajes por dimensión (0–10)."""

    prose_structure: float = 10.0
    tone: float = 10.0
    citations: float = 10.0
    coherence: float = 10.0
    research_question: float = 10.0


class AuditFeedback(BaseModel):
    """Reporte de corrección estructurado para el Expert Academic Writer."""

    overall_score: float = 0.0
    threshold: float = 8.5
    decision: Literal["accept", "revise", "reject"] = "revise"
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    findings: list[AuditFinding] = Field(default_factory=list)
    revision_instructions: str = ""
    summary: str = ""


class AuditVerdict(BaseModel):
    """Dictamen formal de aceptación / revisión para publicación."""

    title: str
    decision: Literal["accept", "revise", "reject"] = "revise"
    overall_score: float = 0.0
    threshold: float = 8.5
    rounds_completed: int = 0
    feedback: AuditFeedback = Field(default_factory=AuditFeedback)
    feedback_history: list[AuditFeedback] = Field(default_factory=list)
    polished_markdown: str = ""
    latex: str | None = None
    paper_draft_path: str = ""
    literature_path: str = ""
    warnings: list[str] = Field(default_factory=list)
    status: Literal["ok", "error", "partial"] = "ok"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
