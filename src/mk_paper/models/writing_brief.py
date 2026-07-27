"""Modelos del Agente Redactor Científico (Scientific Writer)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WritingBrief(BaseModel):
    """Brief de redacción: fusiona literatura local + AnalysisReport."""

    title: str
    literature_review_path: str
    analysis_report_path: str
    authors: list[str] = Field(
        default_factory=list,
        description="Autores meta del manuscrito (no citas bibliográficas).",
    )
    language: Literal["es", "en"] = "es"
    include_latex: bool = True
    cite_all_catalog: bool = False
    """Si True, la bibliografía incluye todo el catálogo; si False, solo citadas."""


class CitationEntry(BaseModel):
    """Entrada APA 7 derivada exclusivamente del Literature Reviewer."""

    cite_key: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    title: str | None = None
    venue: str | None = None
    doi: str | None = None
    level: Literal["core", "conceptual", "seminal"] = "conceptual"
    key_findings: list[str] = Field(default_factory=list)
    citation_context: str = ""
    suggested_section: str = ""
    apa_reference: str = ""
    apa_parenthetical: str = ""
    apa_narrative: str = ""


class CitationCatalog(BaseModel):
    """Catálogo deduplicado de citas permitidas (anti-alucinación)."""

    entries: list[CitationEntry] = Field(default_factory=list)
    by_key: dict[str, CitationEntry] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def allowed_keys(self) -> list[str]:
        return [e.cite_key for e in self.entries]


class CitationValidation(BaseModel):
    """Resultado de validar citas in-text vs catálogo."""

    status: Literal["ok", "error"] = "ok"
    citations_found: list[str] = Field(default_factory=list)
    citations_ok: list[str] = Field(default_factory=list)
    citations_unknown: list[str] = Field(default_factory=list)
    message: str = ""


class PaperDraft(BaseModel):
    """Borrador IMRaD listo para persistir."""

    title: str
    markdown: str = ""
    latex: str | None = None
    citations_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation: CitationValidation = Field(default_factory=CitationValidation)
    literature_path: str = ""
    analysis_path: str = ""
    language: Literal["es", "en"] = "es"
    status: Literal["ok", "error", "partial"] = "ok"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
