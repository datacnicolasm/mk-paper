"""Modelos del embudo de revisión sistemática de literatura."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    """Contexto estructurado de investigación (entrada del Literature Reviewer)."""

    title: str
    objectives: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    variables: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Claves típicas: independent, dependent, controls.",
    )
    methodology: str = ""
    domain: str = ""
    years_back: int = 3
    max_results: int = 20
    seminal_dois: list[str] = Field(
        default_factory=list,
        description=(
            "Whitelist de DOIs fundacionales/seminales. Saltan filtro de "
            "recencia y descarte por baja similitud; van a Seminal Literature."
        ),
    )

    @classmethod
    def from_loose_query(cls, query: str, *, max_results: int = 20) -> ResearchBrief:
        """Envoltorio mínimo cuando solo se dispone de una query suelta."""
        q = (query or "").strip()
        return cls(
            title=q or "Untitled research",
            objectives=[f"Revisar literatura relevante sobre: {q}"],
            research_questions=[q],
            variables={"keywords": [q]},
            methodology="Revisión sistemática exploratoria",
            domain=q,
            max_results=max_results,
        )


class SearchMatrix(BaseModel):
    """Matriz de búsqueda avanzada generada por el LLM."""

    queries: list[str] = Field(
        default_factory=list,
        description="Consultas booleanas/frases listas para APIs académicas.",
    )
    synonyms: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    rationale: str = ""

    @staticmethod
    def _coerce_str_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, dict):
            items: list[str] = []
            for key, val in value.items():
                items.append(str(key))
                if isinstance(val, list):
                    items.extend(str(x) for x in val)
                elif val is not None:
                    items.append(str(val))
            return [x for x in items if x.strip()]
        return [str(value)]

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> SearchMatrix:  # type: ignore[override]
        if isinstance(obj, dict):
            data = dict(obj)
            data["synonyms"] = cls._coerce_str_list(data.get("synonyms"))
            data["method_terms"] = cls._coerce_str_list(data.get("method_terms"))
            queries = data.get("queries")
            if isinstance(queries, dict):
                data["queries"] = cls._coerce_str_list(queries)
            elif isinstance(queries, list):
                data["queries"] = [str(q) for q in queries]
            obj = data
        return super().model_validate(obj, *args, **kwargs)


class ClassifiedPaper(BaseModel):
    """Paper clasificado para el Agente Redactor."""

    doi: str | None = None
    title: str | None = None
    year: int | None = None
    abstract: str | None = None
    citation_count: int = 0
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    is_oa: bool = False
    oa_status: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    sources: list[str] = Field(default_factory=list)
    level: Literal["core", "conceptual", "seminal"] = "conceptual"
    utility: float = 0.5
    relevance_score: float = 0.0
    alignment_score: float = 0.0
    alignment_quadrant: str | None = None
    # core_auto | llm_review | discard | seminal
    historical_centrality: float = 0.0  # citas / años desde publicación
    seminal_reason: str | None = None  # whitelist | historical_centrality
    key_findings: list[str] = Field(default_factory=list)
    citation_context: str = ""
    suggested_section: str = ""
    full_text_excerpt: str | None = None
    full_text_chars: int = 0
    pdf_local_path: str | None = None
    pdf_parse_status: str | None = None  # ok | failed | skipped


class LiteratureReviewOutput(BaseModel):
    """Salida Writer-ready del embudo sistemático."""

    brief_title: str
    search_matrix: SearchMatrix | dict[str, Any] = Field(default_factory=dict)
    primary_sources_used: list[str] = Field(default_factory=list)
    core_findings: list[ClassifiedPaper] = Field(default_factory=list)
    conceptual_references: list[ClassifiedPaper] = Field(default_factory=list)
    seminal_literature: list[ClassifiedPaper] = Field(
        default_factory=list,
        description="Fundamentos teóricos e históricos (no mezclar con Nivel 1).",
    )
    discarded_count: int = 0
    alignment_thresholds: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict JSON-compatible."""
        return self.model_dump(mode="json")
