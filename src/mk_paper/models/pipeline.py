"""Contratos del orquestador end-to-end mk-paper."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuración de una ejecución punta a punta."""

    research_brief_path: str
    method_brief_path: str
    dataset_path: str | None = None
    """Override del dataset_path del MethodBrief (CSV/XLSX local)."""
    literature_review_path: str | None = None
    """Si se provee, salta la búsqueda y usa este review.json."""
    skip_literature: bool = False
    """Salta APIs de literatura; requiere literature_review_path."""
    title: str | None = None
    language: Literal["es", "en"] = "es"
    quality_threshold: float = Field(default=8.5, ge=0.0, le=10.0)
    max_audit_rounds: int = Field(default=2, ge=0, le=5)
    use_llm: bool = True
    via_crew: bool = False
    enrich_analysis_discussion: bool = False
    include_latex: bool = True
    literature_max_results: int | None = None


class PipelineStepResult(BaseModel):
    """Resultado de un paso del pipeline."""

    step: str
    status: Literal["ok", "skipped", "error"] = "ok"
    message: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Salida agregada de ``run_pipeline``."""

    run_id: str
    run_dir: str
    status: Literal["ok", "partial", "error"] = "ok"
    decision: Literal["accept", "revise", "reject", "pending"] = "pending"
    overall_score: float | None = None
    steps: list[PipelineStepResult] = Field(default_factory=list)
    final_manuscript_md: str = ""
    final_manuscript_tex: str = ""
    review_verdict_path: str = ""
    manifest_path: str = ""
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
