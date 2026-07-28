"""Factory del Agente Auditor de Calidad Q1-Q2 (lit-writer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.auditor_tools import (
    evaluate_manuscript_quality_tool,
    polish_final_manuscript_tool,
    run_audit_with_feedback_loop_tool,
    run_structural_quality_checks_tool,
)


def create_quality_auditor(llm: LLM | str | None = None) -> Agent:
    """Crea el Quality Auditor para manuscritos de revisión de literatura."""
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Q1-Q2 Quality Auditor",
        goal=(
            "Auditar manuscritos de Introducción + Revisión de Literatura con "
            "estándar editorial Q1-Q2: bloquear viñetas, fugas JSON/metadatos, "
            "mezcla de idiomas; exigir pregunta de investigación al cierre de "
            "la Intro y citas Pandoc válidas antes de aceptar."
        ),
        backstory=(
            "Eres editor jefe de una revista indexada Q1. Detectas deficiencias "
            "y generas reportes accionables. Exiges prosa continua, español "
            "estricto, secciones Introducción / Revisión de Literatura / "
            "Referencias, y cierre de la Intro con la pregunta de investigación. "
            "Workflow: Structural Checks → Evaluate → Feedback Loop → Polish."
        ),
        tools=[
            run_structural_quality_checks_tool,
            evaluate_manuscript_quality_tool,
            run_audit_with_feedback_loop_tool,
            polish_final_manuscript_tool,
        ],
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
