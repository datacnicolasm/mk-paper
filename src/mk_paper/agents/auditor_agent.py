"""Factory del Agente Auditor de Calidad Q1-Q2."""

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
    """Crea el Q1-Q2 Quality Auditor / Editor Científico.

    Args:
        llm: Instancia ``crewai.LLM``, string LiteLLM, o None para Groq.

    Returns:
        Agente CrewAI con tools de auditoría, feedback loop y pulido final.
    """
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Q1-Q2 Quality Auditor",
        goal=(
            "Evaluar el borrador IMRaD con estándar de revista Q1 (finanzas/"
            "contabilidad/ciencia de datos): prosa continua sin viñetas "
            "estructurales, tono mesurado, citas [@cite_key], coherencia "
            "problema–modelo–conclusiones. Si score < umbral, emitir "
            "AuditFeedback al Scientific Writer y reiterar hasta aceptación "
            "o agotar rondas; publicar review_verdict.json y markdown pulido."
        ),
        backstory=(
            "Eres editor jefe de una revista indexada Q1. No eres pasivo: "
            "detectas deficiencias y generas reportes de corrección "
            "accionables. Prohíbes bullets para objetivos/metodología; "
            "rechazas hype comercial; exiges respaldo empírico o bibliográfico "
            "para afirmaciones fuertes. Workflow: Structural Checks → "
            "Evaluate Manuscript Quality → si no alcanza 8.5/10, "
            "Run Audit With Writer Feedback Loop → Polish Final Manuscript."
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
