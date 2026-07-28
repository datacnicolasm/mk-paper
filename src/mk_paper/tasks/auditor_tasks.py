"""Tasks del Quality Auditor (lit-writer)."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.audit_brief import AuditBrief


def create_audit_task(agent: Agent, brief: AuditBrief) -> Task:
    """Crea la task de auditoría con umbral y feedback loop."""
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Audita el manuscrito de revisión de literatura con estándar Q1-Q2 y, "
            f"si el puntaje es < {brief.quality_threshold}/10, devuelve el "
            "borrador al Writer con AuditFeedback hasta alcanzar el umbral o "
            "agotar rondas.\n\n"
            "INPUT — AuditBrief (pásalo íntegro como audit_brief_json):\n"
            f"{brief_json}\n\n"
            "Workflow:\n"
            "1) Opcional: `Run Structural Quality Checks`.\n"
            "2) `Evaluate Manuscript Quality`.\n"
            "3) Preferente: `Run Audit With Writer Feedback Loop`.\n"
            "4) Si accept: `Polish Final Manuscript`.\n\n"
            "Criterios duros:\n"
            "- Secciones: Introducción, Revisión de Literatura/Marco Teórico, Referencias.\n"
            "- Intro cierra con pregunta de investigación.\n"
            "- Cero viñetas estructurales.\n"
            "- Cero mezcla de idiomas (español).\n"
            "- Cero JSON/código/metadatos internos.\n"
            "- Citas [@cite_key] válidas.\n"
            "- Output: AuditVerdict JSON completo.\n"
        ),
        expected_output=(
            "JSON AuditVerdict con decision (accept|revise|reject), "
            "overall_score, feedback, polished_markdown, warnings, status."
        ),
        agent=agent,
    )
