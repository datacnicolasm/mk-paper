"""Tasks del Quality Auditor Q1-Q2."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.audit_brief import AuditBrief


def create_audit_task(agent: Agent, brief: AuditBrief) -> Task:
    """Crea la task de auditoría con umbral y feedback loop.

    Args:
        agent: Quality Auditor configurado.
        brief: AuditBrief con paths y quality_threshold.

    Returns:
        Task CrewAI que exige AuditVerdict JSON.
    """
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Audita el manuscrito IMRaD con estándar Q1-Q2 y, si el puntaje "
            f"es < {brief.quality_threshold}/10, devuelve el borrador al "
            "Scientific Writer con AuditFeedback estructurado hasta alcanzar "
            "el umbral o agotar rondas.\n\n"
            "INPUT — AuditBrief (pásalo íntegro como audit_brief_json):\n"
            f"{brief_json}\n\n"
            "Workflow:\n"
            "1) Opcional: `Run Structural Quality Checks` sobre el markdown.\n"
            "2) `Evaluate Manuscript Quality` con el AuditBrief.\n"
            "3) Preferente: `Run Audit With Writer Feedback Loop` (orquesta "
            "revisión automática del Writer).\n"
            "4) Si accept: `Polish Final Manuscript`.\n\n"
            "Criterios duros:\n"
            "- Cero viñetas estructurales (objetivos, RQ, método en prosa).\n"
            "- Tono analítico; sin marketing.\n"
            "- Citas [@cite_key]; sin afirmaciones empíricas huérfanas.\n"
            "- Coherencia problema ↔ modelo cuantitativo ↔ conclusiones.\n"
            "- Output: AuditVerdict JSON completo.\n"
        ),
        expected_output=(
            "JSON AuditVerdict con decision (accept|revise|reject), "
            "overall_score, feedback, polished_markdown, warnings, status."
        ),
        agent=agent,
    )
