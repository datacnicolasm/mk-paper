"""Tasks del Literature Reviewer."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.research_brief import ResearchBrief


def create_literature_review_task(
    agent: Agent,
    brief: ResearchBrief,
) -> Task:
    """Crea la task de revisión sistemática alimentada por un ResearchBrief.

    Args:
        agent: Literature Reviewer configurado.
        brief: Contexto estructurado de investigación.

    Returns:
        Task CrewAI que exige JSON clasificado Core vs Conceptual.
    """
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Ejecuta una revisión sistemática completa usando la tool "
            "`Run Systematic Literature Review`.\n\n"
            "INPUT — ResearchBrief (pásalo íntegro como research_brief_json):\n"
            f"{brief_json}\n\n"
            "Debes:\n"
            "1) Invocar la tool con ese JSON.\n"
            "2) No inventar DOIs ni papers fuera del resultado de la tool.\n"
            "3) Devolver el JSON Writer-ready (core_findings + conceptual_references "
            "+ seminal_literature).\n"
            "4) Conservar alignment_score / alignment_quadrant / historical_centrality.\n"
            "5) Tratar seminal_literature solo como marco teórico (nunca como Nivel 1).\n"
        ),
        expected_output=(
            "Un único objeto JSON con keys: brief_title, search_matrix, "
            "primary_sources_used, core_findings, conceptual_references, "
            "seminal_literature, discarded_count, alignment_thresholds, warnings, "
            "candidate_count. Cada paper debe incluir doi, utility, level, "
            "alignment_score y citation_context; los seminales además "
            "historical_centrality y seminal_reason."
        ),
        agent=agent,
    )
