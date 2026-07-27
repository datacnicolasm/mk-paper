"""Crew mínimo del Literature Reviewer."""

from __future__ import annotations

from crewai import Crew, Process

from mk_paper.agents.researcher_agent import create_literature_reviewer
from mk_paper.models.research_brief import LiteratureReviewOutput, ResearchBrief
from mk_paper.tasks.literature_tasks import create_literature_review_task


def build_literature_crew(brief: ResearchBrief) -> Crew:
    """Construye un Crew secuencial de 1 agente / 1 task para el brief dado."""
    agent = create_literature_reviewer()
    task = create_literature_review_task(agent, brief)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def run_literature_crew(brief: ResearchBrief) -> str:
    """Ejecuta el crew y retorna el texto/JSON resultante.

    Args:
        brief: ResearchBrief estructurado.

    Returns:
        Output crudo del crew (idealmente JSON Writer-ready).
    """
    crew = build_literature_crew(brief)
    result = crew.kickoff()
    return str(result)


def parse_crew_review_output(raw: str) -> LiteratureReviewOutput | None:
    """Intenta parsear la salida del crew a LiteratureReviewOutput."""
    import json
    import re

    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return LiteratureReviewOutput.model_validate_json(text[start : end + 1])
    except Exception:  # noqa: BLE001
        try:
            return LiteratureReviewOutput.model_validate(json.loads(text[start : end + 1]))
        except Exception:  # noqa: BLE001
            return None
