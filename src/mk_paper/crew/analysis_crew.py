"""Crew del Quantitative Analyst."""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import Crew, Process

from mk_paper.agents.analyst_agent import create_quantitative_analyst
from mk_paper.models.method_brief import AnalysisReport, MethodBrief
from mk_paper.tasks.analysis_tasks import create_analysis_task

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_analysis_crew(brief: MethodBrief) -> Crew:
    """Construye un crew secuencial de 1 agente / 1 task."""
    agent = create_quantitative_analyst()
    task = create_analysis_task(agent, brief)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def run_analysis_crew(brief: MethodBrief) -> str:
    """Ejecuta el crew y retorna el texto crudo del resultado."""
    result = build_analysis_crew(brief).kickoff()
    return str(result)


def parse_crew_analysis_output(raw: str) -> AnalysisReport | None:
    """Intenta parsear la salida del crew a AnalysisReport."""
    text = (raw or "").strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data: dict[str, Any] = json.loads(text[start : end + 1])
        return AnalysisReport.model_validate(data)
    except Exception:  # noqa: BLE001
        return None
