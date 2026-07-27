"""Crew del Scientific Writer."""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import Crew, Process

from mk_paper.agents.writer_agent import create_scientific_writer
from mk_paper.models.writing_brief import PaperDraft, WritingBrief
from mk_paper.tasks.writer_tasks import create_writer_task

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_writer_crew(brief: WritingBrief) -> Crew:
    """Construye un crew secuencial de 1 agente / 1 task."""
    agent = create_scientific_writer()
    task = create_writer_task(agent, brief)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def run_writer_crew(brief: WritingBrief) -> str:
    """Ejecuta el crew y retorna el texto crudo del resultado."""
    result = build_writer_crew(brief).kickoff()
    return str(result)


def parse_crew_paper_output(raw: str) -> PaperDraft | None:
    """Intenta parsear la salida del crew a PaperDraft."""
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
        return PaperDraft.model_validate(data)
    except Exception:  # noqa: BLE001
        return None
