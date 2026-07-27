"""Crew del Quality Auditor Q1-Q2."""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import Crew, Process

from mk_paper.agents.auditor_agent import create_quality_auditor
from mk_paper.models.audit_brief import AuditBrief, AuditVerdict
from mk_paper.tasks.auditor_tasks import create_audit_task

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_audit_crew(brief: AuditBrief) -> Crew:
    """Construye un crew secuencial de 1 agente / 1 task."""
    agent = create_quality_auditor()
    task = create_audit_task(agent, brief)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def run_audit_crew(brief: AuditBrief) -> str:
    """Ejecuta el crew y retorna el texto crudo del resultado."""
    result = build_audit_crew(brief).kickoff()
    return str(result)


def parse_crew_audit_output(raw: str) -> AuditVerdict | None:
    """Intenta parsear la salida del crew a AuditVerdict."""
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
        return AuditVerdict.model_validate(data)
    except Exception:  # noqa: BLE001
        return None
