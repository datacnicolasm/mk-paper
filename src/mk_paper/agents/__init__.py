"""Factories de agentes CrewAI."""

from mk_paper.agents.analyst_agent import create_quantitative_analyst
from mk_paper.agents.auditor_agent import create_quality_auditor
from mk_paper.agents.researcher_agent import create_literature_reviewer
from mk_paper.agents.writer_agent import create_scientific_writer

__all__ = [
    "create_literature_reviewer",
    "create_quantitative_analyst",
    "create_scientific_writer",
    "create_quality_auditor",
]
