"""Crews CrewAI del proyecto."""

from mk_paper.crew.analysis_crew import (
    build_analysis_crew,
    parse_crew_analysis_output,
    run_analysis_crew,
)
from mk_paper.crew.auditor_crew import (
    build_audit_crew,
    parse_crew_audit_output,
    run_audit_crew,
)
from mk_paper.crew.literature_crew import (
    build_literature_crew,
    parse_crew_review_output,
    run_literature_crew,
)
from mk_paper.crew.main_pipeline import run_pipeline
from mk_paper.crew.writer_crew import (
    build_writer_crew,
    parse_crew_paper_output,
    run_writer_crew,
)

__all__ = [
    "build_literature_crew",
    "run_literature_crew",
    "parse_crew_review_output",
    "build_analysis_crew",
    "run_analysis_crew",
    "parse_crew_analysis_output",
    "build_writer_crew",
    "run_writer_crew",
    "parse_crew_paper_output",
    "build_audit_crew",
    "run_audit_crew",
    "parse_crew_audit_output",
    "run_pipeline",
]
