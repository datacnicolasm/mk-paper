"""Subcomando CLI ``audit`` — Quality Auditor Q1-Q2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mk_paper.config.settings import get_settings
from mk_paper.crew.auditor_crew import parse_crew_audit_output, run_audit_crew
from mk_paper.models.audit_brief import AuditBrief
from mk_paper.persistence.audit_store import save_audit_verdict
from mk_paper.tools.auditor_tools import run_quality_audit

logger = logging.getLogger(__name__)


def add_audit_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``audit``."""
    parser.add_argument(
        "--brief",
        default=None,
        help="Ruta a AuditBrief JSON.",
    )
    parser.add_argument(
        "--draft",
        default=None,
        help="Markdown IMRaD a auditar (override paper_draft_path).",
    )
    parser.add_argument(
        "--writing-brief",
        default=None,
        help="WritingBrief para regenerar/revisar vía feedback loop.",
    )
    parser.add_argument(
        "--literature",
        default=None,
        help="Override literature_review_path.",
    )
    parser.add_argument(
        "--analysis",
        default=None,
        help="Override analysis_report_path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral de calidad (default 8.5).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Máximo de rondas Writer↔Auditor.",
    )
    parser.add_argument(
        "--via-crew",
        action="store_true",
        help="Ejecuta vía CrewAI (por defecto: motor directo).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Auditoría/revisión determinista sin LLM.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="No generar LaTeX final.",
    )


def _load_brief(args: argparse.Namespace) -> AuditBrief:
    if args.brief:
        raw = json.loads(Path(args.brief).read_text(encoding="utf-8"))
    else:
        raw = {"title": "Q1-Q2 quality audit"}
    if args.draft:
        raw["paper_draft_path"] = args.draft
    if args.writing_brief:
        raw["writing_brief_path"] = args.writing_brief
    if args.literature:
        raw["literature_review_path"] = args.literature
    if args.analysis:
        raw["analysis_report_path"] = args.analysis
    if args.threshold is not None:
        raw["quality_threshold"] = args.threshold
    if args.max_rounds is not None:
        raw["max_revision_rounds"] = args.max_rounds
    if args.no_llm:
        raw["use_llm"] = False
    if args.no_latex:
        raw["include_latex"] = False
    return AuditBrief.model_validate(raw)


def _print_summary(verdict_title: str, decision: str, score: float, paths: list[str]) -> None:
    print()
    print("=" * 72)
    print(f"Title    : {verdict_title}")
    print(f"Decision : {decision}")
    print(f"Score    : {score}/10")
    for p in paths:
        print(f"Output   : {p}")
    print("=" * 72)


def run_audit_cli(args: argparse.Namespace) -> int:
    """Handler del subcomando audit."""
    settings = get_settings()
    brief = _load_brief(args)

    if args.via_crew:
        raw = run_audit_crew(brief)
        verdict = parse_crew_audit_output(raw)
        if verdict is None:
            logger.warning("Crew output not parseable; falling back to direct engine")
            verdict = run_quality_audit(brief, settings=settings)
    else:
        verdict = run_quality_audit(brief, settings=settings)

    artifacts = save_audit_verdict(verdict, output_dir=settings.output_dir)
    paths = [str(artifacts.verdict_json), str(artifacts.polished_md)]
    if artifacts.polished_tex:
        paths.append(str(artifacts.polished_tex))
    _print_summary(verdict.title, verdict.decision, verdict.overall_score, paths)
    if verdict.decision == "accept":
        return 0
    if verdict.decision == "revise":
        return 3
    return 2
