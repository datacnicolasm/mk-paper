"""Subcomando CLI ``paper`` — Scientific Writer (IMRaD + APA 7)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mk_paper.config.settings import get_settings
from mk_paper.crew.writer_crew import parse_crew_paper_output, run_writer_crew
from mk_paper.models.writing_brief import WritingBrief
from mk_paper.persistence.paper_store import save_paper_draft
from mk_paper.tools.writer_tools import draft_imrad_paper

logger = logging.getLogger(__name__)


def add_paper_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``paper``."""
    parser.add_argument(
        "--brief",
        default=None,
        help="Ruta a WritingBrief JSON (alternativa a --literature/--analysis).",
    )
    parser.add_argument(
        "--literature",
        default=None,
        help="Ruta a review.json del Literature Reviewer.",
    )
    parser.add_argument(
        "--analysis",
        default=None,
        help="Ruta a AnalysisReport JSON (p.ej. output/analysis/latest_report.json).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Título del manuscrito (override).",
    )
    parser.add_argument(
        "--language",
        choices=("es", "en"),
        default=None,
        help="Idioma del draft (default: es o el del brief).",
    )
    parser.add_argument(
        "--via-crew",
        action="store_true",
        help="Ejecuta vía CrewAI (por defecto: motor directo).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Borrador determinista sin llamar al LLM (skeleton + APA).",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="No generar export LaTeX.",
    )


def _load_brief(args: argparse.Namespace) -> WritingBrief:
    if args.brief:
        raw = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        if args.literature:
            raw["literature_review_path"] = args.literature
        if args.analysis:
            raw["analysis_report_path"] = args.analysis
        if args.title:
            raw["title"] = args.title
        if args.language:
            raw["language"] = args.language
        if args.no_latex:
            raw["include_latex"] = False
        return WritingBrief.model_validate(raw)

    if not args.literature or not args.analysis:
        raise ValueError(
            "Provide --brief or both --literature and --analysis."
        )
    title = args.title or "Untitled scientific manuscript"
    return WritingBrief(
        title=title,
        literature_review_path=args.literature,
        analysis_report_path=args.analysis,
        language=args.language or "es",
        include_latex=not bool(args.no_latex),
    )


def _print_summary(draft_title: str, status: str, paths: list[str], warnings: list[str]) -> None:
    print()
    print("=" * 72)
    print(f"Title  : {draft_title}")
    print(f"Status : {status}")
    for p in paths:
        print(f"Output : {p}")
    if warnings:
        print("Warnings:")
        for w in warnings[:10]:
            print(f"  - {w}")
    print("=" * 72)


def run_paper_cli(args: argparse.Namespace) -> int:
    """Handler del subcomando paper."""
    settings = get_settings()
    brief = _load_brief(args)

    catalog = None
    if args.via_crew:
        raw = run_writer_crew(brief)
        draft = parse_crew_paper_output(raw)
        if draft is None:
            logger.warning("Crew output not parseable; falling back to direct engine")
            draft, catalog = draft_imrad_paper(
                brief, settings=settings, use_llm=not bool(args.no_llm)
            )
    else:
        draft, catalog = draft_imrad_paper(
            brief, settings=settings, use_llm=not bool(args.no_llm)
        )

    artifacts = save_paper_draft(
        draft, catalog, output_dir=settings.output_dir
    )
    paths = [str(artifacts.draft_md), str(artifacts.draft_json)]
    if artifacts.draft_tex:
        paths.append(str(artifacts.draft_tex))
    _print_summary(draft.title, draft.status, paths, draft.warnings)
    return 0 if draft.status != "error" else 2
