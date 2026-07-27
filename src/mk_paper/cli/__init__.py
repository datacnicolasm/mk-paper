"""CLI de mk-paper: componentes aislados + pipeline end-to-end."""

from __future__ import annotations

import argparse
import sys

from mk_paper.cli.analysis import add_analysis_parser, run_analysis_cli
from mk_paper.cli.audit import add_audit_parser, run_audit_cli
from mk_paper.cli.literature import add_literature_parser, run_literature_search_cli
from mk_paper.cli.paper import add_paper_parser, run_paper_cli
from mk_paper.cli.pipeline import add_pipeline_parser, run_pipeline_cli


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser raíz con subcomandos."""
    parser = argparse.ArgumentParser(
        prog="mk-paper",
        description=(
            "CLI de mk-paper: literature, analysis, paper, audit y run-pipeline E2E."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    literature_parser = subparsers.add_parser(
        "literature",
        help=(
            "Revisión sistemática (ResearchBrief + Groq) o búsqueda cruda; "
            "persiste en output/literature/"
        ),
    )
    add_literature_parser(literature_parser)
    literature_parser.set_defaults(handler=run_literature_search_cli)

    analysis_parser = subparsers.add_parser(
        "analysis",
        help=(
            "Análisis cuantitativo determinista (MethodBrief + CSV/XLSX local); "
            "persiste en output/analysis/"
        ),
    )
    add_analysis_parser(analysis_parser)
    analysis_parser.set_defaults(handler=run_analysis_cli)

    paper_parser = subparsers.add_parser(
        "paper",
        help=(
            "Redacción IMRaD + APA 7 (WritingBrief / literature+analysis); "
            "persiste en output/paper/"
        ),
    )
    add_paper_parser(paper_parser)
    paper_parser.set_defaults(handler=run_paper_cli)

    audit_parser = subparsers.add_parser(
        "audit",
        help=(
            "Auditoría Q1-Q2 del manuscrito IMRaD con feedback loop al Writer; "
            "persiste en output/audit/"
        ),
    )
    add_audit_parser(audit_parser)
    audit_parser.set_defaults(handler=run_audit_cli)

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help=(
            "Pipeline E2E Literature → Analysis → Writer → Auditor "
            "(output/runs/{timestamp}_paper_run/)"
        ),
    )
    add_pipeline_parser(pipeline_parser)
    pipeline_parser.set_defaults(handler=run_pipeline_cli)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
