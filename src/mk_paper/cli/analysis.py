"""Subcomando CLI ``analysis`` — Quantitative Analyst determinista."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mk_paper.config.settings import get_settings
from mk_paper.crew.analysis_crew import parse_crew_analysis_output, run_analysis_crew
from mk_paper.models.method_brief import AnalysisReport, MethodBrief
from mk_paper.persistence.analysis_store import save_analysis_report
from mk_paper.tools.analysis_tools import run_quantitative_analysis

logger = logging.getLogger(__name__)


def add_analysis_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``analysis``."""
    parser.add_argument(
        "--brief",
        required=True,
        help="Ruta al MethodBrief JSON (metodología estructurada).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override opcional de dataset_path del brief (CSV/XLSX local).",
    )
    parser.add_argument(
        "--literature",
        default=None,
        help=(
            "Override de literature_review_path (review.json o review.md del "
            "Literature Reviewer, bajo data/workspace/output)."
        ),
    )
    parser.add_argument(
        "--via-crew",
        action="store_true",
        help="Ejecuta vía CrewAI (por defecto: motor directo).",
    )
    parser.add_argument(
        "--no-llm-discussion",
        action="store_true",
        help="Desactiva enriquecimiento Groq de la Analytical Discussion.",
    )


def _load_brief(
    path: str,
    dataset_override: str | None,
    literature_override: str | None,
    *,
    no_llm_discussion: bool = False,
) -> MethodBrief:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset_override:
        raw["dataset_path"] = dataset_override
    if literature_override:
        raw["literature_review_path"] = literature_override
    if no_llm_discussion:
        raw["enrich_discussion_with_llm"] = False
    return MethodBrief.model_validate(raw)


def _print_summary(report: AnalysisReport) -> None:
    print()
    print("=" * 72)
    print(f"Brief : {report.brief_title}")
    print(f"Task  : {report.task_type}")
    print(f"Data  : {report.dataset_path}")
    print(
        f"Best  : {report.best_model}  |  "
        f"{report.primary_metric}={report.best_score}"
    )
    print(
        f"Models: {len(report.model_results)}  |  "
        f"Iterations logged: {len(report.iteration_log)}  |  "
        f"Lit benchmarks: {len(report.literature_benchmarking)}"
    )
    if report.analytical_discussion:
        print("\n## Analytical Discussion (excerpt)")
        print(report.analytical_discussion[:600])
        if len(report.analytical_discussion) > 600:
            print("...")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings[:8]:
            print(f"  - {warning}")
    print("=" * 72)
    for m in report.model_results:
        print(
            f"\n[{m.model_id}] params={m.best_params}  "
            f"test={m.metrics_test}  iters={m.n_iterations}"
        )


def run_analysis_cli(args: argparse.Namespace) -> int:
    """Handler del subcomando analysis."""
    settings = get_settings()
    brief = _load_brief(
        args.brief,
        args.dataset,
        args.literature,
        no_llm_discussion=bool(args.no_llm_discussion),
    )

    if args.via_crew:
        raw = run_analysis_crew(brief)
        report = parse_crew_analysis_output(raw)
        if report is None:
            logger.warning("Crew output not parseable; falling back to direct engine")
            report = run_quantitative_analysis(brief, settings=settings)
    else:
        report = run_quantitative_analysis(brief, settings=settings)

    artifacts = save_analysis_report(report, output_dir=settings.output_dir)
    _print_summary(report)
    print(f"\nSaved: {artifacts.run_dir}")
    print(f"JSON : {artifacts.json_path}")
    print(f"MD   : {artifacts.md_path}")
    return 0 if report.best_model else 1
