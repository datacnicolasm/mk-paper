"""Subcomando CLI ``run-pipeline`` — orquestación end-to-end."""

from __future__ import annotations

import argparse
import logging

from mk_paper.config.settings import get_settings
from mk_paper.crew.main_pipeline import run_pipeline
from mk_paper.models.pipeline import PipelineConfig
from mk_paper.runtime import ensure_directories, setup_logging

logger = logging.getLogger(__name__)


def add_pipeline_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``run-pipeline``."""
    parser.add_argument(
        "--research-brief",
        required=True,
        help="Ruta al ResearchBrief JSON (Literature Reviewer).",
    )
    parser.add_argument(
        "--method-brief",
        required=True,
        help="Ruta al MethodBrief JSON (Quantitative Analyst).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override del dataset local CSV/XLSX del MethodBrief.",
    )
    parser.add_argument(
        "--literature-review",
        default=None,
        help=(
            "Usa un review.json existente y salta la búsqueda en APIs "
            "(equivalente a --skip-literature)."
        ),
    )
    parser.add_argument(
        "--skip-literature",
        action="store_true",
        help="No ejecuta búsqueda; requiere --literature-review.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Título del manuscrito / carpeta de run.",
    )
    parser.add_argument(
        "--language",
        choices=("es", "en"),
        default="es",
        help="Idioma del paper (default: es).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.5,
        help="Umbral del Quality Auditor (default: 8.5).",
    )
    parser.add_argument(
        "--max-audit-rounds",
        type=int,
        default=2,
        help="Rondas máximas Writer↔Auditor (default: 2).",
    )
    parser.add_argument(
        "--literature-max-results",
        type=int,
        default=None,
        help="Override de max_results del ResearchBrief.",
    )
    parser.add_argument(
        "--via-crew",
        action="store_true",
        help="Ejecuta cada paso vía CrewAI cuando sea posible.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Writer/Auditor deterministas (sin Groq). La literatura aún puede usar APIs.",
    )
    parser.add_argument(
        "--enrich-analysis-discussion",
        action="store_true",
        help="Activa enriquecimiento LLM de la discusión analítica.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="No generar export LaTeX final.",
    )


def run_pipeline_cli(args: argparse.Namespace) -> int:
    """Handler del subcomando run-pipeline."""
    settings = get_settings()
    setup_logging(settings.log_level)
    ensure_directories(
        settings.data_dir,
        settings.workspace_dir,
        settings.output_dir,
    )

    skip_lit = bool(args.skip_literature) or bool(args.literature_review)
    if args.skip_literature and not args.literature_review:
        raise SystemExit("--skip-literature requiere --literature-review PATH")

    config = PipelineConfig(
        research_brief_path=args.research_brief,
        method_brief_path=args.method_brief,
        dataset_path=args.dataset,
        literature_review_path=args.literature_review,
        skip_literature=skip_lit,
        title=args.title,
        language=args.language,
        quality_threshold=float(args.threshold),
        max_audit_rounds=int(args.max_audit_rounds),
        use_llm=not bool(args.no_llm),
        via_crew=bool(args.via_crew),
        enrich_analysis_discussion=bool(args.enrich_analysis_discussion),
        include_latex=not bool(args.no_latex),
        literature_max_results=args.literature_max_results,
    )

    logger.info(
        "Starting pipeline research=%s method=%s dataset=%s skip_lit=%s",
        config.research_brief_path,
        config.method_brief_path,
        config.dataset_path,
        config.skip_literature,
    )
    result = run_pipeline(config, settings=settings)

    print()
    print("=" * 72)
    print(f"Run ID   : {result.run_id}")
    print(f"Run dir  : {result.run_dir}")
    print(f"Status   : {result.status}")
    print(f"Decision : {result.decision}")
    if result.overall_score is not None:
        print(f"Score    : {result.overall_score}/10")
    print(f"Manifest : {result.manifest_path}")
    if result.final_manuscript_md:
        print(f"Final MD : {result.final_manuscript_md}")
    if result.final_manuscript_tex:
        print(f"Final TEX: {result.final_manuscript_tex}")
    if result.review_verdict_path:
        print(f"Verdict  : {result.review_verdict_path}")
    print("Steps:")
    for step in result.steps:
        print(f"  - [{step.status}] {step.step}: {step.message}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings[:12]:
            print(f"  - {w}")
    print("=" * 72)

    if result.status == "ok" and result.decision == "accept":
        return 0
    if result.status == "partial" or result.decision == "revise":
        return 3
    return 2
