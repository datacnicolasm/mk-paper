"""Subcomando CLI para revisión sistemática de literatura."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mk_paper.config.settings import get_settings
from mk_paper.crew.literature_crew import parse_crew_review_output, run_literature_crew
from mk_paper.models.research_brief import ResearchBrief
from mk_paper.persistence.literature_store import (
    save_literature_results,
    save_literature_review,
)
from mk_paper.runtime import ensure_directories, setup_logging
from mk_paper.tools.literature_tools import _run_literature_search
from mk_paper.tools.systematic_review import run_systematic_review

logger = logging.getLogger(__name__)


def add_literature_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``literature``."""
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query suelta (modo legado). Preferir --brief.",
    )
    parser.add_argument(
        "--brief",
        type=str,
        default=None,
        help="Ruta a ResearchBrief JSON (modo principal del embudo sistemático).",
    )
    parser.add_argument(
        "--via-crew",
        action="store_true",
        help="Ejecuta el Literature Reviewer vía CrewAI (además de Groq en tools).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Override de max_results del brief (1-150).",
    )
    parser.add_argument(
        "--raw-search",
        action="store_true",
        help="Solo búsqueda API cruda (sin Groq/clasificación). Requiere query.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Solo imprime en consola; no persiste artefactos.",
    )
    parser.add_argument(
        "--quiet-json",
        action="store_true",
        help="No imprime el JSON completo; solo el resumen.",
    )


def _load_brief(args: argparse.Namespace) -> ResearchBrief:
    """Carga ResearchBrief desde --brief o envuelve query suelta."""
    if args.brief:
        path = Path(args.brief)
        data = json.loads(path.read_text(encoding="utf-8"))
        brief = ResearchBrief.model_validate(data)
        if args.limit is not None:
            brief = brief.model_copy(
                update={"max_results": max(1, min(int(args.limit), 150))}
            )
        return brief

    if not args.query:
        raise SystemExit(
            "Debes pasar --brief path/to/brief.json o una query posicional."
        )

    logger.warning(
        "Query suelta detectada: se envuelve en ResearchBrief mínimo. "
        "Preferir --brief para revisión sistemática rigurosa."
    )
    max_results = args.limit if args.limit is not None else 8
    return ResearchBrief.from_loose_query(args.query, max_results=max_results)


def _print_review_summary(payload: dict[str, Any]) -> None:
    """Imprime resumen Core vs Conceptual vs Seminal."""
    core = payload.get("core_findings") or []
    conceptual = payload.get("conceptual_references") or []
    seminal = payload.get("seminal_literature") or []
    warnings = payload.get("warnings") or []
    thr = payload.get("alignment_thresholds") or {}

    print()
    print("=" * 72)
    print(f"Brief : {payload.get('brief_title')}")
    print(
        f"Core  : {len(core)}  |  Conceptual: {len(conceptual)}  |  "
        f"Seminal: {len(seminal)}"
    )
    print(
        f"Screened: {payload.get('candidate_count', 0)}  |  "
        f"Discarded (low alignment): {payload.get('discarded_count', 0)}"
    )
    if thr:
        print(f"Alignment thresholds: high>={thr.get('high')} low<{thr.get('low')}")
    print(f"Sources: {', '.join(payload.get('primary_sources_used') or []) or 'n/a'}")
    if warnings:
        print("Warnings:")
        for warning in warnings[:8]:
            print(f"  - {warning}")
    print("=" * 72)

    print("\n## Core Findings")
    if not core:
        print("(none)")
    for idx, paper in enumerate(core, start=1):
        print(f"\n[{idx}] {paper.get('title')}")
        print(
            f"    doi={paper.get('doi')}  utility={paper.get('utility')}  "
            f"alignment={paper.get('alignment_score')}  year={paper.get('year')}"
        )
        print(f"    cite: {paper.get('citation_context')}")

    print("\n## Conceptual References")
    if not conceptual:
        print("(none)")
    for idx, paper in enumerate(conceptual, start=1):
        print(f"\n[{idx}] {paper.get('title')}")
        print(
            f"    doi={paper.get('doi')}  utility={paper.get('utility')}  "
            f"alignment={paper.get('alignment_score')}  year={paper.get('year')}"
        )
        print(f"    cite: {paper.get('citation_context')}")

    print("\n## Fundamentos Teóricos e Históricos (Seminal Literature)")
    if not seminal:
        print("(none)")
    for idx, paper in enumerate(seminal, start=1):
        print(f"\n[{idx}] {paper.get('title')}")
        print(
            f"    doi={paper.get('doi')}  reason={paper.get('seminal_reason')}  "
            f"centrality={paper.get('historical_centrality')}  "
            f"year={paper.get('year')}"
        )
        print(f"    cite: {paper.get('citation_context')}")


def _print_raw_summary(payload: dict[str, Any]) -> None:
    """Resumen del modo --raw-search."""
    papers = payload.get("papers") or []
    print()
    print("=" * 72)
    print(f"Query : {payload.get('query')}")
    print(f"Count : {payload.get('count', 0)}")
    print(f"Source: {payload.get('primary_source') or 'n/a'}")
    print("=" * 72)
    for idx, paper in enumerate(papers, start=1):
        print(f"\n[{idx}] {paper.get('title')}")
        print(f"    doi={paper.get('doi')}  pdf={paper.get('pdf_url') or '—'}")


def run_literature_search_cli(args: argparse.Namespace) -> int:
    """Ejecuta embudo sistemático (default) o búsqueda cruda."""
    settings = get_settings()
    setup_logging(settings.log_level)
    ensure_directories(
        settings.data_dir,
        settings.workspace_dir,
        settings.output_dir,
    )

    if args.raw_search:
        if not args.query:
            raise SystemExit("--raw-search requiere una query posicional.")
        limit = args.limit if args.limit is not None else 5
        raw = asyncio.run(_run_literature_search(args.query, limit))
        payload = json.loads(raw)
        _print_raw_summary(payload)
        if not args.quiet_json:
            print("\n--- JSON ---")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not args.no_save:
            artifacts = save_literature_results(
                payload, output_dir=settings.output_dir, query=args.query
            )
            print("\n--- Persistencia ---")
            print(f"results.json: {artifacts.json_path}")
        return 0

    brief = _load_brief(args)
    logger.info(
        "Systematic review brief=%r via_crew=%s max_results=%s",
        brief.title,
        args.via_crew,
        brief.max_results,
    )

    if args.via_crew:
        raw = run_literature_crew(brief)
        review = parse_crew_review_output(raw)
        if review is None:
            # Fallback: pipeline directo si el crew no devolvió JSON parseable.
            logger.warning("Crew output not parseable; falling back to direct pipeline")
            review = asyncio.run(run_systematic_review(brief))
            review.warnings.append("Crew output was not valid JSON; used direct pipeline.")
    else:
        review = asyncio.run(run_systematic_review(brief))

    payload = review.to_dict()
    _print_review_summary(payload)

    if not args.quiet_json:
        print("\n--- JSON ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not args.no_save:
        artifacts = save_literature_review(review, output_dir=settings.output_dir)
        print("\n--- Persistencia ---")
        print(f"run_dir        : {artifacts.run_dir}")
        print(f"review.json    : {artifacts.json_path}")
        print(f"review.md      : {artifacts.md_path}")
        print(f"latest_review  : {artifacts.latest_json}")

    return 0
